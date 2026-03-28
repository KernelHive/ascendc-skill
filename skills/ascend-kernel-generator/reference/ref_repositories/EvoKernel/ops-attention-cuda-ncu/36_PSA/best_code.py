import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float max2f(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float max4f(float a, float b, float c, float d) { return max2f(max2f(a,b), max2f(c,d)); }

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float shfl_broadcast(float v, int src_lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    return __shfl_sync(0xffffffff, v, src_lane);
#else
    return v;
#endif
}

// Specialized kernel for the dominant PSA config: S=4, H=W=7 (HW=49), Ci=128.
// One warp handles one (b,ci) and streams 49 pixels; persistent-warp loop over bci.
// Designed to minimize dynamic indexing/register pressure.
__global__ __launch_bounds__(128, 2) void psa_warp_s4_hw49_ci128(
    const float* __restrict__ spc,   // [B,4,128,7,7]
    const float* __restrict__ se,    // [B,4,128,1,1]
    float* __restrict__ out,         // [B,512,7,7]
    int B
) {
    constexpr int S = 4;
    constexpr int Ci = 128;
    constexpr int HW = 49;

    const int lane = (int)(threadIdx.x & 31);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_id_in_block = (int)(threadIdx.x >> 5);
    const int global_warp = (int)blockIdx.x * warps_per_block + warp_id_in_block;

    const int n_items = B * Ci;
    const int stride_grid = (int)gridDim.x * warps_per_block;

    // strides in elements
    const long long spc_stride_b = (long long)S * (long long)Ci * (long long)HW;
    const long long spc_stride_s = (long long)Ci * (long long)HW;
    const long long out_stride_b = (long long)(S * Ci) * (long long)HW;

    for (int bci = global_warp; bci < n_items; bci += stride_grid) {
        const int b = bci >> 7;          // /128
        const int ci = bci & (Ci - 1);   // %128

        float w0=0.f, w1=0.f, w2=0.f, w3=0.f;
        if (lane == 0) {
            const int se_base = b * (S * Ci) + ci;
            const float v0 = ldg_f32(se + se_base + 0 * Ci);
            const float v1 = ldg_f32(se + se_base + 1 * Ci);
            const float v2 = ldg_f32(se + se_base + 2 * Ci);
            const float v3 = ldg_f32(se + se_base + 3 * Ci);

            const float m = max4f(v0, v1, v2, v3);
            const float e0 = __expf(v0 - m);
            const float e1 = __expf(v1 - m);
            const float e2 = __expf(v2 - m);
            const float e3 = __expf(v3 - m);
            const float inv = 1.0f / (e0 + e1 + e2 + e3);
            w0 = e0 * inv; w1 = e1 * inv; w2 = e2 * inv; w3 = e3 * inv;
        }
        w0 = shfl_broadcast(w0, 0);
        w1 = shfl_broadcast(w1, 0);
        w2 = shfl_broadcast(w2, 0);
        w3 = shfl_broadcast(w3, 0);

        const long long base_spc_b = (long long)b * spc_stride_b;
        const long long base_out_b = (long long)b * out_stride_b;

        const float* __restrict__ spc0 = spc + base_spc_b + (long long)(0 * Ci + ci) * HW;
        const float* __restrict__ spc1 = spc + base_spc_b + (long long)(1 * Ci + ci) * HW;
        const float* __restrict__ spc2 = spc + base_spc_b + (long long)(2 * Ci + ci) * HW;
        const float* __restrict__ spc3 = spc + base_spc_b + (long long)(3 * Ci + ci) * HW;

        float* __restrict__ out0 = out + base_out_b + (long long)(0 * Ci + ci) * HW;
        float* __restrict__ out1 = out + base_out_b + (long long)(1 * Ci + ci) * HW;
        float* __restrict__ out2 = out + base_out_b + (long long)(2 * Ci + ci) * HW;
        float* __restrict__ out3 = out + base_out_b + (long long)(3 * Ci + ci) * HW;

        // HW=49; each lane handles up to 2 elements (lane, lane+32)
        int t = lane;
        if (t < HW) {
            const float x0 = ldg_f32(spc0 + t);
            const float x1 = ldg_f32(spc1 + t);
            const float x2 = ldg_f32(spc2 + t);
            const float x3 = ldg_f32(spc3 + t);
            out0[t] = x0 * w0;
            out1[t] = x1 * w1;
            out2[t] = x2 * w2;
            out3[t] = x3 * w3;
        }
        t = lane + 32;
        if (t < HW) {
            const float x0 = ldg_f32(spc0 + t);
            const float x1 = ldg_f32(spc1 + t);
            const float x2 = ldg_f32(spc2 + t);
            const float x3 = ldg_f32(spc3 + t);
            out0[t] = x0 * w0;
            out1[t] = x1 * w1;
            out2[t] = x2 * w2;
            out3[t] = x3 * w3;
        }
    }
}

// Prior fast path: S=4 warp kernel, general H/W, optional float4 pixel vectorization.
__global__ void psa_warp_s4(
    const float* __restrict__ spc,
    const float* __restrict__ se,
    float* __restrict__ out,
    int B, int Ci, int H, int W
) {
    constexpr int S = 4;
    const int lane = (int)(threadIdx.x & 31);
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_id_in_block = (int)(threadIdx.x >> 5);
    const int global_warp = (int)blockIdx.x * warps_per_block + warp_id_in_block;

    const int n_items = B * Ci;
    for (int bci = global_warp; bci < n_items; bci += (int)gridDim.x * warps_per_block) {
        int b = bci / Ci;
        int ci = bci - b * Ci;

        float w0, w1, w2, w3;
        if (lane == 0) {
            int se_base = b * (S * Ci) + ci;
            float v0 = ldg_f32(se + se_base + 0 * Ci);
            float v1 = ldg_f32(se + se_base + 1 * Ci);
            float v2 = ldg_f32(se + se_base + 2 * Ci);
            float v3 = ldg_f32(se + se_base + 3 * Ci);

            float m = max4f(v0, v1, v2, v3);
            float e0 = __expf(v0 - m);
            float e1 = __expf(v1 - m);
            float e2 = __expf(v2 - m);
            float e3 = __expf(v3 - m);
            float inv = 1.0f / (e0 + e1 + e2 + e3);
            w0 = e0 * inv; w1 = e1 * inv; w2 = e2 * inv; w3 = e3 * inv;
        }
        w0 = shfl_broadcast(w0, 0);
        w1 = shfl_broadcast(w1, 0);
        w2 = shfl_broadcast(w2, 0);
        w3 = shfl_broadcast(w3, 0);

        const long long HW = (long long)H * (long long)W;
        const long long base_spc_b = (long long)b * (long long)S * (long long)Ci * HW;
        const long long base_out_b = (long long)b * (long long)(S * Ci) * HW;

        const float* spc0 = spc + base_spc_b + (long long)(0 * Ci + ci) * HW;
        const float* spc1 = spc + base_spc_b + (long long)(1 * Ci + ci) * HW;
        const float* spc2 = spc + base_spc_b + (long long)(2 * Ci + ci) * HW;
        const float* spc3 = spc + base_spc_b + (long long)(3 * Ci + ci) * HW;

        float* out0 = out + base_out_b + (long long)(0 * Ci + ci) * HW;
        float* out1 = out + base_out_b + (long long)(1 * Ci + ci) * HW;
        float* out2 = out + base_out_b + (long long)(2 * Ci + ci) * HW;
        float* out3 = out + base_out_b + (long long)(3 * Ci + ci) * HW;

        bool vec4_ok = ((W & 3) == 0);
        if (vec4_ok) {
            uintptr_t p0 = (uintptr_t)spc0;
            uintptr_t q0 = (uintptr_t)out0;
            vec4_ok = ((p0 & 15) == 0) && ((q0 & 15) == 0);
        }

        if (vec4_ok) {
            const int W4 = W >> 2;
            const int HW4 = H * W4;
            for (int t = lane; t < HW4; t += 32) {
                long long pix4 = (long long)t * 4LL;

                float4 x0 = *reinterpret_cast<const float4*>(spc0 + pix4);
                float4 x1 = *reinterpret_cast<const float4*>(spc1 + pix4);
                float4 x2 = *reinterpret_cast<const float4*>(spc2 + pix4);
                float4 x3 = *reinterpret_cast<const float4*>(spc3 + pix4);

                x0.x *= w0; x0.y *= w0; x0.z *= w0; x0.w *= w0;
                x1.x *= w1; x1.y *= w1; x1.z *= w1; x1.w *= w1;
                x2.x *= w2; x2.y *= w2; x2.z *= w2; x2.w *= w2;
                x3.x *= w3; x3.y *= w3; x3.z *= w3; x3.w *= w3;

                *reinterpret_cast<float4*>(out0 + pix4) = x0;
                *reinterpret_cast<float4*>(out1 + pix4) = x1;
                *reinterpret_cast<float4*>(out2 + pix4) = x2;
                *reinterpret_cast<float4*>(out3 + pix4) = x3;
            }
        } else {
            for (long long t = lane; t < HW; t += 32) {
                float x0 = ldg_f32(spc0 + t) * w0;
                float x1 = ldg_f32(spc1 + t) * w1;
                float x2 = ldg_f32(spc2 + t) * w2;
                float x3 = ldg_f32(spc3 + t) * w3;
                out0[t] = x0;
                out1[t] = x1;
                out2[t] = x2;
                out3[t] = x3;
            }
        }
    }
}

// Generic fallback (S<=4, any W). One block per (b,ci).
__global__ void psa_fused_bci_generic(
    const float* __restrict__ spc,
    const float* __restrict__ se,
    float* __restrict__ out,
    int B, int S, int Ci, int H, int W
) {
    int bci = (int)blockIdx.x;
    int b = bci / Ci;
    int ci = bci - b * Ci;

    int se_base = b * (S * Ci) + ci;
    float v0 = ldg_f32(se + se_base + 0 * Ci);
    float v1 = (S > 1) ? ldg_f32(se + se_base + 1 * Ci) : -INFINITY;
    float v2 = (S > 2) ? ldg_f32(se + se_base + 2 * Ci) : -INFINITY;
    float v3 = (S > 3) ? ldg_f32(se + se_base + 3 * Ci) : -INFINITY;

    float m;
    if (S == 1) m = v0;
    else if (S == 2) m = max2f(v0, v1);
    else if (S == 3) m = max2f(max2f(v0, v1), v2);
    else m = max4f(v0, v1, v2, v3);

    float e0 = __expf(v0 - m);
    float e1 = (S > 1) ? __expf(v1 - m) : 0.0f;
    float e2 = (S > 2) ? __expf(v2 - m) : 0.0f;
    float e3 = (S > 3) ? __expf(v3 - m) : 0.0f;
    float inv = 1.0f / (e0 + e1 + e2 + e3);

    float wgt0 = e0 * inv;
    float wgt1 = e1 * inv;
    float wgt2 = e2 * inv;
    float wgt3 = e3 * inv;

    long long HW = (long long)H * (long long)W;
    long long base_spc_b = (long long)b * (long long)S * (long long)Ci * HW;
    long long base_out_b = (long long)b * (long long)(S * Ci) * HW;

    for (long long t = (long long)threadIdx.x; t < HW; t += (long long)blockDim.x) {
        if (S > 0) out[base_out_b + ((long long)(0 * Ci + ci) * HW + t)] =
            spc[base_spc_b + ((long long)(0 * Ci + ci) * HW + t)] * wgt0;
        if (S > 1) out[base_out_b + ((long long)(1 * Ci + ci) * HW + t)] =
            spc[base_spc_b + ((long long)(1 * Ci + ci) * HW + t)] * wgt1;
        if (S > 2) out[base_out_b + ((long long)(2 * Ci + ci) * HW + t)] =
            spc[base_spc_b + ((long long)(2 * Ci + ci) * HW + t)] * wgt2;
        if (S > 3) out[base_out_b + ((long long)(3 * Ci + ci) * HW + t)] =
            spc[base_spc_b + ((long long)(3 * Ci + ci) * HW + t)] * wgt3;
    }
}

torch::Tensor psa_fused_cuda(torch::Tensor spc, torch::Tensor se) {
    CHECK_INPUT(spc);
    CHECK_INPUT(se);
    TORCH_CHECK(spc.dim() == 5, "spc must be [B,S,Ci,H,W]");
    TORCH_CHECK(se.dim() == 5, "se must be [B,S,Ci,1,1]");

    int B  = (int)spc.size(0);
    int S  = (int)spc.size(1);
    int Ci = (int)spc.size(2);
    int H  = (int)spc.size(3);
    int W  = (int)spc.size(4);

    TORCH_CHECK(se.size(0) == B && se.size(1) == S && se.size(2) == Ci, "se shape mismatch");
    TORCH_CHECK(se.size(3) == 1 && se.size(4) == 1, "se must have spatial 1x1");
    TORCH_CHECK(S <= 4, "This fused kernel supports S <= 4");

    auto out = torch::empty({B, S * Ci, H, W}, spc.options());

    if (S == 4 && H == 7 && W == 7 && Ci == 128) {
        int threads = 128; // 4 warps
        int warps_per_block = threads / 32;
        int n_items = B * Ci;
        int blocks = (n_items + warps_per_block - 1) / warps_per_block;
        blocks = max(1, min(blocks, 8192));
        psa_warp_s4_hw49_ci128<<<blocks, threads>>>(
            (const float*)spc.data_ptr<float>(),
            (const float*)se.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B
        );
    } else if (S == 4) {
        int threads = 128; // 4 warps/block
        int warps_per_block = threads / 32;
        int n_items = B * Ci;
        int blocks = (n_items + warps_per_block - 1) / warps_per_block;
        blocks = max(1, min(blocks, 4096));
        psa_warp_s4<<<blocks, threads>>>(
            (const float*)spc.data_ptr<float>(),
            (const float*)se.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, Ci, H, W
        );
    } else {
        int blocks = B * Ci;
        psa_fused_bci_generic<<<blocks, 128>>>(
            (const float*)spc.data_ptr<float>(),
            (const float*)se.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, S, Ci, H, W
        );
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor psa_fused_cuda(torch::Tensor spc, torch::Tensor se);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_psa_opt9_hw49",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["psa_fused_cuda"],
    extra_cuda_cflags=["--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """PSA module with optimized fused CUDA (softmax over S computed per (B,Ci), applied over HxW)."""
    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.channel = channel
        self.reduction = reduction

        self.convs = nn.ModuleList([])
        for i in range(S):
            self.convs.append(
                nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            )

        self.se_blocks = nn.ModuleList([])
        for i in range(S):
            self.se_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                    nn.Sigmoid(),
                )
            )

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        S = self.S
        ci = c // S

        spc_in = x.view(b, S, ci, h, w).contiguous()

        spc_groups = []
        for idx, conv in enumerate(self.convs):
            spc_groups.append(conv(spc_in[:, idx, :, :, :]))
        spc = torch.stack(spc_groups, dim=1).contiguous()  # [B,S,Ci,H,W]

        se_groups = []
        for idx, seblk in enumerate(self.se_blocks):
            se_groups.append(seblk(spc[:, idx, :, :, :]))
        se = torch.stack(se_groups, dim=1).contiguous()  # [B,S,Ci,1,1]

        out = self.custom_ops_lib.psa_fused_cuda(spc, se)  # [B, C, H, W]
        return out