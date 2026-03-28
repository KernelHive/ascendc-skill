import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.f / (1.f + __expf(-x));
}

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// HW=49 optimized fused kernel, one block per (b,g), 128 threads (4 warps)
// - GN stats over x1 once per (b,g) via warp partials -> shared -> warp0 finalize
// - For channel-attention channels: compute x0 mean just-in-time (warp reduction over 49)
// - For spatial-attention channels: apply GN + affine + sigmoid gate
// - Write directly to channel-shuffled output
__global__ __launch_bounds__(128, 3) void fused_hw49_bg_kernel_opt(
    const float* __restrict__ x,           // [B,C,49] contiguous
    const float* __restrict__ cweight,     // [c_half]
    const float* __restrict__ cbias,       // [c_half]
    const float* __restrict__ sweight,     // [c_half]
    const float* __restrict__ sbias,       // [c_half]
    float* __restrict__ out,               // [B,C,49]
    int B, int C, int G, int Cg, int c_half,
    float eps)
{
    constexpr int HW = 49;
    int bg = (int)blockIdx.x; // 0..B*G-1
    int b = bg / G;
    int g = bg - b * G;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..3
    constexpr int NWARPS = 4;

    const float* __restrict__ x_bg = x + ((b * C + g * Cg) * HW);

    // ---- 1) GN stats over x1: sum/sumsq over N = c_half*49 ----
    const float* __restrict__ x1 = x_bg + c_half * HW;
    int N = c_half * HW;

    float lsum = 0.f, lsum2 = 0.f;
#pragma unroll 4
    for (int i = tid; i < N; i += 128) {
        float v = x1[i];
        lsum += v;
        lsum2 += v * v;
    }

    float wsum = warp_reduce_sum(lsum);
    float wsum2 = warp_reduce_sum(lsum2);

    __shared__ float s_sum[NWARPS];
    __shared__ float s_sum2[NWARPS];
    if (lane == 0) {
        s_sum[warp] = wsum;
        s_sum2[warp] = wsum2;
    }
    __syncthreads();

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (warp == 0) {
        float vs = (lane < NWARPS) ? s_sum[lane] : 0.f;
        float vs2 = (lane < NWARPS) ? s_sum2[lane] : 0.f;
        vs = warp_reduce_sum(vs);
        vs2 = warp_reduce_sum(vs2);
        if (lane == 0) {
            float invN = 1.f / (float)N;
            float mean = vs * invN;
            float var = vs2 * invN - mean * mean;
            if (var < 0.f) var = 0.f;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float mean = s_mean;
    float rstd = s_rstd;

    // ---- 2) Apply gating and write channel-shuffled output ----
    int c_per = C >> 1;

    // warp-strided loop over channels in this group
#pragma unroll 1
    for (int c_in_g = warp; c_in_g < Cg; c_in_g += NWARPS) {
        int c_global = g * Cg + c_in_g;
        int g2 = (c_global >= c_per);
        int within = c_global - g2 * c_per;
        int c_shuf = (within << 1) + g2;

        const float* __restrict__ x_chan = x_bg + c_in_g * HW;
        float* __restrict__ o_chan = out + ((b * C + c_shuf) * HW);

        if (c_in_g < c_half) {
            // channel attention
            int ch = c_in_g;
            // mean over 49 (warp reduction; lanes 0..16 cover 49 with 2 loads)
            float psum = 0.f;
            if (lane < 49) psum += x_chan[lane];
            int idx2 = lane + 32;
            if (idx2 < 49) psum += x_chan[idx2];
            float tot = warp_reduce_sum(psum);
            float m = __shfl_sync(0xffffffff, tot, 0) * (1.f / 49.f);

            float a = sigmoidf_fast(ldg_f(cweight + ch) * m + ldg_f(cbias + ch));

            // apply scalar gate
#pragma unroll
            for (int i = lane; i < 49; i += 32) {
                float v = x_chan[i];
                o_chan[i] = v * a;
            }
        } else {
            // spatial attention
            int ch = c_in_g - c_half;
            float sw = ldg_f(sweight + ch);
            float sb = ldg_f(sbias + ch);
#pragma unroll
            for (int i = lane; i < 49; i += 32) {
                float v = x_chan[i];
                float norm = (v - mean) * rstd;
                float a = sigmoidf_fast(sw * norm + sb);
                o_chan[i] = v * a;
            }
        }
    }
}

// =========================
// Fallback path (unchanged): stats + per-(b,c) forward
// =========================
__global__ void combined_stats_generic_kernel(
    const float* __restrict__ x_bg,  // [BG, Cg, HW]
    float* __restrict__ mean_hw,     // [BG, c_half]
    float* __restrict__ mean_gn,     // [BG]
    float* __restrict__ rstd_gn,     // [BG]
    int BG, int Cg, int HW, int c_half, float eps)
{
    int bg = (int)blockIdx.x;
    if (bg >= BG) return;

    int tid = (int)threadIdx.x;
    int nthreads = (int)blockDim.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (nthreads + 31) >> 5;

    const float* x1 = x_bg + (bg * Cg + c_half) * HW;
    int N = c_half * HW;

    float lsum = 0.f, lsum2 = 0.f;
    uintptr_t addr = (uintptr_t)x1;
    if (((addr & 0xF) == 0) && ((N & 3) == 0)) {
        const float4* p4 = (const float4*)x1;
        int N4 = N >> 2;
        for (int i = tid; i < N4; i += nthreads) {
            float4 v = p4[i];
            lsum += v.x + v.y + v.z + v.w;
            lsum2 += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        for (int i = tid; i < N; i += nthreads) {
            float v = x1[i];
            lsum += v;
            lsum2 += v * v;
        }
    }

    float wsum = warp_reduce_sum(lsum);
    float wsum2 = warp_reduce_sum(lsum2);

    __shared__ float s_sum[8];
    __shared__ float s_sum2[8];
    if (lane == 0) {
        s_sum[warp] = wsum;
        s_sum2[warp] = wsum2;
    }
    __syncthreads();

    if (warp == 0) {
        float vs = (lane < nwarps) ? s_sum[lane] : 0.f;
        float vs2 = (lane < nwarps) ? s_sum2[lane] : 0.f;
        vs = warp_reduce_sum(vs);
        vs2 = warp_reduce_sum(vs2);
        if (lane == 0) {
            float invN = 1.f / (float)N;
            float mean = vs * invN;
            float var = vs2 * invN - mean * mean;
            if (var < 0.f) var = 0.f;
            mean_gn[bg] = mean;
            rstd_gn[bg] = rsqrtf(var + eps);
        }
    }

    const float* x0 = x_bg + (bg * Cg) * HW;
    if (HW <= 128) {
        int warps_total = nthreads >> 5;
        for (int ch = warp; ch < c_half; ch += warps_total) {
            const float* base = x0 + ch * HW;
            float psum = 0.f;
            for (int i = lane; i < HW; i += 32) psum += base[i];
            float tot = warp_reduce_sum(psum);
            if (lane == 0) mean_hw[bg * c_half + ch] = tot / (float)HW;
        }
    } else {
        for (int ch = 0; ch < c_half; ++ch) {
            const float* base = x0 + ch * HW;
            float l = 0.f;
            for (int i = tid; i < HW; i += nthreads) l += base[i];
            float w = warp_reduce_sum(l);
            __shared__ float ws[8];
            if (lane == 0) ws[warp] = w;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < nwarps) ? ws[lane] : 0.f;
                v = warp_reduce_sum(v);
                if (lane == 0) mean_hw[bg * c_half + ch] = v / (float)HW;
            }
            __syncthreads();
        }
    }
}

template<int HW_FIXED>
__global__ __launch_bounds__(128, 3) void forward_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cweight,
    const float* __restrict__ cbias,
    const float* __restrict__ sweight,
    const float* __restrict__ sbias,
    const float* __restrict__ mean_hw,  // [BG, c_half]
    const float* __restrict__ mean_gn,  // [BG]
    const float* __restrict__ rstd_gn,  // [BG]
    float* __restrict__ out,
    int B, int C, int H, int W, int G, int Cg, int c_half)
{
    int bc = (int)blockIdx.x;
    int b = bc / C;
    int c = bc - b * C;
    if (b >= B) return;

    int HW = (HW_FIXED > 0) ? HW_FIXED : (H * W);

    int g = c / Cg;
    int c_in_g = c - g * Cg;
    int bg = b * G + g;

    int c_per = C >> 1;
    int g2 = (c >= c_per);
    int within = c - g2 * c_per;
    int c_shuf = (within << 1) + g2;

    const float* x_base = x + ((b * C + c) * HW);
    float* o_base = out + ((b * C + c_shuf) * HW);

    if (c_in_g < c_half) {
        int ch = c_in_g;
        float m = ldg_f(mean_hw + bg * c_half + ch);
        float z = ldg_f(cweight + ch) * m + ldg_f(cbias + ch);
        float a = sigmoidf_fast(z);

        for (int i = (int)threadIdx.x; i < HW; i += (int)blockDim.x) {
            float v = x_base[i];
            o_base[i] = v * a;
        }
    } else {
        int ch = c_in_g - c_half;
        float mean = ldg_f(mean_gn + bg);
        float rstd = ldg_f(rstd_gn + bg);
        float sw = ldg_f(sweight + ch);
        float sb = ldg_f(sbias + ch);

        for (int i = (int)threadIdx.x; i < HW; i += (int)blockDim.x) {
            float v = x_base[i];
            float norm = (v - mean) * rstd;
            float z = sw * norm + sb;
            float a = sigmoidf_fast(z);
            o_base[i] = v * a;
        }
    }
}

torch::Tensor shuffle_attention_forward_cuda(torch::Tensor x,
                                            torch::Tensor cweight,
                                            torch::Tensor cbias,
                                            torch::Tensor sweight,
                                            torch::Tensor sbias,
                                            int64_t G) {
    CHECK_INPUT(x);
    CHECK_INPUT(cweight);
    CHECK_INPUT(cbias);
    CHECK_INPUT(sweight);
    CHECK_INPUT(sbias);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    TORCH_CHECK(C % (int)G == 0, "C must be divisible by G");
    int Cg = C / (int)G;
    TORCH_CHECK(Cg % 2 == 0, "C/G must be even");
    int c_half = Cg / 2;

    TORCH_CHECK((int)cweight.numel() == c_half, "cweight size mismatch");
    TORCH_CHECK((int)cbias.numel() == c_half, "cbias size mismatch");
    TORCH_CHECK((int)sweight.numel() == c_half, "sweight size mismatch");
    TORCH_CHECK((int)sbias.numel() == c_half, "sbias size mismatch");

    x = x.contiguous();
    cweight = cweight.contiguous();
    cbias = cbias.contiguous();
    sweight = sweight.contiguous();
    sbias = sbias.contiguous();

    auto out = torch::empty_like(x);

    int HW = H * W;
    const float* x_ptr = (const float*)x.data_ptr<float>();

    if (HW == 49) {
        int BG = B * (int)G;
        fused_hw49_bg_kernel_opt<<<BG, 128>>>(
            x_ptr,
            (const float*)cweight.data_ptr<float>(),
            (const float*)cbias.data_ptr<float>(),
            (const float*)sweight.data_ptr<float>(),
            (const float*)sbias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, (int)G, Cg, c_half, 1e-5f
        );
        return out;
    }

    int BG = B * (int)G;
    auto opts = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean_hw = torch::empty({BG, c_half}, opts);
    auto mean_gn = torch::empty({BG}, opts);
    auto rstd_gn = torch::empty({BG}, opts);

    const float* x_bg = x_ptr; // [BG, Cg, HW] since BG*Cg == B*C

    combined_stats_generic_kernel<<<BG, 256>>>(
        x_bg,
        (float*)mean_hw.data_ptr<float>(),
        (float*)mean_gn.data_ptr<float>(),
        (float*)rstd_gn.data_ptr<float>(),
        BG, Cg, HW, c_half, 1e-5f
    );

    int grid_bc = B * C;
    forward_fused_kernel<0><<<grid_bc, 128>>>(
        x_ptr,
        (const float*)cweight.data_ptr<float>(),
        (const float*)cbias.data_ptr<float>(),
        (const float*)sweight.data_ptr<float>(),
        (const float*)sbias.data_ptr<float>(),
        (const float*)mean_hw.data_ptr<float>(),
        (const float*)mean_gn.data_ptr<float>(),
        (const float*)rstd_gn.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, C, H, W, (int)G, Cg, c_half
    );

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor shuffle_attention_forward_cuda(torch::Tensor x,
                                            torch::Tensor cweight,
                                            torch::Tensor cbias,
                                            torch::Tensor sweight,
                                            torch::Tensor sbias,
                                            int64_t G);
"""

custom_ops_lib = load_inline(
    name="custom_shuffle_attention_ops_v9",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["shuffle_attention_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Shuffle Attention Module using an optimized fused custom CUDA op (forward only)."""
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = int(G)
        self.channel = int(channel)

        c_half = channel // (2 * G)
        self.cweight = Parameter(torch.zeros(1, c_half, 1, 1))
        self.cbias = Parameter(torch.ones(1, c_half, 1, 1))
        self.sweight = Parameter(torch.zeros(1, c_half, 1, 1))
        self.sbias = Parameter(torch.ones(1, c_half, 1, 1))

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.shuffle_attention_forward_cuda(
            x.contiguous(),
            self.cweight.view(-1).contiguous(),
            self.cbias.view(-1).contiguous(),
            self.sweight.view(-1).contiguous(),
            self.sbias.view(-1).contiguous(),
            int(self.G),
        )