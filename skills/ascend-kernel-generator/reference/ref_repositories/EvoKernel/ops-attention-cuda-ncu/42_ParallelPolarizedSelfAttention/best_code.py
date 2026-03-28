import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------
# CUDA extension
# - Fast path: fuse sw compute + apply for (C2=256, HW=49) with one block per batch b.
# - Fallback: baseline two-stage (compute sw -> apply) for general shapes.
# ---------------------------

ppsa_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// ------------------------------------------
// Fast path fused kernel for C2=256, HW=49
// One block per b. Compute sw[49] once into shared, then stream channels.
// Mapping:
//  - threads=256 (8 warps)
//  - warps 0..6 each own one hw in each iteration t (7 iters -> covers 49)
//  - after sw computed, all threads iterate over (c,hw) linearized.
// ------------------------------------------
__global__ void ppsa_fused_b_c2_256_hw49(
    const float* __restrict__ x,          // [B,512,49]
    const float* __restrict__ ch_weight,  // [B,512]
    const float* __restrict__ sp_wv,      // [B,256,49]
    const float* __restrict__ sp_q,       // [B,256]
    float* __restrict__ out,              // [B,512,49]
    int B)
{
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;   // 0..255
    int lane = tid & 31;          // 0..31
    int warp = tid >> 5;          // 0..7

    __shared__ float qsh[256];
    __shared__ float swsh[49];

    // stage sp_q[b,:]
    qsh[tid] = __ldg(sp_q + b * 256 + tid);
    __syncthreads();

    // compute sw[49] (each hw computed by one warp; 7 iterations cover 49)
    #pragma unroll
    for (int t = 0; t < 7; ++t) {
        int hw = t * 7 + warp; // use 7 warps (0..6) => 49
        if (warp < 7 && hw < 49) {
            float acc = 0.0f;
            // each lane accumulates 8 elements (256/32)
            #pragma unroll
            for (int it = 0; it < 8; ++it) {
                int k = it * 32 + lane;
                float q = qsh[k];
                float v = __ldg(sp_wv + ((b * 256 + k) * 49 + hw));
                acc = fmaf(q, v, acc);
            }
            float sum = warp_reduce_sum(acc);
            if (lane == 0) swsh[hw] = sigmoidf_fast(sum);
        }
    }
    __syncthreads();

    // stream over all (c,hw): total = 512*49 = 25088 elements
    // each thread handles a strided subset
    int base_x = b * 512 * 49;
    int base_w = b * 512;

    for (int idx = tid; idx < 512 * 49; idx += 256) {
        int c = idx / 49;
        int hw = idx - c * 49;
        float xv = __ldg(x + base_x + idx);
        float cw = __ldg(ch_weight + base_w + c);
        float sw = swsh[hw];
        out[base_x + idx] = xv * (cw + sw);
    }
}

// ------------------------------------------
// Fallback kernels (baseline)
// ------------------------------------------
__global__ void ppsa_compute_sw_block_b_256_49(
    const float* __restrict__ sp_wv,  // [B,256,49]
    const float* __restrict__ sp_q,   // [B,256]
    float* __restrict__ sw,           // [B,49]
    int B)
{
    int b = (int)blockIdx.x;
    if (b >= B) return;

    __shared__ float qsh[256];
    int tid = (int)threadIdx.x;
    if (tid < 256) qsh[tid] = __ldg(sp_q + b * 256 + tid);
    __syncthreads();

    int lane = tid & 31;
    int warp = tid >> 5;  // 0..7

    #pragma unroll
    for (int t = 0; t < 7; ++t) {
        int hw = t * 8 + warp;
        if (hw < 49) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = lane; k < 256; k += 32) {
                float q = qsh[k];
                float v = __ldg(sp_wv + ((b * 256 + k) * 49 + hw));
                acc = fmaf(q, v, acc);
            }
            float sum = warp_reduce_sum(acc);
            if (lane == 0) sw[b * 49 + hw] = sigmoidf_fast(sum);
        }
    }
}

__global__ void ppsa_compute_sw_general(
    const float* __restrict__ sp_wv,  // [B,C2,HW]
    const float* __restrict__ sp_q,   // [B,C2]
    float* __restrict__ sw,           // [B,HW]
    int B, int C2, int HW)
{
    int b = (int)blockIdx.x;
    int hw = (int)blockIdx.y;
    if (b >= B || hw >= HW) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float acc = 0.0f;
    for (int k = tid; k < C2; k += (int)blockDim.x) {
        float q = __ldg(sp_q + b * C2 + k);
        float v = __ldg(sp_wv + ((b * C2 + k) * HW + hw));
        acc = fmaf(q, v, acc);
    }

    acc = warp_reduce_sum(acc);

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    float sum = 0.0f;
    if (warp == 0) {
        int nwarps = ((int)blockDim.x >> 5);
        sum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) sw[b * HW + hw] = sigmoidf_fast(sum);
    }
}

__global__ void ppsa_apply_49(
    const float* __restrict__ x,          // [B,C,49]
    const float* __restrict__ ch_weight,  // [B,C]
    const float* __restrict__ sw,         // [B,49]
    float* __restrict__ out,              // [B,C,49]
    int B, int C)
{
    int b = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (b >= B || c >= C) return;

    float cw = __ldg(ch_weight + b * C + c);

    const float* x_ptr = x + (b * C + c) * 49;
    float* o_ptr = out + (b * C + c) * 49;
    const float* sw_ptr = sw + b * 49;

    int tid = (int)threadIdx.x;

    bool aligned = ((((uintptr_t)x_ptr | (uintptr_t)o_ptr | (uintptr_t)sw_ptr) & 0xF) == 0);
    if (aligned) {
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x_ptr);
        const float4* __restrict__ sw4 = reinterpret_cast<const float4*>(sw_ptr);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(o_ptr);

        for (int i = tid; i < 12; i += (int)blockDim.x) {
            float4 xv = __ldg(x4 + i);
            float4 sv = __ldg(sw4 + i);
            float4 ov;
            ov.x = xv.x * (cw + sv.x);
            ov.y = xv.y * (cw + sv.y);
            ov.z = xv.z * (cw + sv.z);
            ov.w = xv.w * (cw + sv.w);
            o4[i] = ov;
        }
    } else {
        for (int hw = tid; hw < 48; hw += (int)blockDim.x) {
            float xv = __ldg(x_ptr + hw);
            float sv = __ldg(sw_ptr + hw);
            o_ptr[hw] = xv * (cw + sv);
        }
    }

    if (tid == 0) {
        float xv = __ldg(x_ptr + 48);
        float sv = __ldg(sw_ptr + 48);
        o_ptr[48] = xv * (cw + sv);
    }
}

__global__ void ppsa_apply_general(
    const float* __restrict__ x,          // [B,C,HW]
    const float* __restrict__ ch_weight,  // [B,C]
    const float* __restrict__ sw,         // [B,HW]
    float* __restrict__ out,              // [B,C,HW]
    int B, int C, int HW)
{
    int b = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (b >= B || c >= C) return;

    float cw = __ldg(ch_weight + b * C + c);

    const float* x_ptr = x + (b * C + c) * HW;
    float* o_ptr = out + (b * C + c) * HW;
    const float* sw_ptr = sw + b * HW;

    int tid = (int)threadIdx.x;

    int hw4 = (HW / 4) * 4;
    bool aligned = (((uintptr_t)x_ptr | (uintptr_t)o_ptr | (uintptr_t)sw_ptr) & 0xF) == 0;

    if (aligned) {
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x_ptr);
        const float4* __restrict__ sw4 = reinterpret_cast<const float4*>(sw_ptr);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(o_ptr);

        int n4 = hw4 >> 2;
        for (int i = tid; i < n4; i += (int)blockDim.x) {
            float4 xv = __ldg(x4 + i);
            float4 sv = __ldg(sw4 + i);
            float4 ov;
            ov.x = xv.x * (cw + sv.x);
            ov.y = xv.y * (cw + sv.y);
            ov.z = xv.z * (cw + sv.z);
            ov.w = xv.w * (cw + sv.w);
            o4[i] = ov;
        }
    } else {
        for (int hw = tid; hw < hw4; hw += (int)blockDim.x) {
            float xv = __ldg(x_ptr + hw);
            float sv = __ldg(sw_ptr + hw);
            o_ptr[hw] = xv * (cw + sv);
        }
    }

    for (int hw = hw4 + tid; hw < HW; hw += (int)blockDim.x) {
        float xv = __ldg(x_ptr + hw);
        float sv = __ldg(sw_ptr + hw);
        o_ptr[hw] = xv * (cw + sv);
    }
}

torch::Tensor ppsa_fused_out_cuda(torch::Tensor x,
                                 torch::Tensor ch_weight,
                                 torch::Tensor sp_wv,
                                 torch::Tensor sp_q) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(ch_weight.is_cuda(), "ch_weight must be CUDA");
    TORCH_CHECK(sp_wv.is_cuda(), "sp_wv must be CUDA");
    TORCH_CHECK(sp_q.is_cuda(), "sp_q must be CUDA");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(ch_weight.dtype() == torch::kFloat32, "ch_weight must be float32");
    TORCH_CHECK(sp_wv.dtype() == torch::kFloat32, "sp_wv must be float32");
    TORCH_CHECK(sp_q.dtype() == torch::kFloat32, "sp_q must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(ch_weight.is_contiguous(), "ch_weight must be contiguous");
    TORCH_CHECK(sp_wv.is_contiguous(), "sp_wv must be contiguous (NCHW)");
    TORCH_CHECK(sp_q.is_contiguous(), "sp_q must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(ch_weight.dim() == 4, "ch_weight must be [B,C,1,1]");
    TORCH_CHECK(sp_wv.dim() == 4, "sp_wv must be [B,C2,H,W]");
    TORCH_CHECK(sp_q.dim() == 2, "sp_q must be [B,C2]");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;

    TORCH_CHECK(C == (int)ch_weight.size(1), "ch_weight shape mismatch");
    TORCH_CHECK(ch_weight.size(2) == 1 && ch_weight.size(3) == 1, "ch_weight must be [B,C,1,1]");

    int C2 = (int)sp_wv.size(1);
    TORCH_CHECK(sp_wv.size(0) == B && sp_wv.size(2) == H && sp_wv.size(3) == W, "sp_wv shape mismatch");
    TORCH_CHECK(sp_q.size(0) == B && sp_q.size(1) == C2, "sp_q shape mismatch");

    auto out = torch::empty_like(x);

    auto x3 = x.view({B, C, HW});
    auto out3 = out.view({B, C, HW});
    auto ch2 = ch_weight.view({B, C});
    auto sp3 = sp_wv.view({B, C2, HW});

    // Fast path: fuse for (C2=256, HW=49, C=512)
    if (C2 == 256 && HW == 49 && C == 512) {
        dim3 grid((unsigned)B);
        dim3 block(256);
        ppsa_fused_b_c2_256_hw49<<<grid, block>>>(
            (const float*)x3.data_ptr<float>(),
            (const float*)ch2.data_ptr<float>(),
            (const float*)sp3.data_ptr<float>(),
            (const float*)sp_q.data_ptr<float>(),
            (float*)out3.data_ptr<float>(),
            B
        );
        return out;
    }

    // Fallback: baseline two-stage
    auto sw = torch::empty({B, HW}, x.options());

    if (C2 == 256 && HW == 49) {
        dim3 grid1((unsigned)B);
        dim3 block1(256);
        ppsa_compute_sw_block_b_256_49<<<grid1, block1>>>(
            (const float*)sp3.data_ptr<float>(),
            (const float*)sp_q.data_ptr<float>(),
            (float*)sw.data_ptr<float>(),
            B
        );
    } else {
        dim3 grid1((unsigned)B, (unsigned)HW);
        int threads1 = (C2 <= 256) ? 128 : 256;
        dim3 block1((unsigned)threads1);
        ppsa_compute_sw_general<<<grid1, block1>>>(
            (const float*)sp3.data_ptr<float>(),
            (const float*)sp_q.data_ptr<float>(),
            (float*)sw.data_ptr<float>(),
            B, C2, HW
        );
    }

    dim3 grid2((unsigned)B, (unsigned)C);
    dim3 block2(128);
    if (HW == 49) {
        ppsa_apply_49<<<grid2, block2>>>(
            (const float*)x3.data_ptr<float>(),
            (const float*)ch2.data_ptr<float>(),
            (const float*)sw.data_ptr<float>(),
            (float*)out3.data_ptr<float>(),
            B, C
        );
    } else {
        ppsa_apply_general<<<grid2, block2>>>(
            (const float*)x3.data_ptr<float>(),
            (const float*)ch2.data_ptr<float>(),
            (const float*)sw.data_ptr<float>(),
            (float*)out3.data_ptr<float>(),
            B, C, HW
        );
    }

    return out;
}
"""

ppsa_cpp_source = r"""
torch::Tensor ppsa_fused_out_cuda(torch::Tensor x,
                                 torch::Tensor ch_weight,
                                 torch::Tensor sp_wv,
                                 torch::Tensor sp_q);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_ppsa_fused_v9_fuse_b_hw49",
    cpp_sources=ppsa_cpp_source,
    cuda_sources=ppsa_cuda_source,
    functions=["ppsa_fused_out_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Parallel Polarized Self-Attention optimized for reduced layout transforms and a faster
    final gating stage via a custom CUDA kernel.
    """

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=1)
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.custom_ops = custom_ops_lib
        self.channel = int(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA input tensor")
        if x.dtype != torch.float32:
            x = x.float()

        # Keep convs in channels_last to avoid implicit cudnn transforms
        if not x.is_contiguous(memory_format=torch.channels_last):
            x_cl = x.contiguous(memory_format=torch.channels_last)
        else:
            x_cl = x

        b, c, h, w = x_cl.shape
        c2 = c // 2
        hw = h * w

        # ---- Channel-only branch ----
        channel_wv = self.ch_wv(x_cl)  # [B,C2,H,W] channels_last
        channel_wq = self.ch_wq(x_cl)  # [B,1,H,W] channels_last
        channel_wv_r = channel_wv.reshape(b, c2, hw)
        channel_wq_r = channel_wq.reshape(b, hw, 1)
        channel_wq_sm = self.softmax_channel(channel_wq_r)  # softmax over HW
        channel_wz = torch.matmul(channel_wv_r, channel_wq_sm).unsqueeze(-1)  # [B,C2,1,1]

        ch_weight = self.sigmoid(
            self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
        ).permute(0, 2, 1).reshape(b, c, 1, 1)

        # ---- Spatial-only branch ----
        sp_wv = self.sp_wv(x_cl)  # [B,C2,H,W] channels_last
        sp_wq = self.sp_wq(x_cl)  # [B,C2,H,W] channels_last
        sp_wq_pool = self.agp(sp_wq).reshape(b, c2)  # [B,C2]
        sp_q = self.softmax_spatial(sp_wq_pool.unsqueeze(1)).squeeze(1)  # [B,C2]

        # ---- Prepare for CUDA (expects contiguous NCHW) ----
        x_nchw = x_cl.contiguous()
        sp_wv_nchw = sp_wv.contiguous()
        ch_weight_nchw = ch_weight.contiguous()
        sp_q_c = sp_q.contiguous()

        out = self.custom_ops.ppsa_fused_out_cuda(x_nchw, ch_weight_nchw, sp_wv_nchw, sp_q_c)
        return out