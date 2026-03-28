import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA op: CoTAttention fused tail
# out = k1 + reshape( softmax(att, dim=-1) * v )  where att,v are (B,C,S), S=H*W
#
# Improvements over baseline:
# - Dominant S==49 fast path: 4 independent rows per block (4 warps/block), warp-only
#   reductions (no shared, no syncthreads) to boost occupancy/latency hiding.
# - Optional float2 vectorized loads/stores with 8B alignment guard; keeps all lanes useful.
# - Reduced redundant loads by caching per-lane att values.
# - Generic fallback launch tuned to avoid oversized blocks that worsen reg pressure.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __forceinline__ __device__ float ld_g(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __forceinline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

// Generic kernel: one block per (b,c), threads iterate over S elements.
// Shared-memory reduction, safe for any S.
__global__ void cot_attention_fused_generic(
    const float* __restrict__ k1,   // (BC, S) view of contiguous (B,C,H,W)
    const float* __restrict__ att,  // (BC, S)
    const float* __restrict__ v,    // (BC, S)
    float* __restrict__ out,        // (BC, S)
    int BC, int S
) {
    int bc = (int)blockIdx.x;
    if (bc >= BC) return;

    const float* att_ptr = att + (size_t)bc * (size_t)S;
    const float* v_ptr   = v   + (size_t)bc * (size_t)S;
    const float* k1_ptr  = k1  + (size_t)bc * (size_t)S;
    float* out_ptr       = out + (size_t)bc * (size_t)S;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarp = (int)blockDim.x >> 5;

    // max
    float tmax = -INFINITY;
    for (int i = tid; i < S; i += (int)blockDim.x) tmax = fmaxf(tmax, ld_g(att_ptr + i));
    float wmax = warp_reduce_max(tmax);

    __shared__ float s_red[32];
    if (lane == 0) s_red[warp] = wmax;
    __syncthreads();

    float bmax = -INFINITY;
    if (warp == 0) {
        float v0 = (lane < nwarp) ? s_red[lane] : -INFINITY;
        float r = warp_reduce_max(v0);
        if (lane == 0) s_red[0] = r;
    }
    __syncthreads();
    bmax = s_red[0];

    // sumexp
    float tsum = 0.0f;
    for (int i = tid; i < S; i += (int)blockDim.x) tsum += __expf(ld_g(att_ptr + i) - bmax);
    float wsum = warp_reduce_sum(tsum);
    if (lane == 0) s_red[warp] = wsum;
    __syncthreads();

    float bsum = 0.0f;
    if (warp == 0) {
        float v0 = (lane < nwarp) ? s_red[lane] : 0.0f;
        float r = warp_reduce_sum(v0);
        if (lane == 0) s_red[0] = r;
    }
    __syncthreads();
    bsum = s_red[0];

    float inv = 1.0f / (bsum + 1e-20f);

    for (int i = tid; i < S; i += (int)blockDim.x) {
        float a = ld_g(att_ptr + i);
        float p = __expf(a - bmax) * inv;
        out_ptr[i] = ld_g(k1_ptr + i) + p * ld_g(v_ptr + i);
    }
}

// S==49 kernel: 4 rows per block, 1 warp per row. No shared mem, no syncthreads.
// Optional float2 vectorized IO for indices 0..47 when 8B-aligned.
__global__ __launch_bounds__(128, 4) void cot_attention_fused_s49_4rows(
    const float* __restrict__ k1,
    const float* __restrict__ att,
    const float* __restrict__ v,
    float* __restrict__ out,
    int BC
) {
    int tid = (int)threadIdx.x;     // 0..127
    int warp = tid >> 5;            // 0..3
    int lane = tid & 31;            // 0..31

    int bc = ((int)blockIdx.x << 2) + warp; // 4 rows per block
    if (bc >= BC) return;

    const float* att_ptr = att + (size_t)bc * 49u;
    const float* v_ptr   = v   + (size_t)bc * 49u;
    const float* k1_ptr  = k1  + (size_t)bc * 49u;
    float* out_ptr       = out + (size_t)bc * 49u;

    // Load up to two att values per lane for max/sum.
    // lane 0..31 -> idx0 = lane (0..31), idx1 = lane+32 (32..63)
    // For S=49, idx1 valid only for lane 0..16 (32..48).
    float a0 = (lane < 49) ? ld_g(att_ptr + lane) : -INFINITY;
    float a1 = (lane + 32 < 49) ? ld_g(att_ptr + lane + 32) : -INFINITY;

    float tmax = fmaxf(a0, a1);
    float bmax = warp_reduce_max(tmax);
    bmax = __shfl_sync(0xffffffff, bmax, 0);

    float e0 = (lane < 49) ? __expf(a0 - bmax) : 0.0f;
    float e1 = (lane + 32 < 49) ? __expf(a1 - bmax) : 0.0f;
    float tsum = e0 + e1;

    float bsum = warp_reduce_sum(tsum);
    bsum = __shfl_sync(0xffffffff, bsum, 0);
    float inv = 1.0f / (bsum + 1e-20f);

    // Vectorized path over indices 0..47 (24 float2s):
    // lanes 0..23 each handle one float2 at base = lane*2.
    // Tail idx 48 handled by lane 24.
    bool aligned8 =
        (((uintptr_t)att_ptr & 0x7) == 0) &&
        (((uintptr_t)v_ptr   & 0x7) == 0) &&
        (((uintptr_t)k1_ptr  & 0x7) == 0) &&
        (((uintptr_t)out_ptr & 0x7) == 0);

    if (aligned8 && lane < 24) {
        int base = lane * 2; // 0..46
        // Use float2 loads/stores (8B)
        float2 av = *((const float2*)(att_ptr + base));
        float2 vv = *((const float2*)(v_ptr + base));
        float2 kv = *((const float2*)(k1_ptr + base));

        float p0 = __expf(av.x - bmax) * inv;
        float p1 = __expf(av.y - bmax) * inv;

        float2 ov;
        ov.x = kv.x + p0 * vv.x;
        ov.y = kv.y + p1 * vv.y;

        *((float2*)(out_ptr + base)) = ov;
    } else {
        // Scalar stores for indices 0..31 and 32..48 via the a0/a1 already loaded.
        if (lane < 49) {
            float p = __expf(a0 - bmax) * inv;
            out_ptr[lane] = ld_g(k1_ptr + lane) + p * ld_g(v_ptr + lane);
        }
        int j = lane + 32;
        if (j < 49) {
            float p = __expf(a1 - bmax) * inv;
            out_ptr[j] = ld_g(k1_ptr + j) + p * ld_g(v_ptr + j);
        }
    }

    // Tail element 48 (if vectorized handled only 0..47)
    if (aligned8 && lane == 24) {
        int j = 48;
        float aj = ld_g(att_ptr + j);
        float pj = __expf(aj - bmax) * inv;
        out_ptr[j] = ld_g(k1_ptr + j) + pj * ld_g(v_ptr + j);
    }
}

torch::Tensor cot_attention_fused_cuda(torch::Tensor k1, torch::Tensor att, torch::Tensor v) {
    TORCH_CHECK(k1.is_cuda() && att.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(k1.scalar_type() == torch::kFloat32, "k1 must be float32");
    TORCH_CHECK(att.scalar_type() == torch::kFloat32 && v.scalar_type() == torch::kFloat32, "att/v must be float32");
    TORCH_CHECK(k1.is_contiguous(), "k1 must be contiguous");
    TORCH_CHECK(att.is_contiguous() && v.is_contiguous(), "att/v must be contiguous");
    TORCH_CHECK(k1.dim() == 4, "k1 must be (B,C,H,W)");
    TORCH_CHECK(att.dim() == 3 && v.dim() == 3, "att/v must be (B,C,S)");

    int B = (int)k1.size(0);
    int C = (int)k1.size(1);
    int H = (int)k1.size(2);
    int W = (int)k1.size(3);
    int S = H * W;

    TORCH_CHECK(att.size(0) == B && att.size(1) == C && att.size(2) == S, "att must be (B,C,H*W)");
    TORCH_CHECK(v.size(0) == B && v.size(1) == C && v.size(2) == S, "v must be (B,C,H*W)");

    auto out = torch::empty_like(k1);
    int BC = B * C;

    const float* k1_ptr  = (const float*)k1.data_ptr<float>();
    const float* att_ptr = (const float*)att.data_ptr<float>();
    const float* v_ptr   = (const float*)v.data_ptr<float>();
    float* out_ptr       = (float*)out.data_ptr<float>();

    if (S == 49) {
        int blocks = (BC + 3) / 4; // 4 rows per block
        dim3 grid(blocks);
        dim3 block(128); // 4 warps
        cot_attention_fused_s49_4rows<<<grid, block>>>(k1_ptr, att_ptr, v_ptr, out_ptr, BC);
    } else {
        // More conservative block sizing than baseline to reduce reg pressure.
        int threads = 128;
        if (S <= 64) threads = 128;
        else if (S <= 256) threads = 256;
        else threads = 256; // avoid 512/1024 which often reg-limits and hurts occupancy
        dim3 grid(BC);
        cot_attention_fused_generic<<<grid, threads>>>(k1_ptr, att_ptr, v_ptr, out_ptr, BC, S);
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor cot_attention_fused_cuda(torch::Tensor k1, torch::Tensor att, torch::Tensor v);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_cot_attention_opt_s49_4rows_f2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cot_attention_fused_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class CoTAttentionNew(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1),
        )

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        bs, c, h, w = x.shape

        k1 = self.key_embed(x)  # (B,C,H,W)
        v = self.value_embed(x).view(bs, c, -1)  # (B,C,S)

        y = torch.cat([k1, x], dim=1)  # (B,2C,H,W)
        att = self.attention_embed(y)  # (B,C*k*k,H,W)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # (B,C,S)

        if k1.dtype != torch.float32:
            k1 = k1.float()
        if att.dtype != torch.float32:
            att = att.float()
        if v.dtype != torch.float32:
            v = v.float()

        return self.custom_ops_lib.cot_attention_fused_cuda(
            k1.contiguous(), att.contiguous(), v.contiguous()
        )


class ModelNew(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.cot_attention = CoTAttentionNew(dim=dim, kernel_size=kernel_size)

    def forward(self, x):
        return self.cot_attention(x)