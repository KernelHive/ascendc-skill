import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# -----------------------------------------------------------------------------
# Optimized custom CUDA kernel: warp-per-(b,h,t) fused causal ReLU self-attention
# y[b,h,t,d] = sum_{s<=t} relu((q[b,h,t,:]·k[b,h,s,:]) * scale) * v[b,h,s,d]
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __device__ __forceinline__ float load_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

template<int HS>
__global__ __launch_bounds__(128, 2)
void relu_self_attention_warp_kernel_hs(
    const float* __restrict__ q,  // [B,H,T,HS]
    const float* __restrict__ k,  // [B,H,T,HS]
    const float* __restrict__ v,  // [B,H,T,HS]
    float* __restrict__ out,      // [B,H,T,HS]
    int B, int H, int T,
    float scale
) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x >> 5;            // 0..(warps_per_block-1)
    int warps_per_block = blockDim.x >> 5;

    // grid: x = t-tile, y = bh
    int bh = (int)blockIdx.y;
    int b = bh / H;
    int h = bh - b * H;

    int t = (int)blockIdx.x * warps_per_block + warp_id;
    if (b >= B || t >= T) return;

    int64_t base_bh = ((int64_t)b * H + h) * (int64_t)T * (int64_t)HS;
    int64_t q_base = base_bh + (int64_t)t * HS;
    int64_t out_base = q_base;

    // For HS=64: each lane accumulates 2 dims: lane and lane+32
    float acc0 = 0.f, acc1 = 0.f;
    float q0 = 0.f, q1 = 0.f;

    if constexpr (HS == 64) {
        q0 = load_ro(q + q_base + lane);
        q1 = load_ro(q + q_base + lane + 32);
    } else {
        // generic: one dim per lane for first 32 dims only; remaining dims handled in loop below
        if (lane < HS) q0 = load_ro(q + q_base + lane);
    }

    // causal loop over s
    for (int s = 0; s <= t; ++s) {
        int64_t k_base = base_bh + (int64_t)s * HS;

        float partial = 0.f;
        if constexpr (HS == 64) {
            float k0 = load_ro(k + k_base + lane);
            float k1 = load_ro(k + k_base + lane + 32);
            partial = fmaf(q0, k0, q1 * k1);
        } else {
            // generic dot: lanes cover first 32 dims, then iterate remaining dims in chunks of 32
            float sum = 0.f;
            int d = lane;
            if (d < HS) sum += q0 * load_ro(k + k_base + d);
            for (d = lane + WARP; d < HS; d += WARP) {
                sum += load_ro(q + q_base + d) * load_ro(k + k_base + d);
            }
            partial = sum;
        }

        float dot = warp_reduce_sum(partial);
        float w = dot * scale;
        w = __shfl_sync(0xffffffff, w, 0);

        if (w > 0.f) {
            int64_t v_base = k_base;
            if constexpr (HS == 64) {
                float vv0 = load_ro(v + v_base + lane);
                float vv1 = load_ro(v + v_base + lane + 32);
                acc0 = fmaf(w, vv0, acc0);
                acc1 = fmaf(w, vv1, acc1);
            } else {
                // generic: each lane writes its own dim, and loops for remaining dims (strided by warp)
                int d = lane;
                if (d < HS) acc0 = fmaf(w, load_ro(v + v_base + d), acc0);
                // For dims beyond 32, we can't store multiple outputs per thread without more registers.
                // So we only produce correct output for HS<=32 in this path. Enforce in host.
            }
        }
    }

    if constexpr (HS == 64) {
        out[out_base + lane] = acc0;
        out[out_base + lane + 32] = acc1;
    } else {
        if (lane < HS) out[out_base + lane] = acc0;
    }
}

__global__ __launch_bounds__(256, 2)
void relu_self_attention_generic_kernel(
    const float* __restrict__ q,  // [B,H,T,HS]
    const float* __restrict__ k,  // [B,H,T,HS]
    const float* __restrict__ v,  // [B,H,T,HS]
    float* __restrict__ out,      // [B,H,T,HS]
    int B, int H, int T, int HS,
    float scale
) {
    // Fallback: one thread per output element (baseline-like), but with __ldg and fmaf.
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = (int64_t)B * H * T * HS;
    if (idx >= N) return;

    int d = (int)(idx % HS);
    int tmp = (int)(idx / HS);
    int t = tmp % T;
    tmp /= T;
    int h = tmp % H;
    int b = tmp / H;

    int64_t q_base = (((int64_t)b * H + h) * T + t) * HS;

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        int64_t k_base = (((int64_t)b * H + h) * T + s) * HS;

        float dot = 0.0f;
        int j = 0;
        for (; j + 3 < HS; j += 4) {
            float q0 = load_ro(q + q_base + j + 0);
            float q1 = load_ro(q + q_base + j + 1);
            float q2 = load_ro(q + q_base + j + 2);
            float q3 = load_ro(q + q_base + j + 3);

            float k0 = load_ro(k + k_base + j + 0);
            float k1 = load_ro(k + k_base + j + 1);
            float k2 = load_ro(k + k_base + j + 2);
            float k3 = load_ro(k + k_base + j + 3);

            dot = fmaf(q0, k0, dot);
            dot = fmaf(q1, k1, dot);
            dot = fmaf(q2, k2, dot);
            dot = fmaf(q3, k3, dot);
        }
        for (; j < HS; ++j) {
            dot = fmaf(load_ro(q + q_base + j), load_ro(k + k_base + j), dot);
        }

        float w = dot * scale;
        if (w > 0.0f) {
            acc = fmaf(w, load_ro(v + k_base + d), acc);
        }
    }

    out[idx] = acc;
}

torch::Tensor relu_self_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "relu_self_attention_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "relu_self_attention_cuda: q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "relu_self_attention_cuda: k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "relu_self_attention_cuda: v must be float32");
    TORCH_CHECK(q.is_contiguous(), "relu_self_attention_cuda: q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "relu_self_attention_cuda: k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "relu_self_attention_cuda: v must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "relu_self_attention_cuda: inputs must be 4D [B,H,T,HS]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "relu_self_attention_cuda: q,k,v must have same shape");

    int B = (int)q.size(0);
    int H = (int)q.size(1);
    int T = (int)q.size(2);
    int HS = (int)q.size(3);

    auto out = torch::empty_like(q);
    float scale = 1.0f / sqrtf((float)HS);

    // Fast path for HS=64 (common for 768/12)
    if (HS == 64) {
        const int warps_per_block = 4;
        const int threads = warps_per_block * 32; // 128
        dim3 block(threads);
        dim3 grid((T + warps_per_block - 1) / warps_per_block, B * H, 1);

        relu_self_attention_warp_kernel_hs<64><<<grid, block>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, H, T, scale
        );
        return out;
    }

    // Optional HS=32 fast path (only correct for HS==32; generic warp kernel does not produce dims>32)
    if (HS == 32) {
        const int warps_per_block = 4;
        const int threads = warps_per_block * 32; // 128
        dim3 block(threads);
        dim3 grid((T + warps_per_block - 1) / warps_per_block, B * H, 1);

        relu_self_attention_warp_kernel_hs<32><<<grid, block>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, H, T, scale
        );
        return out;
    }

    // Fallback: baseline-like generic kernel for arbitrary HS
    int64_t N = (int64_t)B * H * T * HS;
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);
    relu_self_attention_generic_kernel<<<blocks, threads>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, H, T, HS, scale
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor relu_self_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_relu_self_attention_warp",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_self_attention_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


# -----------------------------------------------------------------------------
# Model using the custom CUDA op
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Multi-head causal self-attention using fused CUDA kernel for:
      (q@k^T)*scale + causal mask + ReLU + (att@v)
    Linear projections remain in PyTorch.
    """
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
            persistent=False
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, nh, hs).transpose(1, 2).contiguous()
        q = q.view(B, T, nh, hs).transpose(1, 2).contiguous()
        v = v.view(B, T, nh, hs).transpose(1, 2).contiguous()

        y = self.custom_ops.relu_self_attention_cuda(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y


# Integration helpers
batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]