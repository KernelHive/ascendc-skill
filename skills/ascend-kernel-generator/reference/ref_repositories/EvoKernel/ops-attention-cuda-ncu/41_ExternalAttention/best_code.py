import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA extension: improved S=64 fast path (all lanes active) + generic fused softmax+L1 ----
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __forceinline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads => 32 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return __shfl_sync(0xffffffff, out, 0);
}

static __forceinline__ __device__ float block_reduce_max(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float out = -INFINITY;
    if (warp == 0) {
        out = (lane < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
        out = warp_reduce_max(out);
    }
    return __shfl_sync(0xffffffff, out, 0);
}

// S==64 fast path: one warp handles one row, but keep multiple warps per CTA for scheduling granularity.
// All 32 lanes participate, each lane loads/stores 2 elements: col=lane and col=lane+32.
// We intentionally do NOT do the explicit L1 renorm in this fast path (softmax already sums to ~1).
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
void softmax64_warp2_multiwarp_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int M) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & (WARP_SIZE - 1);

    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= M) return;

    const float* row_ptr = x + (int64_t)row * 64;
    float* out_ptr = y + (int64_t)row * 64;

    // Read-only cache load when available; __ldg works for global on many archs.
    float v0 = __ldg(row_ptr + lane);
    float v1 = __ldg(row_ptr + lane + 32);

    float tmax = fmaxf(v0, v1);
    float row_max = warp_reduce_max(tmax);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float e0 = __expf(v0 - row_max);
    float e1 = __expf(v1 - row_max);
    float tsum = e0 + e1;

    float sum_exp = warp_reduce_sum(tsum);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    float inv_sum = 1.0f / (sum_exp + 1e-12f);

    out_ptr[lane]      = e0 * inv_sum;
    out_ptr[lane + 32] = e1 * inv_sum;
}

// Generic fused kernel for arbitrary S: softmax then explicit L1 renorm (to match baseline semantics).
__global__ void softmax_fused_generic_kernel(const float* __restrict__ x,
                                            float* __restrict__ y,
                                            int M, int S) {
    int row = blockIdx.x;
    if (row >= M) return;

    // 1) max
    float tmax = -INFINITY;
    for (int col = threadIdx.x; col < S; col += blockDim.x) {
        float v = x[(int64_t)row * S + col];
        tmax = fmaxf(tmax, v);
    }
    float row_max = block_reduce_max(tmax);

    // 2) sum exp
    float tsum = 0.0f;
    for (int col = threadIdx.x; col < S; col += blockDim.x) {
        float v = __expf(x[(int64_t)row * S + col] - row_max);
        tsum += v;
    }
    float sum_exp = block_reduce_sum(tsum);
    float inv_sum = 1.0f / (sum_exp + 1e-12f);

    // 3) write softmax and sum it (explicit L1 renorm)
    float soft_local = 0.0f;
    for (int col = threadIdx.x; col < S; col += blockDim.x) {
        float e = __expf(x[(int64_t)row * S + col] - row_max);
        float s = e * inv_sum;
        y[(int64_t)row * S + col] = s;
        soft_local += s;
    }
    float sum_soft = block_reduce_sum(soft_local);
    float inv_sum_soft = 1.0f / (sum_soft + 1e-12f);

    for (int col = threadIdx.x; col < S; col += blockDim.x) {
        y[(int64_t)row * S + col] *= inv_sum_soft;
    }
}

torch::Tensor softmax_l1_fused_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "softmax_l1_fused_cuda expects a 2D tensor [M, S]");
    int64_t M64 = x.size(0);
    int64_t S64 = x.size(1);
    TORCH_CHECK(M64 > 0 && S64 > 0, "Invalid shape");
    TORCH_CHECK(S64 <= 8192, "S too large for this kernel");

    int M = (int)M64;
    int S = (int)S64;

    auto y = torch::empty_like(x);

    if (S == 64) {
        // Keep multi-warp blocks (avoid tiny-CTA grid effects); all lanes active per warp.
        constexpr int WARPS_PER_BLOCK = 8; // 256 threads
        dim3 threads(WARPS_PER_BLOCK * 32);
        dim3 blocks((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        softmax64_warp2_multiwarp_kernel<WARPS_PER_BLOCK><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            M
        );
        return y;
    }

    int threads = 256;
    if (S <= 64) threads = 128;
    else if (S <= 256) threads = 256;
    else threads = 512;
    if (threads > 1024) threads = 1024;

    dim3 blocks(M);
    softmax_fused_generic_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        M, S
    );
    return y;
}
"""

cpp_src = r"""
torch::Tensor softmax_l1_fused_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_external_attention_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["softmax_l1_fused_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    External Attention with optimized fused CUDA kernel for softmax (+ explicit L1 renorm for generic S).
    S=64 path uses an all-lanes-active warp kernel with multi-warp blocks.
    """
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.custom_ops = custom_ops_lib
        self.S = S

    def forward(self, queries):
        attn = self.mk(queries)  # [bs, n, S]
        bs, n, S = attn.shape
        attn2d = attn.reshape(bs * n, S).contiguous()
        attn2d = self.custom_ops.softmax_l1_fused_cuda(attn2d)
        attn = attn2d.view(bs, n, S)
        out = self.mv(attn)  # [bs, n, d_model]
        return out