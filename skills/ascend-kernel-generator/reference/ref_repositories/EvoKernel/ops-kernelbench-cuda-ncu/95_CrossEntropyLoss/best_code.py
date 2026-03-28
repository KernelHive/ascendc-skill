import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

// ----------------------------
// Warp-level primitives
// ----------------------------
__inline__ __device__ float warp_allreduce_max(float v) {
    // full mask
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}
__inline__ __device__ float warp_allreduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Reduce across warps using a small shared buffer (one per warp).
// Returns valid value in lane0 of warp0; other threads undefined.
template<int WARPS>
__inline__ __device__ float block_reduce_max(float v, float* smem_warp) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_allreduce_max(v);
    if (lane == 0) smem_warp[warp] = v;
    __syncthreads();
    float out = -INFINITY;
    if (warp == 0) {
        out = (lane < WARPS) ? smem_warp[lane] : -INFINITY;
        out = warp_allreduce_max(out);
        if (lane == 0) smem_warp[0] = out;
    }
    __syncthreads();
    return smem_warp[0];
}

template<int WARPS>
__inline__ __device__ float block_reduce_sum(float v, float* smem_warp) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_allreduce_sum(v);
    if (lane == 0) smem_warp[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < WARPS) ? smem_warp[lane] : 0.0f;
        out = warp_allreduce_sum(out);
        if (lane == 0) smem_warp[0] = out;
    }
    __syncthreads();
    return smem_warp[0];
}

// ----------------------------
// Kernel: compute per-block partial sums of per-row CE losses
// Two rows per block to increase ILP/latency hiding.
// One block produces up to 2 row losses, accumulates them locally, then writes one partial sum.
// ----------------------------
template<int THREADS>
__global__ void cross_entropy_partialsum_2rows_f32_i64_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ target,
    float* __restrict__ block_partial_sums, // size = gridDim.x
    int B, int C
) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float smem_warp[WARPS]; // reused for max/sum reductions

    const int block_row0 = (int)blockIdx.x * 2;
    const int block_row1 = block_row0 + 1;

    float block_acc = 0.0f;

    // helper lambda to process one row index if valid
    auto process_row = [&](int row) {
        if (row >= B) return;

        // Load & clamp target safely (avoid OOB)
        int64_t t64 = __ldg(target + row);
        if (t64 < 0) t64 = 0;
        if (t64 >= (int64_t)C) t64 = (int64_t)C - 1;
        const int ti = (int)t64;

        const float* row_ptr = x + (size_t)row * (size_t)C;

        // 1) max
        float local_max = -INFINITY;

        // Try vectorized float4 loads if aligned and C%4==0
        const uintptr_t addr = (uintptr_t)row_ptr;
        const bool aligned16 = ((addr & 0xF) == 0);
        const bool vec4_ok = aligned16 && ((C & 3) == 0);

        if (vec4_ok) {
            const float4* __restrict__ p4 = (const float4*)row_ptr;
            const int C4 = C >> 2;
            for (int i = threadIdx.x; i < C4; i += THREADS) {
                float4 v4 = __ldg(p4 + i);
                local_max = fmaxf(local_max, v4.x);
                local_max = fmaxf(local_max, v4.y);
                local_max = fmaxf(local_max, v4.z);
                local_max = fmaxf(local_max, v4.w);
            }
        } else {
            for (int col = threadIdx.x; col < C; col += THREADS) {
                float v = __ldg(row_ptr + col);
                local_max = fmaxf(local_max, v);
            }
        }

        float row_max = block_reduce_max<WARPS>(local_max, smem_warp);

        // 2) sum exp(x - max)
        float local_sum = 0.0f;
        if (vec4_ok) {
            const float4* __restrict__ p4 = (const float4*)row_ptr;
            const int C4 = C >> 2;
            for (int i = threadIdx.x; i < C4; i += THREADS) {
                float4 v4 = __ldg(p4 + i);
                local_sum += __expf(v4.x - row_max);
                local_sum += __expf(v4.y - row_max);
                local_sum += __expf(v4.z - row_max);
                local_sum += __expf(v4.w - row_max);
            }
        } else {
            for (int col = threadIdx.x; col < C; col += THREADS) {
                float v = __ldg(row_ptr + col);
                local_sum += __expf(v - row_max);
            }
        }

        float row_sum = block_reduce_sum<WARPS>(local_sum, smem_warp);

        // 3) row loss
        float x_t = __ldg(row_ptr + ti);
        float row_loss = logf(fmaxf(row_sum, 1e-20f)) + row_max - x_t;

        // Accumulate into block sum (one thread does it)
        if (threadIdx.x == 0) block_acc += row_loss;
        __syncthreads(); // ensure block_acc updated before possibly reusing reductions
    };

    process_row(block_row0);
    process_row(block_row1);

    if (threadIdx.x == 0) {
        block_partial_sums[blockIdx.x] = block_acc;
    }
}

// Final reduction kernel: reduce block_partial_sums to a single scalar and apply mean (1/B).
// Single block is enough because gridDim.x ~= ceil(B/2) = 16384 here.
// Uses THREADS threads and strided loads.
template<int THREADS>
__global__ void final_reduce_mean_f32_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int N, int B
) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float smem_warp[WARPS];

    float local = 0.0f;
    for (int i = threadIdx.x; i < N; i += THREADS) {
        local += __ldg(partial + i);
    }
    float sum = block_reduce_sum<WARPS>(local, smem_warp);
    if (threadIdx.x == 0) {
        out[0] = sum / (float)B;
    }
}

torch::Tensor cross_entropy_fwd_mean_f32_i64_cuda(torch::Tensor x, torch::Tensor target) {
    CHECK_INPUT(x);
    CHECK_INPUT(target);

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(target.scalar_type() == at::kLong, "target must be int64");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, C]");
    TORCH_CHECK(target.dim() == 1, "target must be 1D [B]");
    TORCH_CHECK(x.size(0) == target.size(0), "x and target must have same batch size");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);

    auto out = torch::zeros({}, x.options()); // scalar float32
    if (B == 0) return out;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    // Kernel config
    constexpr int THREADS = 256;
    const int blocks = (B + 1) / 2; // 2 rows per block
    auto partial = torch::empty({blocks}, x.options());

    cross_entropy_partialsum_2rows_f32_i64_kernel<THREADS><<<blocks, THREADS, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const int64_t*)target.data_ptr<int64_t>(),
        (float*)partial.data_ptr<float>(),
        B, C
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Final reduction (single block)
    final_reduce_mean_f32_kernel<THREADS><<<1, THREADS, 0, stream>>>(
        (const float*)partial.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        blocks,
        B
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor cross_entropy_fwd_mean_f32_i64_cuda(torch::Tensor x, torch::Tensor target);
"""

_ext_name = "custom_ops_lib_cross_entropy_opt2"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cross_entropy_fwd_mean_f32_i64_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    Cross Entropy Loss model using a fused custom CUDA kernel (forward only):
      - predictions: float32 CUDA contiguous, shape [B, C]
      - targets: int64 CUDA contiguous, shape [B]
      - reduction: mean
    Falls back to torch.nn.functional.cross_entropy otherwise.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if (
            predictions.is_cuda and targets.is_cuda
            and predictions.dtype == torch.float32
            and targets.dtype == torch.int64
            and predictions.dim() == 2
            and targets.dim() == 1
            and predictions.is_contiguous()
            and targets.is_contiguous()
            and predictions.size(0) == targets.size(0)
        ):
            return self.custom_ops_lib.cross_entropy_fwd_mean_f32_i64_cuda(predictions, targets)
        return F.cross_entropy(predictions, targets)