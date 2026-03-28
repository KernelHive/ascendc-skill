import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------
# Optimized CUDA smooth_l1_loss (Huber) operator: mean reduce
# - Single-pass sum kernel + tiny finalize kernel
# - float4 vectorized loads when aligned
# - Warp-shuffle reduction + one atomicAdd per block
# - Matches PyTorch semantics for beta==0 (L1)
# -------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float huber_elem(float diff, float beta) {
    float ad = fabsf(diff);
    // beta==0 => L1
    if (beta == 0.0f) return ad;
    if (ad < beta) {
        return 0.5f * (diff * diff) / beta;
    } else {
        return ad - 0.5f * beta;
    }
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

__global__ void smooth_l1_sum_block_atomic_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ out_sum,
    int64_t n,
    float beta
) {
    float local = 0.0f;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Vectorized if both pointers 16B-aligned
    uintptr_t p_addr = (uintptr_t)pred;
    uintptr_t t_addr = (uintptr_t)target;
    bool vec_ok = ((p_addr & 0xF) == 0) && ((t_addr & 0xF) == 0);

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    if (vec_ok) {
        const float4* __restrict__ p4 = reinterpret_cast<const float4*>(pred);
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(target);
        int64_t n4 = n >> 2; // floor(n/4)
        int64_t idx4 = idx;
        int64_t stride4 = stride;

        // process 4-wide chunks
        for (int64_t i = idx4; i < n4; i += stride4) {
            float4 a = p4[i];
            float4 b = t4[i];
            local += huber_elem(a.x - b.x, beta);
            local += huber_elem(a.y - b.y, beta);
            local += huber_elem(a.z - b.z, beta);
            local += huber_elem(a.w - b.w, beta);
        }

        // scalar tail for remaining (n % 4) elements:
        // Use only first few threads of the whole grid-stride domain once.
        int64_t base = (n4 << 2);
        for (int64_t j = base + idx; j < n; j += stride) {
            local += huber_elem(pred[j] - target[j], beta);
        }
    } else {
        for (int64_t i = idx; i < n; i += stride) {
            local += huber_elem(pred[i] - target[i], beta);
        }
    }

    // Reduce within warp
    float wsum = warp_reduce_sum(local);

    // Reduce warp sums to block sum using shared memory (one float per warp)
    __shared__ float warp_sums[8]; // supports up to 256 threads (8 warps)
    if (lane == 0) warp_sums[warp] = wsum;
    __syncthreads();

    float bsum = 0.0f;
    if (warp == 0) {
        // First warp loads up to num_warps values
        int num_warps = blockDim.x >> 5;
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            atomicAdd(out_sum, v); // one atomic per block
        }
    }
}

__global__ void finalize_mean_f32(
    const float* __restrict__ sum,
    float* __restrict__ out,
    int64_t n
) {
    if (threadIdx.x == 0) {
        out[0] = (n == 0) ? 0.0f : (sum[0] / (float)n);
    }
}

torch::Tensor smooth_l1_loss_mean_cuda(torch::Tensor predictions,
                                      torch::Tensor targets,
                                      double beta) {
    TORCH_CHECK(predictions.is_cuda(), "smooth_l1_loss_mean_cuda: predictions must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "smooth_l1_loss_mean_cuda: targets must be CUDA");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "smooth_l1_loss_mean_cuda: only float32 supported");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "smooth_l1_loss_mean_cuda: only float32 supported");
    TORCH_CHECK(predictions.is_contiguous(), "smooth_l1_loss_mean_cuda: predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "smooth_l1_loss_mean_cuda: targets must be contiguous");
    TORCH_CHECK(predictions.numel() == targets.numel(), "smooth_l1_loss_mean_cuda: size mismatch");
    TORCH_CHECK(beta >= 0.0, "smooth_l1_loss_mean_cuda: beta must be >= 0");

    int64_t n = predictions.numel();
    auto out = torch::zeros({}, predictions.options());
    if (n == 0) return out;

    // Global accumulator
    auto sum = torch::zeros({1}, predictions.options());

    // Launch config:
    // Keep blocks moderate to limit atomic contention but enough for latency hiding.
    // Heuristic: min(4096, ceil(n/(threads*4)) * 4)  (scaled up modestly)
    const int threads = 256;
    int64_t blocks64 = (n + (int64_t)threads * 4 - 1) / ((int64_t)threads * 4);
    blocks64 = blocks64 * 4;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 4096) blocks64 = 4096;
    int blocks = (int)blocks64;

    smooth_l1_sum_block_atomic_f32<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        sum.data_ptr<float>(),
        n,
        (float)beta
    );

    finalize_mean_f32<<<1, 32>>>(
        sum.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}
"""

cpp_source = r"""
torch::Tensor smooth_l1_loss_mean_cuda(torch::Tensor predictions,
                                      torch::Tensor targets,
                                      double beta);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_huber_smoothl1_opt_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["smooth_l1_loss_mean_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA Smooth L1 (Huber) loss kernel.
    Implements torch.nn.functional.smooth_l1_loss(predictions, targets)
    with default reduction='mean' and default beta=1.0.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Keep default semantics: reduction='mean', beta=1.0
        if (not predictions.is_cuda) or (not targets.is_cuda):
            return F.smooth_l1_loss(predictions, targets)
        if predictions.dtype != torch.float32 or targets.dtype != torch.float32:
            return F.smooth_l1_loss(predictions, targets)
        if predictions.numel() != targets.numel():
            return F.smooth_l1_loss(predictions, targets)
        if not predictions.is_contiguous():
            predictions = predictions.contiguous()
        if not targets.is_contiguous():
            targets = targets.contiguous()
        return self.custom_ops_lib.smooth_l1_loss_mean_cuda(predictions, targets, 1.0)


# Keep original input helpers for integration consistency
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand((), device="cuda")
    return [
        (torch.rand(batch_size, *input_shape, device="cuda", dtype=torch.float32) * scale),
        torch.rand(batch_size, *input_shape, device="cuda", dtype=torch.float32),
    ]

def get_init_inputs():
    return []