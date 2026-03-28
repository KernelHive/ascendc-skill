import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Optimized CUDA/C++ extension: fused MSE loss (mean squared error) ----
# Computes: mean((predictions - targets)^2) over all elements.

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_DTYPE_FLOAT
#define CHECK_DTYPE_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static __inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __inline__ __device__ float block_reduce_sum(float v) {
    v = warp_reduce_sum(v);
    __shared__ float smem[32]; // max 1024 threads -> 32 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        out = (lane < nwarps) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__global__ void mse_stage1_vec4_kernel(const float* __restrict__ pred,
                                       const float* __restrict__ tgt,
                                       float* __restrict__ partial,
                                       int64_t n) {
    float sum = 0.0f;

    int64_t n4 = n >> 2; // number of float4 items
    const float4* p4 = reinterpret_cast<const float4*>(pred);
    const float4* t4 = reinterpret_cast<const float4*>(tgt);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // grid-stride over float4 chunks
    for (int64_t i = tid; i < n4; i += stride) {
        float4 a = p4[i];
        float4 b = t4[i];
        float dx0 = a.x - b.x; sum += dx0 * dx0;
        float dx1 = a.y - b.y; sum += dx1 * dx1;
        float dx2 = a.z - b.z; sum += dx2 * dx2;
        float dx3 = a.w - b.w; sum += dx3 * dx3;
    }

    // tail (0..3 elements)
    int64_t base = (n4 << 2);
    for (int64_t j = base + tid; j < n; j += stride) {
        float d = pred[j] - tgt[j];
        sum += d * d;
    }

    float bsum = block_reduce_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

__global__ void mse_stage1_scalar_kernel(const float* __restrict__ pred,
                                         const float* __restrict__ tgt,
                                         float* __restrict__ partial,
                                         int64_t n) {
    float sum = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // modest unrolling to reduce loop overhead
    for (int64_t i = tid; i < n; i += stride * 4) {
        int64_t i0 = i;
        int64_t i1 = i + stride;
        int64_t i2 = i + 2 * stride;
        int64_t i3 = i + 3 * stride;

        if (i0 < n) { float d = pred[i0] - tgt[i0]; sum += d * d; }
        if (i1 < n) { float d = pred[i1] - tgt[i1]; sum += d * d; }
        if (i2 < n) { float d = pred[i2] - tgt[i2]; sum += d * d; }
        if (i3 < n) { float d = pred[i3] - tgt[i3]; sum += d * d; }
    }

    float bsum = block_reduce_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = bsum;
}

__global__ void mse_stage2_finalize_kernel(const float* __restrict__ partial,
                                          float* __restrict__ out_mean,
                                          int64_t num_partials,
                                          float inv_n) {
    float sum = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = tid; i < num_partials; i += stride) {
        sum += partial[i];
    }

    float bsum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        // grid is expected small; still use atomic to be safe if >1 block
        atomicAdd(out_mean, bsum * inv_n);
    }
}

torch::Tensor mse_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_CUDA(predictions);
    CHECK_CUDA(targets);
    CHECK_CONTIGUOUS(predictions);
    CHECK_CONTIGUOUS(targets);
    CHECK_DTYPE_FLOAT(predictions);
    CHECK_DTYPE_FLOAT(targets);
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");

    const int64_t n = predictions.numel();
    TORCH_CHECK(n > 0, "Input tensors must have at least one element");

    auto out = torch::zeros({}, torch::TensorOptions().device(predictions.device()).dtype(torch::kFloat32));
    float* out_ptr = out.data_ptr<float>();

    int threads = 256;

    // Stage 1: choose a reasonably large grid to saturate memory BW, but not huge.
    // Use more blocks than SMs for latency hiding; cap to avoid excessive temp size.
    int device = predictions.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm_count = prop.multiProcessorCount;

    int64_t max_blocks = 8192;
    int64_t blocks = (n + threads - 1) / threads;
    int64_t target_blocks = (int64_t)sm_count * 20; // heuristic for memory-bound kernels
    if (blocks < target_blocks) blocks = target_blocks;
    if (blocks < 1) blocks = 1;
    if (blocks > max_blocks) blocks = max_blocks;

    auto partial = torch::empty({blocks}, torch::TensorOptions().device(predictions.device()).dtype(torch::kFloat32));

    const float* pred_ptr = predictions.data_ptr<float>();
    const float* tgt_ptr  = targets.data_ptr<float>();

    bool aligned16 = (((uintptr_t)pred_ptr & 0xF) == 0) && (((uintptr_t)tgt_ptr & 0xF) == 0);
    // Use vec4 path when aligned and n reasonably large
    if (aligned16 && n >= 1024) {
        mse_stage1_vec4_kernel<<<(int)blocks, threads>>>(pred_ptr, tgt_ptr, partial.data_ptr<float>(), n);
    } else {
        mse_stage1_scalar_kernel<<<(int)blocks, threads>>>(pred_ptr, tgt_ptr, partial.data_ptr<float>(), n);
    }

    // Stage 2: reduce partials to final mean (fuse division)
    int64_t num_partials = blocks;

    // Keep stage2 grid small; ideally 1 block, but allow a few blocks for large partial counts.
    int64_t blocks2 = (num_partials + threads - 1) / threads;
    int64_t max_blocks2 = (int64_t)sm_count * 2;
    if (blocks2 < 1) blocks2 = 1;
    if (blocks2 > max_blocks2) blocks2 = max_blocks2;

    // out already zero; stage2 atomics accumulate mean
    float inv_n = 1.0f / (float)n;
    mse_stage2_finalize_kernel<<<(int)blocks2, threads>>>(partial.data_ptr<float>(), out_ptr, num_partials, inv_n);

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mse_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

custom_ops_lib = load_inline(
    name="custom_ops_mse_loss_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mse_loss_mean_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    MSE loss using optimized custom CUDA kernels:
      loss = mean((predictions - targets)^2)
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.is_cuda and targets.is_cuda:
            if predictions.dtype != torch.float32:
                predictions = predictions.float()
            if targets.dtype != torch.float32:
                targets = targets.float()
            if not predictions.is_contiguous():
                predictions = predictions.contiguous()
            if not targets.is_contiguous():
                targets = targets.contiguous()
            return self.custom_ops_lib.mse_loss_mean_cuda(predictions, targets)

        return torch.mean((predictions - targets) ** 2)