import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float tanh_fast_f32(float x) {
    // --use_fast_math makes this a fast approx on most GPUs
    return tanhf(x);
}

__device__ __forceinline__ float2 tanh_fast_f32x2(float2 v) {
    v.x = tanh_fast_f32(v.x);
    v.y = tanh_fast_f32(v.y);
    return v;
}

__device__ __forceinline__ float4 tanh_fast_f32x4(float4 v) {
    v.x = tanh_fast_f32(v.x);
    v.y = tanh_fast_f32(v.y);
    v.z = tanh_fast_f32(v.z);
    v.w = tanh_fast_f32(v.w);
    return v;
}

template<int UNROLL>
__global__ __launch_bounds__(256, 3)
void tanh_fwd_vec4_aligned_fulltiles(const float4* __restrict__ x4,
                                     float4* __restrict__ y4,
                                     int64_t n4_full) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // Process only full tiles so we can avoid per-iteration bounds checks in the hot loop.
    int64_t total = n4_full; // multiple of (stride*UNROLL) not guaranteed, so do a main loop + tail
    int64_t step = stride * (int64_t)UNROLL;

    // Main unrolled loop for indices where j is always in range
    int64_t limit = total - (total % step);
    for (int64_t base = tid; base < limit; base += step) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = base + (int64_t)u * stride;
            float4 v = x4[j];
            y4[j] = tanh_fast_f32x4(v);
        }
    }

    // Small tail with bounds checks (only up to step-1 iterations per thread across grid)
    for (int64_t j = limit + tid; j < total; j += stride) {
        float4 v = x4[j];
        y4[j] = tanh_fast_f32x4(v);
    }
}

__global__ __launch_bounds__(256, 3)
void tanh_fwd_vec2_aligned(const float2* __restrict__ x2,
                           float2* __restrict__ y2,
                           int64_t n2) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n2; i += stride) {
        float2 v = x2[i];
        y2[i] = tanh_fast_f32x2(v);
    }
}

__global__ __launch_bounds__(256, 3)
void tanh_fwd_scalar(const float* __restrict__ x,
                     float* __restrict__ y,
                     int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        y[i] = tanh_fast_f32(x[i]);
    }
}

static inline int clamp_grid_x(int64_t blocks64) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int max_x = prop.maxGridSize[0];
    if (max_x <= 0) max_x = 2147483647;

    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > (int64_t)max_x) blocks64 = (int64_t)max_x;
    return (int)blocks64;
}

static inline int heuristic_blocks_sm_aware(int64_t work_items, int threads) {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int sm = prop.multiProcessorCount;

    // cover work (1 item per thread)
    int64_t blocks_work = (work_items + (int64_t)threads - 1) / (int64_t)threads;

    // aim for ~6 waves per SM for latency hiding on streaming kernels
    int64_t blocks_target = (int64_t)sm * 6;
    int64_t blocks = blocks_work;
    if (blocks < blocks_target) blocks = blocks_target;

    return clamp_grid_x(blocks);
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "tanh_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tanh_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "tanh_cuda: input must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty_like(x);

    const int64_t n = x.numel();
    if (n == 0) return y;

    constexpr int THREADS = 256;
    constexpr int UNROLL4 = 4;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const float* xp = (const float*)x.data_ptr<float>();
    float* yp = (float*)y.data_ptr<float>();

    // Fast path: both pointers are 16B aligned, so float4 is safe and fully coalesced.
    const uintptr_t xpu = (uintptr_t)xp;
    const uintptr_t ypu = (uintptr_t)yp;
    const bool aligned16 = (((xpu | ypu) & 0xF) == 0);

    if (aligned16) {
        // float4 chunk
        int64_t n4 = n >> 2;            // number of float4
        int64_t r = n & 3;              // remaining scalars after float4

        if (n4 > 0) {
            // Choose blocks based on number of float4 items (work_items = n4)
            int blocks4 = heuristic_blocks_sm_aware(n4, THREADS);
            const float4* x4 = reinterpret_cast<const float4*>(xp);
            float4* y4 = reinterpret_cast<float4*>(yp);
            tanh_fwd_vec4_aligned_fulltiles<UNROLL4><<<blocks4, THREADS, 0, stream>>>(x4, y4, n4);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // Handle remainder using float2 then scalar to avoid an extra full unaligned kernel.
        int64_t offset = (n4 << 2);
        if (r >= 2) {
            const float2* x2 = reinterpret_cast<const float2*>(xp + offset);
            float2* y2 = reinterpret_cast<float2*>(yp + offset);
            // exactly 1 float2 when r==2 or r==3
            int blocks2 = heuristic_blocks_sm_aware(1, THREADS);
            tanh_fwd_vec2_aligned<<<blocks2, THREADS, 0, stream>>>(x2, y2, 1);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            offset += 2;
            r -= 2;
        }
        if (r > 0) {
            // 1 scalar left
            int blocks1 = 1;
            tanh_fwd_scalar<<<blocks1, THREADS, 0, stream>>>(xp + offset, yp + offset, r);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        return y;
    }

    // Fallback: fully scalar streaming kernel (safe for any alignment)
    int blocks = heuristic_blocks_sm_aware(n, THREADS);
    tanh_fwd_scalar<<<blocks, THREADS, 0, stream>>>(xp, yp, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor tanh_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_tanh_vec4_vec2_scalar_dispatch_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["tanh_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32):
            return torch.tanh(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.tanh_cuda(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []