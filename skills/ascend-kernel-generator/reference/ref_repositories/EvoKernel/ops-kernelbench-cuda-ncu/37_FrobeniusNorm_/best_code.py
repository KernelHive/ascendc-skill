import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__inline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Warp-aggregated atomic add: one atomic per warp instead of per thread.
__inline__ __device__ void warp_agg_atomic_add(float* addr, float val) {
    int lane = threadIdx.x & 31;
    // sum within warp
    float sum = warp_reduce_sum(val);
    if (lane == 0) atomicAdd(addr, sum);
}

// Single-kernel reduction to global scalar (sumsq[0]).
__global__ void frob_sumsq_atomic_vec4_kernel(const float* __restrict__ x,
                                              float* __restrict__ sumsq,
                                              int64_t n) {
    float local = 0.0f;

    // Prefer float4 path if base pointer is 16B aligned.
    const uintptr_t base_addr = (uintptr_t)x;
    const bool aligned16 = (base_addr & 0xF) == 0;

    if (aligned16) {
        const int64_t n4 = n >> 2; // number of float4
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);

        int64_t idx4 = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t stride4 = (int64_t)blockDim.x * (int64_t)gridDim.x;

        for (int64_t i4 = idx4; i4 < n4; i4 += stride4) {
            float4 v = x4[i4];
            local = fmaf(v.x, v.x, local);
            local = fmaf(v.y, v.y, local);
            local = fmaf(v.z, v.z, local);
            local = fmaf(v.w, v.w, local);
        }

        // Tail (n % 4) handled by first few threads
        int64_t tail = n - (n4 << 2);
        if (tail) {
            int64_t base = (n4 << 2);
            for (int t = (int)threadIdx.x; t < (int)tail; t += blockDim.x) {
                float vv = ldg_f32(x + base + t);
                local = fmaf(vv, vv, local);
            }
        }
    } else {
        // Scalar path with modest unrolling for ILP
        int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

        for (int64_t i = idx; i < n; i += 4 * stride) {
            float v0 = ldg_f32(x + i);
            local = fmaf(v0, v0, local);

            int64_t i1 = i + stride;
            if (i1 < n) { float v1 = ldg_f32(x + i1); local = fmaf(v1, v1, local); }

            int64_t i2 = i + 2 * stride;
            if (i2 < n) { float v2 = ldg_f32(x + i2); local = fmaf(v2, v2, local); }

            int64_t i3 = i + 3 * stride;
            if (i3 < n) { float v3 = ldg_f32(x + i3); local = fmaf(v3, v3, local); }
        }
    }

    // Reduce within block then warp-aggregated atomic to global.
    float bsum = block_reduce_sum(local);
    // Only warp0 participates; still one atomic per warp (here effectively one atomic per block)
    if ((threadIdx.x >> 5) == 0) {
        float v = (threadIdx.x < 32) ? bsum : 0.0f;
        warp_agg_atomic_add(sumsq, v);
    }
}

__global__ void frob_normalize_vec4_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int64_t n,
                                          const float* __restrict__ sumsq_scalar) {
    float sumsq = sumsq_scalar[0];
    // Preserve IEEE behavior: if sumsq==0 => inv_norm=inf; 0*inf => NaN as IEEE dictates.
    float inv_norm = rsqrtf(sumsq);

    const uintptr_t ax = (uintptr_t)x;
    const uintptr_t ay = (uintptr_t)y;
    const bool aligned16 = ((ax | ay) & 0xF) == 0;

    if (aligned16) {
        const int64_t n4 = n >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

        int64_t idx4 = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t stride4 = (int64_t)blockDim.x * (int64_t)gridDim.x;

        for (int64_t i4 = idx4; i4 < n4; i4 += stride4) {
            float4 v = x4[i4];
            float4 o;
            o.x = v.x * inv_norm;
            o.y = v.y * inv_norm;
            o.z = v.z * inv_norm;
            o.w = v.w * inv_norm;
            y4[i4] = o;
        }

        int64_t tail = n - (n4 << 2);
        if (tail) {
            int64_t base = (n4 << 2);
            for (int t = (int)threadIdx.x; t < (int)tail; t += blockDim.x) {
                y[base + t] = ldg_f32(x + base + t) * inv_norm;
            }
        }
    } else {
        int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

        // unroll by 4 for better ILP
        for (int64_t i = idx; i < n; i += 4 * stride) {
            float v0 = ldg_f32(x + i);
            y[i] = v0 * inv_norm;

            int64_t i1 = i + stride;
            if (i1 < n) { float v1 = ldg_f32(x + i1); y[i1] = v1 * inv_norm; }

            int64_t i2 = i + 2 * stride;
            if (i2 < n) { float v2 = ldg_f32(x + i2); y[i2] = v2 * inv_norm; }

            int64_t i3 = i + 3 * stride;
            if (i3 < n) { float v3 = ldg_f32(x + i3); y[i3] = v3 * inv_norm; }
        }
    }
}

torch::Tensor frobenius_norm_normalize_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "frobenius_norm_normalize_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "frobenius_norm_normalize_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "frobenius_norm_normalize_cuda: x must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;

    // scalar on device
    auto sumsq_scalar = torch::zeros({1}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));

    const int threads = 256;

    // SM-aware grid sizing to limit atomic contention while keeping enough MLP.
    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    // For reduction: too many blocks => atomic contention; too few => low MLP.
    int blocks_red = sm * 6;
    if (blocks_red < 1) blocks_red = 1;
    if (blocks_red > 4096) blocks_red = 4096;

    frob_sumsq_atomic_vec4_kernel<<<(unsigned int)blocks_red, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)sumsq_scalar.data_ptr<float>(),
        n
    );

    // Normalize: bandwidth bound; allow more blocks, but cap reasonably.
    int blocks_norm = (int)((n + threads - 1) / threads);
    int max_blocks_norm = sm * 20;
    if (blocks_norm > max_blocks_norm) blocks_norm = max_blocks_norm;
    if (blocks_norm < 1) blocks_norm = 1;
    if (blocks_norm > 4096) blocks_norm = 4096;

    frob_normalize_vec4_kernel<<<(unsigned int)blocks_norm, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        n,
        (const float*)sumsq_scalar.data_ptr<float>()
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor frobenius_norm_normalize_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_frobenius_norm_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["frobenius_norm_normalize_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Frobenius norm normalization using optimized custom CUDA kernels.
    Fast path: contiguous CUDA float32 input.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return self.custom_ops_lib.frobenius_norm_normalize_cuda(x)

        # Fallback/adaptation
        if not x.is_cuda:
            norm = torch.norm(x, p="fro")
            return x / norm
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.frobenius_norm_normalize_cuda(x)