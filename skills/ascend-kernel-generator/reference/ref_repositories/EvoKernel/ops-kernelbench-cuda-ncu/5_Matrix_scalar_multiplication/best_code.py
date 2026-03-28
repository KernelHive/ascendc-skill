import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __launch_bounds__(t, b)
#endif

static __forceinline__ __device__ float ro_load_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float4 ro_load_f32x4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    // __ldg for float4 is supported on modern toolchains; if not, compiler falls back.
    return __ldg(p);
#else
    return *p;
#endif
}

template<int UNROLL>
__global__ __launch_bounds__(256, 2)
void mul_scalar_f32_kernel(const float* __restrict__ A,
                           float* __restrict__ C,
                           float s,
                           int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t base = tid; base < n; base += stride * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t idx = base + (int64_t)u * stride;
            if (idx < n) {
                float a = ro_load_f32(A + idx);
                C[idx] = a * s;
            }
        }
    }
}

template<int UNROLL>
__global__ __launch_bounds__(256, 2)
void mul_scalar_f32x4_kernel(const float4* __restrict__ A4,
                             float4* __restrict__ C4,
                             float s,
                             int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t base = tid; base < n4; base += stride * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t idx = base + (int64_t)u * stride;
            if (idx < n4) {
                float4 v = ro_load_f32x4(A4 + idx);
                v.x *= s; v.y *= s; v.z *= s; v.w *= s;
                C4[idx] = v;
            }
        }
    }
}

static inline int div_up_int64(int64_t a, int64_t b) {
    return (int)((a + b - 1) / b);
}

static int get_sm_count() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return prop.multiProcessorCount;
}

template <typename KernelT>
static inline int occupancy_blocks_per_sm(KernelT kernel, int threads) {
    int blocks_per_sm = 0;
    // dynamic shared mem = 0
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0);
    return blocks_per_sm;
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, double s) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");

    auto C = torch::empty_like(A);
    const int64_t n = A.numel();
    if (n == 0) return C;

    const int threads = 256;
    const int sms = get_sm_count();
    const float sf = (float)s;

    const uintptr_t ap = (uintptr_t)A.data_ptr<float>();
    const uintptr_t cp = (uintptr_t)C.data_ptr<float>();

    // Prefer float4 when 16B aligned and divisible by 4.
    if (((ap | cp) & 15u) == 0u && (n % 4 == 0)) {
        const int64_t n4 = n / 4;
        int blocks_per_sm = occupancy_blocks_per_sm(mul_scalar_f32x4_kernel<2>, threads);
        if (blocks_per_sm < 1) blocks_per_sm = 1;
        int grid = blocks_per_sm * sms;
        // ensure at least enough blocks to cover without relying on stride too much
        int cover = div_up_int64(n4, threads);
        if (grid < cover) grid = cover;

        mul_scalar_f32x4_kernel<2><<<grid, threads>>>(
            (const float4*)A.data_ptr<float>(),
            (float4*)C.data_ptr<float>(),
            sf,
            n4
        );
    } else {
        int blocks_per_sm = occupancy_blocks_per_sm(mul_scalar_f32_kernel<2>, threads);
        if (blocks_per_sm < 1) blocks_per_sm = 1;
        int grid = blocks_per_sm * sms;
        int cover = div_up_int64(n, threads);
        if (grid < cover) grid = cover;

        mul_scalar_f32_kernel<2><<<grid, threads>>>(
            (const float*)A.data_ptr<float>(),
            (float*)C.data_ptr<float>(),
            sf,
            n
        );
    }
    return C;
}

torch::Tensor matrix_scalar_mul_out_cuda(torch::Tensor A, double s, torch::Tensor out) {
    TORCH_CHECK(A.is_cuda() && out.is_cuda(), "A and out must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && out.dtype() == torch::kFloat32, "A and out must be float32");
    TORCH_CHECK(A.is_contiguous() && out.is_contiguous(), "A and out must be contiguous");
    TORCH_CHECK(A.numel() == out.numel(), "out must have same number of elements as A");

    const int64_t n = A.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int sms = get_sm_count();
    const float sf = (float)s;

    const uintptr_t ap = (uintptr_t)A.data_ptr<float>();
    const uintptr_t op = (uintptr_t)out.data_ptr<float>();

    if (((ap | op) & 15u) == 0u && (n % 4 == 0)) {
        const int64_t n4 = n / 4;
        int blocks_per_sm = occupancy_blocks_per_sm(mul_scalar_f32x4_kernel<2>, threads);
        if (blocks_per_sm < 1) blocks_per_sm = 1;
        int grid = blocks_per_sm * sms;
        int cover = div_up_int64(n4, threads);
        if (grid < cover) grid = cover;

        mul_scalar_f32x4_kernel<2><<<grid, threads>>>(
            (const float4*)A.data_ptr<float>(),
            (float4*)out.data_ptr<float>(),
            sf,
            n4
        );
    } else {
        int blocks_per_sm = occupancy_blocks_per_sm(mul_scalar_f32_kernel<2>, threads);
        if (blocks_per_sm < 1) blocks_per_sm = 1;
        int grid = blocks_per_sm * sms;
        int cover = div_up_int64(n, threads);
        if (grid < cover) grid = cover;

        mul_scalar_f32_kernel<2><<<grid, threads>>>(
            (const float*)A.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            sf,
            n
        );
    }
    return out;
}
"""

cpp_source = r"""
torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, double s);
torch::Tensor matrix_scalar_mul_out_cuda(torch::Tensor A, double s, torch::Tensor out);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matrix_scalar_mul_opt",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matrix_scalar_mul_cuda", "matrix_scalar_mul_out_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replaces A * s with an optimized custom CUDA kernel (occupancy-sized grid-stride + float4 fast path).
    Optionally reuses an output buffer to reduce allocation overhead.
    """
    def __init__(self, reuse_output: bool = False):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib
        self.reuse_output = reuse_output
        self._out = None

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        if not A.is_cuda:
            return A * s
        if A.dtype != torch.float32:
            A = A.float()
        if not A.is_contiguous():
            A = A.contiguous()

        if not self.reuse_output:
            return self.custom_ops_lib.matrix_scalar_mul_cuda(A, float(s))

        if self._out is None or self._out.numel() != A.numel() or self._out.device != A.device:
            self._out = torch.empty_like(A)
        elif self._out.shape != A.shape:
            self._out = self._out.view_as(A)

        return self.custom_ops_lib.matrix_scalar_mul_out_cuda(A, float(s), self._out)