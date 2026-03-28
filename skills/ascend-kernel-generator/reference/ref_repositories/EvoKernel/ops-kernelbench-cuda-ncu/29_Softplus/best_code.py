import torch
import torch.nn as nn
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
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

__device__ __forceinline__ float softplus_stable_f32(float x) {
    // softplus(x) = max(x, 0) + log1p(exp(-abs(x)))
    float ax = fabsf(x);
    float m = fmaxf(x, 0.0f);
    // fast-math enabled: expf/log1pf may map to faster intrinsics where allowed
    return m + log1pf(expf(-ax));
}

template<int UNROLL_VEC>
__global__ void softplus_f32_vec4_kernel(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int64_t n) {
    // n is number of float elements
    int64_t n4 = n >> 2;  // number of float4s
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    // grid-stride over float4 units, unrolled
    int64_t i = tid;
    for (; i + (UNROLL_VEC - 1) * stride < n4; i += stride * UNROLL_VEC) {
        #pragma unroll
        for (int u = 0; u < UNROLL_VEC; ++u) {
            int64_t idx4 = i + (int64_t)u * stride;
            float4 v = x4[idx4];
            float4 o;
            o.x = softplus_stable_f32(v.x);
            o.y = softplus_stable_f32(v.y);
            o.z = softplus_stable_f32(v.z);
            o.w = softplus_stable_f32(v.w);
            y4[idx4] = o;
        }
    }
    // tail for remaining float4s
    for (; i < n4; i += stride) {
        float4 v = x4[i];
        float4 o;
        o.x = softplus_stable_f32(v.x);
        o.y = softplus_stable_f32(v.y);
        o.z = softplus_stable_f32(v.z);
        o.w = softplus_stable_f32(v.w);
        y4[i] = o;
    }

    // scalar tail (n not divisible by 4): handled by separate kernel to keep vec kernel clean
}

template<int UNROLL_SCALAR>
__global__ void softplus_f32_scalar_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // grid-stride loop with a small unroll to increase ILP
    int64_t i = tid;
    for (; i + (UNROLL_SCALAR - 1) * stride < n; i += stride * UNROLL_SCALAR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_SCALAR; ++u) {
            int64_t idx = i + (int64_t)u * stride;
#if __CUDA_ARCH__ >= 350
            float xv = __ldg(x + idx);
#else
            float xv = x[idx];
#endif
            y[idx] = softplus_stable_f32(xv);
        }
    }
    for (; i < n; i += stride) {
#if __CUDA_ARCH__ >= 350
        float xv = __ldg(x + i);
#else
        float xv = x[i];
#endif
        y[i] = softplus_stable_f32(xv);
    }
}

torch::Tensor softplus_f32_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "softplus_f32_cuda only supports float32");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    // Launch config: try larger blocks to reduce overhead and improve memory throughput.
    // Cap grid to avoid excessive blocks when n is huge; still enough to fill the GPU.
    const int threads = 256;

    // Heuristic grid size: up to 65535 blocks (1D grid limit for older CUDA),
    // but typically fewer based on workload.
    int64_t max_blocks = 65535;
    int64_t blocks_for_n = (n + threads - 1) / threads;
    int blocks = (int)min(blocks_for_n, max_blocks);

    const uintptr_t x_addr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t y_addr = (uintptr_t)y.data_ptr<float>();
    const bool aligned16 = ((x_addr & 0xF) == 0) && ((y_addr & 0xF) == 0);

    // Vectorized path if aligned; handle scalar tail separately if needed.
    if (aligned16 && (n >= 4)) {
        int64_t n_vec = (n / 4) * 4;  // largest multiple of 4
        // Use vec4 kernel for bulk
        constexpr int UNROLL_VEC = 2;
        int64_t n4 = n_vec >> 2;
        int64_t blocks_for_n4 = (n4 + threads - 1) / threads;
        int blocks4 = (int)min(blocks_for_n4, max_blocks);

        softplus_f32_vec4_kernel<UNROLL_VEC><<<blocks4, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            n_vec
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Tail (0..3 elements)
        int64_t tail = n - n_vec;
        if (tail) {
            const float* xt = (const float*)x.data_ptr<float>() + n_vec;
            float* yt = (float*)y.data_ptr<float>() + n_vec;
            constexpr int UNROLL_SCALAR = 4;
            int64_t blocks_tail = (tail + threads - 1) / threads;
            int btail = (int)min(blocks_tail, max_blocks);
            softplus_f32_scalar_kernel<UNROLL_SCALAR><<<btail, threads, 0, stream>>>(xt, yt, tail);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return y;
    }

    // Scalar fallback
    constexpr int UNROLL_SCALAR = 4;
    softplus_f32_scalar_kernel<UNROLL_SCALAR><<<blocks, threads, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor softplus_f32_cuda(torch::Tensor x);
"""

_ext_name = "custom_ops_lib_softplus_opt2"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softplus_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    Softplus model using an optimized custom CUDA kernel (float32, contiguous CUDA tensors).
    Falls back to torch.nn.functional.softplus otherwise.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return self.custom_ops_lib.softplus_f32_cuda(x)
        return torch.nn.functional.softplus(x)