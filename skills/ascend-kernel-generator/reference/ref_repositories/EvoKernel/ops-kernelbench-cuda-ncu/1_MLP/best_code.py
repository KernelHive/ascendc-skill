import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------
# Custom CUDA: in-place ReLU for contiguous float32 CUDA
# Vectorized float4 path + safe scalar tail
# -------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

static __forceinline__ __device__ float relu_f(float v) {
    return v > 0.0f ? v : 0.0f;
}

__global__ void relu_inplace_f32_vec4_kernel(float* __restrict__ x, int64_t n) {
    // Process float4 chunks when possible. n is element count.
    int64_t n4 = n >> 2; // number of float4s
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // float4 path
    float4* x4 = reinterpret_cast<float4*>(x);

    // Unroll by 2 to increase ILP a bit.
    for (int64_t i = tid; i < n4; i += stride * 2) {
        int64_t i0 = i;
        int64_t i1 = i + stride;

        if (i0 < n4) {
            // read-only cache hint; works on supported arch; harmless otherwise
            float4 v = __ldg(&x4[i0]);
            v.x = relu_f(v.x);
            v.y = relu_f(v.y);
            v.z = relu_f(v.z);
            v.w = relu_f(v.w);
            x4[i0] = v;
        }
        if (i1 < n4) {
            float4 v = __ldg(&x4[i1]);
            v.x = relu_f(v.x);
            v.y = relu_f(v.y);
            v.z = relu_f(v.z);
            v.w = relu_f(v.w);
            x4[i1] = v;
        }
    }
}

__global__ void relu_inplace_f32_scalar_kernel(float* __restrict__ x, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        float v = __ldg(&x[i]);
        x[i] = relu_f(v);
    }
}

torch::Tensor relu_inplace_f32_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_inplace_f32_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "relu_inplace_f32_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "relu_inplace_f32_cuda: input must be contiguous");

    const int64_t n = x.numel();
    if (n == 0) return x;

    // Tune threads/blocks: memory-bound kernel; use moderate threads and enough blocks.
    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    // Cap to avoid oversubscription/launch overhead, but keep enough to fill SMs.
    if (blocks > 8192) blocks = 8192;
    if (blocks < 1) blocks = 1;

    float* ptr = x.data_ptr<float>();

    // Vectorized float4 path only if 16B aligned and at least 4 elements.
    bool aligned16 = (((uintptr_t)ptr) & 0xF) == 0;
    if (aligned16 && n >= 4) {
        // Launch vec4 kernel over n4 float4 items
        relu_inplace_f32_vec4_kernel<<<blocks, threads>>>(ptr, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Handle tail (n % 4) scalars if any
        int64_t tail = n & 3;
        if (tail) {
            int64_t start = n - tail;
            // small scalar kernel for the tail only; use 1 block
            relu_inplace_f32_scalar_kernel<<<1, 32>>>(ptr + start, tail);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        relu_inplace_f32_scalar_kernel<<<blocks, threads>>>(ptr, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return x;
}
"""

cpp_source = r"""
torch::Tensor relu_inplace_f32_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mlp_relu_inplace_vec4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_inplace_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    MLP using cuBLAS GEMMs (nn.Linear) and a faster custom in-place ReLU kernel.
    """
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

        layers = []
        current = input_size
        for h in layer_sizes:
            layers.append(nn.Linear(current, h))
            current = h
        self.linears = nn.ModuleList(layers)
        self.out_linear = nn.Linear(current, output_size)

    def _relu_inplace(self, x: torch.Tensor) -> torch.Tensor:
        # In-place activation to reduce memory traffic and allocations.
        if x.is_cuda and x.dtype == torch.float32:
            if not x.is_contiguous():
                x = x.contiguous()
            return self.custom_ops_lib.relu_inplace_f32_cuda(x)
        return torch.relu_(x) if x.is_floating_point() else torch.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.linears:
            x = lin(x)
            x = self._relu_inplace(x)
        x = self.out_linear(x)
        return x