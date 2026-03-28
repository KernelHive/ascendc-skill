import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA op: out = x * sigmoid(x) * s
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // fast exp is enabled by --use_fast_math
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void swish_scale_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  float s,
                                  int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float sig = sigmoidf_fast(v);
        y[idx] = (v * sig) * s;
    }
}

torch::Tensor swish_scale_cuda(torch::Tensor x, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    const unsigned int blocks = (unsigned int)blocks64;

    swish_scale_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (float)scaling_factor,
        n
    );

    return y;
}
"""

cpp_source = r"""
torch::Tensor swish_scale_cuda(torch::Tensor x, double scaling_factor);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_swish_scaling",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["swish_scale_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps the matmul (nn.Linear) but fuses Swish + scaling into a single custom CUDA op:
        y = linear(x)
        y = y * sigmoid(y) * scaling_factor
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.matmul(x)

        # Fallback to PyTorch on CPU; keep predictable constraints on CUDA path
        if not y.is_cuda:
            y = y * torch.sigmoid(y)
            return y * self.scaling_factor

        if y.dtype != torch.float32:
            y = y.float()
        if not y.is_contiguous():
            y = y.contiguous()

        return self.custom_ops_lib.swish_scale_cuda(y, self.scaling_factor)