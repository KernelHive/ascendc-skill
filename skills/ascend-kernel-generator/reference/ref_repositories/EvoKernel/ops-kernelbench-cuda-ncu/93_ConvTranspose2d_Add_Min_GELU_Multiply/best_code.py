import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: fused add -> min(v,0) -> GELU -> multiply
# Input:  x [N, C, H, W] float32 CUDA contiguous (NCHW)
# Output: y [N, C, H, W] float32 CUDA contiguous
# Scalars: add_value, multiply_value (float)
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float gelu_approx(float x) {
    // tanh-based GELU approximation
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    float x2 = x * x;
    float x3 = x2 * x;
    float t = kAlpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float op_fused(float v, float add_value, float multiply_value) {
    v = v + add_value;
    v = fminf(v, 0.0f);
    v = gelu_approx(v);
    v = v * multiply_value;
    return v;
}

__global__ __launch_bounds__(256, 2) void add_min_gelu_mul_kernel_vec4(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t total,
    float add_value,
    float multiply_value
) {
    // Each thread processes float4 chunks when aligned.
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // reinterpret as float4
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);
    int64_t total4 = total >> 2; // total / 4

    for (int64_t i = tid; i < total4; i += stride) {
        float4 v = __ldg(reinterpret_cast<const float4*>(x4 + i));

        v.x = op_fused(v.x, add_value, multiply_value);
        v.y = op_fused(v.y, add_value, multiply_value);
        v.z = op_fused(v.z, add_value, multiply_value);
        v.w = op_fused(v.w, add_value, multiply_value);

        y4[i] = v;
    }
}

__global__ __launch_bounds__(256, 2) void add_min_gelu_mul_kernel_scalar(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t total,
    float add_value,
    float multiply_value
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = tid; i < total; i += stride) {
        float v = __ldg(x + i);
        y[i] = op_fused(v, add_value, multiply_value);
    }
}

torch::Tensor add_min_gelu_mul_cuda(
    torch::Tensor x,
    double add_value,
    double multiply_value
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW contiguous)");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    const int64_t total = x.numel();
    TORCH_CHECK(total > 0, "empty tensor");

    auto y = torch::empty_like(x);

    // Heuristic: use 256 threads, enough blocks to cover SMs; cap at a reasonable max.
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    const float addv = (float)add_value;
    const float mulv = (float)multiply_value;

    const uintptr_t x_addr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t y_addr = (uintptr_t)y.data_ptr<float>();
    const bool aligned16 = ((x_addr & 0xF) == 0) && ((y_addr & 0xF) == 0);

    // Vectorized path when:
    //  - pointers are 16-byte aligned
    //  - total divisible by 4
    // Otherwise fall back to scalar.
    if (aligned16 && ((total & 3LL) == 0)) {
        add_min_gelu_mul_kernel_vec4<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            total,
            addv,
            mulv
        );
    } else {
        add_min_gelu_mul_kernel_scalar<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            total,
            addv,
            mulv
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor add_min_gelu_mul_cuda(torch::Tensor x, double add_value, double multiply_value);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_add_min_gelu_multiply_opt2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_min_gelu_mul_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps nn.ConvTranspose2d on cuDNN, fuses:
      x + add_value -> min(x, 0) -> GELU -> * multiply_value
    into a single optimized custom CUDA kernel (vectorized + grid-stride).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.custom_ops = custom_ops_lib
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        # ensure contiguous NCHW for vectorized path
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.add_min_gelu_mul_cuda(x, self.add_value, self.multiply_value)