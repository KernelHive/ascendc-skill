import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused subtract (broadcast bias) + tanh
#   y = tanh(x - bias[c])
# Input/Output: contiguous NCHW float32.
# Bias: [C] float32 contiguous on same device.
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sub_bias_tanh_kernel(
    const float* __restrict__ x,    // [N,C,H,W]
    float* __restrict__ y,          // [N,C,H,W]
    const float* __restrict__ bias, // [C]
    int N, int C, int HxW
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * HxW;
    if (idx >= total) return;

    int c = (int)((idx / HxW) % C);
    float v = x[idx] - bias[c];
    y[idx] = tanhf(v);
}

torch::Tensor sub_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias_c) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias_c.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(bias_c.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(bias_c.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(bias_c.dim() == 1, "bias must be 1D [C]");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK((int)bias_c.numel() == C, "bias must have C elements");

    auto y = torch::empty_like(x);
    int HxW = H * W;
    int64_t total = (int64_t)N * C * HxW;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    sub_bias_tanh_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (const float*)bias_c.data_ptr<float>(),
        N, C, HxW
    );
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sub_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias_c);
"""

custom_ops_lib = load_inline(
    name="custom_conv_transpose2d_subtract_tanh_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sub_bias_tanh_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keep ConvTranspose2d (cuDNN) and replace:
      x - bias -> tanh
    with a fused CUDA kernel (float32, contiguous NCHW).
    Includes CPU fallback to eager ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,  # preserve PyTorch default semantics
        )
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        # Normalize at boundary: float32 + contiguous NCHW
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        # Flatten broadcast bias [C,1,1] -> [C]
        bias_c = self.bias
        if bias_c.dtype != torch.float32:
            bias_c = bias_c.float()
        bias_c = bias_c.contiguous().view(-1)

        # CPU fallback
        if not x.is_cuda:
            y = x - bias_c.view(1, -1, 1, 1)
            return torch.tanh(y)

        # Ensure bias on same CUDA device
        if not bias_c.is_cuda or bias_c.device != x.device:
            bias_c = bias_c.to(device=x.device)

        return self.custom_ops_lib.sub_bias_tanh_cuda(x, bias_c)