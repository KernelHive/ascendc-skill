import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source: forward-only average pooling for NCHW float32 on CUDA
avg_pool2d_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static inline int64_t div_up_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__global__ void avg_pool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * outH * outW;
    if (idx >= total) return;

    int ow = (int)(idx % outW);
    int tmp = (int)(idx / outW);
    int oh = (int)(tmp % outH);
    tmp = tmp / outH;
    int c = (int)(tmp % C);
    int n = (int)(tmp / C);

    int hstart = oh * sH - pH;
    int wstart = ow * sW - pW;
    int hend = hstart + kH;
    int wend = wstart + kW;

    // Exclude padding from the average (PyTorch AvgPool2d default: count_include_pad=False)
    int h0 = hstart < 0 ? 0 : hstart;
    int w0 = wstart < 0 ? 0 : wstart;
    int h1 = hend > H ? H : hend;
    int w1 = wend > W ? W : wend;

    float sum = 0.0f;
    int count = 0;

    const int64_t base_nc = ((int64_t)n * C + c) * (int64_t)H * (int64_t)W;
    for (int ih = h0; ih < h1; ++ih) {
        int64_t row = base_nc + (int64_t)ih * W;
        for (int iw = w0; iw < w1; ++iw) {
            sum += x[row + iw];
            count++;
        }
    }

    // If the window lies completely in padding (can happen with large padding), define output as 0.
    float out = (count > 0) ? (sum / (float)count) : 0.0f;
    y[idx] = out;
}

torch::Tensor avg_pool2d_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW
) {
    TORCH_CHECK(x.is_cuda(), "avg_pool2d_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avg_pool2d_forward_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 4, "avg_pool2d_forward_cuda: expected NCHW 4D input");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    int64_t sH = sH_opt.has_value() ? sH_opt.value() : kH;
    int64_t sW = sW_opt.has_value() ? sW_opt.value() : kW;

    TORCH_CHECK(kH > 0 && kW > 0, "kernel sizes must be > 0");
    TORCH_CHECK(sH > 0 && sW > 0, "strides must be > 0");
    TORCH_CHECK(pH >= 0 && pW >= 0, "paddings must be >= 0");

    // Match PyTorch's output size formula (ceil_mode=False default):
    // out = floor((in + 2*pad - kernel)/stride) + 1
    const int64_t outH = (H + 2 * pH - kH) / sH + 1;
    const int64_t outW = (W + 2 * pW - kW) / sW + 1;
    TORCH_CHECK(outH >= 0 && outW >= 0, "computed output size is negative");

    auto y = torch::empty({N, C, outH, outW}, x.options());

    const int threads = 256;
    const int64_t total = N * C * outH * outW;
    const int blocks = (int)div_up_int64(total, threads);

    avg_pool2d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)outH, (int)outW,
        (int)kH, (int)kW,
        (int)sH, (int)sW,
        (int)pH, (int)pW
    );

    return y;
}
"""

# C++ declarations for load_inline
avg_pool2d_cpp_src = r"""
torch::Tensor avg_pool2d_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW
);
"""

# Compile into custom_ops_lib
custom_ops_lib = load_inline(
    name="custom_ops_lib_avgpool2d",
    cpp_sources=avg_pool2d_cpp_src,
    cuda_sources=avg_pool2d_cuda_src,
    functions=["avg_pool2d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replacement model using a custom CUDA kernel for AvgPool2d forward.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kH = int(kernel_size)
        self.kW = int(kernel_size)
        self.sH = None if stride is None else int(stride)
        self.sW = None if stride is None else int(stride)
        self.pH = int(padding)
        self.pW = int(padding)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Falls back to PyTorch if not CUDA/float32 for correctness.
        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4):
            s = self.kH if self.sH is None else self.sH
            return torch.nn.functional.avg_pool2d(
                x, kernel_size=self.kH, stride=s, padding=self.pH, ceil_mode=False, count_include_pad=False
            )
        return self.custom_ops.avg_pool2d_forward_cuda(
            x, self.kH, self.kW, self.sH, self.sW, self.pH, self.pW
        )