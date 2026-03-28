import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: ConvTranspose3d forward (groups=1, bias=False)
# Input:  x [N, Cin, Din, Hin, Win]
# Weight: w [Cin, Cout, K, K, K]  (PyTorch ConvTranspose3d layout)
# Output: y [N, Cout, Dout, Hout, Wout]
# where:
# Dout = (Din - 1) * stride - 2*padding + K + output_padding
# similarly for H/W.
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static inline __host__ __device__ int div_floor(int a, int b) {
    // b>0
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) q--;
    return q;
}

__global__ void conv_transposed3d_square_input_square_kernel(
    const float* __restrict__ x,      // [N, Cin, Din, Hin, Win]
    const float* __restrict__ w,      // [Cin, Cout, K, K, K]
    float* __restrict__ y,            // [N, Cout, Dout, Hout, Wout]
    int N, int Cin, int Cout,
    int Din, int Hin, int Win,
    int K,
    int stride, int padding, int output_padding,
    int Dout, int Hout, int Wout
) {
    // Flattened index over output tensor
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cout * Dout * Hout * Wout;
    if (idx >= total) return;

    // Decode idx -> (n, co, od, oh, ow)
    int64_t t = idx;
    int ow = (int)(t % Wout); t /= Wout;
    int oh = (int)(t % Hout); t /= Hout;
    int od = (int)(t % Dout); t /= Dout;
    int co = (int)(t % Cout); t /= Cout;
    int n  = (int)t;

    float acc = 0.0f;

    // For each kernel tap (kd,kh,kw), input coordinate is:
    // id = (od + padding - kd) / stride  must be integer and in [0, Din)
    // similarly for h/w.
    // So we loop kd/kh/kw and check divisibility.
    for (int kd = 0; kd < K; ++kd) {
        int a = od + padding - kd;
        if (a < 0) continue;
        if (a % stride != 0) continue;
        int id = a / stride;
        if ((unsigned)id >= (unsigned)Din) continue;

        for (int kh = 0; kh < K; ++kh) {
            int b = oh + padding - kh;
            if (b < 0) continue;
            if (b % stride != 0) continue;
            int ih = b / stride;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            for (int kw = 0; kw < K; ++kw) {
                int c = ow + padding - kw;
                if (c < 0) continue;
                if (c % stride != 0) continue;
                int iw = c / stride;
                if ((unsigned)iw >= (unsigned)Win) continue;

                // Accumulate over input channels
                // x index: (((n*Cin + ci)*Din + id)*Hin + ih)*Win + iw
                int64_t x_base = (((int64_t)n * Cin + 0) * Din + id) * Hin * (int64_t)Win + (int64_t)ih * Win + iw;

                // w index: ((((ci*Cout + co)*K + kd)*K + kh)*K + kw)
                // layout: [Cin, Cout, K, K, K] contiguous
                int64_t w_base = (((int64_t)0 * Cout + co) * K + kd) * (int64_t)K * K + (int64_t)kh * K + kw;

                for (int ci = 0; ci < Cin; ++ci) {
                    float xv = x[x_base + (int64_t)ci * Din * (int64_t)Hin * Win];
                    float wv = w[w_base + (int64_t)ci * Cout * (int64_t)K * K * K];
                    acc += xv * wv;
                }
            }
        }
    }

    // Write output
    y[idx] = acc;
}

torch::Tensor conv_transposed3d_square_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [Cin,Cout,K,K,K] (ConvTranspose3d weight layout)");
    TORCH_CHECK(w.size(2) == w.size(3) && w.size(3) == w.size(4), "Kernel must be cubic KxKxK");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int64_t N   = x_c.size(0);
    const int64_t Cin = x_c.size(1);
    const int64_t Din = x_c.size(2);
    const int64_t Hin = x_c.size(3);
    const int64_t Win = x_c.size(4);

    const int64_t wCin  = w_c.size(0);
    const int64_t Cout  = w_c.size(1);
    const int64_t K     = w_c.size(2);

    TORCH_CHECK(wCin == Cin, "Weight Cin must match input channels");

    TORCH_CHECK(stride >= 1, "stride must be >= 1");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");
    TORCH_CHECK(output_padding >= 0 && output_padding < stride, "output_padding must be in [0, stride-1] for typical conv_transpose");

    const int64_t Dout = (Din - 1) * stride - 2 * padding + K + output_padding;
    const int64_t Hout = (Hin - 1) * stride - 2 * padding + K + output_padding;
    const int64_t Wout = (Win - 1) * stride - 2 * padding + K + output_padding;

    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Computed output shape must be positive");

    auto y = torch::zeros({N, Cout, Dout, Hout, Wout}, x_c.options());

    int64_t total = N * Cout * Dout * Hout * Wout;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    conv_transposed3d_square_input_square_kernel<<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Cout,
        (int)Din, (int)Hin, (int)Win,
        (int)K,
        (int)stride, (int)padding, (int)output_padding,
        (int)Dout, (int)Hout, (int)Wout
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transposed3d_square_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transposed3d_square",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transposed3d_square_input_square_kernel_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d replacement using a custom CUDA kernel.
    Assumptions aligned with the provided model defaults:
      - groups=1
      - bias=False
      - cubic kernel KxKxK
      - float32 CUDA tensors
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if groups != 1:
            raise ValueError("Custom kernel currently supports groups=1 only")
        if bias:
            raise ValueError("Custom kernel currently supports bias=False only")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.groups = int(groups)
        self.bias = bool(bias)

        # Match PyTorch ConvTranspose3d weight layout: [Cin, Cout, K, K, K]
        w = torch.empty(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.conv_transposed3d_square_input_square_kernel_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding
        )