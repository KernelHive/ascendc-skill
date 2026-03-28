import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source: direct convolution forward specialized for float32 CUDA tensors.
# Notes:
# - Assumes input/layout: NCHW contiguous.
# - Computes output in NCHW.
# - Supports general H/W (still expects square-ish but not required), fixed kernel=11, stride=4, padding=2.
# - Cin=3, Cout arbitrary (here 96), groups=1.
# - Bias optional (pass empty tensor to skip).
conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

// Kernel params fixed for this operator
constexpr int K = 11;
constexpr int STRIDE = 4;
constexpr int PAD = 2;

__global__ void conv2d_fwd_nchw_k11s4p2(
    const float* __restrict__ x,       // [N, Cin, Hin, Win]
    const float* __restrict__ w,       // [Cout, Cin, K, K]
    const float* __restrict__ b,       // [Cout] or nullptr
    float* __restrict__ y,             // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    // 3D mapping: (n, oc, oh, ow) flattened
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    float acc = 0.0f;
    if (b != nullptr) acc = b[oc];

    // Input top-left corner for this output
    int in_y0 = oh * STRIDE - PAD;
    int in_x0 = ow * STRIDE - PAD;

    // Weight base pointer for this output channel
    // w layout: [Cout, Cin, K, K]
    int w_oc_base = oc * Cin * K * K;

    // Cin expected small (3), but keep generic
    for (int ic = 0; ic < Cin; ++ic) {
        int w_ic_base = w_oc_base + ic * K * K;
        int x_ic_base = ((n * Cin + ic) * Hin) * Win;

        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_ic_base + iy * Win;

            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                int ix = in_x0 + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;
                float xv = x[x_row + ix];
                float wv = w[w_ic_base + ky * K + kx];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = acc;
}

torch::Tensor conv2d_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_FLOAT(x); CHECK_FLOAT(w);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW");
    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);

    int64_t Cout = w.size(0);
    TORCH_CHECK(w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK(w.size(2) == K && w.size(3) == K, "weight must be 11x11");

    // Output shape for conv2d: floor((H + 2P - K)/S) + 1
    int64_t Hout = (Hin + 2 * PAD - K) / STRIDE + 1;
    int64_t Wout = (Win + 2 * PAD - K) / STRIDE + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        CHECK_CUDA(b);
        CHECK_CONTIGUOUS(b);
        CHECK_FLOAT(b);
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias must be [Cout]");
        bptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    int total = (int)(N * Cout * Hout * Wout);
    const int block = 256;
    const int grid = (total + block - 1) / block;

    conv2d_fwd_nchw_k11s4p2<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Hin, (int)Win,
        (int)Cout, (int)Hout, (int)Wout
    );

    return y;
}
"""

conv2d_cpp_source = r"""
torch::Tensor conv2d_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

# Compile into custom_ops_lib
custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_k11s4p2",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Keep parameters identical to nn.Conv2d(3->96, k=11, s=4, p=2)
        self.weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.empty(96))
        # Initialize similarly to PyTorch default Conv2d init
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure contiguous for the custom kernel
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        return custom_ops_lib.conv2d_forward_cuda(x, w, b)