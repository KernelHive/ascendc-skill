import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension
# -------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#define CUDA_KERNEL_CHECK() do { \
  cudaError_t err = cudaGetLastError(); \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err)); \
} while(0)

__device__ __forceinline__ float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

// Fused conv2d (k=11,s=4,p=2) + bias + ReLU for NCHW float32 specialized to Cin=3.
// Compute 2 output channels per thread to reuse input loads and increase ILP.
// w: [Cout,Cin,11,11], b: [Cout]
__global__ __launch_bounds__(256, 2)
void conv2d_k11s4p2_bias_relu_nchw_oc2(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,11,11]
    const float* __restrict__ b,   // [Cout]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    // Flatten over (n, oh, ow, oc_pair)
    // oc_pair indexes oc in steps of 2.
    int oc_pairs = (Cout + 1) >> 1;
    int64_t total_pairs = (int64_t)N * Hout * Wout * oc_pairs;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = tid; linear < total_pairs; linear += stride) {
        int tmp = (int)(linear % oc_pairs);
        int oc0 = tmp << 1;
        int64_t t1 = linear / oc_pairs;

        int ow = (int)(t1 % Wout);
        int64_t t2 = t1 / Wout;
        int oh = (int)(t2 % Hout);
        int n  = (int)(t2 / Hout);

        int oc1 = oc0 + 1;

        // stride=4 pad=2
        int in_y0 = oh * 4 - 2;
        int in_x0 = ow * 4 - 2;

        float acc0 = (oc0 < Cout) ? __ldg(&b[oc0]) : 0.0f;
        float acc1 = (oc1 < Cout) ? __ldg(&b[oc1]) : 0.0f;

        // Weight base offsets
        // w layout: [oc][ic][ky][kx] contiguous
        int w_oc0_base = oc0 * 3 * 121;
        int w_oc1_base = oc1 * 3 * 121;

        // x base per ic
        int x_n_base = (n * 3 * Hin) * Win;

        #pragma unroll
        for (int ic = 0; ic < 3; ++ic) {
            const float* xptr = x + x_n_base + (ic * Hin) * Win;
            const float* w0 = w + w_oc0_base + ic * 121;
            const float* w1 = w + w_oc1_base + ic * 121;

            #pragma unroll
            for (int ky = 0; ky < 11; ++ky) {
                int iy = in_y0 + ky;
                if ((unsigned)iy >= (unsigned)Hin) continue;
                const float* xrow = xptr + iy * Win;

                #pragma unroll
                for (int kx = 0; kx < 11; ++kx) {
                    int ix = in_x0 + kx;
                    if ((unsigned)ix >= (unsigned)Win) continue;
                    float xv = __ldg(&xrow[ix]);
                    if (oc0 < Cout) acc0 = fmaf(xv, __ldg(&w0[ky * 11 + kx]), acc0);
                    if (oc1 < Cout) acc1 = fmaf(xv, __ldg(&w1[ky * 11 + kx]), acc1);
                }
            }
        }

        // ReLU
        if (oc0 < Cout) {
            acc0 = relu_f(acc0);
            y[(((n * Cout + oc0) * Hout + oh) * Wout) + ow] = acc0;
        }
        if (oc1 < Cout) {
            acc1 = relu_f(acc1);
            y[(((n * Cout + oc1) * Hout + oh) * Wout) + ow] = acc1;
        }
    }
}

torch::Tensor conv1_relu_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(b);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(b);
    CHECK_FLOAT(x); CHECK_FLOAT(w); CHECK_FLOAT(b);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW");
    TORCH_CHECK(b.dim() == 1, "b must be [Cout]");

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);

    int64_t Cout = w.size(0);
    TORCH_CHECK(w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK(w.size(2) == 11 && w.size(3) == 11, "weight must be 11x11");
    TORCH_CHECK(b.size(0) == Cout, "bias Cout mismatch");

    // Specialized fast kernel assumes Cin==3, but keep checks here for safety.
    TORCH_CHECK(Cin == 3, "conv1 fast-path expects Cin=3");

    // Hout = floor((Hin + 2*pad - k)/stride) + 1 with pad=2,k=11,stride=4
    int64_t Hout = (Hin + 4 - 11) / 4 + 1;
    int64_t Wout = (Win + 4 - 11) / 4 + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    // grid based on total pairs, but cap grid to avoid oversubscription overhead
    int oc_pairs = (int)((Cout + 1) >> 1);
    int64_t total_pairs = N * Hout * Wout * (int64_t)oc_pairs;

    const int block = 256;
    int grid = (int)((total_pairs + block - 1) / block);
    // cap grid (helps very large N); still uses grid-stride loop
    grid = grid > 65535 ? 65535 : grid;
    if (grid < 1) grid = 1;

    conv2d_k11s4p2_bias_relu_nchw_oc2<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)Hin, (int)Win,
        (int)Cout, (int)Hout, (int)Wout
    );
    CUDA_KERNEL_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv1_relu_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_alex_net_conv1relu_v2_oc2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv1_relu_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition (uses fused conv1+relu op)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        use_fast = (
            x.is_cuda
            and x.dtype == torch.float32
            and (not self.training)
            and x.dim() == 4
            and x.shape[1] == 3
            and x.is_contiguous()
            and self.conv1.weight.is_cuda
            and self.conv1.bias is not None
            and self.conv1.bias.is_cuda
            and self.conv1.weight.is_contiguous()
            and self.conv1.bias.is_contiguous()
            and tuple(self.conv1.weight.shape) == (96, 3, 11, 11)
        )

        if use_fast:
            x = custom_ops_lib.conv1_relu_forward_cuda(x, self.conv1.weight, self.conv1.bias)
        else:
            x = self.conv1(x)
            x = self.relu1(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x