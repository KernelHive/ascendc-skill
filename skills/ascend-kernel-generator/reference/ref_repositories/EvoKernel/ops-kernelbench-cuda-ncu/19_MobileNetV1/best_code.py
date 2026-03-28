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

// Specialization for MobileNetV1 stem common shape:
// Cin=3, Cout<=32, K=3, stride=2, pad=1. NCHW float32.
// Put weights and BN params in constant memory to reduce global traffic and improve cache behavior.
__constant__ float c_w[32 * 3 * 3 * 3];   // [Cout, Cin=3, 3, 3] flattened as oc*(27)+ic*9+ky*3+kx
__constant__ float c_gamma[32];
__constant__ float c_beta[32];
__constant__ float c_mean[32];
__constant__ float c_var[32];

static inline void copy_to_const_checked(const float* src, size_t bytes, const void* symbol, const char* what) {
  cudaError_t err = cudaMemcpyToSymbol(symbol, src, bytes, 0, cudaMemcpyDeviceToDevice);
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for ", what, ": ", cudaGetErrorString(err));
}

__device__ __forceinline__ float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

// Each block covers one (n, oc, oh) and a stripe of ow.
// blockDim.x = threads over ow; grid.y covers (n, oc); grid.z covers oh.
__global__ __launch_bounds__(128, 3)
void stem_conv3x3s2_bn_relu_cmem_nchw(
    const float* __restrict__ x, // [N,3,H,W]
    float* __restrict__ y,       // [N,Cout,Hout,Wout]
    int N, int Hin, int Win,
    int Cout, int Hout, int Wout,
    float eps
) {
    int ow = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ow >= Wout) return;

    int oh = (int)blockIdx.z;
    if (oh >= Hout) return;

    int nc = (int)blockIdx.y; // 0..N*Cout-1
    int n = nc / Cout;
    int oc = nc - n * Cout;
    if (n >= N || oc >= Cout) return;

    int in_y0 = oh * 2 - 1;
    int in_x0 = ow * 2 - 1;

    // Load BN params from constant memory (broadcast-friendly)
    float g = c_gamma[oc];
    float b = c_beta[oc];
    float m = c_mean[oc];
    float v = c_var[oc];
    float inv_std = rsqrtf(v + eps);

    // Base offsets
    int x_n_base = n * 3 * Hin * Win;
    int y_idx = (((n * Cout + oc) * Hout + oh) * Wout + ow);

    // Compute 3x3 over Cin=3 (27 taps), with boundary checks.
    float acc = 0.0f;
    int w_base = oc * 27;

    // ic=0..2
    #pragma unroll
    for (int ic = 0; ic < 3; ++ic) {
        int x_c_base = x_n_base + ic * Hin * Win;
        int w_ic_base = w_base + ic * 9;

        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_c_base + iy * Win;

            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int ix = in_x0 + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;
                float xv = x[x_row + ix];
                float wv = c_w[w_ic_base + ky * 3 + kx];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    float outv = (acc - m) * inv_std * g + b;
    y[y_idx] = relu_f(outv);
}

torch::Tensor stem_conv_bn_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps
) {
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CUDA(gamma); CHECK_CUDA(beta); CHECK_CUDA(mean); CHECK_CUDA(var);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta); CHECK_CONTIGUOUS(mean); CHECK_CONTIGUOUS(var);
    CHECK_FLOAT(x); CHECK_FLOAT(w);
    CHECK_FLOAT(gamma); CHECK_FLOAT(beta); CHECK_FLOAT(mean); CHECK_FLOAT(var);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW");
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "w must be 3x3");
    TORCH_CHECK(x.size(1) == 3, "specialized stem expects Cin=3");
    TORCH_CHECK(w.size(1) == 3, "specialized stem expects weight Cin=3");
    TORCH_CHECK(w.size(0) <= 32, "specialized stem expects Cout<=32");

    int64_t N = x.size(0);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);
    int64_t Cout = w.size(0);

    TORCH_CHECK(gamma.numel() == Cout, "gamma must be [Cout]");
    TORCH_CHECK(beta.numel() == Cout, "beta must be [Cout]");
    TORCH_CHECK(mean.numel() == Cout, "mean must be [Cout]");
    TORCH_CHECK(var.numel() == Cout, "var must be [Cout]");

    int64_t Hout = (Hin + 2 - 3) / 2 + 1;
    int64_t Wout = (Win + 2 - 3) / 2 + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    // Copy weights and BN params into constant memory (device-to-device).
    // Weight layout in memory is contiguous OIHW; flatten and copy first Cout*27 floats.
    size_t w_bytes = (size_t)Cout * 27 * sizeof(float);
    size_t c_bytes = (size_t)Cout * sizeof(float);

    // c_w symbol is sized for 32*27; we copy only the used prefix.
    cudaError_t err;
    err = cudaMemcpyToSymbol(c_w, w.data_ptr<float>(), w_bytes, 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for weights: ", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(c_gamma, gamma.data_ptr<float>(), c_bytes, 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for gamma: ", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(c_beta, beta.data_ptr<float>(), c_bytes, 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for beta: ", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(c_mean, mean.data_ptr<float>(), c_bytes, 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for mean: ", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(c_var, var.data_ptr<float>(), c_bytes, 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for var: ", cudaGetErrorString(err));

    // Launch
    const int block = 128;
    dim3 blockDim(block, 1, 1);
    dim3 gridDim((unsigned)((Wout + block - 1) / block), (unsigned)(N * Cout), (unsigned)Hout);

    stem_conv3x3s2_bn_relu_cmem_nchw<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)Hin, (int)Win,
        (int)Cout, (int)Hout, (int)Wout,
        (float)eps
    );
    CUDA_KERNEL_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor stem_conv_bn_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stem_conv_bn_relu_forward_cuda", &stem_conv_bn_relu_forward_cuda, "stem conv+bn+relu forward (CUDA, specialized)");
}
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mobilenetv1_stem_cmem",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=None,
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model definition (uses fused stem op)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        c32 = int(32 * alpha)
        c64 = int(64 * alpha)
        c128 = int(128 * alpha)
        c256 = int(256 * alpha)
        c512 = int(512 * alpha)
        c1024 = int(1024 * alpha)

        self.model = nn.Sequential(
            conv_bn(input_channels, c32, 2),
            conv_dw(c32, c64, 1),
            conv_dw(c64, c128, 2),
            conv_dw(c128, c128, 1),
            conv_dw(c128, c256, 2),
            conv_dw(c256, c256, 1),
            conv_dw(c256, c512, 2),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c1024, 2),
            conv_dw(c1024, c1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(c1024, num_classes)

    def forward(self, x):
        # Use fused specialized stem only when it exactly matches the specialization and BN is inference.
        use_fast = (
            x.is_cuda and x.dtype == torch.float32 and (not self.training) and x.is_contiguous()
        )
        if use_fast:
            stem = self.model[0]
            conv = stem[0]
            bn = stem[1]
            use_fast = (
                x.size(1) == 3 and
                conv.weight.is_cuda and conv.weight.dtype == torch.float32 and conv.weight.is_contiguous() and
                conv.weight.size(1) == 3 and conv.weight.size(0) <= 32 and
                bn.weight.is_cuda and bn.bias.is_cuda and bn.running_mean.is_cuda and bn.running_var.is_cuda and
                bn.weight.dtype == torch.float32 and bn.bias.dtype == torch.float32 and
                bn.running_mean.dtype == torch.float32 and bn.running_var.dtype == torch.float32 and
                bn.weight.is_contiguous() and bn.bias.is_contiguous() and
                bn.running_mean.is_contiguous() and bn.running_var.is_contiguous()
            )

        if use_fast:
            x = custom_ops_lib.stem_conv_bn_relu_forward_cuda(
                x,
                self.model[0][0].weight,
                self.model[0][1].weight,
                self.model[0][1].bias,
                self.model[0][1].running_mean,
                self.model[0][1].running_var,
                float(self.model[0][1].eps),
            )
            x = self.model[1:](x)
        else:
            x = self.model(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x