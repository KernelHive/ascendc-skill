import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------- CUDA/C++ Extension --------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float softplus_fast(float x) {
    float ax = fabsf(x);
    float m = fmaxf(x, 0.0f);
    // stable: m + log1p(exp(-|x|))
    return m + log1pf(expf(-ax));
}
static __device__ __forceinline__ float mish_fast(float x) {
    return x * tanhf(softplus_fast(x));
}
static __device__ __forceinline__ float mish2_fast(float x) {
    // mish(mish(x))
    float y = mish_fast(x);
    return mish_fast(y);
}

// ------------------------- Generic fallback -------------------------
// Supports: stride=1, padding=0, dilation=1, groups=1 (enforced in Python wrapper)
__global__ void conv2d_mish_mish_generic_kernel(
    const float* __restrict__ x,   // [N, Cin, H, W]
    const float* __restrict__ w,   // [Cout, Cin, kH, kW]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, Cout, outH, outW]
    int N, int Cin, int H, int W,
    int Cout, int kH, int kW,
    int outH, int outW
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = (idx / (outW * outH)) % Cout;
    int n  = idx / (outW * outH * Cout);

    float acc = b ? __ldg(b + oc) : 0.0f;

    int x_n_base = (n * Cin) * H * W;
    int w_oc_base = (oc * Cin) * kH * kW;

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        int x_c_base = x_n_base + ic * H * W;
        int w_ic_base = w_oc_base + ic * kH * kW;

        #pragma unroll 1
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh + kh;
            int x_row = x_c_base + ih * W;
            int w_row = w_ic_base + kh * kW;

            #pragma unroll 1
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow + kw;
                float xv = __ldg(x + x_row + iw);
                float wv = __ldg(w + w_row + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = mish2_fast(acc);
}

// ------------------------- Specialized constant-memory path -------------------------
// Target specialization: Cin=64, Cout=128, k=3x3, float32 contiguous.
// Constant memory sizes:
// - weights: 128*64*9 = 73728 floats (~288KB) -> too large for constant memory (64KB).
// So we *cannot* store full weights in constant memory.
// Alternative: store only bias in constant memory and rely on L2 for weights; keep compute fully unrolled.
// (We still keep a smaller const path for bias; weight remains in global memory.)

__constant__ float CB_128[128];

__global__ __launch_bounds__(256, 2) void conv2d_k3_cin64_cout128_mish2_unrolled_kernel(
    const float* __restrict__ x,   // [N, 64, H, W]
    const float* __restrict__ w,   // [128, 64, 3, 3] (global)
    float* __restrict__ y,         // [N, 128, outH, outW]
    int N, int H, int W,
    int outH, int outW,
    bool has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * 128 * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = (idx / (outW * outH)) % 128;
    int n  = idx / (outW * outH * 128);

    float acc = has_bias ? CB_128[oc] : 0.0f;

    int x_n_base = (n * 64) * H * W;
    // w layout: [oc][ic][kh][kw] contiguous
    int w_oc_base = (oc * 64) * 9;

    // Unroll kh/kw fully (3x3). Loop ic remains (64).
    #pragma unroll
    for (int ic = 0; ic < 64; ++ic) {
        const float* xp = x + x_n_base + ic * H * W + oh * W + ow;
        const float* wp = w + w_oc_base + ic * 9;

        float x00 = __ldg(xp + 0);
        float x01 = __ldg(xp + 1);
        float x02 = __ldg(xp + 2);
        float x10 = __ldg(xp + W + 0);
        float x11 = __ldg(xp + W + 1);
        float x12 = __ldg(xp + W + 2);
        float x20 = __ldg(xp + 2*W + 0);
        float x21 = __ldg(xp + 2*W + 1);
        float x22 = __ldg(xp + 2*W + 2);

        float w00 = __ldg(wp + 0);
        float w01 = __ldg(wp + 1);
        float w02 = __ldg(wp + 2);
        float w10 = __ldg(wp + 3);
        float w11 = __ldg(wp + 4);
        float w12 = __ldg(wp + 5);
        float w20 = __ldg(wp + 6);
        float w21 = __ldg(wp + 7);
        float w22 = __ldg(wp + 8);

        acc = fmaf(x00, w00, acc); acc = fmaf(x01, w01, acc); acc = fmaf(x02, w02, acc);
        acc = fmaf(x10, w10, acc); acc = fmaf(x11, w11, acc); acc = fmaf(x12, w12, acc);
        acc = fmaf(x20, w20, acc); acc = fmaf(x21, w21, acc); acc = fmaf(x22, w22, acc);
    }

    y[idx] = mish2_fast(acc);
}

static void upload_bias_128(torch::Tensor bias) {
    cudaMemcpyToSymbol(CB_128, bias.data_ptr<float>(), sizeof(float) * 128, 0, cudaMemcpyDeviceToDevice);
}

torch::Tensor conv2d_mish_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be OIHW (4D)");

    auto bias = bias_opt.has_value() ? bias_opt.value() : torch::Tensor();
    bool has_bias = bias.defined();
    if (has_bias) {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    }

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    int Cout = (int)weight.size(0);
    int wCin = (int)weight.size(1);
    int kH = (int)weight.size(2);
    int kW = (int)weight.size(3);

    TORCH_CHECK(Cin == wCin, "in_channels mismatch between x and weight");
    if (has_bias) TORCH_CHECK((int)bias.size(0) == Cout, "bias size must equal out_channels");
    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be > 0");

    int outH = H - kH + 1;
    int outW = W - kW + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "conv output size is non-positive");

    auto y = torch::empty({N, Cout, outH, outW}, x.options());

    // Specialized fast path for the target model: Cin=64, Cout=128, k=3
    if (kH == 3 && kW == 3 && Cin == 64 && Cout == 128) {
        if (has_bias) upload_bias_128(bias);

        int total = N * 128 * outH * outW;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        conv2d_k3_cin64_cout128_mish2_unrolled_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            y.data_ptr<float>(),
            N, H, W, outH, outW,
            has_bias
        );
        return y;
    }

    // Generic fallback
    const float* bptr = has_bias ? bias.data_ptr<float>() : nullptr;
    int total = N * Cout * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_mish_mish_generic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, Cin, H, W,
        Cout, kH, kW,
        outH, outW
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv2d_mish_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_mish_mish_opt_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_mish_mish_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


# -------------------- PyTorch Module Wrapper --------------------
class ModelNew(nn.Module):
    """
    Conv2d -> Mish -> Mish fused into a custom CUDA forward op.

    Supported by fused op ONLY for:
    - stride=1, padding=0, dilation=1, groups=1
    - float32 contiguous CUDA tensors

    Has a specialized fast path for Cin=64, Cout=128, k=3, and a generic fallback.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        if self.conv.groups != 1:
            raise RuntimeError("Fused kernel supports groups=1 only")
        if tuple(self.conv.stride) != (1, 1):
            raise RuntimeError("Fused kernel supports stride=(1,1) only")
        if tuple(self.conv.padding) != (0, 0):
            raise RuntimeError("Fused kernel supports padding=(0,0) only")
        if tuple(self.conv.dilation) != (1, 1):
            raise RuntimeError("Fused kernel supports dilation=(1,1) only")

        w = self.conv.weight
        b = self.conv.bias

        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        bias_opt = None
        if b is not None:
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()
            bias_opt = b

        return self.custom_ops.conv2d_mish_mish_forward_cuda(x, w, bias_opt)