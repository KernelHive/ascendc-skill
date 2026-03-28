import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

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
    // stable softplus: max(x,0) + log1p(exp(-abs(x)))
    float ax = fabsf(x);
    float m = fmaxf(x, 0.0f);
    return m + log1pf(expf(-ax));
}

static __device__ __forceinline__ float mish_fast(float x) {
    // mish(x) = x * tanh(softplus(x))
    return x * tanhf(softplus_fast(x));
}

__global__ void conv2d_sub_sub_mish_generic_kernel(
    const float* __restrict__ x,   // [N, Cin, H, W]
    const float* __restrict__ w,   // [Cout, Cin, kH, kW]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, Cout, outH, outW]
    int N, int Cin, int H, int W,
    int Cout, int kH, int kW,
    int outH, int outW,
    float sub12
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = (idx / (outW * outH)) % Cout;
    int n  = idx / (outW * outH * Cout);

    float acc = b ? __ldg(b + oc) : 0.0f;

    int w_base = ((oc * Cin) * kH) * kW;
    int x_n_base = (n * Cin) * H * W;

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        int x_c_base = x_n_base + ic * H * W;
        int w_ic_base = w_base + ic * kH * kW;

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

    y[idx] = mish_fast(acc - sub12);
}

// Specialized k=3:
// - blockIdx.z: batch n
// - blockIdx.y: oc_pair (2 output channels per block)
// - blockIdx.x: spatial pairs in flattened outH*outW domain (2 outputs per thread)
// We stage weights for (oc0, oc1) into shared once per block to reduce register pressure and global weight loads.
__global__ __launch_bounds__(128, 4) void conv2d_k3_oc2_ow2_sub_sub_mish_shw_kernel(
    const float* __restrict__ x,   // [N, Cin, H, W]
    const float* __restrict__ w,   // [Cout, Cin, 3, 3]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N, Cout, outH, outW]
    int Cin, int H, int W,
    int Cout, int outH, int outW,
    float sub12
) {
    // shared weights: [2][Cin][9] flattened
    extern __shared__ float shw[]; // size = 2*Cin*9
    float* shw0 = shw;
    float* shw1 = shw + Cin * 9;

    int n = (int)blockIdx.z;
    int oc_pair = (int)blockIdx.y;
    int oc0 = oc_pair * 2;
    int oc1 = oc0 + 1;

    // stage weights for oc0 and oc1
    int t = (int)threadIdx.x;
    int total_w = Cin * 9;
    for (int i = t; i < total_w; i += (int)blockDim.x) {
        int ic = i / 9;
        int k  = i - ic * 9;
        float v0 = 0.0f, v1 = 0.0f;
        if (oc0 < Cout) v0 = __ldg(w + (oc0 * Cin + ic) * 9 + k);
        if (oc1 < Cout) v1 = __ldg(w + (oc1 * Cin + ic) * 9 + k);
        shw0[i] = v0;
        shw1[i] = v1;
    }
    __syncthreads();

    // each thread handles two linear output positions
    int lin_pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lin0 = lin_pair * 2;
    int total_lin = outH * outW;
    if (lin0 >= total_lin) return;
    int lin1 = lin0 + 1;
    bool in1 = (lin1 < total_lin);

    int oh0 = lin0 / outW;
    int ow0 = lin0 - oh0 * outW;

    int oh1 = in1 ? (lin1 / outW) : 0;
    int ow1 = in1 ? (lin1 - oh1 * outW) : 0;

    // Bias
    float acc00 = (b && oc0 < Cout) ? __ldg(b + oc0) : 0.0f;
    float acc01 = acc00;
    float acc10 = (b && oc1 < Cout) ? __ldg(b + oc1) : 0.0f;
    float acc11 = acc10;

    // common-case neighbor in same row
    bool neigh = in1 && (oh1 == oh0) && (ow1 == ow0 + 1);

    int x_n_base = (n * Cin) * H * W;

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        const float* xbase = x + x_n_base + ic * H * W;

        // weights from shared
        const float* w0 = shw0 + ic * 9;
        const float* w1p = shw1 + ic * 9;

        float w00 = w0[0], w01 = w0[1], w02 = w0[2];
        float w10 = w0[3], w11 = w0[4], w12 = w0[5];
        float w20 = w0[6], w21 = w0[7], w22 = w0[8];

        float v00 = w1p[0], v01 = w1p[1], v02 = w1p[2];
        float v10 = w1p[3], v11 = w1p[4], v12 = w1p[5];
        float v20 = w1p[6], v21 = w1p[7], v22 = w1p[8];

        const float* p0 = xbase + (oh0 * W + ow0);

        if (neigh) {
            // load 3x4 patch once (always in-bounds for valid conv when ow0<=outW-2)
            float a00 = __ldg(p0 + 0);
            float a01 = __ldg(p0 + 1);
            float a02 = __ldg(p0 + 2);
            float a03 = __ldg(p0 + 3);

            float a10 = __ldg(p0 + W + 0);
            float a11 = __ldg(p0 + W + 1);
            float a12 = __ldg(p0 + W + 2);
            float a13 = __ldg(p0 + W + 3);

            float a20 = __ldg(p0 + 2 * W + 0);
            float a21 = __ldg(p0 + 2 * W + 1);
            float a22 = __ldg(p0 + 2 * W + 2);
            float a23 = __ldg(p0 + 2 * W + 3);

            // oc0 lin0
            acc00 = fmaf(a00, w00, acc00); acc00 = fmaf(a01, w01, acc00); acc00 = fmaf(a02, w02, acc00);
            acc00 = fmaf(a10, w10, acc00); acc00 = fmaf(a11, w11, acc00); acc00 = fmaf(a12, w12, acc00);
            acc00 = fmaf(a20, w20, acc00); acc00 = fmaf(a21, w21, acc00); acc00 = fmaf(a22, w22, acc00);
            // oc0 lin1 (shifted)
            acc01 = fmaf(a01, w00, acc01); acc01 = fmaf(a02, w01, acc01); acc01 = fmaf(a03, w02, acc01);
            acc01 = fmaf(a11, w10, acc01); acc01 = fmaf(a12, w11, acc01); acc01 = fmaf(a13, w12, acc01);
            acc01 = fmaf(a21, w20, acc01); acc01 = fmaf(a22, w21, acc01); acc01 = fmaf(a23, w22, acc01);

            if (oc1 < Cout) {
                // oc1 lin0
                acc10 = fmaf(a00, v00, acc10); acc10 = fmaf(a01, v01, acc10); acc10 = fmaf(a02, v02, acc10);
                acc10 = fmaf(a10, v10, acc10); acc10 = fmaf(a11, v11, acc10); acc10 = fmaf(a12, v12, acc10);
                acc10 = fmaf(a20, v20, acc10); acc10 = fmaf(a21, v21, acc10); acc10 = fmaf(a22, v22, acc10);
                // oc1 lin1
                acc11 = fmaf(a01, v00, acc11); acc11 = fmaf(a02, v01, acc11); acc11 = fmaf(a03, v02, acc11);
                acc11 = fmaf(a11, v10, acc11); acc11 = fmaf(a12, v11, acc11); acc11 = fmaf(a13, v12, acc11);
                acc11 = fmaf(a21, v20, acc11); acc11 = fmaf(a22, v21, acc11); acc11 = fmaf(a23, v22, acc11);
            }
        } else {
            // lin0 always valid
            {
                float a00 = __ldg(p0 + 0), a01 = __ldg(p0 + 1), a02 = __ldg(p0 + 2);
                float a10 = __ldg(p0 + W + 0), a11 = __ldg(p0 + W + 1), a12 = __ldg(p0 + W + 2);
                float a20 = __ldg(p0 + 2 * W + 0), a21 = __ldg(p0 + 2 * W + 1), a22 = __ldg(p0 + 2 * W + 2);

                acc00 = fmaf(a00, w00, acc00); acc00 = fmaf(a01, w01, acc00); acc00 = fmaf(a02, w02, acc00);
                acc00 = fmaf(a10, w10, acc00); acc00 = fmaf(a11, w11, acc00); acc00 = fmaf(a12, w12, acc00);
                acc00 = fmaf(a20, w20, acc00); acc00 = fmaf(a21, w21, acc00); acc00 = fmaf(a22, w22, acc00);

                if (oc1 < Cout) {
                    acc10 = fmaf(a00, v00, acc10); acc10 = fmaf(a01, v01, acc10); acc10 = fmaf(a02, v02, acc10);
                    acc10 = fmaf(a10, v10, acc10); acc10 = fmaf(a11, v11, acc10); acc10 = fmaf(a12, v12, acc10);
                    acc10 = fmaf(a20, v20, acc10); acc10 = fmaf(a21, v21, acc10); acc10 = fmaf(a22, v22, acc10);
                }
            }

            if (in1) {
                const float* p1 = xbase + (oh1 * W + ow1);
                float b00 = __ldg(p1 + 0), b01 = __ldg(p1 + 1), b02 = __ldg(p1 + 2);
                float b10 = __ldg(p1 + W + 0), b11 = __ldg(p1 + W + 1), b12 = __ldg(p1 + W + 2);
                float b20 = __ldg(p1 + 2 * W + 0), b21 = __ldg(p1 + 2 * W + 1), b22 = __ldg(p1 + 2 * W + 2);

                acc01 = fmaf(b00, w00, acc01); acc01 = fmaf(b01, w01, acc01); acc01 = fmaf(b02, w02, acc01);
                acc01 = fmaf(b10, w10, acc01); acc01 = fmaf(b11, w11, acc01); acc01 = fmaf(b12, w12, acc01);
                acc01 = fmaf(b20, w20, acc01); acc01 = fmaf(b21, w21, acc01); acc01 = fmaf(b22, w22, acc01);

                if (oc1 < Cout) {
                    acc11 = fmaf(b00, v00, acc11); acc11 = fmaf(b01, v01, acc11); acc11 = fmaf(b02, v02, acc11);
                    acc11 = fmaf(b10, v10, acc11); acc11 = fmaf(b11, v11, acc11); acc11 = fmaf(b12, v12, acc11);
                    acc11 = fmaf(b20, v20, acc11); acc11 = fmaf(b21, v21, acc11); acc11 = fmaf(b22, v22, acc11);
                }
            }
        }
    }

    // fused subtract + mish
    acc00 = mish_fast(acc00 - sub12);
    if (oc1 < Cout) acc10 = mish_fast(acc10 - sub12);
    if (in1) {
        acc01 = mish_fast(acc01 - sub12);
        if (oc1 < Cout) acc11 = mish_fast(acc11 - sub12);
    }

    // store
    if (oc0 < Cout) {
        int base0 = ((n * Cout + oc0) * outH + oh0) * outW + ow0;
        y[base0] = acc00;
        if (in1) {
            int base0b = ((n * Cout + oc0) * outH + oh1) * outW + ow1;
            y[base0b] = acc01;
        }
    }
    if (oc1 < Cout) {
        int base1 = ((n * Cout + oc1) * outH + oh0) * outW + ow0;
        y[base1] = acc10;
        if (in1) {
            int base1b = ((n * Cout + oc1) * outH + oh1) * outW + ow1;
            y[base1b] = acc11;
        }
    }
}

torch::Tensor conv2d_subtract_subtract_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    double subtract_value_1,
    double subtract_value_2
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be OIHW (4D)");

    auto bias = bias_opt.has_value() ? bias_opt.value() : torch::Tensor();
    if (bias.defined()) {
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
    if (bias.defined()) TORCH_CHECK((int)bias.size(0) == Cout, "bias size must equal out_channels");
    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be > 0");

    int outH = H - kH + 1;
    int outW = W - kW + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "conv output size is non-positive");

    auto y = torch::empty({N, Cout, outH, outW}, x.options());

    const float* bptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float sub12 = (float)(subtract_value_1 + subtract_value_2);

    // Specialized path for k=3
    if (kH == 3 && kW == 3) {
        int total_lin = outH * outW;
        int total_pairs = (total_lin + 1) / 2;
        int threads = 128;
        int blocks_x = (total_pairs + threads - 1) / threads;
        dim3 grid(blocks_x, (Cout + 1) / 2, N);
        size_t shmem = (size_t)(2 * Cin * 9) * sizeof(float);

        conv2d_k3_oc2_ow2_sub_sub_mish_shw_kernel<<<grid, threads, shmem>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bptr,
            y.data_ptr<float>(),
            Cin, H, W,
            Cout, outH, outW,
            sub12
        );
        return y;
    }

    // Generic fallback
    int64_t total = (int64_t)N * Cout * outH * outW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    conv2d_sub_sub_mish_generic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, Cin, H, W,
        Cout, kH, kW,
        outH, outW,
        sub12
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv2d_subtract_subtract_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    double subtract_value_1,
    double subtract_value_2
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_sub_sub_mish_opt9",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_subtract_subtract_mish_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv2d -> subtract -> subtract -> mish, using a fused custom CUDA forward op.
    This fused op supports ONLY: stride=1, padding=0, dilation=1, groups=1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = float(subtract_value_1)
        self.subtract_value_2 = float(subtract_value_2)
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

        return self.custom_ops.conv2d_subtract_subtract_mish_forward_cuda(
            x, w, bias_opt, self.subtract_value_1, self.subtract_value_2
        )