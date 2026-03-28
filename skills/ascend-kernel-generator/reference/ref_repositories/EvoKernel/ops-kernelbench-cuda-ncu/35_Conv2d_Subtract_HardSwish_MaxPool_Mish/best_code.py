import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

static __device__ __forceinline__ float hardswish(float x) {
    float t = clampf(x + 3.0f, 0.0f, 6.0f);
    return x * (t * (1.0f / 6.0f));
}

// numerically stable softplus for mish
static __device__ __forceinline__ float softplus_fast(float x) {
    float ax = fabsf(x);
    float m = fmaxf(x, 0.0f);
    return m + log1pf(expf(-ax));
}
static __device__ __forceinline__ float mish_fast(float x) {
    return x * tanhf(softplus_fast(x));
}

// -------------------- Generic baseline kernel (fallback) --------------------
__global__ void conv2d_sub_hsw_pool_mish_kernel_generic(
    const float* __restrict__ x,        // [N, Cin, H, W]
    const float* __restrict__ w,        // [Cout, Cin, kH, kW]
    const float* __restrict__ b,        // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, outPH, outPW]
    int N, int Cin, int H, int W,
    int Cout,
    int kH, int kW,
    int outH, int outW,
    int poolK,
    int outPH, int outPW,
    float subtract_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * outPH * outPW;
    if (idx >= total) return;

    int opw = idx % outPW;
    int oph = (idx / outPW) % outPH;
    int oc  = (idx / (outPW * outPH)) % Cout;
    int n   = idx / (outPW * outPH * Cout);

    int conv_h0 = oph * poolK;
    int conv_w0 = opw * poolK;

    float maxval = -INFINITY;

    for (int ph = 0; ph < poolK; ++ph) {
        int oh = conv_h0 + ph;
        if ((unsigned)oh >= (unsigned)outH) continue;
        for (int pw = 0; pw < poolK; ++pw) {
            int ow = conv_w0 + pw;
            if ((unsigned)ow >= (unsigned)outW) continue;

            float acc = (b != nullptr) ? __ldg(&b[oc]) : 0.0f;

            int ih0 = oh;
            int iw0 = ow;

            int w_base = ((oc * Cin) * kH) * kW;
            int x_n_base = (n * Cin) * H * W;

            #pragma unroll 1
            for (int ic = 0; ic < Cin; ++ic) {
                int x_c_base = x_n_base + ic * H * W;
                int w_ic_base = w_base + ic * kH * kW;
                #pragma unroll 1
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh;
                    int x_row = x_c_base + ih * W;
                    int w_row = w_ic_base + kh * kW;
                    #pragma unroll 1
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw;
                        float xv = __ldg(&x[x_row + iw]);
                        float wv = __ldg(&w[w_row + kw]);
                        acc = fmaf(xv, wv, acc);
                    }
                }
            }

            acc -= subtract_value;
            acc = hardswish(acc);
            maxval = acc > maxval ? acc : maxval;
        }
    }

    y[((n * Cout + oc) * outPH + oph) * outPW + opw] = mish_fast(maxval);
}

// -------------------- Optimized fast path for fixed hot shape --------------------
// Assumptions: k=3x3, poolK=2, stride=1, padding=0, dilation=1, groups=1
// Target common model: Cin=64, Cout=128, H=W=128 (but H/W can vary as long as outPH/outPW computed).
// Mapping: each block handles one (n, oc_pair) and a tile of pooled spatial positions.
// For each ic: load 2*9 weights into shared, then each thread computes its spatial point (2 outputs) fully in registers.
__constant__ float CB_128[128]; // bias only (small, fits)

__global__ __launch_bounds__(256, 2)
void conv3x3_pool2_ocpair_tiled_kernel(
    const float* __restrict__ x,        // [N, 64, H, W]
    const float* __restrict__ w,        // [128, 64, 3, 3] (global, read-only)
    float* __restrict__ y,              // [N, 128, outPH, outPW]
    int H, int W,
    int outH, int outW,
    int outPH, int outPW,
    float subtract_value,
    bool has_bias
) {
    int n = (int)blockIdx.z;
    int oc_pair = (int)blockIdx.y; // 0..63
    int oc0 = oc_pair * 2;

    // spatial linear index over pooled output
    int tid = (int)threadIdx.x; // 0..255
    int tile_base = ((int)blockIdx.x) * 256 + tid;
    int total_sp = outPH * outPW;
    if (tile_base >= total_sp) return;

    int oph = tile_base / outPW;
    int opw = tile_base - oph * outPW;

    int oh0 = oph * 2;
    int ow0 = opw * 2;

    // shared weights for this oc-pair and one ic at a time: 2 channels * 9 weights
    __shared__ float shw[18];

    // Accumulators for maxpooled value per output channel
    float max0 = -INFINITY;
    float max1 = -INFINITY;

    // base pointers
    int x_n_base = (n * 64) * H * W;

    // loop over 2x2 pool points (conv output coords)
    #pragma unroll
    for (int ph = 0; ph < 2; ++ph) {
        #pragma unroll
        for (int pw = 0; pw < 2; ++pw) {
            int oh = oh0 + ph;
            int ow = ow0 + pw;

            float acc0 = has_bias ? CB_128[oc0 + 0] : 0.0f;
            float acc1 = has_bias ? CB_128[oc0 + 1] : 0.0f;

            if ((unsigned)oh < (unsigned)outH && (unsigned)ow < (unsigned)outW) {
                int off0 = (oh + 0) * W + ow;
                int off1 = (oh + 1) * W + ow;
                int off2 = (oh + 2) * W + ow;

                #pragma unroll 1
                for (int ic = 0; ic < 64; ++ic) {
                    // cooperative load weights for this ic into shared
                    // thread 0..17 load one weight each (others idle) to keep registers low
                    if (tid < 18) {
                        int t = tid;
                        int oc_sel = t / 9; // 0 or 1
                        int k = t - oc_sel * 9;
                        const float* wptr = w + (((oc0 + oc_sel) * 64 + ic) * 9);
                        shw[t] = __ldg(&wptr[k]);
                    }
                    __syncthreads();

                    const float* xptr = x + x_n_base + ic * H * W;
                    float x00 = __ldg(&xptr[off0 + 0]);
                    float x01 = __ldg(&xptr[off0 + 1]);
                    float x02 = __ldg(&xptr[off0 + 2]);
                    float x10 = __ldg(&xptr[off1 + 0]);
                    float x11 = __ldg(&xptr[off1 + 1]);
                    float x12 = __ldg(&xptr[off1 + 2]);
                    float x20 = __ldg(&xptr[off2 + 0]);
                    float x21 = __ldg(&xptr[off2 + 1]);
                    float x22 = __ldg(&xptr[off2 + 2]);

                    // oc0 weights
                    float w0 = shw[0], w1 = shw[1], w2 = shw[2];
                    float w3 = shw[3], w4 = shw[4], w5 = shw[5];
                    float w6 = shw[6], w7 = shw[7], w8 = shw[8];
                    acc0 = fmaf(x00, w0, acc0); acc0 = fmaf(x01, w1, acc0); acc0 = fmaf(x02, w2, acc0);
                    acc0 = fmaf(x10, w3, acc0); acc0 = fmaf(x11, w4, acc0); acc0 = fmaf(x12, w5, acc0);
                    acc0 = fmaf(x20, w6, acc0); acc0 = fmaf(x21, w7, acc0); acc0 = fmaf(x22, w8, acc0);

                    // oc1 weights
                    w0 = shw[9], w1 = shw[10], w2 = shw[11];
                    w3 = shw[12], w4 = shw[13], w5 = shw[14];
                    w6 = shw[15], w7 = shw[16], w8 = shw[17];
                    acc1 = fmaf(x00, w0, acc1); acc1 = fmaf(x01, w1, acc1); acc1 = fmaf(x02, w2, acc1);
                    acc1 = fmaf(x10, w3, acc1); acc1 = fmaf(x11, w4, acc1); acc1 = fmaf(x12, w5, acc1);
                    acc1 = fmaf(x20, w6, acc1); acc1 = fmaf(x21, w7, acc1); acc1 = fmaf(x22, w8, acc1);

                    __syncthreads();
                }

                acc0 = hardswish(acc0 - subtract_value);
                acc1 = hardswish(acc1 - subtract_value);

                max0 = fmaxf(max0, acc0);
                max1 = fmaxf(max1, acc1);
            } else {
                // outside conv bounds contributes -inf to maxpool; no-op
            }
        }
    }

    float out0 = mish_fast(max0);
    float out1 = mish_fast(max1);

    int out_sp = outPH * outPW;
    int idx0 = ((n * 128 + (oc0 + 0)) * outPH + oph) * outPW + opw;
    int idx1 = idx0 + out_sp; // next channel plane
    y[idx0] = out0;
    y[idx1] = out1;
}

static void upload_bias_128(torch::Tensor bias) {
    cudaMemcpyToSymbol(CB_128, bias.data_ptr<float>(), sizeof(float) * 128, 0, cudaMemcpyDeviceToDevice);
}

torch::Tensor conv2d_subtract_hard_swish_max_pool_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    double subtract_value,
    int64_t pool_kernel_size
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

    int poolK = (int)pool_kernel_size;
    TORCH_CHECK(poolK > 0, "pool_kernel_size must be > 0");
    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be > 0");

    int outH = H - kH + 1;
    int outW = W - kW + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "conv output size is non-positive");

    int outPH = (outH - poolK) / poolK + 1;
    int outPW = (outW - poolK) / poolK + 1;
    TORCH_CHECK(outPH > 0 && outPW > 0, "pool output size is non-positive");

    auto y = torch::empty({N, Cout, outPH, outPW}, x.options());
    const float* bptr = has_bias ? bias.data_ptr<float>() : nullptr;

    // Hot specialization: Cin=64, Cout=128, k=3, pool=2
    if (kH == 3 && kW == 3 && poolK == 2 && Cin == 64 && Cout == 128) {
        if (has_bias) upload_bias_128(bias);

        int total_sp = outPH * outPW;
        int blocks_x = (total_sp + 256 - 1) / 256;
        dim3 grid(blocks_x, 128 / 2, N);
        conv3x3_pool2_ocpair_tiled_kernel<<<grid, 256>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            y.data_ptr<float>(),
            H, W,
            outH, outW,
            outPH, outPW,
            (float)subtract_value,
            has_bias
        );
        return y;
    }

    // Fallback generic kernel
    const int threads = 256;
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)outPH * (int64_t)outPW;
    int blocks = (int)((total + threads - 1) / threads);

    conv2d_sub_hsw_pool_mish_kernel_generic<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, Cin, H, W,
        Cout,
        kH, kW,
        outH, outW,
        poolK,
        outPH, outPW,
        (float)subtract_value
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv2d_subtract_hard_swish_max_pool_mish_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    double subtract_value,
    int64_t pool_kernel_size
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_sub_hsw_pool_mish_v7",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv2d_subtract_hard_swish_max_pool_mish_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv2d -> subtract -> hardswish -> maxpool2d -> mish, using a fused custom CUDA forward op.
    Fast-path specialized for (Cin=64, Cout=128, k=3, pool=2) plus generic fallback.
    Forward-only op (no autograd for the fused kernel).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)
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

        return self.custom_ops.conv2d_subtract_hard_swish_max_pool_mish_forward_cuda(
            x, w, bias_opt, self.subtract_value, self.pool_kernel_size
        )