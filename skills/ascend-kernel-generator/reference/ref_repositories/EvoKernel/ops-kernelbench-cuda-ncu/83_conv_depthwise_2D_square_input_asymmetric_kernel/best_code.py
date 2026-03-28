import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float4 ldg_f32x4(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(reinterpret_cast<const float4*>(p));
#else
    return *reinterpret_cast<const float4*>(p);
#endif
}

// Generic fallback: one thread computes one output element
__global__ void dwconv2d_kHx1_generic(
    const float* __restrict__ x,
    const float* __restrict__ w, // [C,1,kH,1] contiguous
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH,
    int sH, int sW,
    int pH, int pW,
    int dH,
    bool has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int n  = idx / (outW * outH * C);

    int ih0 = oh * sH - pH;
    int iw0 = ow * sW - pW;

    float acc = has_bias ? ldg_f32(b + c) : 0.0f;
    int x_base = (n * C + c) * H * W;
    int w_base = c * kH;

    #pragma unroll 1
    for (int kh = 0; kh < kH; ++kh) {
        int ih = ih0 + kh * dH;
        if ((unsigned)ih >= (unsigned)H) continue;
        int iw = iw0; // kW=1
        if ((unsigned)iw >= (unsigned)W) continue;
        float xv = ldg_f32(x + x_base + ih * W + iw);
        float wv = ldg_f32(w + w_base + kh);
        acc = fmaf(xv, wv, acc);
    }

    y[((n * C + c) * outH + oh) * outW + ow] = acc;
}

// Vectorized fast path for sW==1, pW==0, any sH/pH/dH.
// Each thread computes WPT=4 output columns. No shared memory, no barrier.
// Weights loaded once into registers; input loads are scalar or float4 when aligned and in-bounds.
template<int WPT>
__global__ __launch_bounds__(256, 2)
void dwconv2d_kHx1_swp0_vecwpt_anykh(
    const float* __restrict__ x,
    const float* __restrict__ w, // treated as [C*kH]
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH,
    int sH, int pH, int dH,
    bool has_bias
) {
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (oh >= outH) return;

    int ow0 = (((int)blockIdx.x * (int)blockDim.x) + (int)threadIdx.x) * WPT;
    if (ow0 >= outW) return;

    const float biasv = has_bias ? ldg_f32(b + c) : 0.0f;

    float acc0 = biasv, acc1 = biasv, acc2 = biasv, acc3 = biasv;

    const int ow1 = ow0 + 1;
    const int ow2 = ow0 + 2;
    const int ow3 = ow0 + 3;

    const bool in0 = (unsigned)ow0 < (unsigned)outW;
    const bool in1 = (unsigned)ow1 < (unsigned)outW;
    const bool in2 = (unsigned)ow2 < (unsigned)outW;
    const bool in3 = (unsigned)ow3 < (unsigned)outW;

    const int x_nc = (n * C + c) * H * W;
    const int ih0 = oh * sH - pH;
    const float* __restrict__ wptr = w + c * kH;

    #pragma unroll 1
    for (int kh = 0; kh < kH; ++kh) {
        int ih = ih0 + kh * dH;
        if ((unsigned)ih >= (unsigned)H) continue;

        const float wv = ldg_f32(wptr + kh);
        const float* __restrict__ rowp = x + x_nc + ih * W + ow0;

        // Prefer float4 input load if:
        //  - ow0..ow0+3 are all in [0, W)
        //  - pointer aligned to 16B
        // For this operator outW is usually <= W (pW==0,sW==1,kW==1 => outW==W).
        if (in0 && in3 && ((unsigned)(ow0 + 3) < (unsigned)W)) {
            uintptr_t addr = (uintptr_t)(rowp);
            if ((addr & 0xF) == 0) {
                float4 v = ldg_f32x4(rowp);
                acc0 = fmaf(v.x, wv, acc0);
                acc1 = fmaf(v.y, wv, acc1);
                acc2 = fmaf(v.z, wv, acc2);
                acc3 = fmaf(v.w, wv, acc3);
            } else {
                acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
                acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
                acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
                acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
            }
        } else {
            if (in0 && (unsigned)ow0 < (unsigned)W) acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
            if (in1 && (unsigned)ow1 < (unsigned)W) acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
            if (in2 && (unsigned)ow2 < (unsigned)W) acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
            if (in3 && (unsigned)ow3 < (unsigned)W) acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
        }
    }

    float* __restrict__ yptr = y + ((n * C + c) * outH + oh) * outW + ow0;

    if (ow0 + (WPT - 1) < outW) {
        uintptr_t yaddr = (uintptr_t)yptr;
        if ((yaddr & 0xF) == 0) {
            float4 out4; out4.x = acc0; out4.y = acc1; out4.z = acc2; out4.w = acc3;
            *reinterpret_cast<float4*>(yptr) = out4;
        } else {
            yptr[0] = acc0; yptr[1] = acc1; yptr[2] = acc2; yptr[3] = acc3;
        }
    } else {
        if (ow0 < outW) yptr[0] = acc0;
        if (ow1 < outW) yptr[1] = acc1;
        if (ow2 < outW) yptr[2] = acc2;
        if (ow3 < outW) yptr[3] = acc3;
    }
}

// kH=3 specialization: fully unrolled and slightly less control flow.
template<int WPT>
__global__ __launch_bounds__(256, 2)
void dwconv2d_k3x1_swp0_vecwpt(
    const float* __restrict__ x,
    const float* __restrict__ w, // [C*3]
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int sH, int pH, int dH,
    bool has_bias
) {
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (oh >= outH) return;

    int ow0 = (((int)blockIdx.x * (int)blockDim.x) + (int)threadIdx.x) * WPT;
    if (ow0 >= outW) return;

    const float biasv = has_bias ? ldg_f32(b + c) : 0.0f;
    float acc0 = biasv, acc1 = biasv, acc2 = biasv, acc3 = biasv;

    const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
    const bool full4 = (ow0 + 3) < outW && (ow0 + 3) < W;

    const float* __restrict__ wptr = w + c * 3;
    const float w0 = ldg_f32(wptr + 0);
    const float w1 = ldg_f32(wptr + 1);
    const float w2 = ldg_f32(wptr + 2);

    const int x_nc = (n * C + c) * H * W;
    const int ih0 = oh * sH - pH;

    auto step = [&](int ih, float wv) {
        if ((unsigned)ih >= (unsigned)H) return;
        const float* __restrict__ rowp = x + x_nc + ih * W + ow0;
        if (full4) {
            uintptr_t addr = (uintptr_t)rowp;
            if ((addr & 0xF) == 0) {
                float4 v = ldg_f32x4(rowp);
                acc0 = fmaf(v.x, wv, acc0);
                acc1 = fmaf(v.y, wv, acc1);
                acc2 = fmaf(v.z, wv, acc2);
                acc3 = fmaf(v.w, wv, acc3);
            } else {
                acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
                acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
                acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
                acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
            }
        } else {
            if (ow0 < outW && ow0 < W) acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
            if (ow1 < outW && ow1 < W) acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
            if (ow2 < outW && ow2 < W) acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
            if (ow3 < outW && ow3 < W) acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
        }
    };

    step(ih0 + 0 * dH, w0);
    step(ih0 + 1 * dH, w1);
    step(ih0 + 2 * dH, w2);

    float* __restrict__ yptr = y + ((n * C + c) * outH + oh) * outW + ow0;
    if (ow0 + (WPT - 1) < outW) {
        uintptr_t yaddr = (uintptr_t)yptr;
        if ((yaddr & 0xF) == 0) {
            float4 out4; out4.x = acc0; out4.y = acc1; out4.z = acc2; out4.w = acc3;
            *reinterpret_cast<float4*>(yptr) = out4;
        } else {
            yptr[0] = acc0; yptr[1] = acc1; yptr[2] = acc2; yptr[3] = acc3;
        }
    } else {
        if (ow0 < outW) yptr[0] = acc0;
        if (ow1 < outW) yptr[1] = acc1;
        if (ow2 < outW) yptr[2] = acc2;
        if (ow3 < outW) yptr[3] = acc3;
    }
}

// kH=5 specialization
template<int WPT>
__global__ __launch_bounds__(256, 2)
void dwconv2d_k5x1_swp0_vecwpt(
    const float* __restrict__ x,
    const float* __restrict__ w, // [C*5]
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outH, int outW,
    int sH, int pH, int dH,
    bool has_bias
) {
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (oh >= outH) return;

    int ow0 = (((int)blockIdx.x * (int)blockDim.x) + (int)threadIdx.x) * WPT;
    if (ow0 >= outW) return;

    const float biasv = has_bias ? ldg_f32(b + c) : 0.0f;
    float acc0 = biasv, acc1 = biasv, acc2 = biasv, acc3 = biasv;

    const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
    const bool full4 = (ow0 + 3) < outW && (ow0 + 3) < W;

    const float* __restrict__ wptr = w + c * 5;
    const float w0 = ldg_f32(wptr + 0);
    const float w1 = ldg_f32(wptr + 1);
    const float w2 = ldg_f32(wptr + 2);
    const float w3 = ldg_f32(wptr + 3);
    const float w4 = ldg_f32(wptr + 4);

    const int x_nc = (n * C + c) * H * W;
    const int ih0 = oh * sH - pH;

    auto step = [&](int ih, float wv) {
        if ((unsigned)ih >= (unsigned)H) return;
        const float* __restrict__ rowp = x + x_nc + ih * W + ow0;
        if (full4) {
            uintptr_t addr = (uintptr_t)rowp;
            if ((addr & 0xF) == 0) {
                float4 v = ldg_f32x4(rowp);
                acc0 = fmaf(v.x, wv, acc0);
                acc1 = fmaf(v.y, wv, acc1);
                acc2 = fmaf(v.z, wv, acc2);
                acc3 = fmaf(v.w, wv, acc3);
            } else {
                acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
                acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
                acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
                acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
            }
        } else {
            if (ow0 < outW && ow0 < W) acc0 = fmaf(ldg_f32(rowp + 0), wv, acc0);
            if (ow1 < outW && ow1 < W) acc1 = fmaf(ldg_f32(rowp + 1), wv, acc1);
            if (ow2 < outW && ow2 < W) acc2 = fmaf(ldg_f32(rowp + 2), wv, acc2);
            if (ow3 < outW && ow3 < W) acc3 = fmaf(ldg_f32(rowp + 3), wv, acc3);
        }
    };

    step(ih0 + 0 * dH, w0);
    step(ih0 + 1 * dH, w1);
    step(ih0 + 2 * dH, w2);
    step(ih0 + 3 * dH, w3);
    step(ih0 + 4 * dH, w4);

    float* __restrict__ yptr = y + ((n * C + c) * outH + oh) * outW + ow0;
    if (ow0 + (WPT - 1) < outW) {
        uintptr_t yaddr = (uintptr_t)yptr;
        if ((yaddr & 0xF) == 0) {
            float4 out4; out4.x = acc0; out4.y = acc1; out4.z = acc2; out4.w = acc3;
            *reinterpret_cast<float4*>(yptr) = out4;
        } else {
            yptr[0] = acc0; yptr[1] = acc1; yptr[2] = acc2; yptr[3] = acc3;
        }
    } else {
        if (ow0 < outW) yptr[0] = acc0;
        if (ow1 < outW) yptr[1] = acc1;
        if (ow2 < outW) yptr[2] = acc2;
        if (ow3 < outW) yptr[3] = acc3;
    }
}

torch::Tensor conv_depthwise2d_square_input_asymmetric_kernel_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be [C,1,kH,1]");
    TORCH_CHECK(w.size(1) == 1, "w second dim must be 1 for depthwise");
    TORCH_CHECK(w.size(3) == 1, "w fourth dim must be 1 for asymmetric kHx1");
    TORCH_CHECK(sH > 0 && sW > 0, "stride must be > 0");
    TORCH_CHECK(dH > 0, "dilation must be > 0");

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    TORCH_CHECK(w.size(0) == C64, "w.size(0) must equal input channels");
    const int64_t kH64 = w.size(2);
    TORCH_CHECK(kH64 > 0, "kH must be > 0");

    const int64_t outH64 = (H64 + 2 * pH - dH * (kH64 - 1) - 1) / sH + 1;
    const int64_t outW64 = (W64 + 2 * pW - 1) / sW + 1; // kW=1
    TORCH_CHECK(outH64 > 0 && outW64 > 0, "computed output size is non-positive");

    const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    const float* b_ptr = nullptr;
    if (has_bias) {
        auto b = bias_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1 && b.size(0) == C64, "bias must be [C]");
        b_ptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N64, C64, outH64, outW64}, x.options());

    const int N = (int)N64, C = (int)C64, H = (int)H64, W = (int)W64;
    const int outH = (int)outH64, outW = (int)outW64;
    const int kH = (int)kH64;

    // Vectorized fast path when sW=1,pW=0
    if (sW == 1 && pW == 0) {
        constexpr int WPT = 4;
        dim3 block(32, 8, 1); // 256 threads
        dim3 grid(
            (outW + (block.x * WPT) - 1) / (block.x * WPT),
            (outH + block.y - 1) / block.y,
            N * C
        );

        if (kH == 3) {
            dwconv2d_k3x1_swp0_vecwpt<WPT><<<grid, block>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, C, H, W,
                outH, outW,
                (int)sH, (int)pH, (int)dH,
                has_bias
            );
            return y;
        } else if (kH == 5) {
            dwconv2d_k5x1_swp0_vecwpt<WPT><<<grid, block>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, C, H, W,
                outH, outW,
                (int)sH, (int)pH, (int)dH,
                has_bias
            );
            return y;
        } else {
            dwconv2d_kHx1_swp0_vecwpt_anykh<WPT><<<grid, block>>>(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, C, H, W,
                outH, outW,
                kH,
                (int)sH, (int)pH, (int)dH,
                has_bias
            );
            return y;
        }
    }

    // Generic fallback
    const int threads = 256;
    const int64_t total = (int64_t)N * C * outH * outW;
    const int blocks = (int)((total + threads - 1) / threads);
    dwconv2d_kHx1_generic<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kH,
        (int)sH, (int)sW,
        (int)pH, (int)pW,
        (int)dH,
        has_bias
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_depthwise2d_square_input_asymmetric_kernel_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dwconv_khx1_v8_vec4_nosmem",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_depthwise2d_square_input_asymmetric_kernel_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Forward-only replacement for depthwise nn.Conv2d(in_channels->in_channels, groups=in_channels)
    with kernel shape (kH, 1), optimized CUDA kernel.
    Assumes CUDA input; converts to float32 contiguous as needed.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.kH = int(kernel_size)
        self.sH = int(stride)
        self.sW = int(stride)
        self.pH = int(padding)
        self.pW = int(padding)
        self.dH = int(dilation)
        self.bias_enabled = bool(bias)

        w = torch.empty(self.in_channels, 1, self.kH, 1, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = nn.Parameter(w)

        if self.bias_enabled:
            self.bias = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32))
        else:
            self.bias = None

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        b = None
        if self.bias is not None:
            b = self.bias
            if not b.is_cuda:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()

        return self.custom_ops.conv_depthwise2d_square_input_asymmetric_kernel_forward_cuda(
            x, w, b, self.sH, self.sW, self.pH, self.pW, self.dH
        )