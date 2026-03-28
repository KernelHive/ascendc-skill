import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------
# CUDA/C++ extension
# -------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline __device__ float4 ldg_f32x4(const float* p) {
#if __CUDA_ARCH__ >= 350
    // __ldg supports vector types on many toolchains; if not, it still compiles as global load.
    return __ldg(reinterpret_cast<const float4*>(p));
#else
    return *reinterpret_cast<const float4*>(p);
#endif
}

// Generic kernel (fallback): one thread per output
__global__ void depthwise_conv2d_forward_generic(
    const float* __restrict__ x,      // [N, C, H, W]
    const float* __restrict__ w,      // [C, 1, K, K]
    const float* __restrict__ b,      // [C] or nullptr
    float* __restrict__ y,            // [N, C, outH, outW]
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int outH, int outW,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int n  = idx / (outW * outH * C);

    int in_h0 = oh * stride - padding;
    int in_w0 = ow * stride - padding;

    float acc = has_bias ? b[c] : 0.0f;
    int w_base = (c * K) * K;

    for (int kh = 0; kh < K; ++kh) {
        int ih = in_h0 + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        int x_row = ((n * C + c) * H + ih) * W;
        int w_row = w_base + kh * K;
        for (int kw = 0; kw < K; ++kw) {
            int iw = in_w0 + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            acc = fmaf(ldg_f32(x + x_row + iw), ldg_f32(w + w_row + kw), acc);
        }
    }

    int y_off = ((n * C + c) * outH + oh) * outW + ow;
    y[y_off] = acc;
}

// Fast path: K=3, stride=1, padding=0
// Compute WPT=4 output columns per thread.
// Vectorize loads along width using float4 when aligned and in-bounds.
template<int WPT>
__global__ __launch_bounds__(256, 2) void depthwise_conv2d_k3s1p0_vecwpt(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ w,      // [C,1,3,3]
    const float* __restrict__ b,      // [C] or nullptr
    float* __restrict__ y,            // [N,C,outH,outW]
    int N, int C, int H, int W,
    int outH, int outW,
    bool has_bias
) {
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (oh >= outH) return;

    // Each thread covers 4 output columns
    int ow0 = (((int)blockIdx.x * (int)blockDim.x) + (int)threadIdx.x) * WPT;
    if (ow0 >= outW) return;

    // p0/s1/K3 => ih0=oh, iw0=ow
    int ih0 = oh;
    const float* xbase = x + ((n * C + c) * H + ih0) * W;

    const float* r0 = xbase + ow0;
    const float* r1 = r0 + W;
    const float* r2 = r1 + W;

    // Load weights into registers (9 floats)
    const float* wptr = w + (c * 9);
    float w00 = ldg_f32(wptr + 0), w01 = ldg_f32(wptr + 1), w02 = ldg_f32(wptr + 2);
    float w10 = ldg_f32(wptr + 3), w11 = ldg_f32(wptr + 4), w12 = ldg_f32(wptr + 5);
    float w20 = ldg_f32(wptr + 6), w21 = ldg_f32(wptr + 7), w22 = ldg_f32(wptr + 8);

    float biasv = has_bias ? ldg_f32(b + c) : 0.0f;

    // For WPT=4, need (WPT+2)=6 values per row: cols [0..5]
    // We'll try to load first 4 with float4, then 2 scalars.
    float r0v[6], r1v[6], r2v[6];

    auto load_row6 = [&](const float* rp, float* outv) {
        // For most threads in the interior, ow0+5 < outW+1 == (W-2)+1 => ow0+5 < W-1, still in bounds
        // But we must guard the tail block.
        int max_col = ow0 + 5;
        if (max_col < W) {
            uintptr_t addr = (uintptr_t)rp;
            if ((addr & 0xF) == 0) {
                float4 v4 = ldg_f32x4(rp);
                outv[0] = v4.x; outv[1] = v4.y; outv[2] = v4.z; outv[3] = v4.w;
            } else {
                outv[0] = ldg_f32(rp + 0);
                outv[1] = ldg_f32(rp + 1);
                outv[2] = ldg_f32(rp + 2);
                outv[3] = ldg_f32(rp + 3);
            }
            outv[4] = ldg_f32(rp + 4);
            outv[5] = ldg_f32(rp + 5);
        } else {
            // Tail-safe scalar path (rare for large W)
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                int col = ow0 + i;
                outv[i] = (col < W) ? ldg_f32(rp + i) : 0.0f;
            }
        }
    };

    load_row6(r0, r0v);
    load_row6(r1, r1v);
    load_row6(r2, r2v);

    // Compute 4 outputs. For output i, window uses columns i,i+1,i+2 of each row.
    float acc0 = biasv, acc1 = biasv, acc2 = biasv, acc3 = biasv;

    // row 0
    acc0 = fmaf(r0v[0], w00, acc0); acc0 = fmaf(r0v[1], w01, acc0); acc0 = fmaf(r0v[2], w02, acc0);
    acc1 = fmaf(r0v[1], w00, acc1); acc1 = fmaf(r0v[2], w01, acc1); acc1 = fmaf(r0v[3], w02, acc1);
    acc2 = fmaf(r0v[2], w00, acc2); acc2 = fmaf(r0v[3], w01, acc2); acc2 = fmaf(r0v[4], w02, acc2);
    acc3 = fmaf(r0v[3], w00, acc3); acc3 = fmaf(r0v[4], w01, acc3); acc3 = fmaf(r0v[5], w02, acc3);

    // row 1
    acc0 = fmaf(r1v[0], w10, acc0); acc0 = fmaf(r1v[1], w11, acc0); acc0 = fmaf(r1v[2], w12, acc0);
    acc1 = fmaf(r1v[1], w10, acc1); acc1 = fmaf(r1v[2], w11, acc1); acc1 = fmaf(r1v[3], w12, acc1);
    acc2 = fmaf(r1v[2], w10, acc2); acc2 = fmaf(r1v[3], w11, acc2); acc2 = fmaf(r1v[4], w12, acc2);
    acc3 = fmaf(r1v[3], w10, acc3); acc3 = fmaf(r1v[4], w11, acc3); acc3 = fmaf(r1v[5], w12, acc3);

    // row 2
    acc0 = fmaf(r2v[0], w20, acc0); acc0 = fmaf(r2v[1], w21, acc0); acc0 = fmaf(r2v[2], w22, acc0);
    acc1 = fmaf(r2v[1], w20, acc1); acc1 = fmaf(r2v[2], w21, acc1); acc1 = fmaf(r2v[3], w22, acc1);
    acc2 = fmaf(r2v[2], w20, acc2); acc2 = fmaf(r2v[3], w21, acc2); acc2 = fmaf(r2v[4], w22, acc2);
    acc3 = fmaf(r2v[3], w20, acc3); acc3 = fmaf(r2v[4], w21, acc3); acc3 = fmaf(r2v[5], w22, acc3);

    // Store: try vectorized float4 store if all 4 outputs in-bounds and aligned.
    int ybase = ((n * C + c) * outH + oh) * outW + ow0;

    if (ow0 + (WPT - 1) < outW) {
        uintptr_t yaddr = (uintptr_t)(y + ybase);
        if ((yaddr & 0xF) == 0) {
            float4 out4;
            out4.x = acc0; out4.y = acc1; out4.z = acc2; out4.w = acc3;
            *reinterpret_cast<float4*>(y + ybase) = out4;
        } else {
            y[ybase + 0] = acc0;
            y[ybase + 1] = acc1;
            y[ybase + 2] = acc2;
            y[ybase + 3] = acc3;
        }
    } else {
        if (ow0 + 0 < outW) y[ybase + 0] = acc0;
        if (ow0 + 1 < outW) y[ybase + 1] = acc1;
        if (ow0 + 2 < outW) y[ybase + 2] = acc2;
        if (ow0 + 3 < outW) y[ybase + 3] = acc3;
    }
}

torch::Tensor conv_depthwise2d_asymmetric_input_square_kernel(
    torch::Tensor x,       // [N,C,H,W]
    torch::Tensor w,       // [C,1,K,K]
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be [C,1,K,K]");
    TORCH_CHECK(w.size(1) == 1, "w second dim must be 1 for depthwise");
    TORCH_CHECK(x.size(1) == w.size(0), "x channels must match w.size(0)");
    TORCH_CHECK(w.size(2) == w.size(3), "kernel must be square");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int K = (int)w.size(2);

    const int outH = (H + 2 * (int)padding - K) / (int)stride + 1;
    const int outW = (W + 2 * (int)padding - K) / (int)stride + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    const float* b_ptr = nullptr;
    bool has_bias = false;
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must be shape [C]");
        b_ptr = bias.data_ptr<float>();
        has_bias = true;
    }

    auto y = torch::empty({N, C, outH, outW}, x.options());

    // Fast path for typical model: K=3, stride=1, padding=0
    if (K == 3 && stride == 1 && padding == 0) {
        constexpr int WPT = 4;
        dim3 block(32, 8, 1); // 256 threads
        dim3 grid(
            (outW + (block.x * WPT) - 1) / (block.x * WPT),
            (outH + block.y - 1) / block.y,
            N * C
        );
        depthwise_conv2d_k3s1p0_vecwpt<WPT><<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            N, C, H, W,
            outH, outW,
            has_bias
        );
        return y;
    }

    // Fallback generic
    const int threads = 256;
    const int64_t total = (int64_t)N * C * outH * outW;
    const int blocks = (int)((total + threads - 1) / threads);

    depthwise_conv2d_forward_generic<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        N, C, H, W,
        K, (int)stride, (int)padding,
        outH, outW,
        has_bias
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_depthwise2d_asymmetric_input_square_kernel(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_depthwise_conv2d_opt4_vec4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_depthwise2d_asymmetric_input_square_kernel"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

# -------------------------
# Model replacement
# -------------------------
class ModelNew(nn.Module):
    """
    Depthwise Conv2d (groups=in_channels) implemented with a custom CUDA kernel (forward-only).
    Assumes CUDA, contiguous, float32 in forward (casts/contiguous enforced).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        if out_channels != in_channels:
            raise ValueError("Depthwise conv requires out_channels == in_channels for this implementation.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.has_bias = bool(bias)

        w = torch.empty(self.out_channels, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

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

        return self.custom_ops.conv_depthwise2d_asymmetric_input_square_kernel(
            x, w, b, self.stride, self.padding
        )