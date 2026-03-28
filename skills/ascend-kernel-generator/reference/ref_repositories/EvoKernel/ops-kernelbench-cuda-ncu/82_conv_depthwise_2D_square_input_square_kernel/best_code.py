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

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ float ro_load_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float2 ro_load_f32x2(const float* p) {
    // Caller must ensure 8B alignment
    return *reinterpret_cast<const float2*>(p);
}
static __device__ __forceinline__ void store_f32x2(float* p, const float2& v) {
    // Caller must ensure 8B alignment
    *reinterpret_cast<float2*>(p) = v;
}

__global__ void depthwise_conv2d_forward_generic(
    const float* __restrict__ x,      // [N, C, H, W]
    const float* __restrict__ w,      // [C, 1, kH, kW]
    const float* __restrict__ b,      // [C] or nullptr
    float* __restrict__ y,            // [N, C, outH, outW]
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int stride, int padding,
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

    float acc = has_bias ? ro_load_f32(b + c) : 0.0f;
    int w_base = c * kH * kW;

    for (int kh = 0; kh < kH; ++kh) {
        int ih = in_h0 + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        int x_row = ((n * C + c) * H + ih) * W;
        int w_row = w_base + kh * kW;
        #pragma unroll 1
        for (int kw = 0; kw < kW; ++kw) {
            int iw = in_w0 + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            acc = fmaf(ro_load_f32(x + x_row + iw), ro_load_f32(w + w_row + kw), acc);
        }
    }

    int y_off = ((n * C + c) * outH + oh) * outW + ow;
    y[y_off] = acc;
}

// Fast path: K=3, stride=1, padding=0
// Row-streaming kernel with safe float2 vectorization and per-thread multi-output.
template<int VEC2_PER_THREAD, bool HAS_BIAS>
__global__ __launch_bounds__(256, 2)
void dwconv2d_k3s1p0_rowstream_vec2(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ w,   // [C,1,3,3]
    const float* __restrict__ b,   // [C] or nullptr
    float* __restrict__ y,         // [N,C,outH,outW]
    int N, int C, int H, int W,
    int outH, int outW,
    int total_rows
) {
    constexpr int VEC_ELEMS = 2;
    constexpr int OUTS_PER_THREAD = VEC2_PER_THREAD * VEC_ELEMS; // outputs per thread

    int tid = (int)threadIdx.x;

    int ow_block0 = (int)blockIdx.x * ((int)blockDim.x * OUTS_PER_THREAD);
    int ow0 = ow_block0 + tid * OUTS_PER_THREAD;
    if (ow0 >= outW) return;

    // Iterate rows in a grid-stride manner over blockIdx.y
    for (int row = (int)blockIdx.y; row < total_rows; row += (int)gridDim.y) {
        int t = row;
        int oh = t % outH; t /= outH;
        int c  = t % C;    t /= C;
        int n  = t;

        // weights in regs
        const float* wptr = w + (c * 3) * 3;
        const float w00 = ro_load_f32(wptr + 0); const float w01 = ro_load_f32(wptr + 1); const float w02 = ro_load_f32(wptr + 2);
        const float w10 = ro_load_f32(wptr + 3); const float w11 = ro_load_f32(wptr + 4); const float w12 = ro_load_f32(wptr + 5);
        const float w20 = ro_load_f32(wptr + 6); const float w21 = ro_load_f32(wptr + 7); const float w22 = ro_load_f32(wptr + 8);
        const float bv = HAS_BIAS ? ro_load_f32(b + c) : 0.0f;

        int x_base_nc = (n * C + c) * H * W;
        const float* __restrict__ x0 = x + x_base_nc + (oh + 0) * W + ow0;
        const float* __restrict__ x1 = x + x_base_nc + (oh + 1) * W + ow0;
        const float* __restrict__ x2 = x + x_base_nc + (oh + 2) * W + ow0;

        float* __restrict__ yrow = y + ((n * C + c) * outH + oh) * outW + ow0;

        // We need:
        // - store OUTS_PER_THREAD outputs (to outW)
        // - load up to +2 columns for the 3x3 window (to W)
        // For vector loads, we will use float2 at addresses (x? + o), (x? + o + 1), (x? + o + 2),
        // which require each address to be 8-byte aligned. Alignment depends on (ow0 + o + shift) being even.
        const bool inbounds = (ow0 + OUTS_PER_THREAD) <= outW && (ow0 + OUTS_PER_THREAD + 2) <= W;

        // Alignment checks for float2 on shifted pointers:
        // require (ow0 + o + shift) even for shift in {0,1,2}.
        // It's sufficient to ensure ow0 is even (=> shift parity decides), but for shift=1 we'd need odd.
        // So we only vectorize across shift=0 and shift=2 with float2 loads at aligned addresses,
        // and handle shift=1 via scalar loads. This keeps vectorization safe and still beneficial.
        const bool ow0_even = ((ow0 & 1) == 0);

        // Accumulators: each holds 2 outputs
        float2 acc[VEC2_PER_THREAD];
        #pragma unroll
        for (int v = 0; v < VEC2_PER_THREAD; ++v) {
            acc[v].x = bv;
            acc[v].y = bv;
        }

        if (inbounds && ow0_even) {
            #pragma unroll
            for (int v = 0; v < VEC2_PER_THREAD; ++v) {
                int o = v * VEC_ELEMS; // even
                // For outputs at columns (ow0+o) and (ow0+o+1)
                // Need inputs at (o + 0,1,2,3) for each row, but window uses +0..+2 per output:
                // output col (o): needs input cols (o,o+1,o+2)
                // output col (o+1): needs input cols (o+1,o+2,o+3)
                //
                // We'll vector-load:
                // - x? at (o) -> provides (o,o+1)
                // - x? at (o+2) -> provides (o+2,o+3)
                // And scalar-load x? at (o+1) and (o+3?)? Actually we can derive:
                // For output (o): need x(o), x(o+1), x(o+2)
                // For output (o+1): need x(o+1), x(o+2), x(o+3)
                // We already have:
                //   v0 = load2(o): [x(o), x(o+1)]
                //   v2 = load2(o+2): [x(o+2), x(o+3)]
                // Scalar x(o+1) comes from v0.y, scalar x(o+2) from v2.x, scalar x(o+3) from v2.y.
                // So no unaligned shift=1 vector load is needed.

                const float2 a0 = ro_load_f32x2(x0 + o);      // [o, o+1]
                const float2 a2 = ro_load_f32x2(x0 + o + 2);  // [o+2, o+3]
                const float2 b0 = ro_load_f32x2(x1 + o);
                const float2 b2 = ro_load_f32x2(x1 + o + 2);
                const float2 c0 = ro_load_f32x2(x2 + o);
                const float2 c2 = ro_load_f32x2(x2 + o + 2);

                float r0 = acc[v].x;
                float r1 = acc[v].y;

                // output at (o)
                r0 = fmaf(a0.x, w00, r0); r0 = fmaf(a0.y, w01, r0); r0 = fmaf(a2.x, w02, r0);
                r0 = fmaf(b0.x, w10, r0); r0 = fmaf(b0.y, w11, r0); r0 = fmaf(b2.x, w12, r0);
                r0 = fmaf(c0.x, w20, r0); r0 = fmaf(c0.y, w21, r0); r0 = fmaf(c2.x, w22, r0);

                // output at (o+1)
                r1 = fmaf(a0.y, w00, r1); r1 = fmaf(a2.x, w01, r1); r1 = fmaf(a2.y, w02, r1);
                r1 = fmaf(b0.y, w10, r1); r1 = fmaf(b2.x, w11, r1); r1 = fmaf(b2.y, w12, r1);
                r1 = fmaf(c0.y, w20, r1); r1 = fmaf(c2.x, w21, r1); r1 = fmaf(c2.y, w22, r1);

                acc[v].x = r0;
                acc[v].y = r1;
            }

            #pragma unroll
            for (int v = 0; v < VEC2_PER_THREAD; ++v) {
                store_f32x2(yrow + v * VEC_ELEMS, acc[v]);
            }
        } else {
            // Scalar safe path (handles odd ow0, edges)
            #pragma unroll
            for (int i = 0; i < OUTS_PER_THREAD; ++i) {
                int ow = ow0 + i;
                if (ow >= outW) break;

                float r = bv;
                const float x00 = ro_load_f32(x0 + i + 0);
                const float x01 = ro_load_f32(x0 + i + 1);
                const float x02 = ro_load_f32(x0 + i + 2);
                const float x10 = ro_load_f32(x1 + i + 0);
                const float x11 = ro_load_f32(x1 + i + 1);
                const float x12 = ro_load_f32(x1 + i + 2);
                const float x20 = ro_load_f32(x2 + i + 0);
                const float x21 = ro_load_f32(x2 + i + 1);
                const float x22 = ro_load_f32(x2 + i + 2);

                r = fmaf(x00, w00, r); r = fmaf(x01, w01, r); r = fmaf(x02, w02, r);
                r = fmaf(x10, w10, r); r = fmaf(x11, w11, r); r = fmaf(x12, w12, r);
                r = fmaf(x20, w20, r); r = fmaf(x21, w21, r); r = fmaf(x22, w22, r);

                yrow[i] = r;
            }
        }
    }
}

torch::Tensor conv_depthwise2d_square_input_square_kernel(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C,1,kH,kW]");
    TORCH_CHECK(weight.size(1) == 1, "weight second dim must be 1 for depthwise");
    TORCH_CHECK(x.size(1) == weight.size(0), "channels mismatch");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int kH = (int)weight.size(2);
    int kW = (int)weight.size(3);

    int outH = (H + 2 * (int)padding - kH) / (int)stride + 1;
    int outW = (W + 2 * (int)padding - kW) / (int)stride + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    const float* b_ptr = nullptr;
    bool has_bias = false;
    torch::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must be [C]");
        b_ptr = bias.data_ptr<float>();
        has_bias = true;
    }

    auto y = torch::empty({N, C, outH, outW}, x.options());

    if (kH == 3 && kW == 3 && stride == 1 && padding == 0) {
        const int total_rows = N * C * outH;

        constexpr int threads = 256;
        constexpr int VEC2_PER_THREAD = 2; // 4 outputs/thread
        constexpr int OUTS_PER_THREAD = VEC2_PER_THREAD * 2;
        const int ow_per_block = threads * OUTS_PER_THREAD;

        const int grid_x = (outW + ow_per_block - 1) / ow_per_block;

        int grid_y = 4096;
        if (grid_y > total_rows) grid_y = total_rows;
        if (grid_y < 1) grid_y = 1;

        dim3 block(threads, 1, 1);
        dim3 grid(grid_x, grid_y, 1);

        if (has_bias) {
            dwconv2d_k3s1p0_rowstream_vec2<VEC2_PER_THREAD, true><<<grid, block>>>(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                b_ptr,
                y.data_ptr<float>(),
                N, C, H, W,
                outH, outW,
                total_rows
            );
        } else {
            dwconv2d_k3s1p0_rowstream_vec2<VEC2_PER_THREAD, false><<<grid, block>>>(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                nullptr,
                y.data_ptr<float>(),
                N, C, H, W,
                outH, outW,
                total_rows
            );
        }
        return y;
    }

    int threads = 256;
    int64_t total = (int64_t)N * C * outH * outW;
    int blocks = (int)((total + threads - 1) / threads);

    depthwise_conv2d_forward_generic<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kH, kW,
        (int)stride, (int)padding,
        has_bias
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_depthwise2d_square_input_square_kernel(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_depthwise_conv2d_square_opt_v3_vec2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_depthwise2d_square_input_square_kernel"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Depthwise Conv2d replacement (forward-only) using a custom CUDA kernel.
    Matches: nn.Conv2d(in_channels, in_channels, k, stride=stride, padding=padding, groups=in_channels, bias=bias)
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        w = torch.empty(self.in_channels, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32))
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

        b_opt = None
        if self.bias is not None:
            b = self.bias
            if not b.is_cuda:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()
            b_opt = b

        return self.custom_ops.conv_depthwise2d_square_input_square_kernel(
            x,
            w,
            b_opt,
            self.stride,
            self.padding,
        )