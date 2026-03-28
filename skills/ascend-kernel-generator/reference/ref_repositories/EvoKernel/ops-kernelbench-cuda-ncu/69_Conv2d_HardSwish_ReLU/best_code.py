import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Fused op: conv2d + hardswish + relu
# - Specialized fast path for:
#     x: float32 NCHW contiguous CUDA
#     w: float32 OIHW contiguous CUDA
#     Cin=8, Cout=64, Kh=Kw=3, stride=1, pad=0
#
# Improvements over baseline:
#   * OWx2 per thread (2 adjacent outputs) to reduce register pressure vs OWx4
#   * vectorized input loads via float2 where safe (improves ILP, reduces load inst count)
#   * optional constant-memory bias only (tiny, low overhead) with pointer-cached update
#   * tuned block size (128 threads) + launch bounds to encourage lower registers/thread
#
# Generic fallback remains a scalar direct conv for correctness.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float hardswish(float x) {
    float t = clampf(x + 3.0f, 0.0f, 6.0f);
    return x * (t * (1.0f / 6.0f));
}

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float2 ldg_f32x2(const float* p) {
#if __CUDA_ARCH__ >= 350
    float2 v;
    v.x = __ldg(p + 0);
    v.y = __ldg(p + 1);
    return v;
#else
    return *reinterpret_cast<const float2*>(p);
#endif
}

// Small constant bias cache (64 floats = 256B). No constant weights to avoid update overhead.
__device__ __constant__ float kBias64[64];

// ------------------------------------
// Generic kernel (scalar, any Kh/Kw/stride/pad)
// ------------------------------------
__global__ __launch_bounds__(256, 2)
void conv2d_hardswish_relu_generic_fwd_f32(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, Kh, Kw]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ out,          // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int Hout, int Wout
) {
    int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % Wout; tmp /= Wout;
    int oh = tmp % Hout; tmp /= Hout;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    float acc = (b != nullptr) ? ldg_f32(b + oc) : 0.0f;

    int ih0 = oh * stride_h - pad_h;
    int iw0 = ow * stride_w - pad_w;

    const float* __restrict__ w_oc = w + (oc * Cin * Kh * Kw);

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        const float* __restrict__ x_ic = x + ((n * Cin + ic) * Hin * Win);
        const float* __restrict__ w_ic = w_oc + ic * (Kh * Kw);

        #pragma unroll 1
        for (int kh = 0; kh < Kh; ++kh) {
            int ih = ih0 + kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;
            const float* __restrict__ x_row = x_ic + ih * Win;
            const float* __restrict__ w_row = w_ic + kh * Kw;

            #pragma unroll 1
            for (int kw = 0; kw < Kw; ++kw) {
                int iw = iw0 + kw;
                if ((unsigned)iw >= (unsigned)Win) continue;
                acc = fmaf(ldg_f32(x_row + iw), ldg_f32(w_row + kw), acc);
            }
        }
    }

    float y = hardswish(acc);
    y = (y > 0.0f) ? y : 0.0f;
    out[idx] = y;
}

// ------------------------------------
// Fast path kernel for:
//   Cin=8, Cout=64, Kh=Kw=3, stride=1, pad=0
// Each thread computes 2 adjacent W outputs (OWx2) for fixed (n, oc, oh).
// Grid: x over (n, oc, oh) rows; y tiles over Wout in chunks of (blockDim.x*2).
// ------------------------------------
__global__ __launch_bounds__(128, 3)
void conv2d_hardswish_relu_k3s1p0_cin8_cout64_ow2_fwd_f32(
    const float* __restrict__ x,   // [N, 8, Hin, Win]
    const float* __restrict__ w,   // [64, 8, 3, 3]
    const float* __restrict__ b,   // [64] or nullptr (if null, try kBias64 when use_const_bias=1)
    float* __restrict__ out,       // [N, 64, Hout, Wout]
    int N, int Hin, int Win,
    int Hout, int Wout,
    int use_const_bias
) {
    int row = (int)blockIdx.x;
    int oh = row % Hout; row /= Hout;
    int oc = row % 64;   row /= 64;
    int n  = row;

    int tid = (int)threadIdx.x;
    int ow0 = (int)blockIdx.y * (int)(blockDim.x * 2) + tid * 2;
    if (ow0 >= Wout) return;

    float base = 0.0f;
    if (b != nullptr) base = ldg_f32(b + oc);
    else if (use_const_bias) base = kBias64[oc];

    float acc0 = base;
    float acc1 = base;

    int ih0 = oh;
    int iw0 = ow0;

    const float* __restrict__ w_oc = w + (oc * 8 * 9);
    float* __restrict__ out_row = out + ((n * 64 + oc) * Hout + oh) * Wout + ow0;

    // Main path: only do vectorized loads when both outputs fully in-bounds and we can read up to +3.
    // For two outputs at ow0 and ow0+1 with 3x3 kernel, we need to read x at offsets [0..(1+2)] = [0..3]
    // plus for fused reuse we also read pairs; so require iw0 + 3 < Win and ow0 + 1 < Wout.
    bool vec_ok = (ow0 + 1 < Wout) && (iw0 + 3 < Win);

    #pragma unroll
    for (int ic = 0; ic < 8; ++ic) {
        const float* __restrict__ x_ic = x + ((n * 8 + ic) * Hin + ih0) * Win + iw0;
        const float* __restrict__ w_ic = w_oc + ic * 9;

        float w00 = ldg_f32(w_ic + 0), w01 = ldg_f32(w_ic + 1), w02 = ldg_f32(w_ic + 2);
        float w10 = ldg_f32(w_ic + 3), w11 = ldg_f32(w_ic + 4), w12 = ldg_f32(w_ic + 5);
        float w20 = ldg_f32(w_ic + 6), w21 = ldg_f32(w_ic + 7), w22 = ldg_f32(w_ic + 8);

        if (vec_ok) {
            // kh=0 row
            const float* r0 = x_ic;
            float2 x01 = ldg_f32x2(r0 + 0); // [x0,x1]
            float2 x23 = ldg_f32x2(r0 + 2); // [x2,x3]
            float x2 = x23.x;

            // out0 uses x0,x1,x2 ; out1 uses x1,x2,x3
            acc0 = fmaf(x01.x, w00, acc0);
            acc0 = fmaf(x01.y, w01, acc0);
            acc0 = fmaf(x2,    w02, acc0);

            acc1 = fmaf(x01.y, w00, acc1);
            acc1 = fmaf(x2,    w01, acc1);
            acc1 = fmaf(x23.y, w02, acc1);

            // kh=1 row
            const float* r1 = r0 + Win;
            float2 y01 = ldg_f32x2(r1 + 0);
            float2 y23 = ldg_f32x2(r1 + 2);
            float y2 = y23.x;

            acc0 = fmaf(y01.x, w10, acc0);
            acc0 = fmaf(y01.y, w11, acc0);
            acc0 = fmaf(y2,    w12, acc0);

            acc1 = fmaf(y01.y, w10, acc1);
            acc1 = fmaf(y2,    w11, acc1);
            acc1 = fmaf(y23.y, w12, acc1);

            // kh=2 row
            const float* r2 = r1 + Win;
            float2 z01 = ldg_f32x2(r2 + 0);
            float2 z23 = ldg_f32x2(r2 + 2);
            float z2 = z23.x;

            acc0 = fmaf(z01.x, w20, acc0);
            acc0 = fmaf(z01.y, w21, acc0);
            acc0 = fmaf(z2,    w22, acc0);

            acc1 = fmaf(z01.y, w20, acc1);
            acc1 = fmaf(z2,    w21, acc1);
            acc1 = fmaf(z23.y, w22, acc1);
        } else {
            // Scalar guarded path (near right edge)
            // For each output t in {0,1} if in bounds
            int max_t = (ow0 + 1 < Wout) ? 2 : 1;
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                if (t >= max_t) break;
                float a = (t == 0) ? acc0 : acc1;
                const float* r0 = x_ic + t;
                const float* r1 = r0 + Win;
                const float* r2 = r1 + Win;

                // We are in pad=0, stride=1, and Wout = Win-2, so r*+2 should be valid for ow<Wout,
                // but keep minimal guarding on Win for safety.
                float x0 = (iw0 + t + 0 < Win) ? ldg_f32(r0 + 0) : 0.0f;
                float x1 = (iw0 + t + 1 < Win) ? ldg_f32(r0 + 1) : 0.0f;
                float x2 = (iw0 + t + 2 < Win) ? ldg_f32(r0 + 2) : 0.0f;

                float y0 = (iw0 + t + 0 < Win) ? ldg_f32(r1 + 0) : 0.0f;
                float y1 = (iw0 + t + 1 < Win) ? ldg_f32(r1 + 1) : 0.0f;
                float y2 = (iw0 + t + 2 < Win) ? ldg_f32(r1 + 2) : 0.0f;

                float z0 = (iw0 + t + 0 < Win) ? ldg_f32(r2 + 0) : 0.0f;
                float z1 = (iw0 + t + 1 < Win) ? ldg_f32(r2 + 1) : 0.0f;
                float z2 = (iw0 + t + 2 < Win) ? ldg_f32(r2 + 2) : 0.0f;

                a = fmaf(x0, w00, a); a = fmaf(x1, w01, a); a = fmaf(x2, w02, a);
                a = fmaf(y0, w10, a); a = fmaf(y1, w11, a); a = fmaf(y2, w12, a);
                a = fmaf(z0, w20, a); a = fmaf(z1, w21, a); a = fmaf(z2, w22, a);

                if (t == 0) acc0 = a; else acc1 = a;
            }
        }
    }

    // Activations + store (only store valid outputs)
    float y0 = hardswish(acc0); y0 = (y0 > 0.0f) ? y0 : 0.0f;
    out_row[0] = y0;

    if (ow0 + 1 < Wout) {
        float y1 = hardswish(acc1); y1 = (y1 > 0.0f) ? y1 : 0.0f;
        out_row[1] = y1;
    }
}

// -------------------------
// Host entry + optional const-bias updater (cached by storage pointer)
// -------------------------
static uintptr_t g_last_b_ptr = 0;

static void maybe_update_const_bias(torch::Tensor b, bool has_bias) {
    if (!has_bias) return;
    uintptr_t bptr = (uintptr_t)b.data_ptr<float>();
    if (bptr != g_last_b_ptr) {
        // bias is tiny; D2D memcpy of 256B. Still avoid doing it every call.
        cudaMemcpyToSymbol(kBias64, b.data_ptr<float>(), 64 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        g_last_b_ptr = bptr;
    }
}

torch::Tensor conv2d_hard_swish_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w
) {
    TORCH_CHECK(x.is_cuda(), "conv2d_hard_swish_relu_cuda: x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "conv2d_hard_swish_relu_cuda: w must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "conv2d_hard_swish_relu_cuda: x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "conv2d_hard_swish_relu_cuda: w must be float32");
    TORCH_CHECK(x.is_contiguous(), "conv2d_hard_swish_relu_cuda: x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "conv2d_hard_swish_relu_cuda: w must be contiguous");
    TORCH_CHECK(x.dim() == 4, "conv2d_hard_swish_relu_cuda: x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "conv2d_hard_swish_relu_cuda: w must be OIHW (4D)");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);
    int wCin = (int)w.size(1);
    int Kh = (int)w.size(2);
    int Kw = (int)w.size(3);
    TORCH_CHECK(wCin == Cin, "conv2d_hard_swish_relu_cuda: weight Cin must match x Cin");

    bool has_bias = b.defined() && (b.numel() > 0);
    const float* b_ptr = nullptr;
    if (has_bias) {
        TORCH_CHECK(b.is_cuda(), "conv2d_hard_swish_relu_cuda: bias must be CUDA if provided");
        TORCH_CHECK(b.scalar_type() == at::kFloat, "conv2d_hard_swish_relu_cuda: bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "conv2d_hard_swish_relu_cuda: bias must be contiguous");
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == Cout, "conv2d_hard_swish_relu_cuda: bias must be [Cout]");
        b_ptr = (const float*)b.data_ptr<float>();
    }

    int Hout = (Hin + 2 * (int)pad_h - Kh) / (int)stride_h + 1;
    int Wout = (Win + 2 * (int)pad_w - Kw) / (int)stride_w + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "conv2d_hard_swish_relu_cuda: output dims must be positive");

    auto out = torch::empty({N, Cout, Hout, Wout}, x.options());

    // Fast path specialized to the prompt's typical case
    if (Cin == 8 && Cout == 64 && Kh == 3 && Kw == 3 &&
        stride_h == 1 && stride_w == 1 &&
        pad_h == 0 && pad_w == 0) {

        // Optionally populate constant bias cache (small and pointer-cached).
        // Kernel can still read bias from global b_ptr if present; we pass both and choose inside.
        // For best latency, we prefer constant bias and pass b_ptr as nullptr to remove a global read.
        int use_const_bias = 0;
        const float* kernel_b_ptr = nullptr;
        if (has_bias) {
            maybe_update_const_bias(b, true);
            use_const_bias = 1;
            kernel_b_ptr = nullptr;
        } else {
            use_const_bias = 0;
            kernel_b_ptr = nullptr;
        }

        int rows = N * 64 * Hout;

        int threads = 128;
        int chunk = threads * 2;
        int wy = (Wout + chunk - 1) / chunk;

        dim3 block(threads, 1, 1);
        dim3 grid(rows, wy, 1);

        conv2d_hardswish_relu_k3s1p0_cin8_cout64_ow2_fwd_f32<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            kernel_b_ptr,
            (float*)out.data_ptr<float>(),
            N, Hin, Win,
            Hout, Wout,
            use_const_bias
        );
        return out;
    }

    // Fallback: generic 1D kernel
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Hout * (int64_t)Wout;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    conv2d_hardswish_relu_generic_fwd_f32<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        b_ptr,
        (float*)out.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, Kh, Kw,
        (int)stride_h, (int)stride_w,
        (int)pad_h, (int)pad_w,
        Hout, Wout
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor conv2d_hard_swish_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_hardswish_relu_v5_ow2_vec2_biasconst",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_hard_swish_relu_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU
    using a fused custom CUDA kernel with an improved specialized fast path.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input.")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.conv.weight
        b = self.conv.bias

        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        if b is None:
            b_arg = torch.empty((0,), device=x.device, dtype=torch.float32)
        else:
            b_arg = b
            if b_arg.dtype != torch.float32:
                b_arg = b_arg.float()
            if not b_arg.is_contiguous():
                b_arg = b_arg.contiguous()

        stride_h, stride_w = self.conv.stride
        pad_h, pad_w = self.conv.padding

        return self.custom_ops_lib.conv2d_hard_swish_relu_cuda(
            x, w, b_arg, int(stride_h), int(stride_w), int(pad_h), int(pad_w)
        )