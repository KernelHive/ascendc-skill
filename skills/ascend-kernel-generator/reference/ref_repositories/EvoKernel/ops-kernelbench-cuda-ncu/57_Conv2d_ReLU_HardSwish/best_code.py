import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Custom CUDA op: conv2d + ReLU + HardSwish (float32, NCHW, groups=1)
# Fast path specialized for: Kh=Kw=3, stride=1, pad=1, Cin=8, dilation=1, groups=1.
# Improvements vs baseline:
#   - Stage INPUT tile (3 x (OW_TILE+2)) into shared memory per (n,oh,ow-tile) to reuse across OC tile.
#   - Keep WEIGHTS staged in shared memory per OC tile (amortize global loads).
#   - Vectorized (float4) cooperative loads for input and weights where aligned/possible.
#   - __launch_bounds__ to help cap registers and improve occupancy.
# Fallback: generic direct conv for other shapes.
# ---------------------------------------------------------------------

conv2d_relu_hardswish_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
__device__ __forceinline__ float relu(float v) {
    return v > 0.0f ? v : 0.0f;
}
__device__ __forceinline__ float hardswish(float v) {
    float t = (v + 3.0f) * (1.0f / 6.0f);
    t = clamp01(t);
    return v * t;
}

// -------------------------
// Generic fallback kernel
// -------------------------
__global__ void conv2d_relu_hardswish_f32_generic(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, Kh, Kw]
    const float* __restrict__ b,      // [Cout] (may be null)
    float* __restrict__ out,          // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int t = idx;
    int ow = t % Wout; t /= Wout;
    int oh = t % Hout; t /= Hout;
    int oc = t % Cout; t /= Cout;
    int n  = t;

    float acc = (b ? LDG(b + oc) : 0.0f);

    int ih0 = oh * stride_h - pad_h;
    int iw0 = ow * stride_w - pad_w;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* x_base = x + ((n * Cin + ic) * Hin * Win);
        const float* w_base = w + ((oc * Cin + ic) * Kh * Kw);
        for (int kh = 0; kh < Kh; ++kh) {
            int ih = ih0 + kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;
            int row = ih * Win;
            #pragma unroll 1
            for (int kw = 0; kw < Kw; ++kw) {
                int iw = iw0 + kw;
                if ((unsigned)iw >= (unsigned)Win) continue;
                float xv = LDG(x_base + row + iw);
                float wv = LDG(w_base + kh * Kw + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    out[idx] = hardswish(relu(acc));
}

// ------------------------------------------------------------
// Fast path: K=3, stride=1, pad=1, Cin=8, Hout=Hin, Wout=Win
// Block computes:
//   - OC_TILE output channels tile (each thread computes 2 OCs)
//   - OW_TILE contiguous output x positions
//   - for one (n, oh)
// Shared memory:
//   - sW: OC_TILE*8*9 floats
//   - sX: 8 * 3 * (OW_TILE+2) floats  (input tile for (oh-1..oh+1, ow_base-1..ow_base+OW_TILE))
// ------------------------------------------------------------
template<int OC_TILE, int OW_TILE>
__global__ __launch_bounds__(128, 4)
void conv2d_relu_hardswish_f32_k3s1p1_cin8_ocx2_smem_w_smem_x(
    const float* __restrict__ x,      // [N, 8, H, W]
    const float* __restrict__ w,      // [Cout, 8, 3, 3]
    const float* __restrict__ b,      // [Cout] (may be null)
    float* __restrict__ out,          // [N, Cout, H, W]
    int N, int H, int W,
    int Cout
) {
    extern __shared__ float smem[];
    float* sW = smem;
    float* sX = sW + (OC_TILE * 8 * 9);

    // indices
    int oc_base = (int)blockIdx.x * OC_TILE;
    int ow_base = (int)blockIdx.y * OW_TILE;
    int ow = ow_base + (int)threadIdx.x; // threadIdx.x in [0, OW_TILE)
    int noz = (int)blockIdx.z;
    int oh = noz % H;
    int n  = noz / H;
    if (n >= N) return;

    // Thread layout: blockDim = (OW_TILE, OC_TILE/2, 1) => OW_TILE*(OC_TILE/2) threads
    // We'll require OW_TILE*(OC_TILE/2) == 128 via chosen params.
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int linear = ty * OW_TILE + tx;
    int nthreads = OW_TILE * (OC_TILE / 2);

    // 1) Load weights tile to shared memory: OC_TILE*8*9 floats
    int w_elems = OC_TILE * 8 * 9;
    // vectorize by 4 when possible: treat as float4
    int w_vec_elems = w_elems / 4;
    for (int i4 = linear; i4 < w_vec_elems; i4 += nthreads) {
        int base = i4 * 4;
        // decode base element for global load address; load 4 consecutive k's in flattened order
        int k = base % 9;
        int t1 = base / 9;
        int ic = t1 % 8;
        int oc_t = t1 / 8;
        int oc = oc_base + oc_t;

        float4 v = make_float4(0,0,0,0);
        if (oc < Cout) {
            // We can only safely vectorize if k <= 5 (4 consecutive within 0..8)
            // Otherwise fall back to scalar path handled later.
            if (k <= 5) {
                const float* g = w + (((oc * 8 + ic) * 3) * 3 + k);
                v = *reinterpret_cast<const float4*>(g);
            } else {
                // will be covered by scalar tail
            }
        }
        // store; if k>5 this stores zeros (tail fixes)
        *reinterpret_cast<float4*>(sW + base) = v;
    }
    // scalar tail and vector-misaligned segments (k=6..8 for each (oc,ic))
    for (int i = linear; i < w_elems; i += nthreads) {
        int k = i % 9;
        if (k <= 5) continue; // already covered by vector path
        int t1 = i / 9;
        int ic = t1 % 8;
        int oc_t = t1 / 8;
        int oc = oc_base + oc_t;
        float val = 0.0f;
        if (oc < Cout) {
            val = LDG(w + (((oc * 8 + ic) * 3) * 3 + k));
        }
        sW[i] = val;
    }

    // 2) Load input tile sX: shape [ic(8), kh(3), ow(OW_TILE+2)]
    // We need input coordinates:
    //   ih = oh - 1 + kh
    //   iw = ow_base - 1 + t, where t in [0..OW_TILE+1]
    // Load cooperatively; vectorize along width where possible.
    int XW = OW_TILE + 2;
    int x_elems = 8 * 3 * XW;
    int x_vec_elems = x_elems / 4;
    const float* x_n = x + (n * 8) * (H * W);

    for (int i4 = linear; i4 < x_vec_elems; i4 += nthreads) {
        int base = i4 * 4;
        int t = base % XW;           // width offset within tile
        int t1 = base / XW;
        int kh = t1 % 3;
        int ic = t1 / 3;

        int ih = oh - 1 + kh;
        int iw = ow_base - 1 + t;

        float4 v = make_float4(0,0,0,0);
        // Only vectorize if within bounds for all 4 and width contiguous
        if ((unsigned)ih < (unsigned)H && (iw >= 0) && (iw + 3 < W) && (t + 3 < XW)) {
            const float* g = x_n + ic * (H * W) + ih * W + iw;
            v = *reinterpret_cast<const float4*>(g);
        } else {
            // scalar tail handles out-of-bounds
        }
        *reinterpret_cast<float4*>(sX + base) = v;
    }
    // scalar fix-up for everything not guaranteed in-bounds
    for (int i = linear; i < x_elems; i += nthreads) {
        int t = i % XW;
        int t1 = i / XW;
        int kh = t1 % 3;
        int ic = t1 / 3;

        int ih = oh - 1 + kh;
        int iw = ow_base - 1 + t;

        float val = 0.0f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            val = LDG(x_n + ic * (H * W) + ih * W + iw);
        }
        sX[i] = val;
    }

    __syncthreads();

    if (ow >= W) return;

    int oc_pair = (int)threadIdx.y; // [0, OC_TILE/2)
    int oc0 = oc_base + oc_pair * 2;
    int oc1 = oc0 + 1;
    if (oc0 >= Cout) return;

    float acc0 = (b ? LDG(b + oc0) : 0.0f);
    float acc1 = (oc1 < Cout) ? (b ? LDG(b + oc1) : 0.0f) : 0.0f;

    int oc0_t = oc0 - oc_base;
    int oc1_t = oc1 - oc_base;

    // sX indexing helper:
    // sX[(ic*3 + kh)*XW + t] where t = (ow - ow_base) + kw
    int t0 = (ow - ow_base); // in [0..OW_TILE-1]

    #pragma unroll
    for (int ic = 0; ic < 8; ++ic) {
        const float* w0 = sW + (oc0_t * 8 + ic) * 9;
        const float* w1 = sW + (oc1_t * 8 + ic) * 9;

        // three rows in tile for this ic
        const float* x0 = sX + ((ic * 3 + 0) * XW + t0);
        const float* x1 = sX + ((ic * 3 + 1) * XW + t0);
        const float* x2 = sX + ((ic * 3 + 2) * XW + t0);

        // Load 3x3 from shared for this output pixel (contiguous in width)
        float a0 = x0[0], a1 = x0[1], a2 = x0[2];
        float b0 = x1[0], b1 = x1[1], b2 = x1[2];
        float c0 = x2[0], c1 = x2[1], c2 = x2[2];

        acc0 = fmaf(a0, w0[0], acc0); acc0 = fmaf(a1, w0[1], acc0); acc0 = fmaf(a2, w0[2], acc0);
        acc0 = fmaf(b0, w0[3], acc0); acc0 = fmaf(b1, w0[4], acc0); acc0 = fmaf(b2, w0[5], acc0);
        acc0 = fmaf(c0, w0[6], acc0); acc0 = fmaf(c1, w0[7], acc0); acc0 = fmaf(c2, w0[8], acc0);

        if (oc1 < Cout) {
            acc1 = fmaf(a0, w1[0], acc1); acc1 = fmaf(a1, w1[1], acc1); acc1 = fmaf(a2, w1[2], acc1);
            acc1 = fmaf(b0, w1[3], acc1); acc1 = fmaf(b1, w1[4], acc1); acc1 = fmaf(b2, w1[5], acc1);
            acc1 = fmaf(c0, w1[6], acc1); acc1 = fmaf(c1, w1[7], acc1); acc1 = fmaf(c2, w1[8], acc1);
        }
    }

    int out0 = ((n * Cout + oc0) * H + oh) * W + ow;
    out[out0] = hardswish(relu(acc0));
    if (oc1 < Cout) out[out0 + (H * W)] = hardswish(relu(acc1));
}

torch::Tensor conv2d_relu_hardswish_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w
) {
    TORCH_CHECK(x.is_cuda(), "conv2d_relu_hardswish_cuda: x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "conv2d_relu_hardswish_cuda: w must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "conv2d_relu_hardswish_cuda: only float32 supported");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "conv2d_relu_hardswish_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "conv2d_relu_hardswish_cuda: x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "conv2d_relu_hardswish_cuda: w must be contiguous");
    TORCH_CHECK(x.dim() == 4, "conv2d_relu_hardswish_cuda: x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "conv2d_relu_hardswish_cuda: w must be OIHW");

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);

    int64_t Cout = w.size(0);
    int64_t wCin = w.size(1);
    int64_t Kh = w.size(2);
    int64_t Kw = w.size(3);
    TORCH_CHECK(wCin == Cin, "conv2d_relu_hardswish_cuda: weight Cin must match input Cin");

    const float* b_ptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        TORCH_CHECK(b.is_cuda(), "conv2d_relu_hardswish_cuda: b must be CUDA if defined");
        TORCH_CHECK(b.scalar_type() == at::kFloat, "conv2d_relu_hardswish_cuda: only float32 bias supported");
        TORCH_CHECK(b.is_contiguous(), "conv2d_relu_hardswish_cuda: b must be contiguous");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "conv2d_relu_hardswish_cuda: b must be [Cout]");
        b_ptr = (const float*)b.data_ptr<float>();
    }

    TORCH_CHECK(stride_h > 0 && stride_w > 0, "conv2d_relu_hardswish_cuda: stride must be > 0");

    int64_t Hout = (Hin + 2 * pad_h - Kh) / stride_h + 1;
    int64_t Wout = (Win + 2 * pad_w - Kw) / stride_w + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "conv2d_relu_hardswish_cuda: invalid output size");

    auto out = torch::empty({N, Cout, Hout, Wout}, x.options());

    bool fast = (Kh == 3 && Kw == 3 &&
                 stride_h == 1 && stride_w == 1 &&
                 pad_h == 1 && pad_w == 1 &&
                 Cin == 8 &&
                 Hout == Hin && Wout == Win);

    if (fast) {
        // Keep same overall tiling shape but add shared-memory input staging.
        // Choose params so block is 128 threads: OW_TILE*(OC_TILE/2) == 128
        constexpr int OC_TILE = 32;
        constexpr int OW_TILE = 8;   // 8 * 16 = 128 threads

        dim3 block(OW_TILE, OC_TILE / 2, 1);
        dim3 grid(
            (unsigned)((Cout + OC_TILE - 1) / OC_TILE),
            (unsigned)((Wout + OW_TILE - 1) / OW_TILE),
            (unsigned)(N * Hout)
        );

        size_t smem_w = (size_t)OC_TILE * 8 * 9 * sizeof(float);
        size_t smem_x = (size_t)8 * 3 * (OW_TILE + 2) * sizeof(float);
        size_t smem = smem_w + smem_x;

        conv2d_relu_hardswish_f32_k3s1p1_cin8_ocx2_smem_w_smem_x<OC_TILE, OW_TILE><<<grid, block, smem>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b_ptr,
            out.data_ptr<float>(),
            (int)N, (int)Hin, (int)Win,
            (int)Cout
        );
    } else {
        int64_t total = N * Cout * Hout * Wout;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        conv2d_relu_hardswish_f32_generic<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b_ptr,
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout, (int)Kh, (int)Kw,
            (int)Hout, (int)Wout,
            (int)stride_h, (int)stride_w,
            (int)pad_h, (int)pad_w
        );
    }

    return out;
}
"""

conv2d_relu_hardswish_cpp_source = r"""
torch::Tensor conv2d_relu_hardswish_cuda(
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
    name="custom_ops_lib_conv2d_relu_hardswish_v6",
    cpp_sources=conv2d_relu_hardswish_cpp_source,
    cuda_sources=conv2d_relu_hardswish_cuda_source,
    functions=["conv2d_relu_hardswish_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Convolution + ReLU + HardSwish using a fused custom CUDA kernel.
    Assumptions: NCHW, float32, CUDA, contiguous. groups=1, dilation=1.
    Includes a fast path for 3x3 stride=1 pad=1 with Cin=8.
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
        if not w.is_cuda:
            w = w.cuda()
        if not w.is_contiguous():
            w = w.contiguous()

        if b is None:
            b_arg = torch.tensor([], device=w.device, dtype=w.dtype)
        else:
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_cuda:
                b = b.cuda()
            if not b.is_contiguous():
                b = b.contiguous()
            b_arg = b

        sh, sw = self.conv.stride
        ph, pw = self.conv.padding
        dh, dw = self.conv.dilation
        groups = self.conv.groups
        if dh != 1 or dw != 1:
            raise RuntimeError("ModelNew fused op supports dilation=1 only.")
        if groups != 1:
            raise RuntimeError("ModelNew fused op supports groups=1 only.")

        return self.custom_ops_lib.conv2d_relu_hardswish_cuda(
            x, w, b_arg, int(sh), int(sw), int(ph), int(pw)
        )