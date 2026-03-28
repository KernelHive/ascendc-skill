import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard2d_asymmetric_input_square_kernel ---------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

// ---------------- Generic baseline (correctness for any square K) ----------------
__global__ void conv2d_forward_nchw_f32_square_generic(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, K, K]
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int K,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int Hout, int Wout
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % Wout; tmp /= Wout;
    int oh = tmp % Hout; tmp /= Hout;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    float acc = 0.0f;

    int in_h0 = oh * stride_h - pad_h;
    int in_w0 = ow * stride_w - pad_w;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* w_oc_ic = w + ((oc * Cin + ic) * K * K);
        const float* x_n_ic  = x + ((n * Cin + ic) * Hin * Win);

        #pragma unroll 1
        for (int kh = 0; kh < K; ++kh) {
            int ih = in_h0 + kh * dil_h;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            #pragma unroll 1
            for (int kw = 0; kw < K; ++kw) {
                int iw = in_w0 + kw * dil_w;
                if ((unsigned)iw >= (unsigned)Win) continue;

                float xv = ldg_f32(x_n_ic + ih * Win + iw);
                float wv = ldg_f32(w_oc_ic + kh * K + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = acc;
}

// ---------------- Previous fast path: K=3,S=1,D=1 (global weight loads), OCx4 ----------------
__global__ __launch_bounds__(256, 2)
void conv2d_forward_nchw_f32_k3_s1d1_oc4_gs(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, 3, 3]
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout
) {
    int ocv = (int)blockIdx.y;     // output channel vector index
    int n   = (int)blockIdx.z;     // batch
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= Cout) return;

    int tid = (int)threadIdx.x;
    int total_pix = Hout * Wout;
    int base = (int)blockIdx.x * blockDim.x + tid;
    int stride = (int)(gridDim.x * blockDim.x);

    bool has1 = (oc0 + 1) < Cout;
    bool has2 = (oc0 + 2) < Cout;
    bool has3 = (oc0 + 3) < Cout;

    for (int p = base; p < total_pix; p += stride) {
        int ow = p % Wout;
        int oh = p / Wout;

        int ih0 = oh - pad_h;
        int iw0 = ow - pad_w;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            const float* x_n_ic = x + ((n * Cin + ic) * Hin * Win);

            const float* w0 = w + ((oc0 * Cin + ic) * 9);
            const float* w1 = has1 ? (w + (((oc0 + 1) * Cin + ic) * 9)) : nullptr;
            const float* w2 = has2 ? (w + (((oc0 + 2) * Cin + ic) * 9)) : nullptr;
            const float* w3 = has3 ? (w + (((oc0 + 3) * Cin + ic) * 9)) : nullptr;

            // Row 0
            int ih = ih0;
            if ((unsigned)ih < (unsigned)Hin) {
                int iw = iw0;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 0), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 0), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 0), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 0), acc3);
                }
                iw = iw0 + 1;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 1), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 1), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 1), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 1), acc3);
                }
                iw = iw0 + 2;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 2), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 2), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 2), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 2), acc3);
                }
            }

            // Row 1
            ih = ih0 + 1;
            if ((unsigned)ih < (unsigned)Hin) {
                int iw = iw0;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 3), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 3), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 3), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 3), acc3);
                }
                iw = iw0 + 1;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 4), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 4), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 4), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 4), acc3);
                }
                iw = iw0 + 2;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 5), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 5), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 5), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 5), acc3);
                }
            }

            // Row 2
            ih = ih0 + 2;
            if ((unsigned)ih < (unsigned)Hin) {
                int iw = iw0;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 6), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 6), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 6), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 6), acc3);
                }
                iw = iw0 + 1;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 7), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 7), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 7), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 7), acc3);
                }
                iw = iw0 + 2;
                if ((unsigned)iw < (unsigned)Win) {
                    float xv = ldg_f32(x_n_ic + ih * Win + iw);
                    acc0 = fmaf(xv, ldg_f32(w0 + 8), acc0);
                    if (has1) acc1 = fmaf(xv, ldg_f32(w1 + 8), acc1);
                    if (has2) acc2 = fmaf(xv, ldg_f32(w2 + 8), acc2);
                    if (has3) acc3 = fmaf(xv, ldg_f32(w3 + 8), acc3);
                }
            }
        }

        int out_base = ((n * Cout + oc0) * Hout + oh) * Wout + ow;
        y[out_base] = acc0;
        if (has1) y[out_base + (Hout * Wout)] = acc1;
        if (has2) y[out_base + 2 * (Hout * Wout)] = acc2;
        if (has3) y[out_base + 3 * (Hout * Wout)] = acc3;
    }
}

// ---------------- New fast path: K=3,S=1,D=1, interior without bounds checks, weights in shared ----------------
// Layout: for each (n, oc4) block (grid.y, grid.z), stage weights [Cin][9] as float4 in shared mem once.
// Then threads grid-stride over interior pixels (oh,ow) where full 3x3 is in-bounds.
__global__ __launch_bounds__(128, 4)
void conv2d_forward_nchw_f32_k3_s1d1_oc4_interior_smemw(
    const float* __restrict__ x,   // [N,Cin,Hin,Win]
    const float* __restrict__ w,   // [Cout,Cin,3,3]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int oh0, int oh1,              // interior oh range [oh0, oh1)
    int ow0, int ow1               // interior ow range [ow0, ow1)
) {
    int ocv = (int)blockIdx.y;
    int n = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= Cout) return;

    bool has1 = (oc0 + 1) < Cout;
    bool has2 = (oc0 + 2) < Cout;
    bool has3 = (oc0 + 3) < Cout;

    // Shared weights: Cin * 9 float4
    extern __shared__ float4 sw4[];
    int tid = (int)threadIdx.x;

    // Stage weights: sw4[(ic*9 + t)] = (w[oc0+0], w[oc0+1], w[oc0+2], w[oc0+3]) for that (ic,t)
    // For missing channels, store 0.
    int total_w4 = Cin * 9;
    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / 9;
        int t  = i - ic * 9;

        float a = ldg_f32(w + ((oc0 + 0) * Cin + ic) * 9 + t);
        float b = has1 ? ldg_f32(w + ((oc0 + 1) * Cin + ic) * 9 + t) : 0.f;
        float c = has2 ? ldg_f32(w + ((oc0 + 2) * Cin + ic) * 9 + t) : 0.f;
        float d = has3 ? ldg_f32(w + ((oc0 + 3) * Cin + ic) * 9 + t) : 0.f;
        sw4[i] = make_float4(a, b, c, d);
    }
    __syncthreads();

    int ih_off = -pad_h;
    int iw_off = -pad_w;

    int intH = oh1 - oh0;
    int intW = ow1 - ow0;
    int interior_pix = intH * intW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    // Precompute base pointers
    int64_t x_n_base = (int64_t)n * Cin * (int64_t)Hin * Win;
    int64_t y_n_base = (int64_t)n * Cout * (int64_t)Hout * Wout;

    for (int p = base; p < interior_pix; p += step) {
        int local_ow = p % intW;
        int local_oh = p / intW;

        int oh = oh0 + local_oh;
        int ow = ow0 + local_ow;

        int ih0 = oh + ih_off;
        int iw0 = ow + iw_off;

        // Now guaranteed in-bounds for 3x3: ih0..ih0+2, iw0..iw0+2
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * Hin * (int64_t)Win;

            int64_t r0 = x_c_base + (int64_t)(ih0 + 0) * Win + (iw0 + 0);
            int64_t r1 = x_c_base + (int64_t)(ih0 + 1) * Win + (iw0 + 0);
            int64_t r2 = x_c_base + (int64_t)(ih0 + 2) * Win + (iw0 + 0);

            float x00 = ldg_f32(x + r0 + 0);
            float x01 = ldg_f32(x + r0 + 1);
            float x02 = ldg_f32(x + r0 + 2);

            float x10 = ldg_f32(x + r1 + 0);
            float x11 = ldg_f32(x + r1 + 1);
            float x12 = ldg_f32(x + r1 + 2);

            float x20 = ldg_f32(x + r2 + 0);
            float x21 = ldg_f32(x + r2 + 1);
            float x22 = ldg_f32(x + r2 + 2);

            const float4* w4 = sw4 + ic * 9;

            float4 v;
            v = w4[0]; acc0 = fmaf(x00, v.x, acc0); acc1 = fmaf(x00, v.y, acc1); acc2 = fmaf(x00, v.z, acc2); acc3 = fmaf(x00, v.w, acc3);
            v = w4[1]; acc0 = fmaf(x01, v.x, acc0); acc1 = fmaf(x01, v.y, acc1); acc2 = fmaf(x01, v.z, acc2); acc3 = fmaf(x01, v.w, acc3);
            v = w4[2]; acc0 = fmaf(x02, v.x, acc0); acc1 = fmaf(x02, v.y, acc1); acc2 = fmaf(x02, v.z, acc2); acc3 = fmaf(x02, v.w, acc3);

            v = w4[3]; acc0 = fmaf(x10, v.x, acc0); acc1 = fmaf(x10, v.y, acc1); acc2 = fmaf(x10, v.z, acc2); acc3 = fmaf(x10, v.w, acc3);
            v = w4[4]; acc0 = fmaf(x11, v.x, acc0); acc1 = fmaf(x11, v.y, acc1); acc2 = fmaf(x11, v.z, acc2); acc3 = fmaf(x11, v.w, acc3);
            v = w4[5]; acc0 = fmaf(x12, v.x, acc0); acc1 = fmaf(x12, v.y, acc1); acc2 = fmaf(x12, v.z, acc2); acc3 = fmaf(x12, v.w, acc3);

            v = w4[6]; acc0 = fmaf(x20, v.x, acc0); acc1 = fmaf(x20, v.y, acc1); acc2 = fmaf(x20, v.z, acc2); acc3 = fmaf(x20, v.w, acc3);
            v = w4[7]; acc0 = fmaf(x21, v.x, acc0); acc1 = fmaf(x21, v.y, acc1); acc2 = fmaf(x21, v.z, acc2); acc3 = fmaf(x21, v.w, acc3);
            v = w4[8]; acc0 = fmaf(x22, v.x, acc0); acc1 = fmaf(x22, v.y, acc1); acc2 = fmaf(x22, v.z, acc2); acc3 = fmaf(x22, v.w, acc3);
        }

        int64_t out_pix = (int64_t)oh * Wout + ow;
        int64_t out_base0 = y_n_base + (int64_t)(oc0 + 0) * Hout * (int64_t)Wout + out_pix;
        y[out_base0] = acc0;
        if (has1) y[out_base0 + (int64_t)Hout * Wout] = acc1;
        if (has2) y[out_base0 + 2LL * (int64_t)Hout * Wout] = acc2;
        if (has3) y[out_base0 + 3LL * (int64_t)Hout * Wout] = acc3;
    }
}

// ---------------- Border kernel: K=3,S=1,D=1 with bounds checks, weights in shared ----------------
// Computes pixels in rectangles around the interior. Uses same weight staging to reduce weight bandwidth.
__global__ __launch_bounds__(128, 4)
void conv2d_forward_nchw_f32_k3_s1d1_oc4_border_smemw(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int N, int Cin, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int oh_begin, int oh_end,   // [oh_begin, oh_end)
    int ow_begin, int ow_end    // [ow_begin, ow_end)
) {
    int ocv = (int)blockIdx.y;
    int n = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= Cout) return;

    bool has1 = (oc0 + 1) < Cout;
    bool has2 = (oc0 + 2) < Cout;
    bool has3 = (oc0 + 3) < Cout;

    extern __shared__ float4 sw4[];
    int tid = (int)threadIdx.x;

    int total_w4 = Cin * 9;
    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / 9;
        int t  = i - ic * 9;

        float a = ldg_f32(w + ((oc0 + 0) * Cin + ic) * 9 + t);
        float b = has1 ? ldg_f32(w + ((oc0 + 1) * Cin + ic) * 9 + t) : 0.f;
        float c = has2 ? ldg_f32(w + ((oc0 + 2) * Cin + ic) * 9 + t) : 0.f;
        float d = has3 ? ldg_f32(w + ((oc0 + 3) * Cin + ic) * 9 + t) : 0.f;
        sw4[i] = make_float4(a, b, c, d);
    }
    __syncthreads();

    int rectH = oh_end - oh_begin;
    int rectW = ow_end - ow_begin;
    int rect_pix = rectH * rectW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    int64_t x_n_base = (int64_t)n * Cin * (int64_t)Hin * Win;
    int64_t y_n_base = (int64_t)n * Cout * (int64_t)Hout * Wout;

    for (int p = base; p < rect_pix; p += step) {
        int local_ow = p % rectW;
        int local_oh = p / rectW;

        int oh = oh_begin + local_oh;
        int ow = ow_begin + local_ow;

        int ih0 = oh - pad_h;
        int iw0 = ow - pad_w;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * Hin * (int64_t)Win;
            const float4* w4 = sw4 + ic * 9;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih0 + kh;
                if ((unsigned)ih >= (unsigned)Hin) continue;
                int64_t row = x_c_base + (int64_t)ih * Win;

                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = iw0 + kw;
                    if ((unsigned)iw >= (unsigned)Win) continue;

                    float xv = ldg_f32(x + row + iw);
                    float4 v = w4[kh * 3 + kw];
                    acc0 = fmaf(xv, v.x, acc0);
                    acc1 = fmaf(xv, v.y, acc1);
                    acc2 = fmaf(xv, v.z, acc2);
                    acc3 = fmaf(xv, v.w, acc3);
                }
            }
        }

        int64_t out_pix = (int64_t)oh * Wout + ow;
        int64_t out_base0 = y_n_base + (int64_t)(oc0 + 0) * Hout * (int64_t)Wout + out_pix;
        y[out_base0] = acc0;
        if (has1) y[out_base0 + (int64_t)Hout * Wout] = acc1;
        if (has2) y[out_base0 + 2LL * (int64_t)Hout * Wout] = acc2;
        if (has3) y[out_base0 + 3LL * (int64_t)Hout * Wout] = acc3;
    }
}

torch::Tensor conv_standard2d_asymmetric_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW (4D)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous (OIHW)");
    TORCH_CHECK(w.size(2) == w.size(3), "kernel must be square");

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);
    int wCin = (int)w.size(1);
    int K    = (int)w.size(2);

    TORCH_CHECK(wCin == Cin, "weight Cin must match input Cin");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be > 0");
    TORCH_CHECK(dil_h > 0 && dil_w > 0, "dilation must be > 0");

    int Hout = (int)((Hin + 2 * (int)pad_h - (int)dil_h * (K - 1) - 1) / (int)stride_h + 1);
    int Wout = (int)((Win + 2 * (int)pad_w - (int)dil_w * (K - 1) - 1) / (int)stride_w + 1);
    TORCH_CHECK(Hout >= 0 && Wout >= 0, "Invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool fast = (K == 3 &&
                 stride_h == 1 && stride_w == 1 &&
                 dil_h == 1 && dil_w == 1);

    if (fast) {
        // Try shared-weight + interior/border split only if shared memory fits
        // smem bytes per block = Cin*9*sizeof(float4) = Cin*9*16
        size_t smem_bytes = (size_t)Cin * 9u * sizeof(float4);

        // Conservative cap: if smem too large, fall back to previous OCx4 kernel
        // (many GPUs have 48-100KB shared per block; keep margin).
        bool use_smem = (smem_bytes <= 48u * 1024u);

        int oc_vecs = (Cout + 3) / 4;

        // Compute interior output region where 3x3 footprint always in-bounds
        // Input top-left = oh - pad_h, ow - pad_w. Need 0 <= ih0 and ih0+2 < Hin => pad_h <= oh <= Hin + pad_h - 3
        int oh0 = (int)pad_h;
        int oh1 = (Hin + (int)pad_h - 2);
        int ow0 = (int)pad_w;
        int ow1 = (Win + (int)pad_w - 2);

        // Clamp to output bounds
        if (oh0 < 0) oh0 = 0;
        if (ow0 < 0) ow0 = 0;
        if (oh1 > Hout) oh1 = Hout;
        if (ow1 > Wout) ow1 = Wout;

        int interiorH = oh1 - oh0;
        int interiorW = ow1 - ow0;

        if (use_smem && interiorH > 0 && interiorW > 0) {
            const int threads = 128;

            // Interior launch
            int interior_pix = interiorH * interiorW;
            int blocks_x = div_up_int(interior_pix, threads);
            if (blocks_x > 4096) blocks_x = 4096;
            if (blocks_x < 1) blocks_x = 1;

            dim3 grid_in((unsigned)blocks_x, (unsigned)oc_vecs, (unsigned)N);
            conv2d_forward_nchw_f32_k3_s1d1_oc4_interior_smemw<<<grid_in, threads, smem_bytes, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                N, Cin, Hin, Win,
                Cout,
                (int)pad_h, (int)pad_w,
                Hout, Wout,
                oh0, oh1, ow0, ow1
            );

            // Border: top, bottom, left, right rectangles (skip empty and avoid double-compute by carving carefully)
            // Top: [0, oh0) x [0, Wout)
            // Bottom: [oh1, Hout) x [0, Wout)
            // Middle-left: [oh0, oh1) x [0, ow0)
            // Middle-right: [oh0, oh1) x [ow1, Wout)
            auto launch_border = [&](int ob, int oe, int wb, int we) {
                int rh = oe - ob;
                int rw = we - wb;
                if (rh <= 0 || rw <= 0) return;
                int rect_pix = rh * rw;
                int bx = div_up_int(rect_pix, threads);
                if (bx > 4096) bx = 4096;
                if (bx < 1) bx = 1;
                dim3 grid((unsigned)bx, (unsigned)oc_vecs, (unsigned)N);
                conv2d_forward_nchw_f32_k3_s1d1_oc4_border_smemw<<<grid, threads, smem_bytes, stream>>>(
                    (const float*)x.data_ptr<float>(),
                    (const float*)w.data_ptr<float>(),
                    (float*)y.data_ptr<float>(),
                    N, Cin, Hin, Win,
                    Cout,
                    (int)pad_h, (int)pad_w,
                    Hout, Wout,
                    ob, oe, wb, we
                );
            };

            launch_border(0, oh0, 0, Wout);
            launch_border(oh1, Hout, 0, Wout);
            launch_border(oh0, oh1, 0, ow0);
            launch_border(oh0, oh1, ow1, Wout);

        } else {
            // Fallback to previous OCx4 (single kernel, bounds-checked)
            const int threads = 256;
            int total_pix = Hout * Wout;
            int blocks_x = div_up_int(total_pix, threads);
            if (blocks_x > 4096) blocks_x = 4096;
            if (blocks_x < 1) blocks_x = 1;

            dim3 grid((unsigned)blocks_x, (unsigned)oc_vecs, (unsigned)N);
            conv2d_forward_nchw_f32_k3_s1d1_oc4_gs<<<grid, threads, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                N, Cin, Hin, Win,
                Cout,
                (int)pad_h, (int)pad_w,
                Hout, Wout
            );
        }
    } else {
        int total = N * Cout * Hout * Wout;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        conv2d_forward_nchw_f32_square_generic<<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, K,
            (int)stride_h, (int)stride_w,
            (int)pad_h, (int)pad_w,
            (int)dil_h, (int)dil_w,
            Hout, Wout
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv_standard2d_asymmetric_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_standard2d_asym_square_v3_smemw",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard2d_asymmetric_input_square_kernel_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Custom CUDA forward for Conv2d (square kernel).
    Assumptions: NCHW float32 CUDA tensors, groups=1, bias=False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if groups != 1:
            raise ValueError("ModelNew custom kernel supports groups=1 only")
        if bias:
            raise ValueError("ModelNew custom kernel supports bias=False only")

        self.custom_ops_lib = custom_ops_lib
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (int(kernel_size), int(kernel_size)),
            stride=int(stride),
            padding=int(padding),
            dilation=int(dilation),
            groups=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != torch.float32:
            raise ValueError("Input must be float32")
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.conv2d.weight
        if w.device != x.device:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        sh, sw = self.conv2d.stride
        ph, pw = self.conv2d.padding
        dh, dw = self.conv2d.dilation

        return self.custom_ops_lib.conv_standard2d_asymmetric_input_square_kernel_cuda(
            x, w, int(sh), int(sw), int(ph), int(pw), int(dh), int(dw)
        )