import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard3d_asymmetric_input_square_kernel ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---------------- Generic kernel (covers all supported params) ----------------
__global__ __launch_bounds__(256, 2)
void conv3d_khw1_forward_generic_f32_kernel(
    const float* __restrict__ x,      // [N, Cin, Hin, Win, Din]
    const float* __restrict__ w,      // [Cout, Cin/groups, kH, kW, 1]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Hout, Wout, Dout]
    int N, int Cin, int Hin, int Win, int Din,
    int Cout,
    int kH, int kW,
    int stride, int padding, int dilation,
    int groups,
    int Hout, int Wout, int Dout
) {
    int64_t total = (int64_t)N * Cout * Hout * Wout * Dout;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;

    int cin_per_group = Cin / groups;
    int cout_per_group = Cout / groups;

    for (int64_t linear = tid; linear < total; linear += step) {
        int64_t idx = linear;

        int od = (int)(idx % Dout); idx /= Dout;
        int ow = (int)(idx % Wout); idx /= Wout;
        int oh = (int)(idx % Hout); idx /= Hout;
        int oc = (int)(idx % Cout);
        int n  = (int)(idx / Cout);

        float acc = (b != nullptr) ? ldg_f32(b + oc) : 0.0f;

        int g = oc / cout_per_group;
        int cin_start = g * cin_per_group;

        int id0 = od * stride - padding; // kd=0 only

        if ((unsigned)id0 < (unsigned)Din) {
            int oh_base = oh * stride - padding;
            int ow_base = ow * stride - padding;

            for (int icg = 0; icg < cin_per_group; ++icg) {
                int ic = cin_start + icg;

                int64_t x_nic_base = (((int64_t)n * Cin + ic) * (int64_t)Hin * Win * Din);
                int64_t w_ocic_base = (((int64_t)oc * cin_per_group + icg) * (int64_t)kH * kW);

#pragma unroll 1
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh_base + kh * dilation;
                    if ((unsigned)ih >= (unsigned)Hin) continue;
#pragma unroll 1
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ow_base + kw * dilation;
                        if ((unsigned)iw >= (unsigned)Win) continue;

                        int64_t x_idx = x_nic_base + ((int64_t)ih * Win + iw) * Din + id0;
                        int64_t w_idx = w_ocic_base + (int64_t)kh * kW + kw;
                        acc = fmaf(ldg_f32(x + x_idx), ldg_f32(w + w_idx), acc);
                    }
                }
            }
        }

        int64_t y_idx = (((((int64_t)n * Cout + oc) * Hout + oh) * Wout + ow) * Dout + od);
        y[y_idx] = acc;
    }
}

// ---------------- Fast path: k=3, stride=1, dilation=1, padding=0, groups=1 ----------------
// Depth-vectorized kernel computing OCx4 per thread and OD vector (4 or 2) when possible.
// grid.x spans (n,oh,ow,od_vec) pixels; grid.y spans oc tiles of 4.
template<int VEC>
__global__ __launch_bounds__(256, 2)
void conv3d_k3s1p0d1_g1_oc4_odvec_f32_kernel(
    const float* __restrict__ x,   // [N,Cin,H,W,D]
    const float* __restrict__ w,   // [Cout,Cin,3,3,1] contiguous
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N,Cout,Hout,Wout,Dout]
    int N, int Cin, int H, int W, int D,
    int Cout,
    int Hout, int Wout, int Dout
) {
    // oc tile
    int oc0 = (int)(blockIdx.y * 4);

    // work over (n,oh,ow,od_base) where od_base increments by VEC
    int64_t n_hw = (int64_t)N * Hout * Wout;
    int64_t od_tiles = (int64_t)Dout / VEC; // only full tiles here
    int64_t total = n_hw * od_tiles;

    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;

    for (int64_t lin = t; lin < total; lin += step) {
        int64_t idx = lin;

        int64_t odt = idx % od_tiles; idx /= od_tiles;
        int ow = (int)(idx % Wout); idx /= Wout;
        int oh = (int)(idx % Hout);
        int n  = (int)(idx / Hout);

        int od0 = (int)(odt * VEC);

        // Input coords (k=3, s=1, p=0)
        int ih0 = oh;
        int iw0 = ow;

        int64_t x_n_base = (int64_t)n * Cin * (int64_t)H * W * D;
        int64_t y_n_base = (int64_t)n * Cout * (int64_t)Hout * Wout * Dout;
        int64_t out_pix_base = ((int64_t)oh * Wout + ow) * Dout + od0;

        // accumulators: 4 oc, each has VEC lanes
        float acc0[VEC], acc1[VEC], acc2[VEC], acc3[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            acc0[i] = 0.f; acc1[i] = 0.f; acc2[i] = 0.f; acc3[i] = 0.f;
        }

        if (b != nullptr) {
            float bb0 = (oc0 + 0 < Cout) ? ldg_f32(b + (oc0 + 0)) : 0.f;
            float bb1 = (oc0 + 1 < Cout) ? ldg_f32(b + (oc0 + 1)) : 0.f;
            float bb2 = (oc0 + 2 < Cout) ? ldg_f32(b + (oc0 + 2)) : 0.f;
            float bb3 = (oc0 + 3 < Cout) ? ldg_f32(b + (oc0 + 3)) : 0.f;
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc0[i] = bb0; acc1[i] = bb1; acc2[i] = bb2; acc3[i] = bb3;
            }
        }

        // accumulate
        for (int ic = 0; ic < Cin; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * H * (int64_t)W * D;

            // For each of 3 rows, load 3 columns; each load is VEC contiguous floats along D.
            // Use vector types when aligned.
            float x00[VEC], x01[VEC], x02[VEC];
            float x10[VEC], x11[VEC], x12[VEC];
            float x20[VEC], x21[VEC], x22[VEC];

            int64_t base00 = x_c_base + ((int64_t)(ih0 + 0) * W + (iw0 + 0)) * D + od0;
            int64_t base01 = x_c_base + ((int64_t)(ih0 + 0) * W + (iw0 + 1)) * D + od0;
            int64_t base02 = x_c_base + ((int64_t)(ih0 + 0) * W + (iw0 + 2)) * D + od0;

            int64_t base10 = x_c_base + ((int64_t)(ih0 + 1) * W + (iw0 + 0)) * D + od0;
            int64_t base11 = x_c_base + ((int64_t)(ih0 + 1) * W + (iw0 + 1)) * D + od0;
            int64_t base12 = x_c_base + ((int64_t)(ih0 + 1) * W + (iw0 + 2)) * D + od0;

            int64_t base20 = x_c_base + ((int64_t)(ih0 + 2) * W + (iw0 + 0)) * D + od0;
            int64_t base21 = x_c_base + ((int64_t)(ih0 + 2) * W + (iw0 + 1)) * D + od0;
            int64_t base22 = x_c_base + ((int64_t)(ih0 + 2) * W + (iw0 + 2)) * D + od0;

            if constexpr (VEC == 4) {
                float4 v00 = *reinterpret_cast<const float4*>(x + base00);
                float4 v01 = *reinterpret_cast<const float4*>(x + base01);
                float4 v02 = *reinterpret_cast<const float4*>(x + base02);

                float4 v10 = *reinterpret_cast<const float4*>(x + base10);
                float4 v11 = *reinterpret_cast<const float4*>(x + base11);
                float4 v12 = *reinterpret_cast<const float4*>(x + base12);

                float4 v20 = *reinterpret_cast<const float4*>(x + base20);
                float4 v21 = *reinterpret_cast<const float4*>(x + base21);
                float4 v22 = *reinterpret_cast<const float4*>(x + base22);

                x00[0]=v00.x; x00[1]=v00.y; x00[2]=v00.z; x00[3]=v00.w;
                x01[0]=v01.x; x01[1]=v01.y; x01[2]=v01.z; x01[3]=v01.w;
                x02[0]=v02.x; x02[1]=v02.y; x02[2]=v02.z; x02[3]=v02.w;

                x10[0]=v10.x; x10[1]=v10.y; x10[2]=v10.z; x10[3]=v10.w;
                x11[0]=v11.x; x11[1]=v11.y; x11[2]=v11.z; x11[3]=v11.w;
                x12[0]=v12.x; x12[1]=v12.y; x12[2]=v12.z; x12[3]=v12.w;

                x20[0]=v20.x; x20[1]=v20.y; x20[2]=v20.z; x20[3]=v20.w;
                x21[0]=v21.x; x21[1]=v21.y; x21[2]=v21.z; x21[3]=v21.w;
                x22[0]=v22.x; x22[1]=v22.y; x22[2]=v22.z; x22[3]=v22.w;
            } else { // VEC == 2
                float2 v00 = *reinterpret_cast<const float2*>(x + base00);
                float2 v01 = *reinterpret_cast<const float2*>(x + base01);
                float2 v02 = *reinterpret_cast<const float2*>(x + base02);

                float2 v10 = *reinterpret_cast<const float2*>(x + base10);
                float2 v11 = *reinterpret_cast<const float2*>(x + base11);
                float2 v12 = *reinterpret_cast<const float2*>(x + base12);

                float2 v20 = *reinterpret_cast<const float2*>(x + base20);
                float2 v21 = *reinterpret_cast<const float2*>(x + base21);
                float2 v22 = *reinterpret_cast<const float2*>(x + base22);

                x00[0]=v00.x; x00[1]=v00.y;
                x01[0]=v01.x; x01[1]=v01.y;
                x02[0]=v02.x; x02[1]=v02.y;

                x10[0]=v10.x; x10[1]=v10.y;
                x11[0]=v11.x; x11[1]=v11.y;
                x12[0]=v12.x; x12[1]=v12.y;

                x20[0]=v20.x; x20[1]=v20.y;
                x21[0]=v21.x; x21[1]=v21.y;
                x22[0]=v22.x; x22[1]=v22.y;
            }

            int64_t w_ic_off = (int64_t)ic * 9;

            if (oc0 + 0 < Cout) {
                const float* w0 = w + ((int64_t)(oc0 + 0) * Cin * 9 + w_ic_off);
                float ww0 = ldg_f32(w0 + 0), ww1 = ldg_f32(w0 + 1), ww2 = ldg_f32(w0 + 2);
                float ww3 = ldg_f32(w0 + 3), ww4 = ldg_f32(w0 + 4), ww5 = ldg_f32(w0 + 5);
                float ww6 = ldg_f32(w0 + 6), ww7 = ldg_f32(w0 + 7), ww8 = ldg_f32(w0 + 8);
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc0[i] = fmaf(x00[i], ww0, acc0[i]);
                    acc0[i] = fmaf(x01[i], ww1, acc0[i]);
                    acc0[i] = fmaf(x02[i], ww2, acc0[i]);
                    acc0[i] = fmaf(x10[i], ww3, acc0[i]);
                    acc0[i] = fmaf(x11[i], ww4, acc0[i]);
                    acc0[i] = fmaf(x12[i], ww5, acc0[i]);
                    acc0[i] = fmaf(x20[i], ww6, acc0[i]);
                    acc0[i] = fmaf(x21[i], ww7, acc0[i]);
                    acc0[i] = fmaf(x22[i], ww8, acc0[i]);
                }
            }
            if (oc0 + 1 < Cout) {
                const float* w1 = w + ((int64_t)(oc0 + 1) * Cin * 9 + w_ic_off);
                float ww0 = ldg_f32(w1 + 0), ww1 = ldg_f32(w1 + 1), ww2 = ldg_f32(w1 + 2);
                float ww3 = ldg_f32(w1 + 3), ww4 = ldg_f32(w1 + 4), ww5 = ldg_f32(w1 + 5);
                float ww6 = ldg_f32(w1 + 6), ww7 = ldg_f32(w1 + 7), ww8 = ldg_f32(w1 + 8);
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc1[i] = fmaf(x00[i], ww0, acc1[i]);
                    acc1[i] = fmaf(x01[i], ww1, acc1[i]);
                    acc1[i] = fmaf(x02[i], ww2, acc1[i]);
                    acc1[i] = fmaf(x10[i], ww3, acc1[i]);
                    acc1[i] = fmaf(x11[i], ww4, acc1[i]);
                    acc1[i] = fmaf(x12[i], ww5, acc1[i]);
                    acc1[i] = fmaf(x20[i], ww6, acc1[i]);
                    acc1[i] = fmaf(x21[i], ww7, acc1[i]);
                    acc1[i] = fmaf(x22[i], ww8, acc1[i]);
                }
            }
            if (oc0 + 2 < Cout) {
                const float* w2 = w + ((int64_t)(oc0 + 2) * Cin * 9 + w_ic_off);
                float ww0 = ldg_f32(w2 + 0), ww1 = ldg_f32(w2 + 1), ww2 = ldg_f32(w2 + 2);
                float ww3 = ldg_f32(w2 + 3), ww4 = ldg_f32(w2 + 4), ww5 = ldg_f32(w2 + 5);
                float ww6 = ldg_f32(w2 + 6), ww7 = ldg_f32(w2 + 7), ww8 = ldg_f32(w2 + 8);
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc2[i] = fmaf(x00[i], ww0, acc2[i]);
                    acc2[i] = fmaf(x01[i], ww1, acc2[i]);
                    acc2[i] = fmaf(x02[i], ww2, acc2[i]);
                    acc2[i] = fmaf(x10[i], ww3, acc2[i]);
                    acc2[i] = fmaf(x11[i], ww4, acc2[i]);
                    acc2[i] = fmaf(x12[i], ww5, acc2[i]);
                    acc2[i] = fmaf(x20[i], ww6, acc2[i]);
                    acc2[i] = fmaf(x21[i], ww7, acc2[i]);
                    acc2[i] = fmaf(x22[i], ww8, acc2[i]);
                }
            }
            if (oc0 + 3 < Cout) {
                const float* w3 = w + ((int64_t)(oc0 + 3) * Cin * 9 + w_ic_off);
                float ww0 = ldg_f32(w3 + 0), ww1 = ldg_f32(w3 + 1), ww2 = ldg_f32(w3 + 2);
                float ww3 = ldg_f32(w3 + 3), ww4 = ldg_f32(w3 + 4), ww5 = ldg_f32(w3 + 5);
                float ww6 = ldg_f32(w3 + 6), ww7 = ldg_f32(w3 + 7), ww8 = ldg_f32(w3 + 8);
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc3[i] = fmaf(x00[i], ww0, acc3[i]);
                    acc3[i] = fmaf(x01[i], ww1, acc3[i]);
                    acc3[i] = fmaf(x02[i], ww2, acc3[i]);
                    acc3[i] = fmaf(x10[i], ww3, acc3[i]);
                    acc3[i] = fmaf(x11[i], ww4, acc3[i]);
                    acc3[i] = fmaf(x12[i], ww5, acc3[i]);
                    acc3[i] = fmaf(x20[i], ww6, acc3[i]);
                    acc3[i] = fmaf(x21[i], ww7, acc3[i]);
                    acc3[i] = fmaf(x22[i], ww8, acc3[i]);
                }
            }
        }

        // vectorized stores along D
        int64_t y_plane = (int64_t)Hout * Wout * Dout;
        if (oc0 + 0 < Cout) {
            float* yp = y + y_n_base + (int64_t)(oc0 + 0) * y_plane + out_pix_base;
            if constexpr (VEC == 4) {
                *reinterpret_cast<float4*>(yp) = make_float4(acc0[0], acc0[1], acc0[2], acc0[3]);
            } else {
                *reinterpret_cast<float2*>(yp) = make_float2(acc0[0], acc0[1]);
            }
        }
        if (oc0 + 1 < Cout) {
            float* yp = y + y_n_base + (int64_t)(oc0 + 1) * y_plane + out_pix_base;
            if constexpr (VEC == 4) {
                *reinterpret_cast<float4*>(yp) = make_float4(acc1[0], acc1[1], acc1[2], acc1[3]);
            } else {
                *reinterpret_cast<float2*>(yp) = make_float2(acc1[0], acc1[1]);
            }
        }
        if (oc0 + 2 < Cout) {
            float* yp = y + y_n_base + (int64_t)(oc0 + 2) * y_plane + out_pix_base;
            if constexpr (VEC == 4) {
                *reinterpret_cast<float4*>(yp) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);
            } else {
                *reinterpret_cast<float2*>(yp) = make_float2(acc2[0], acc2[1]);
            }
        }
        if (oc0 + 3 < Cout) {
            float* yp = y + y_n_base + (int64_t)(oc0 + 3) * y_plane + out_pix_base;
            if constexpr (VEC == 4) {
                *reinterpret_cast<float4*>(yp) = make_float4(acc3[0], acc3[1], acc3[2], acc3[3]);
            } else {
                *reinterpret_cast<float2*>(yp) = make_float2(acc3[0], acc3[1]);
            }
        }
    }
}

// Scalar tail for remaining od elements (or when no vectorization possible), still OCx4 per thread.
__global__ __launch_bounds__(256, 2)
void conv3d_k3s1p0d1_g1_oc4_scalar_f32_kernel(
    const float* __restrict__ x,   // [N,Cin,H,W,D]
    const float* __restrict__ w,   // [Cout,Cin,3,3,1]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N,Cout,Hout,Wout,Dout]
    int N, int Cin, int H, int W, int D,
    int Cout,
    int Hout, int Wout, int Dout,
    int od_start
) {
    int64_t pixels = (int64_t)N * Hout * Wout * (int64_t)(Dout - od_start);
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;

    int oc0 = (int)(blockIdx.y * 4);

    for (int64_t p = t; p < pixels; p += step) {
        int64_t idx = p;
        int od = (int)(idx % (Dout - od_start)); idx /= (Dout - od_start);
        int ow = (int)(idx % Wout); idx /= Wout;
        int oh = (int)(idx % Hout);
        int n  = (int)(idx / Hout);
        od += od_start;

        int ih0 = oh;
        int iw0 = ow;
        int id0 = od;

        int64_t x_n_base = (int64_t)n * Cin * (int64_t)H * W * D;
        int64_t y_n_base = (int64_t)n * Cout * (int64_t)Hout * Wout * Dout;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        if (b != nullptr) {
            if (oc0 + 0 < Cout) acc0 = ldg_f32(b + (oc0 + 0));
            if (oc0 + 1 < Cout) acc1 = ldg_f32(b + (oc0 + 1));
            if (oc0 + 2 < Cout) acc2 = ldg_f32(b + (oc0 + 2));
            if (oc0 + 3 < Cout) acc3 = ldg_f32(b + (oc0 + 3));
        }

        for (int ic = 0; ic < Cin; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * H * (int64_t)W * D;

            int64_t row0 = x_c_base + ((int64_t)(ih0 + 0) * W + (iw0 + 0)) * D + id0;
            int64_t row1 = x_c_base + ((int64_t)(ih0 + 1) * W + (iw0 + 0)) * D + id0;
            int64_t row2 = x_c_base + ((int64_t)(ih0 + 2) * W + (iw0 + 0)) * D + id0;

            float x00 = ldg_f32(x + row0 + 0 * (int64_t)D);
            float x01 = ldg_f32(x + row0 + 1 * (int64_t)D);
            float x02 = ldg_f32(x + row0 + 2 * (int64_t)D);

            float x10 = ldg_f32(x + row1 + 0 * (int64_t)D);
            float x11 = ldg_f32(x + row1 + 1 * (int64_t)D);
            float x12 = ldg_f32(x + row1 + 2 * (int64_t)D);

            float x20 = ldg_f32(x + row2 + 0 * (int64_t)D);
            float x21 = ldg_f32(x + row2 + 1 * (int64_t)D);
            float x22 = ldg_f32(x + row2 + 2 * (int64_t)D);

            int64_t w_ic_off = (int64_t)ic * 9;

            if (oc0 + 0 < Cout) {
                const float* w0 = w + ((int64_t)(oc0 + 0) * Cin * 9 + w_ic_off);
                acc0 = fmaf(x00, ldg_f32(w0 + 0), acc0);
                acc0 = fmaf(x01, ldg_f32(w0 + 1), acc0);
                acc0 = fmaf(x02, ldg_f32(w0 + 2), acc0);
                acc0 = fmaf(x10, ldg_f32(w0 + 3), acc0);
                acc0 = fmaf(x11, ldg_f32(w0 + 4), acc0);
                acc0 = fmaf(x12, ldg_f32(w0 + 5), acc0);
                acc0 = fmaf(x20, ldg_f32(w0 + 6), acc0);
                acc0 = fmaf(x21, ldg_f32(w0 + 7), acc0);
                acc0 = fmaf(x22, ldg_f32(w0 + 8), acc0);
            }
            if (oc0 + 1 < Cout) {
                const float* w1 = w + ((int64_t)(oc0 + 1) * Cin * 9 + w_ic_off);
                acc1 = fmaf(x00, ldg_f32(w1 + 0), acc1);
                acc1 = fmaf(x01, ldg_f32(w1 + 1), acc1);
                acc1 = fmaf(x02, ldg_f32(w1 + 2), acc1);
                acc1 = fmaf(x10, ldg_f32(w1 + 3), acc1);
                acc1 = fmaf(x11, ldg_f32(w1 + 4), acc1);
                acc1 = fmaf(x12, ldg_f32(w1 + 5), acc1);
                acc1 = fmaf(x20, ldg_f32(w1 + 6), acc1);
                acc1 = fmaf(x21, ldg_f32(w1 + 7), acc1);
                acc1 = fmaf(x22, ldg_f32(w1 + 8), acc1);
            }
            if (oc0 + 2 < Cout) {
                const float* w2 = w + ((int64_t)(oc0 + 2) * Cin * 9 + w_ic_off);
                acc2 = fmaf(x00, ldg_f32(w2 + 0), acc2);
                acc2 = fmaf(x01, ldg_f32(w2 + 1), acc2);
                acc2 = fmaf(x02, ldg_f32(w2 + 2), acc2);
                acc2 = fmaf(x10, ldg_f32(w2 + 3), acc2);
                acc2 = fmaf(x11, ldg_f32(w2 + 4), acc2);
                acc2 = fmaf(x12, ldg_f32(w2 + 5), acc2);
                acc2 = fmaf(x20, ldg_f32(w2 + 6), acc2);
                acc2 = fmaf(x21, ldg_f32(w2 + 7), acc2);
                acc2 = fmaf(x22, ldg_f32(w2 + 8), acc2);
            }
            if (oc0 + 3 < Cout) {
                const float* w3 = w + ((int64_t)(oc0 + 3) * Cin * 9 + w_ic_off);
                acc3 = fmaf(x00, ldg_f32(w3 + 0), acc3);
                acc3 = fmaf(x01, ldg_f32(w3 + 1), acc3);
                acc3 = fmaf(x02, ldg_f32(w3 + 2), acc3);
                acc3 = fmaf(x10, ldg_f32(w3 + 3), acc3);
                acc3 = fmaf(x11, ldg_f32(w3 + 4), acc3);
                acc3 = fmaf(x12, ldg_f32(w3 + 5), acc3);
                acc3 = fmaf(x20, ldg_f32(w3 + 6), acc3);
                acc3 = fmaf(x21, ldg_f32(w3 + 7), acc3);
                acc3 = fmaf(x22, ldg_f32(w3 + 8), acc3);
            }
        }

        int64_t out_pix = ((int64_t)oh * Wout + ow) * Dout + od;
        int64_t y_plane = (int64_t)Hout * Wout * Dout;
        if (oc0 + 0 < Cout) y[y_n_base + (int64_t)(oc0 + 0) * y_plane + out_pix] = acc0;
        if (oc0 + 1 < Cout) y[y_n_base + (int64_t)(oc0 + 1) * y_plane + out_pix] = acc1;
        if (oc0 + 2 < Cout) y[y_n_base + (int64_t)(oc0 + 2) * y_plane + out_pix] = acc2;
        if (oc0 + 3 < Cout) y[y_n_base + (int64_t)(oc0 + 3) * y_plane + out_pix] = acc3;
    }
}

torch::Tensor conv_standard3d_asymmetric_input_square_kernel_cuda(
    torch::Tensor x,      // [N,Cin,H,W,D]
    torch::Tensor w,      // [Cout,Cin/groups,kH,kW,1]
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [Cout, Cin/groups, kH, kW, 1]");
    TORCH_CHECK((int)w.size(4) == 1, "kernel depth must be 1");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");
    TORCH_CHECK(x.size(1) % groups == 0, "Cin must be divisible by groups");
    TORCH_CHECK(w.size(0) % groups == 0, "Cout must be divisible by groups");
    TORCH_CHECK(w.size(1) * groups == x.size(1), "w Cin/groups mismatch with x Cin");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);
    const int Din = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int kH   = (int)w.size(2);
    const int kW   = (int)w.size(3);

    const int s = (int)stride;
    const int p = (int)padding;
    const int d = (int)dilation;
    const int g = (int)groups;

    const int Hout = (Hin + 2*p - d*(kH - 1) - 1) / s + 1;
    const int Wout = (Win + 2*p - d*(kW - 1) - 1) / s + 1;
    const int Dout = (Din + 2*p - d*(1 - 1) - 1) / s + 1;

    TORCH_CHECK(Hout > 0 && Wout > 0 && Dout > 0, "Invalid output shape");

    auto y = torch::empty({N, Cout, Hout, Wout, Dout}, x.options());

    const float* b_ptr = nullptr;
    torch::Tensor b;
    if (b_opt.has_value()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias must be [Cout]");
        if (!b.is_contiguous()) b = b.contiguous();
        b_ptr = (const float*)b.data_ptr<float>();
    }

    const float* x_ptr = (const float*)x.data_ptr<float>();
    const float* w_ptr = (const float*)w.data_ptr<float>();
    float* y_ptr = (float*)y.data_ptr<float>();

    bool fast =
        (g == 1) &&
        (kH == 3 && kW == 3) &&
        (s == 1) &&
        (d == 1) &&
        (p == 0);

    if (fast) {
        int threads = 256;
        int blocks_y = (Cout + 3) / 4;

        // Prefer vec4 if possible and aligned; else vec2; then scalar tail.
        // Since D is innermost and x is contiguous, alignment depends on base pointer and od0.
        // We only launch vec kernels for the full tiles; tail is scalar.
        bool can_vec4 = ((Dout % 4) == 0) && (((uintptr_t)x_ptr & 0xF) == 0) && (((uintptr_t)y_ptr & 0xF) == 0);
        bool can_vec2 = ((Dout % 2) == 0) && (((uintptr_t)x_ptr & 0x7) == 0) && (((uintptr_t)y_ptr & 0x7) == 0);

        if (can_vec4) {
            int64_t total = (int64_t)N * Hout * Wout * (int64_t)(Dout / 4);
            int blocks_x = (int)div_up_i64(total, threads);
            if (blocks_x > 65535) blocks_x = 65535;
            dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);
            conv3d_k3s1p0d1_g1_oc4_odvec_f32_kernel<4><<<grid, threads>>>(
                x_ptr, w_ptr, b_ptr, y_ptr,
                N, Cin, Hin, Win, Din,
                Cout,
                Hout, Wout, Dout
            );
            return y;
        }

        int od_start = 0;
        if (can_vec2) {
            int64_t full2 = (int64_t)(Dout / 2) * 2;
            int64_t total = (int64_t)N * Hout * Wout * (int64_t)(full2 / 2);
            int blocks_x = (int)div_up_i64(total, threads);
            if (blocks_x > 65535) blocks_x = 65535;
            dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);
            conv3d_k3s1p0d1_g1_oc4_odvec_f32_kernel<2><<<grid, threads>>>(
                x_ptr, w_ptr, b_ptr, y_ptr,
                N, Cin, Hin, Win, Din,
                Cout,
                Hout, Wout, Dout
            );
            od_start = (int)full2;
        }

        if (od_start < Dout) {
            int64_t pixels = (int64_t)N * Hout * Wout * (int64_t)(Dout - od_start);
            int blocks_x = (int)div_up_i64(pixels, threads);
            if (blocks_x > 65535) blocks_x = 65535;
            dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);
            conv3d_k3s1p0d1_g1_oc4_scalar_f32_kernel<<<grid, threads>>>(
                x_ptr, w_ptr, b_ptr, y_ptr,
                N, Cin, Hin, Win, Din,
                Cout,
                Hout, Wout, Dout,
                od_start
            );
        }
        return y;
    }

    // Generic fallback
    int64_t total = (int64_t)N * Cout * Hout * Wout * Dout;
    int threads = 256;
    int blocks = (int)div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;

    conv3d_khw1_forward_generic_f32_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, Cin, Hin, Win, Din,
        Cout,
        kH, kW,
        s, p, d,
        g,
        Hout, Wout, Dout
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv_standard3d_asymmetric_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_khw1_opt_oc4_odvec_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard3d_asymmetric_input_square_kernel_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Conv3d forward replaced with a custom CUDA kernel specialized for kernel depth=1 (kH,kW,1), NCDHW.
    Fast path: k=3, stride=1, dilation=1, padding=0, groups=1 using OCx4 per thread and depth vectorization (float4/float2).
    Generic fallback supports all parameterizations.
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
        self.custom_ops_lib = custom_ops_lib

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if not isinstance(stride, int) or not isinstance(padding, int) or not isinstance(dilation, int):
            raise TypeError("This optimized ModelNew expects stride/padding/dilation as ints.")
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv3d.weight
        b = self.conv3d.bias if self.conv3d.bias is not None else None
        return self.custom_ops_lib.conv_standard3d_asymmetric_input_square_kernel_cuda(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )