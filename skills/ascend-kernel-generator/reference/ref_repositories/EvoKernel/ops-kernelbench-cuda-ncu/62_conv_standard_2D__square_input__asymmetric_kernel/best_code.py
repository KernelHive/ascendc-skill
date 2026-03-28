import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard2d_square_input_asymmetric_kernel ---------

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

// ---------------- Generic baseline (correctness for any Kh x Kw) ----------------
__global__ void conv2d_forward_nchw_f32_generic(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, Kh, Kw]
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
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
        const float* w_oc_ic = w + ((oc * Cin + ic) * Kh * Kw);
        const float* x_n_ic  = x + ((n * Cin + ic) * Hin * Win);

        #pragma unroll 1
        for (int kh = 0; kh < Kh; ++kh) {
            int ih = in_h0 + kh * dil_h;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            #pragma unroll 1
            for (int kw = 0; kw < Kw; ++kw) {
                int iw = in_w0 + kw * dil_w;
                if ((unsigned)iw >= (unsigned)Win) continue;

                float xv = ldg_f32(x_n_ic + ih * Win + iw);
                float wv = ldg_f32(w_oc_ic + kh * Kw + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = acc;
}

// ---------------- Previous fast path: Cin=32, Kh=5, Kw=9, stride=1, dilation=1 (input tile per-ic) ----------------
template<int BLOCK_W, int BLOCK_H>
__global__ __launch_bounds__(BLOCK_W*BLOCK_H, 2)
void conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc2_shmem(
    const float* __restrict__ x,      // [N, 32, Hin, Win]
    const float* __restrict__ w,      // [Cout, 32, 5, 9]
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int tiles_per_n                 // = ceil_div(Cout, 2)
) {
    int tile_ow = (int)blockIdx.x;
    int tile_oh = (int)blockIdx.y;
    int gz      = (int)blockIdx.z;

    int n   = gz / tiles_per_n;
    int oc2 = gz - n * tiles_per_n;
    int oc0 = oc2 * 2;
    int oc1 = oc0 + 1;

    if (n >= N || oc0 >= Cout) return;

    int tx = (int)threadIdx.x; // 0..BLOCK_W*BLOCK_H-1
    int t_ow = tx % BLOCK_W;
    int t_oh = tx / BLOCK_W;

    int ow = tile_ow * BLOCK_W + t_ow;
    int oh = tile_oh * BLOCK_H + t_oh;
    bool vout = (ow < Wout) && (oh < Hout);

    constexpr int KH = 5;
    constexpr int KW = 9;
    constexpr int SH_H = BLOCK_H + (KH - 1);
    constexpr int SH_W = BLOCK_W + (KW - 1);

    extern __shared__ float shmem[];
    float* sh = shmem;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int base_ih0 = tile_oh * BLOCK_H - pad_h;
    int base_iw0 = tile_ow * BLOCK_W - pad_w;

    #pragma unroll 1
    for (int ic = 0; ic < 32; ++ic) {
        const float* x_n_ic = x + ((n * 32 + ic) * Hin * Win);

        int sh_elems = SH_H * SH_W;
        for (int i = tx; i < sh_elems; i += (BLOCK_W * BLOCK_H)) {
            int r = i / SH_W;
            int c = i - r * SH_W;
            int ih = base_ih0 + r;
            int iw = base_iw0 + c;

            float v = 0.0f;
            if ((unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
                v = ldg_f32(x_n_ic + ih * Win + iw);
            }
            sh[i] = v;
        }
        __syncthreads();

        if (vout) {
            int sh_r0 = t_oh;
            int sh_c0 = t_ow;

            const float* w0 = w + ((oc0 * 32 + ic) * (KH * KW));
            const float* w1 = (oc1 < Cout) ? (w + ((oc1 * 32 + ic) * (KH * KW))) : nullptr;

            #pragma unroll
            for (int kh = 0; kh < KH; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KW; ++kw) {
                    float xv = sh[(sh_r0 + kh) * SH_W + (sh_c0 + kw)];
                    float wv0 = ldg_f32(w0 + kh * KW + kw);
                    acc0 = fmaf(xv, wv0, acc0);
                    if (oc1 < Cout) {
                        float wv1 = ldg_f32(w1 + kh * KW + kw);
                        acc1 = fmaf(xv, wv1, acc1);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (vout) {
        int out_base0 = ((n * Cout + oc0) * Hout + oh) * Wout + ow;
        y[out_base0] = acc0;
        if (oc1 < Cout) {
            y[out_base0 + Hout * Wout] = acc1;
        }
    }
}

// ---------------- New fast path: Cin=32, K=5x9, S=1, D=1, OCx4 weights in shared, interior/border split ----------------
constexpr int K5 = 5;
constexpr int K9 = 9;
constexpr int K59 = K5 * K9;

__global__ __launch_bounds__(128, 4)
void conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc4_interior_smemw(
    const float* __restrict__ x,   // [N,32,Hin,Win]
    const float* __restrict__ w,   // [Cout,32,5,9]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int oh0, int oh1,              // interior oh range [oh0, oh1)
    int ow0, int ow1               // interior ow range [ow0, ow1)
) {
    int ocv = (int)blockIdx.y;
    int n   = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= Cout) return;

    bool has1 = (oc0 + 1) < Cout;
    bool has2 = (oc0 + 2) < Cout;
    bool has3 = (oc0 + 3) < Cout;

    extern __shared__ float4 sw4[];
    int tid = (int)threadIdx.x;

    // stage weights: 32 * 45 float4 = 1440 float4 = 23040 bytes
    int total_w4 = 32 * K59;
    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / K59;
        int t  = i - ic * K59;

        float a = ldg_f32(w + ((oc0 + 0) * 32 + ic) * K59 + t);
        float b = has1 ? ldg_f32(w + ((oc0 + 1) * 32 + ic) * K59 + t) : 0.f;
        float c = has2 ? ldg_f32(w + ((oc0 + 2) * 32 + ic) * K59 + t) : 0.f;
        float d = has3 ? ldg_f32(w + ((oc0 + 3) * 32 + ic) * K59 + t) : 0.f;
        sw4[i] = make_float4(a, b, c, d);
    }
    __syncthreads();

    int intH = oh1 - oh0;
    int intW = ow1 - ow0;
    int interior_pix = intH * intW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    int64_t x_n_base = (int64_t)n * 32LL * (int64_t)Hin * Win;
    int64_t y_n_base = (int64_t)n * (int64_t)Cout * (int64_t)Hout * Wout;

    for (int p = base; p < interior_pix; p += step) {
        int local_ow = p % intW;
        int local_oh = p / intW;

        int oh = oh0 + local_oh;
        int ow = ow0 + local_ow;

        int ih0 = oh - pad_h;
        int iw0 = ow - pad_w;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll
        for (int ic = 0; ic < 32; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * Hin * (int64_t)Win;
            const float4* w4 = sw4 + ic * K59;

            // Unrolled 5x9; no bounds checks in interior
            #pragma unroll
            for (int kh = 0; kh < K5; ++kh) {
                int64_t row = x_c_base + (int64_t)(ih0 + kh) * Win + iw0;
                int wt_base = kh * K9;

                // encourage contiguous loads along width
                #pragma unroll
                for (int kw = 0; kw < K9; ++kw) {
                    float xv = ldg_f32(x + row + kw);
                    float4 v = w4[wt_base + kw];
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

__global__ __launch_bounds__(128, 4)
void conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc4_border_smemw(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int N, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int oh_begin, int oh_end,   // [oh_begin, oh_end)
    int ow_begin, int ow_end    // [ow_begin, ow_end)
) {
    int ocv = (int)blockIdx.y;
    int n   = (int)blockIdx.z;
    int oc0 = ocv * 4;
    if (n >= N || oc0 >= Cout) return;

    bool has1 = (oc0 + 1) < Cout;
    bool has2 = (oc0 + 2) < Cout;
    bool has3 = (oc0 + 3) < Cout;

    extern __shared__ float4 sw4[];
    int tid = (int)threadIdx.x;

    int total_w4 = 32 * K59;
    for (int i = tid; i < total_w4; i += (int)blockDim.x) {
        int ic = i / K59;
        int t  = i - ic * K59;

        float a = ldg_f32(w + ((oc0 + 0) * 32 + ic) * K59 + t);
        float b = has1 ? ldg_f32(w + ((oc0 + 1) * 32 + ic) * K59 + t) : 0.f;
        float c = has2 ? ldg_f32(w + ((oc0 + 2) * 32 + ic) * K59 + t) : 0.f;
        float d = has3 ? ldg_f32(w + ((oc0 + 3) * 32 + ic) * K59 + t) : 0.f;
        sw4[i] = make_float4(a, b, c, d);
    }
    __syncthreads();

    int rectH = oh_end - oh_begin;
    int rectW = ow_end - ow_begin;
    int rect_pix = rectH * rectW;

    int base = (int)blockIdx.x * (int)blockDim.x + tid;
    int step = (int)gridDim.x * (int)blockDim.x;

    int64_t x_n_base = (int64_t)n * 32LL * (int64_t)Hin * Win;
    int64_t y_n_base = (int64_t)n * (int64_t)Cout * (int64_t)Hout * Wout;

    for (int p = base; p < rect_pix; p += step) {
        int local_ow = p % rectW;
        int local_oh = p / rectW;

        int oh = oh_begin + local_oh;
        int ow = ow_begin + local_ow;

        int ih0 = oh - pad_h;
        int iw0 = ow - pad_w;

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        #pragma unroll
        for (int ic = 0; ic < 32; ++ic) {
            int64_t x_c_base = x_n_base + (int64_t)ic * Hin * (int64_t)Win;
            const float4* w4 = sw4 + ic * K59;

            #pragma unroll
            for (int kh = 0; kh < K5; ++kh) {
                int ih = ih0 + kh;
                if ((unsigned)ih >= (unsigned)Hin) continue;
                int64_t row = x_c_base + (int64_t)ih * Win;

                #pragma unroll
                for (int kw = 0; kw < K9; ++kw) {
                    int iw = iw0 + kw;
                    if ((unsigned)iw >= (unsigned)Win) continue;
                    float xv = ldg_f32(x + row + iw);
                    float4 v = w4[kh * K9 + kw];
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

torch::Tensor conv_standard2d_square_input_asymmetric_kernel_cuda(
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

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);
    int wCin = (int)w.size(1);
    int Kh   = (int)w.size(2);
    int Kw   = (int)w.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin must match input Cin");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be > 0");
    TORCH_CHECK(dil_h > 0 && dil_w > 0, "dilation must be > 0");

    int Hout = (int)((Hin + 2 * (int)pad_h - (int)dil_h * (Kh - 1) - 1) / (int)stride_h + 1);
    int Wout = (int)((Win + 2 * (int)pad_w - (int)dil_w * (Kw - 1) - 1) / (int)stride_w + 1);
    TORCH_CHECK(Hout >= 0 && Wout >= 0, "Invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool fast =
        (Cin == 32) &&
        (Kh == 5 && Kw == 9) &&
        (stride_h == 1 && stride_w == 1) &&
        (dil_h == 1 && dil_w == 1);

    if (fast) {
        int oc_vecs = (Cout + 3) / 4;

        // Interior region where full 5x9 footprint is in-bounds:
        // ih0 = oh - pad_h, need 0 <= ih0 and ih0+4 < Hin => pad_h <= oh <= Hin + pad_h - 5
        // iw0 = ow - pad_w, need 0 <= iw0 and iw0+8 < Win => pad_w <= ow <= Win + pad_w - 9
        int oh0 = (int)pad_h;
        int oh1 = (Hin + (int)pad_h - 4);  // exclusive upper bound is (Hin+pad_h-4)
        int ow0 = (int)pad_w;
        int ow1 = (Win + (int)pad_w - 8);

        if (oh0 < 0) oh0 = 0;
        if (ow0 < 0) ow0 = 0;
        if (oh1 > Hout) oh1 = Hout;
        if (ow1 > Wout) ow1 = Wout;

        int interiorH = oh1 - oh0;
        int interiorW = ow1 - ow0;

        // shared weights bytes for OCx4
        size_t smem_bytes = (size_t)32u * (size_t)K59 * sizeof(float4); // 23040 bytes

        // Use new path if interior exists; else fall back to border-only (or previous kernel)
        if (interiorH > 0 && interiorW > 0) {
            const int threads = 128;
            int interior_pix = interiorH * interiorW;
            int blocks_x = div_up_int(interior_pix, threads);
            if (blocks_x > 4096) blocks_x = 4096;
            if (blocks_x < 1) blocks_x = 1;

            dim3 grid_in((unsigned)blocks_x, (unsigned)oc_vecs, (unsigned)N);
            conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc4_interior_smemw<<<grid_in, threads, smem_bytes, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                N, Hin, Win,
                Cout,
                (int)pad_h, (int)pad_w,
                Hout, Wout,
                oh0, oh1, ow0, ow1
            );

            auto launch_border = [&](int ob, int oe, int wb, int we) {
                int rh = oe - ob;
                int rw = we - wb;
                if (rh <= 0 || rw <= 0) return;
                int rect_pix = rh * rw;
                int bx = div_up_int(rect_pix, threads);
                if (bx > 4096) bx = 4096;
                if (bx < 1) bx = 1;
                dim3 grid((unsigned)bx, (unsigned)oc_vecs, (unsigned)N);
                conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc4_border_smemw<<<grid, threads, smem_bytes, stream>>>(
                    (const float*)x.data_ptr<float>(),
                    (const float*)w.data_ptr<float>(),
                    (float*)y.data_ptr<float>(),
                    N, Hin, Win,
                    Cout,
                    (int)pad_h, (int)pad_w,
                    Hout, Wout,
                    ob, oe, wb, we
                );
            };

            // Top, bottom, left, right borders (avoid overlap)
            launch_border(0, oh0, 0, Wout);
            launch_border(oh1, Hout, 0, Wout);
            launch_border(oh0, oh1, 0, ow0);
            launch_border(oh0, oh1, ow1, Wout);
        } else {
            // If interior is empty, border is everything. Use OCx4 border kernel.
            const int threads = 128;
            int rect_pix = Hout * Wout;
            int blocks_x = div_up_int(rect_pix, threads);
            if (blocks_x > 4096) blocks_x = 4096;
            if (blocks_x < 1) blocks_x = 1;
            dim3 grid((unsigned)blocks_x, (unsigned)oc_vecs, (unsigned)N);
            conv2d_forward_nchw_f32_cin32_k5x9_s1d1_oc4_border_smemw<<<grid, threads, smem_bytes, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                N, Hin, Win,
                Cout,
                (int)pad_h, (int)pad_w,
                Hout, Wout,
                0, Hout, 0, Wout
            );
        }
    } else {
        // fallback generic
        int total = N * Cout * Hout * Wout;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        conv2d_forward_nchw_f32_generic<<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Kh, Kw,
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
torch::Tensor conv_standard2d_square_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_standard2d_square_in_asym_k_v2_smemw_oc4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard2d_square_input_asymmetric_kernel_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Custom CUDA forward for Conv2d with asymmetric kernel (e.g., 5x9).
    Assumptions: NCHW float32 CUDA tensors, groups=1, bias=False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
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

        if isinstance(kernel_size, int):
            kh, kw = int(kernel_size), int(kernel_size)
        else:
            kh, kw = int(kernel_size[0]), int(kernel_size[1])

        self.custom_ops_lib = custom_ops_lib
        self.conv2d = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            (kh, kw),
            stride=int(stride),
            padding=padding,
            dilation=dilation,
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

        return self.custom_ops_lib.conv_standard2d_square_input_asymmetric_kernel_cuda(
            x, w, int(sh), int(sw), int(ph), int(pw), int(dh), int(dw)
        )