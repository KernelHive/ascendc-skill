import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// Output size for this specialization: stride=1, pad=0, k=3 => Dout=Din+2, etc.

enum FaceType : int {
    FACE_ZLOW  = 0,
    FACE_ZHIGH = 1,
    FACE_YLOW  = 2,
    FACE_YHIGH = 3,
    FACE_XLOW  = 4,
    FACE_XHIGH = 5
};

// -----------------------------------------------------------------------------
// Interior kernel (kept from v4; shared-memory weight staging, computes only interior)
// -----------------------------------------------------------------------------
__device__ __forceinline__ int64_t idx_ncdhw(
    int n, int c, int d, int h, int w,
    int C, int D, int H, int W
) {
    return (((((int64_t)n * C + c) * D + d) * H + h) * W + w);
}

__global__ __launch_bounds__(128, 4)
void conv_t3d_k3s1p0_g1_interior_co2_smemw(
    const float* __restrict__ x,  // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w,  // [Cin,Cout,3,3,3]
    float* __restrict__ y,        // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout
) {
    const int Dout = Din + 2;
    const int Hout = Hin + 2;
    const int Wout = Win + 2;

    const int64_t x_hw  = (int64_t)Hin * Win;
    const int64_t x_dhw = (int64_t)Din * x_hw;
    const int64_t y_hw  = (int64_t)Hout * Wout;
    const int64_t y_dhw = (int64_t)Dout * y_hw;

    const int od0 = 2, od1 = Din - 1;
    const int oh0 = 2, oh1 = Hin - 1;
    const int ow0 = 2, ow1 = Win - 1;
    const int Dint = od1 - od0 + 1;
    const int Hint = oh1 - oh0 + 1;
    const int Wint = ow1 - ow0 + 1;

    if (Dint <= 0 || Hint <= 0 || Wint <= 0) return;

    const int co_pairs = (Cout + 1) >> 1;
    const int64_t spatial_int = (int64_t)Dint * Hint * Wint;

    int co_pair = (int)(blockIdx.x % co_pairs);
    int co0 = co_pair * 2;
    int co1 = co0 + 1;
    bool v1 = (co1 < Cout);

    extern __shared__ float smem[];
    float* sm_w0 = smem;                       // Cin*27
    float* sm_w1 = smem + (int64_t)Cin * 27;   // Cin*27

    {
        int64_t elems = (int64_t)Cin * 27;
        int64_t total_elems = v1 ? (elems * 2) : elems;

        for (int64_t i = (int64_t)threadIdx.x; i < total_elems; i += (int64_t)blockDim.x) {
            if (i < elems) {
                int64_t t = i;
                int ci = (int)(t / 27);
                int r  = (int)(t - (int64_t)ci * 27);
                sm_w0[i] = LDG(w + ((int64_t)ci * Cout + co0) * 27 + r);
            } else {
                int64_t j = i - elems;
                int64_t t = j;
                int ci = (int)(t / 27);
                int r  = (int)(t - (int64_t)ci * 27);
                sm_w1[j] = LDG(w + ((int64_t)ci * Cout + co1) * 27 + r);
            }
        }
    }
    __syncthreads();

    int blk_group = (int)(blockIdx.x / co_pairs);
    int grid_groups = (int)((gridDim.x + co_pairs - 1) / co_pairs);

    for (int64_t linear_group = (int64_t)blk_group * blockDim.x + threadIdx.x;
         linear_group < (int64_t)N * spatial_int;
         linear_group += (int64_t)blockDim.x * grid_groups) {

        int64_t t = linear_group;
        int iw = (int)(t % Wint); t /= Wint;
        int ih = (int)(t % Hint); t /= Hint;
        int id = (int)(t % Dint); t /= Dint;
        int n  = (int)t;

        const int ow = iw + ow0;
        const int oh = ih + oh0;
        const int od = id + od0;

        float acc0 = 0.0f, acc1 = 0.0f;

        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_base = x + ((int64_t)n * Cin + ci) * x_dhw;
            const float* __restrict__ w0_base = sm_w0 + (int64_t)ci * 27;
            const float* __restrict__ w1_base = v1 ? (sm_w1 + (int64_t)ci * 27) : nullptr;

            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                const int id_in = od - kd;
                const float* __restrict__ x_d = x_base + (int64_t)id_in * x_hw;
                const float* __restrict__ w0_kd = w0_base + kd * 9;
                const float* __restrict__ w1_kd = v1 ? (w1_base + kd * 9) : nullptr;

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    const int ih_in = oh - kh;
                    const float* __restrict__ x_dh = x_d + (int64_t)ih_in * Win;
                    const float* __restrict__ w0_kh = w0_kd + kh * 3;
                    const float* __restrict__ w1_kh = v1 ? (w1_kd + kh * 3) : nullptr;

                    float xv0 = LDG(x_dh + (ow - 0));
                    float xv1 = LDG(x_dh + (ow - 1));
                    float xv2 = LDG(x_dh + (ow - 2));

                    float ww0 = w0_kh[0];
                    float ww1 = w0_kh[1];
                    float ww2 = w0_kh[2];

                    acc0 = fmaf(xv0, ww0, acc0);
                    acc0 = fmaf(xv1, ww1, acc0);
                    acc0 = fmaf(xv2, ww2, acc0);

                    if (v1) {
                        float vv0 = w1_kh[0];
                        float vv1 = w1_kh[1];
                        float vv2 = w1_kh[2];

                        acc1 = fmaf(xv0, vv0, acc1);
                        acc1 = fmaf(xv1, vv1, acc1);
                        acc1 = fmaf(xv2, vv2, acc1);
                    }
                }
            }
        }

        const int64_t out0 = idx_ncdhw(n, co0, od, oh, ow, Cout, Dout, Hout, Wout);
        y[out0] = acc0;
        if (v1) y[out0 + y_dhw] = acc1;
    }
}

// -----------------------------------------------------------------------------
// Face kernels: disjoint border coverage, no interior-skip branch.
// Each block handles one (n, co_pair) (or (n, co_pair) for pairs), and one FaceType.
// We stage weights (co0,co1) into shared memory once per block.
// -----------------------------------------------------------------------------

template<int FACE, bool EVEN_COUT>
__global__ __launch_bounds__(256, 2)
void conv_t3d_k3s1p0_g1_face_pairs_smemw(
    const float* __restrict__ x,  // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w,  // [Cin,Cout,3,3,3]
    float* __restrict__ y,        // [N,Cout,Dout,Hout,Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout
) {
    const int Dout = Din + 2;
    const int Hout = Hin + 2;
    const int Wout = Win + 2;

    // Interior ranges in output
    const int od0 = 2, od1 = Din - 1;
    const int oh0 = 2, oh1 = Hin - 1;
    const int ow0 = 2, ow1 = Win - 1;

    // Derive face dimensions, ensuring disjoint partition:
    // Z faces cover all (oh,ow).
    // Y faces exclude Z-face regions (od in interior only), cover all ow.
    // X faces exclude Z and Y face regions (od,oh in interior only).
    int od_start=0, od_end=0, oh_start=0, oh_end=0, ow_start=0, ow_end=0;

    if constexpr (FACE == FACE_ZLOW) {
        if (Dout <= 0) return;
        od_start = 0; od_end = min(1, Dout-1);
        oh_start = 0; oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if constexpr (FACE == FACE_ZHIGH) {
        if (Dout <= 0) return;
        od_start = max(Dout-2, 0); od_end = Dout-1;
        oh_start = 0; oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if constexpr (FACE == FACE_YLOW) {
        // only if interior depth exists
        if (!(Din >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = 0; oh_end = min(1, Hout-1);
        ow_start = 0; ow_end = Wout-1;
    } else if constexpr (FACE == FACE_YHIGH) {
        if (!(Din >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = max(Hout-2, 0); oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if constexpr (FACE == FACE_XLOW) {
        if (!(Din >= 3 && Hin >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = oh0; oh_end = oh1;
        ow_start = 0; ow_end = min(1, Wout-1);
    } else { // FACE_XHIGH
        if (!(Din >= 3 && Hin >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = oh0; oh_end = oh1;
        ow_start = max(Wout-2, 0); ow_end = Wout-1;
    }

    const int faceD = od_end - od_start + 1;
    const int faceH = oh_end - oh_start + 1;
    const int faceW = ow_end - ow_start + 1;
    if (faceD <= 0 || faceH <= 0 || faceW <= 0) return;

    const int co_pairs = EVEN_COUT ? (Cout >> 1) : ((Cout + 1) >> 1);

    // grid: x = N * co_pairs, y unused (face fixed per launch)
    int blk = (int)blockIdx.x;
    int n = blk / co_pairs;
    int co_pair = blk - n * co_pairs;
    if ((unsigned)n >= (unsigned)N) return;

    int co0 = co_pair * 2;
    int co1 = co0 + 1;
    if constexpr (EVEN_COUT) {
        // co1 always valid
    } else {
        if (co0 >= Cout) return;
    }
    bool v1 = (co1 < Cout);

    // stage weights for this co_pair
    extern __shared__ float smem[];
    float* sm_w0 = smem;
    float* sm_w1 = smem + (int64_t)Cin * 27;

    {
        int64_t elems = (int64_t)Cin * 27;
        int64_t total_elems = (EVEN_COUT ? (elems * 2) : (v1 ? (elems * 2) : elems));
        for (int64_t i = (int64_t)threadIdx.x; i < total_elems; i += (int64_t)blockDim.x) {
            if (i < elems) {
                int64_t t = i;
                int ci = (int)(t / 27);
                int r  = (int)(t - (int64_t)ci * 27);
                sm_w0[i] = LDG(w + ((int64_t)ci * Cout + co0) * 27 + r);
            } else {
                int64_t j = i - elems;
                int64_t t = j;
                int ci = (int)(t / 27);
                int r  = (int)(t - (int64_t)ci * 27);
                sm_w1[j] = LDG(w + ((int64_t)ci * Cout + co1) * 27 + r);
            }
        }
    }
    __syncthreads();

    const int64_t x_hw  = (int64_t)Hin * Win;
    const int64_t x_dhw = (int64_t)Din * x_hw;
    const int64_t y_hw  = (int64_t)Hout * Wout;
    const int64_t y_dhw = (int64_t)Dout * y_hw;

    const int64_t face_spatial = (int64_t)faceD * faceH * faceW;

    // iterate face points
    for (int64_t s = (int64_t)threadIdx.x; s < face_spatial; s += (int64_t)blockDim.x) {
        int64_t t = s;
        int ow = (int)(t % faceW); t /= faceW;
        int oh = (int)(t % faceH); t /= faceH;
        int od = (int)t;

        ow += ow_start;
        oh += oh_start;
        od += od_start;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        // bounds for (kd,kh,kw) given od/oh/ow; same as border logic but with fixed k=3
        int kd_min = od - (Din - 1); if (kd_min < 0) kd_min = 0;
        int kd_max = od;            if (kd_max > 2) kd_max = 2;

        int kh_min = oh - (Hin - 1); if (kh_min < 0) kh_min = 0;
        int kh_max = oh;             if (kh_max > 2) kh_max = 2;

        int kw_min = ow - (Win - 1); if (kw_min < 0) kw_min = 0;
        int kw_max = ow;             if (kw_max > 2) kw_max = 2;

        #pragma unroll 1
        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_base = x + ((int64_t)n * Cin + ci) * x_dhw;
            const float* __restrict__ w0_base = sm_w0 + (int64_t)ci * 27;
            const float* __restrict__ w1_base = sm_w1 + (int64_t)ci * 27;

            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                if (kd < kd_min || kd > kd_max) continue;
                int id_in = od - kd;
                const float* __restrict__ x_d = x_base + (int64_t)id_in * x_hw;
                const float* __restrict__ w0_kd = w0_base + kd * 9;
                const float* __restrict__ w1_kd = w1_base + kd * 9;

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    if (kh < kh_min || kh > kh_max) continue;
                    int ih_in = oh - kh;
                    const float* __restrict__ x_dh = x_d + (int64_t)ih_in * Win;
                    const float* __restrict__ w0_kh = w0_kd + kh * 3;
                    const float* __restrict__ w1_kh = w1_kd + kh * 3;

                    // kw loop (small), keep branch checks (only border points)
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        if (kw < kw_min || kw > kw_max) continue;
                        int iw_in = ow - kw;
                        float xv = LDG(x_dh + iw_in);
                        acc0 = fmaf(xv, w0_kh[kw], acc0);
                        if constexpr (EVEN_COUT) {
                            acc1 = fmaf(xv, w1_kh[kw], acc1);
                        } else {
                            if (v1) acc1 = fmaf(xv, w1_kh[kw], acc1);
                        }
                    }
                }
            }
        }

        const int64_t out0 = idx_ncdhw(n, co0, od, oh, ow, Cout, Dout, Hout, Wout);
        y[out0] = acc0;
        if constexpr (EVEN_COUT) {
            y[out0 + y_dhw] = acc1;
        } else {
            if (v1) y[out0 + y_dhw] = acc1;
        }
    }
}

__global__ __launch_bounds__(256, 2)
void conv_t3d_k3s1p0_g1_face_tail_lastc(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int N, int Cin, int Din, int Hin, int Win,
    int Cout,
    int face // 0..5
) {
    // Computes only co = Cout-1 across the given face, for odd Cout tail.
    const int co = Cout - 1;
    if (co < 0) return;

    const int Dout = Din + 2;
    const int Hout = Hin + 2;
    const int Wout = Win + 2;

    const int od0 = 2, od1 = Din - 1;
    const int oh0 = 2, oh1 = Hin - 1;

    int od_start=0, od_end=0, oh_start=0, oh_end=0, ow_start=0, ow_end=0;
    if (face == FACE_ZLOW) {
        od_start = 0; od_end = min(1, Dout-1);
        oh_start = 0; oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if (face == FACE_ZHIGH) {
        od_start = max(Dout-2, 0); od_end = Dout-1;
        oh_start = 0; oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if (face == FACE_YLOW) {
        if (!(Din >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = 0; oh_end = min(1, Hout-1);
        ow_start = 0; ow_end = Wout-1;
    } else if (face == FACE_YHIGH) {
        if (!(Din >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = max(Hout-2, 0); oh_end = Hout-1;
        ow_start = 0; ow_end = Wout-1;
    } else if (face == FACE_XLOW) {
        if (!(Din >= 3 && Hin >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = oh0; oh_end = oh1;
        ow_start = 0; ow_end = min(1, Wout-1);
    } else {
        if (!(Din >= 3 && Hin >= 3)) return;
        od_start = od0; od_end = od1;
        oh_start = oh0; oh_end = oh1;
        ow_start = max(Wout-2, 0); ow_end = Wout-1;
    }

    const int faceD = od_end - od_start + 1;
    const int faceH = oh_end - oh_start + 1;
    const int faceW = ow_end - ow_start + 1;
    if (faceD <= 0 || faceH <= 0 || faceW <= 0) return;

    int n = (int)blockIdx.x;
    if ((unsigned)n >= (unsigned)N) return;

    const int64_t x_hw  = (int64_t)Hin * Win;
    const int64_t x_dhw = (int64_t)Din * x_hw;
    const int64_t y_hw  = (int64_t)Hout * Wout;
    const int64_t spatial = (int64_t)Dout * y_hw;

    const int64_t face_spatial = (int64_t)faceD * faceH * faceW;
    const int64_t out_base = ((int64_t)n * Cout + co) * spatial;

    for (int64_t s = (int64_t)threadIdx.x; s < face_spatial; s += (int64_t)blockDim.x) {
        int64_t t = s;
        int ow = (int)(t % faceW); t /= faceW;
        int oh = (int)(t % faceH); t /= faceH;
        int od = (int)t;

        ow += ow_start;
        oh += oh_start;
        od += od_start;

        float acc = 0.0f;

        int kd_min = od - (Din - 1); if (kd_min < 0) kd_min = 0;
        int kd_max = od;            if (kd_max > 2) kd_max = 2;

        int kh_min = oh - (Hin - 1); if (kh_min < 0) kh_min = 0;
        int kh_max = oh;             if (kh_max > 2) kh_max = 2;

        int kw_min = ow - (Win - 1); if (kw_min < 0) kw_min = 0;
        int kw_max = ow;             if (kw_max > 2) kw_max = 2;

        for (int ci = 0; ci < Cin; ++ci) {
            const float* __restrict__ x_base = x + ((int64_t)n * Cin + ci) * x_dhw;
            const float* __restrict__ w_base = w + ((int64_t)ci * Cout + co) * 27;

            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                if (kd < kd_min || kd > kd_max) continue;
                int id_in = od - kd;
                const float* __restrict__ x_d = x_base + (int64_t)id_in * x_hw;
                const float* __restrict__ w_kd = w_base + kd * 9;

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    if (kh < kh_min || kh > kh_max) continue;
                    int ih_in = oh - kh;
                    const float* __restrict__ x_dh = x_d + (int64_t)ih_in * Win;
                    const float* __restrict__ w_kh = w_kd + kh * 3;

                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        if (kw < kw_min || kw > kw_max) continue;
                        int iw_in = ow - kw;
                        float xv = LDG(x_dh + iw_in);
                        acc = fmaf(xv, LDG(w_kh + kw), acc);
                    }
                }
            }
        }

        int64_t out_idx = out_base + ((int64_t)od * y_hw + (int64_t)oh * Wout + ow);
        y[out_idx] = acc;
    }
}

torch::Tensor conv_transpose3d_asymmetric_input_square_cuda(torch::Tensor x, torch::Tensor w) {
    CHECK_CUDA(x);
    CHECK_CUDA(w);
    CHECK_FLOAT(x);
    CHECK_FLOAT(w);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w);

    TORCH_CHECK(x.dim() == 5, "x must be 5D NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be 5D [Cin,Cout,3,3,3]");

    int64_t N   = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Din = x.size(2);
    int64_t Hin = x.size(3);
    int64_t Win = x.size(4);

    TORCH_CHECK(w.size(0) == Cin, "w.size(0) must match Cin");
    int64_t Cout = w.size(1);
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3 && w.size(4) == 3, "Only kernel_size=3 supported");

    int64_t Dout = Din + 2;
    int64_t Hout = Hin + 2;
    int64_t Wout = Win + 2;

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());

    int device = x.get_device();
    cudaDeviceProp prop;
    int sm_count = 80;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) sm_count = prop.multiProcessorCount;

    // -------- Interior launch --------
    if (Din >= 3 && Hin >= 3 && Win >= 3) {
        const int threads_int = 128;
        const int64_t spatial_int = (int64_t)(Din - 2) * (Hin - 2) * (Win - 2);
        const int co_pairs = (int)((Cout + 1) >> 1);

        int max_blocks = sm_count * 20;
        int64_t work_items = (int64_t)N * spatial_int;
        int groups = (int)((work_items + threads_int - 1) / threads_int);
        if (groups < 1) groups = 1;
        if (groups > max_blocks) groups = max_blocks;
        int blocks_int = groups * co_pairs;
        if (blocks_int < co_pairs) blocks_int = co_pairs;
        if (blocks_int > max_blocks) blocks_int = max_blocks;

        size_t smem_bytes = (size_t)2 * (size_t)Cin * 27 * sizeof(float);

        conv_t3d_k3s1p0_g1_interior_co2_smemw<<<blocks_int, threads_int, smem_bytes>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
            (int)Cout
        );
    }

    // -------- Face launches (disjoint border coverage) --------
    const int threads_face = 256;
    size_t smem_face = (size_t)2 * (size_t)Cin * 27 * sizeof(float);

    const bool even_cout = ((Cout & 1) == 0);
    const int co_pairs_even = (int)(Cout >> 1);
    const int co_pairs_all  = (int)((Cout + 1) >> 1);

    // Helper lambda-ish: launch for a face
    auto launch_face_even = [&](int face) {
        int blocks = (int)(N * co_pairs_even);
        if (blocks <= 0) return;
        switch(face) {
            case FACE_ZLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_ZLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
            case FACE_ZHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_ZHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
            case FACE_YLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_YLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
            case FACE_YHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_YHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
            case FACE_XLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_XLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
            case FACE_XHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_XHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout
                ); break;
        }
    };

    auto launch_face_oddpairs = [&](int face, int cout_for_pairs) {
        // cout_for_pairs is Cout-1 (even) so we can use EVEN_COUT=true kernel over pairs
        int blocks = (int)(N * (cout_for_pairs >> 1));
        if (blocks <= 0) return;
        switch(face) {
            case FACE_ZLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_ZLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
            case FACE_ZHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_ZHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
            case FACE_YLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_YLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
            case FACE_YHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_YHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
            case FACE_XLOW:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_XLOW, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
            case FACE_XHIGH:
                conv_t3d_k3s1p0_g1_face_pairs_smemw<FACE_XHIGH, true><<<blocks, threads_face, smem_face>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                    (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)cout_for_pairs
                ); break;
        }
    };

    if (even_cout) {
        // pairs for all Cout
        launch_face_even(FACE_ZLOW);
        launch_face_even(FACE_ZHIGH);
        launch_face_even(FACE_YLOW);
        launch_face_even(FACE_YHIGH);
        launch_face_even(FACE_XLOW);
        launch_face_even(FACE_XHIGH);
    } else {
        // run pairs on Cout-1 (even), then tail last channel per face
        int cout_pairs = (int)(Cout - 1);
        if (cout_pairs > 0) {
            launch_face_oddpairs(FACE_ZLOW,  cout_pairs);
            launch_face_oddpairs(FACE_ZHIGH, cout_pairs);
            launch_face_oddpairs(FACE_YLOW,  cout_pairs);
            launch_face_oddpairs(FACE_YHIGH, cout_pairs);
            launch_face_oddpairs(FACE_XLOW,  cout_pairs);
            launch_face_oddpairs(FACE_XHIGH, cout_pairs);
        }
        // tail channel, one block per n per face
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_ZLOW
        );
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_ZHIGH
        );
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_YLOW
        );
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_YHIGH
        );
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_XLOW
        );
        conv_t3d_k3s1p0_g1_face_tail_lastc<<<(int)N, threads_face>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win, (int)Cout, FACE_XHIGH
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose3d_asymmetric_input_square_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transposed3d_asym_input_square_v5_faces",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_asymmetric_input_square_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-maxrregcount=64",
    ],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Forward-only replacement for nn.ConvTranspose3d specialized to:
      kernel_size=(3,3,3), stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if int(kernel_size) != 3:
            raise ValueError("Custom kernel supports only kernel_size=3")
        if stride != 1:
            raise ValueError("Custom kernel supports only stride=1")
        if padding != 0:
            raise ValueError("Custom kernel supports only padding=0")
        if output_padding != 0:
            raise ValueError("Custom kernel supports only output_padding=0")
        if dilation != 1:
            raise ValueError("Custom kernel supports only dilation=1")
        if groups != 1:
            raise ValueError("Custom kernel supports only groups=1")
        if bias:
            raise ValueError("Custom kernel supports only bias=False")

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, 3, 3, 3, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.conv_transpose3d_asymmetric_input_square_cuda(
            x.contiguous(), self.weight.contiguous()
        )