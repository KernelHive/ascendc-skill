import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline __host__ __device__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// ------------------------
// Generic kernel (robust)
// ------------------------
template<int VEC>
__global__ void conv_transpose2d_generic_vec(
    const float* __restrict__ x,      // [N,Cin,Hin,Win]
    const float* __restrict__ w,      // [Cin,Cout,kH,kW]
    float* __restrict__ y,            // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int outPadH, int outPadW,
    int dilH, int dilW,
    int Hout, int Wout
) {
    int oh = (int)blockIdx.y;
    int nz = (int)blockIdx.z;
    int oc = nz % Cout;
    int n  = nz / Cout;

    int ow_base = ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x) * VEC;

    float acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    if (n >= N || oh >= Hout) return;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* w_ic_oc = w + (((ic * Cout + oc) * kH) * kW);

        for (int kh = 0; kh < kH; ++kh) {
            int oh_in = oh + padH - kh * dilH;
            if (oh_in < 0) continue;
            if (oh_in % strideH != 0) continue;
            int ih = oh_in / strideH;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            const float* xrow = x + (((n * Cin + ic) * Hin + ih) * Win);

            #pragma unroll
            for (int vi = 0; vi < VEC; ++vi) {
                int ow = ow_base + vi;
                if ((unsigned)ow >= (unsigned)Wout) continue;

                float sum = 0.0f;
                #pragma unroll 1
                for (int kw = 0; kw < kW; ++kw) {
                    int ow_in = ow + padW - kw * dilW;
                    if (ow_in < 0) continue;
                    if (ow_in % strideW != 0) continue;
                    int iw = ow_in / strideW;
                    if ((unsigned)iw >= (unsigned)Win) continue;

                    float xv = LDG(xrow + iw);
                    float wv = LDG(w_ic_oc + kh * kW + kw);
                    sum = fmaf(xv, wv, sum);
                }
                acc[vi] += sum;
            }
        }
    }

    float* yptr = y + (((n * Cout + oc) * Hout + oh) * Wout);
    #pragma unroll
    for (int vi = 0; vi < VEC; ++vi) {
        int ow = ow_base + vi;
        if ((unsigned)ow < (unsigned)Wout) {
            yptr[ow] = acc[vi];
        }
    }
}

// -----------------------------------------
// Specialized vec kernel (kH=3,kW=5,s1,d1)
// -----------------------------------------
template<int VEC>
__global__ void conv_transpose2d_k3k5_s1_d1_vec(
    const float* __restrict__ x,      // [N,Cin,Hin,Win]
    const float* __restrict__ w,      // [Cin,Cout,3,5]
    float* __restrict__ y,            // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout,
    int padH, int padW,
    int Hout, int Wout
) {
    int oh = (int)blockIdx.y;
    int nz = (int)blockIdx.z;
    int oc = nz % Cout;
    int n  = nz / Cout;

    int ow_base = ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x) * VEC;

    if (n >= N || oh >= Hout) return;

    float acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    int t = oh + padH;
    int kh_min = t - (Hin - 1);
    int kh_max = t;
    if (kh_min < 0) kh_min = 0;
    if (kh_max > 2) kh_max = 2;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* w_ic_oc = w + (((ic * Cout + oc) * 3) * 5);

        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            if (kh < kh_min || kh > kh_max) continue;
            int ih = t - kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            const float* xrow = x + (((n * Cin + ic) * Hin + ih) * Win);
            const float* wrow = w_ic_oc + kh * 5;

            #pragma unroll
            for (int vi = 0; vi < VEC; ++vi) {
                int ow = ow_base + vi;
                if ((unsigned)ow >= (unsigned)Wout) continue;

                int base = ow + padW;
                int kw_lo = base - (Win - 1);
                int kw_hi = base;
                if (kw_lo < 0) kw_lo = 0;
                if (kw_hi > 4) kw_hi = 4;

                float sum = 0.0f;

                #pragma unroll
                for (int kw = 0; kw < 5; ++kw) {
                    if (kw < kw_lo || kw > kw_hi) continue;
                    int iw = base - kw;
                    if ((unsigned)iw >= (unsigned)Win) continue;
                    float xv = LDG(xrow + iw);
                    float wv = LDG(wrow + kw);
                    sum = fmaf(xv, wv, sum);
                }
                acc[vi] += sum;
            }
        }
    }

    float* yptr = y + (((n * Cout + oc) * Hout + oh) * Wout);
    #pragma unroll
    for (int vi = 0; vi < VEC; ++vi) {
        int ow = ow_base + vi;
        if ((unsigned)ow < (unsigned)Wout) {
            yptr[ow] = acc[vi];
        }
    }
}

// -------------------------------------------------------
// New fast kernel: kH=3,kW=5,stride=1,dilation=1,outPad=0
// 2D tiled + OC blocking + shared input halo + shared weight block.
// 128-thread CTA (4 warps) to improve latency hiding and reduce reg pressure.
// Each thread computes exactly one output pixel (oh,ow) for OC_BLOCK channels.
// -------------------------------------------------------
template<int TILE_H, int TILE_W, int OC_BLOCK>
__global__ __launch_bounds__(128, 3) void convt2d_k3k5_s1_tiled_ocblock(
    const float* __restrict__ x,  // [N,Cin,Hin,Win]
    const float* __restrict__ w,  // [Cin,Cout,3,5]
    float* __restrict__ y,        // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int pH, int pW
) {
    constexpr int kH = 3;
    constexpr int kW = 5;

    const int oh0 = (int)blockIdx.y * TILE_H;
    const int ow0 = (int)blockIdx.x * TILE_W;

    const int num_oc_blocks = div_up_int(Cout, OC_BLOCK);
    const int z = (int)blockIdx.z;
    const int n = z / num_oc_blocks;
    const int ocb = z - n * num_oc_blocks;
    const int oc0 = ocb * OC_BLOCK;
    if (n >= N) return;

    // shared tile sizes
    constexpr int SH_H = TILE_H + (kH - 1); // +2
    constexpr int SH_W = TILE_W + (kW - 1); // +4
    constexpr int SH_SIZE = SH_H * SH_W;
    constexpr int WBLK = OC_BLOCK * kH * kW;

    extern __shared__ float smem[];
    float* sx = smem;              // SH_SIZE
    float* sw = smem + SH_SIZE;    // WBLK

    // 128 threads = TILE_H*TILE_W (require that at compile time)
    const int tx = (int)threadIdx.x & (TILE_W - 1);
    const int ty = (int)threadIdx.x >> 5; // assumes TILE_W=32, TILE_H=4 => ty in [0..3]
    const int lane = (int)threadIdx.x;

    // output coords for this thread
    const int oh = oh0 + ty;
    const int ow = ow0 + tx;

    float acc[OC_BLOCK];
    #pragma unroll
    for (int oci = 0; oci < OC_BLOCK; ++oci) acc[oci] = 0.f;

    // shared origin in input coordinates for the tile
    const int ih0 = oh0 + pH - (kH - 1); // oh0 + pH - 2
    const int iw0 = ow0 + pW - (kW - 1); // ow0 + pW - 4

    for (int ic = 0; ic < Cin; ++ic) {
        const int x_base = ((n * Cin + ic) * Hin) * Win;

        // load input tile+halo
        for (int idx = lane; idx < SH_SIZE; idx += 128) {
            int shy = idx / SH_W;
            int shx = idx - shy * SH_W;
            int ih = ih0 + shy;
            int iw = iw0 + shx;
            float v = 0.f;
            if ((unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
                v = LDG(x + x_base + ih * Win + iw);
            }
            sx[idx] = v;
        }

        // load weights for this ic and oc-block
        for (int idx = lane; idx < WBLK; idx += 128) {
            int tmp = idx;
            int kw = tmp % kW; tmp /= kW;
            int kh = tmp % kH; tmp /= kH;
            int oci = tmp;
            int oc = oc0 + oci;
            float v = 0.f;
            if (oc < Cout) {
                int w_off = ((ic * Cout + oc) * (kH * kW)) + kh * kW + kw;
                v = LDG(w + w_off);
            }
            sw[idx] = v;
        }

        __syncthreads();

        if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
            const int sh_y = (oh + pH) - ih0;
            const int sh_x = (ow + pW) - iw0;

            // fixed 3x5
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                const int row = (sh_y - kh) * SH_W;

                float x0 = sx[row + (sh_x - 0)];
                float x1 = sx[row + (sh_x - 1)];
                float x2 = sx[row + (sh_x - 2)];
                float x3 = sx[row + (sh_x - 3)];
                float x4 = sx[row + (sh_x - 4)];

                #pragma unroll
                for (int oci = 0; oci < OC_BLOCK; ++oci) {
                    const float* wrow = sw + ((oci * 3 + kh) * 5);
                    float sum = 0.f;
                    sum = fmaf(x0, wrow[0], sum);
                    sum = fmaf(x1, wrow[1], sum);
                    sum = fmaf(x2, wrow[2], sum);
                    sum = fmaf(x3, wrow[3], sum);
                    sum = fmaf(x4, wrow[4], sum);
                    acc[oci] += sum;
                }
            }
        }

        __syncthreads();
    }

    if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
        const int hw = Hout * Wout;
        const int out_base = (n * Cout * hw) + (oh * Wout + ow);
        #pragma unroll
        for (int oci = 0; oci < OC_BLOCK; ++oci) {
            int oc = oc0 + oci;
            if (oc < Cout) {
                y[out_base + oc * hw] = acc[oci];
            }
        }
    }
}

torch::Tensor conv_transposed2d_asymmetric_input_asymmetric_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW,
    int64_t outPadH, int64_t outPadW,
    int64_t dilH, int64_t dilW
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cin,Cout,kH,kW]");

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Hin = x.size(2);
    const int64_t Win = x.size(3);

    TORCH_CHECK(w.size(0) == Cin, "w.size(0) must equal Cin");
    const int64_t Cout = w.size(1);
    const int64_t kH = w.size(2);
    const int64_t kW = w.size(3);

    const int64_t Hout = (Hin - 1) * strideH - 2 * padH + dilH * (kH - 1) + outPadH + 1;
    const int64_t Wout = (Win - 1) * strideW - 2 * padW + dilW * (kW - 1) + outPadW + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Invalid output size");

    auto y = torch::zeros({N, Cout, Hout, Wout}, x.options());

    const bool fast =
        (strideH == 1 && strideW == 1 &&
         dilH == 1 && dilW == 1 &&
         outPadH == 0 && outPadW == 0 &&
         kH == 3 && kW == 5);

    if (fast) {
        // tuned for TILE_W=32, TILE_H=4 so 128 threads = 4 warps
        constexpr int TILE_H = 4;
        constexpr int TILE_W = 32;
        constexpr int OC_BLOCK = 4;

        dim3 block(128, 1, 1);
        dim3 grid(
            (unsigned)div_up_int((int)Wout, TILE_W),
            (unsigned)div_up_int((int)Hout, TILE_H),
            (unsigned)((int)N * div_up_int((int)Cout, OC_BLOCK))
        );

        constexpr int SH_H = TILE_H + 2;
        constexpr int SH_W = TILE_W + 4;
        constexpr int SH_SIZE = SH_H * SH_W;
        constexpr int WBLK = OC_BLOCK * 3 * 5;
        size_t shmem_bytes = (size_t)(SH_SIZE + WBLK) * sizeof(float);

        convt2d_k3k5_s1_tiled_ocblock<TILE_H, TILE_W, OC_BLOCK>
            <<<grid, block, shmem_bytes>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (int)N, (int)Cin, (int)Hin, (int)Win,
                (int)Cout, (int)Hout, (int)Wout,
                (int)padH, (int)padW
            );
        return y;
    }

    // Fallbacks
    constexpr int VEC = 2;
    constexpr int THREADS = 256;

    int tilesW = (int)(((int)Wout + (THREADS * VEC) - 1) / (THREADS * VEC));
    dim3 block(THREADS, 1, 1);
    dim3 grid(tilesW, (unsigned)Hout, (unsigned)(N * Cout));

    const bool vec_fast =
        (strideH == 1 && strideW == 1 &&
         dilH == 1 && dilW == 1 &&
         outPadH == 0 && outPadW == 0 &&
         kH == 3 && kW == 5);

    if (vec_fast) {
        conv_transpose2d_k3k5_s1_d1_vec<VEC><<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout,
            (int)padH, (int)padW,
            (int)Hout, (int)Wout
        );
    } else {
        conv_transpose2d_generic_vec<VEC><<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout, (int)kH, (int)kW,
            (int)strideH, (int)strideW,
            (int)padH, (int)padW,
            (int)outPadH, (int)outPadW,
            (int)dilH, (int)dilW,
            (int)Hout, (int)Wout
        );
    }

    return y;
}
"""

cpp_source = r"""
torch::Tensor conv_transposed2d_asymmetric_input_asymmetric_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW,
    int64_t outPadH, int64_t outPadW,
    int64_t dilH, int64_t dilW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_asym_opt_tiled_ocblock_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transposed2d_asymmetric_input_asymmetric_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose2d replaced by a custom CUDA forward kernel.
    Supports groups=1 and bias=False only (falls back otherwise).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.custom_ops = custom_ops_lib
        self._supported = (groups == 1 and (not bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self._supported) or (not x.is_cuda) or (x.dtype != torch.float32):
            return self.conv_transpose2d(x)

        x_c = x.contiguous()
        w_c = self.conv_transpose2d.weight.contiguous()

        strideH, strideW = self.conv_transpose2d.stride
        padH, padW = self.conv_transpose2d.padding
        outPadH, outPadW = self.conv_transpose2d.output_padding
        dilH, dilW = self.conv_transpose2d.dilation

        return self.custom_ops.conv_transposed2d_asymmetric_input_asymmetric_cuda(
            x_c, w_c,
            int(strideH), int(strideW),
            int(padH), int(padW),
            int(outPadH), int(outPadW),
            int(dilH), int(dilW),
        )