import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Further-optimized ConvTranspose2d forward (stride=1,pad=0,groups=1,bias=False)
# Main improvement over baseline:
# - Stage weights for current ic and current OC_BLOCK into shared memory (per-CTA)
#   to reduce repeated global weight loads.
# - Keep baseline-correct transposed-conv mapping and indexing.
# - Provide specialized k3x7 kernel with shared-weight staging and unrolled math.
# - Generic fallback also uses shared-weight staging (small: OC_BLOCK*KH*KW).
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

#ifndef __CUDA_ARCH__
#define __device_builtin__ __attribute__((device_builtin))
#define __cudart_builtin__ __attribute__((cudart_builtin))
#define __noinline__ __attribute__((noinline))
#define __forceinline__ __attribute__((forceinline))
#endif

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
static inline float ldg_f32(const float* p) { return *p; }
#endif

// --------------------------------------
// Specialized kernel: KH=3, KW=7
// - Block computes TILE_H x TILE_W pixels and OC_BLOCK output channels.
// - Shared x tile (with halo) per ic.
// - Shared weight slab per ic: [OC_BLOCK, 21] for this CTA's oc_block0.
// --------------------------------------
__global__ void convt2d_k3x7_tiled_sharedw(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cin,Cout,3,7]
    float* __restrict__ out,       // [N,Cout,Hout,Wout]
    int N, int Cin, int H, int W,
    int Cout, int Hout, int Wout
) {
    constexpr int KH = 3;
    constexpr int KW = 7;

    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int OC_BLOCK = 8;

    const int ow0 = (int)blockIdx.x * TILE_W;
    const int oh0 = (int)blockIdx.y * TILE_H;

    const int num_oc_blocks = div_up(Cout, OC_BLOCK);
    const int z = (int)blockIdx.z;
    const int n = z / num_oc_blocks;
    const int oc_block_idx = z - n * num_oc_blocks;
    const int oc_block0 = oc_block_idx * OC_BLOCK;
    if (n >= N) return;

    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int lane = ty * TILE_W + tx; // 0..255

    // shared x tile with halo
    const int SH_W = TILE_W + (KW - 1); // 16+6=22
    const int SH_H = TILE_H + (KH - 1); // 16+2=18
    const int SH_X_SIZE = SH_W * SH_H;

    // shared weights slab: OC_BLOCK * 21
    constexpr int KAREA = KH * KW; // 21
    const int SH_W_SIZE = OC_BLOCK * KAREA;

    extern __shared__ float smem[];
    float* sx = smem;
    float* sw = smem + SH_X_SIZE; // weights follow x tile

    const int ih_base = oh0 - (KH - 1);
    const int iw_base = ow0 - (KW - 1);

    float acc[OC_BLOCK];
    #pragma unroll
    for (int i = 0; i < OC_BLOCK; ++i) acc[i] = 0.f;

    const int x_n_base = n * Cin * H * W;
    const int hw_out = Hout * Wout;
    const int out_n_base = n * Cout * hw_out;

    for (int ic = 0; ic < Cin; ++ic) {
        const int x_base = x_n_base + ic * H * W;

        // 1) Load x tile (with halo) into shared (scalar, simple and correct)
        for (int idx = lane; idx < SH_X_SIZE; idx += 256) {
            int shy = idx / SH_W;
            int shx = idx - shy * SH_W;
            int ih = ih_base + shy;
            int iw = iw_base + shx;
            float val = 0.f;
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                val = ldg_f32(x + x_base + ih * W + iw);
            }
            sx[idx] = val;
        }

        // 2) Load weight slab for this ic and oc_block into shared
        // sw[oci*KAREA + k] where k = kh*KW + kw
        for (int idx = lane; idx < SH_W_SIZE; idx += 256) {
            int oci = idx / KAREA;
            int k = idx - oci * KAREA;
            int oc = oc_block0 + oci;
            float val = 0.f;
            if (oc < Cout) {
                // w index: ((ic*Cout + oc)*KAREA + k)
                val = ldg_f32(w + ((ic * Cout + oc) * KAREA + k));
            }
            sw[idx] = val;
        }

        __syncthreads();

        const int oh = oh0 + ty;
        const int ow = ow0 + tx;
        if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
            const int sh_oy = (KH - 1) + ty; // 2+ty
            const int sh_ox = (KW - 1) + tx; // 6+tx

            #pragma unroll
            for (int kh = 0; kh < KH; ++kh) {
                const int sh_y = sh_oy - kh;
                const int sh_row = sh_y * SH_W;

                // Load 7 taps from shared x
                float xv0 = sx[sh_row + (sh_ox - 0)];
                float xv1 = sx[sh_row + (sh_ox - 1)];
                float xv2 = sx[sh_row + (sh_ox - 2)];
                float xv3 = sx[sh_row + (sh_ox - 3)];
                float xv4 = sx[sh_row + (sh_ox - 4)];
                float xv5 = sx[sh_row + (sh_ox - 5)];
                float xv6 = sx[sh_row + (sh_ox - 6)];

                const int kbase = kh * KW; // 0,7,14

                #pragma unroll
                for (int oci = 0; oci < OC_BLOCK; ++oci) {
                    const float* wptr = sw + oci * KAREA + kbase;
                    float sum = 0.f;
                    sum = fmaf(xv0, wptr[0], sum);
                    sum = fmaf(xv1, wptr[1], sum);
                    sum = fmaf(xv2, wptr[2], sum);
                    sum = fmaf(xv3, wptr[3], sum);
                    sum = fmaf(xv4, wptr[4], sum);
                    sum = fmaf(xv5, wptr[5], sum);
                    sum = fmaf(xv6, wptr[6], sum);
                    acc[oci] += sum;
                }
            }
        }

        __syncthreads();
    }

    // Store results
    const int oh = oh0 + ty;
    const int ow = ow0 + tx;
    if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
        const int pix = oh * Wout + ow;
        #pragma unroll
        for (int oci = 0; oci < OC_BLOCK; ++oci) {
            int oc = oc_block0 + oci;
            if (oc < Cout) {
                out[out_n_base + oc * hw_out + pix] = acc[oci];
            }
        }
    }
}

// --------------------------------------
// Generic kernel: any KH/KW
// - Same tiling for x.
// - Stage weights slab [OC_BLOCK, KH*KW] into shared per ic.
// --------------------------------------
__global__ void convt2d_tiled_generic_sharedw(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cin,Cout,KH,KW]
    float* __restrict__ out,       // [N,Cout,Hout,Wout]
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int Hout, int Wout
) {
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int OC_BLOCK = 4;

    const int ow0 = (int)blockIdx.x * TILE_W;
    const int oh0 = (int)blockIdx.y * TILE_H;

    const int num_oc_blocks = div_up(Cout, OC_BLOCK);
    const int z = (int)blockIdx.z;
    const int n = z / num_oc_blocks;
    const int oc_block_idx = z - n * num_oc_blocks;
    const int oc_block0 = oc_block_idx * OC_BLOCK;
    if (n >= N) return;

    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int lane = ty * TILE_W + tx; // 0..255

    const int SH_W = TILE_W + (KW - 1);
    const int SH_H = TILE_H + (KH - 1);
    const int SH_X_SIZE = SH_W * SH_H;

    const int KAREA = KH * KW;
    const int SH_W_SIZE = OC_BLOCK * KAREA;

    extern __shared__ float smem[];
    float* sx = smem;
    float* sw = smem + SH_X_SIZE;

    const int ih_base = oh0 - (KH - 1);
    const int iw_base = ow0 - (KW - 1);

    float acc[OC_BLOCK];
    #pragma unroll
    for (int i = 0; i < OC_BLOCK; ++i) acc[i] = 0.f;

    const int x_n_base = n * Cin * H * W;
    const int hw_out = Hout * Wout;
    const int out_n_base = n * Cout * hw_out;

    for (int ic = 0; ic < Cin; ++ic) {
        const int x_base = x_n_base + ic * H * W;

        for (int idx = lane; idx < SH_X_SIZE; idx += 256) {
            int shy = idx / SH_W;
            int shx = idx - shy * SH_W;
            int ih = ih_base + shy;
            int iw = iw_base + shx;
            float val = 0.f;
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
                val = ldg_f32(x + x_base + ih * W + iw);
            }
            sx[idx] = val;
        }

        for (int idx = lane; idx < SH_W_SIZE; idx += 256) {
            int oci = idx / KAREA;
            int k = idx - oci * KAREA;
            int oc = oc_block0 + oci;
            float val = 0.f;
            if (oc < Cout) {
                val = ldg_f32(w + ((ic * Cout + oc) * KAREA + k));
            }
            sw[idx] = val;
        }

        __syncthreads();

        const int oh = oh0 + ty;
        const int ow = ow0 + tx;
        if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
            const int sh_oy = (KH - 1) + ty;
            const int sh_ox = (KW - 1) + tx;

            for (int kh = 0; kh < KH; ++kh) {
                const int sh_y = sh_oy - kh;
                const int sh_row = sh_y * SH_W;
                const int kbase = kh * KW;

                for (int kw = 0; kw < KW; ++kw) {
                    float xv = sx[sh_row + (sh_ox - kw)];
                    int k = kbase + kw;

                    #pragma unroll
                    for (int oci = 0; oci < OC_BLOCK; ++oci) {
                        float wf = sw[oci * KAREA + k];
                        acc[oci] = fmaf(xv, wf, acc[oci]);
                    }
                }
            }
        }

        __syncthreads();
    }

    const int oh = oh0 + ty;
    const int ow = ow0 + tx;
    if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
        const int pix = oh * Wout + ow;
        #pragma unroll
        for (int oci = 0; oci < OC_BLOCK; ++oci) {
            int oc = oc_block0 + oci;
            if (oc < Cout) {
                out[out_n_base + oc * hw_out + pix] = acc[oci];
            }
        }
    }
}

torch::Tensor conv_transposed2d_square_input_asymmetric_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be 4D [Cin, Cout, KH, KW]");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int N   = (int)x_c.size(0);
    const int Cin = (int)x_c.size(1);
    const int H   = (int)x_c.size(2);
    const int W   = (int)x_c.size(3);

    TORCH_CHECK((int)w_c.size(0) == Cin, "weight Cin must match input Cin");
    const int Cout = (int)w_c.size(1);
    const int KH   = (int)w_c.size(2);
    const int KW   = (int)w_c.size(3);

    TORCH_CHECK(KH >= 1 && KW >= 1, "invalid kernel size");

    const int Hout = H + KH - 1;
    const int Wout = W + KW - 1;

    auto out = torch::zeros({N, Cout, Hout, Wout}, x_c.options());

    if (KH == 3 && KW == 7) {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 16;
        constexpr int OC_BLOCK = 8;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid(
            div_up(Wout, TILE_W),
            div_up(Hout, TILE_H),
            (unsigned)(N * div_up(Cout, OC_BLOCK))
        );

        const int SH_W = TILE_W + (KW - 1); // 22
        const int SH_H = TILE_H + (KH - 1); // 18
        const int SH_X_SIZE = SH_W * SH_H;  // 396
        const int SH_W_SIZE = OC_BLOCK * (KH * KW); // 8*21=168
        const size_t shmem_bytes = (size_t)(SH_X_SIZE + SH_W_SIZE) * sizeof(float);

        convt2d_k3x7_tiled_sharedw<<<grid, block, shmem_bytes>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            out.data_ptr<float>(),
            N, Cin, H, W,
            Cout, Hout, Wout
        );
    } else {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 16;
        constexpr int OC_BLOCK = 4;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid(
            div_up(Wout, TILE_W),
            div_up(Hout, TILE_H),
            (unsigned)(N * div_up(Cout, OC_BLOCK))
        );

        const int SH_W = TILE_W + (KW - 1);
        const int SH_H = TILE_H + (KH - 1);
        const int SH_X_SIZE = SH_W * SH_H;
        const int SH_W_SIZE = OC_BLOCK * (KH * KW);
        const size_t shmem_bytes = (size_t)(SH_X_SIZE + SH_W_SIZE) * sizeof(float);

        convt2d_tiled_generic_sharedw<<<grid, block, shmem_bytes>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            out.data_ptr<float>(),
            N, Cin, H, W,
            Cout, KH, KW,
            Hout, Wout
        );
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transposed2d_square_input_asymmetric_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_asym_opt_sharedw_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transposed2d_square_input_asymmetric_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Drop-in replacement using an optimized custom CUDA kernel for ConvTranspose2d forward.

    Supported configuration:
    - groups=1, bias=False, stride=1, padding=0, output_padding=0, dilation=1
    - weight layout: [Cin, Cout, KH, KW]
    - float32 CUDA contiguous NCHW
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.custom_ops = custom_ops_lib

        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

        if stride != 1:
            raise ValueError("ModelNew custom kernel supports stride=1 only")
        if padding != 0:
            raise ValueError("ModelNew custom kernel supports padding=0 only")
        if output_padding != 0:
            raise ValueError("ModelNew custom kernel supports output_padding=0 only")
        if groups != 1:
            raise ValueError("ModelNew custom kernel supports groups=1 only")
        if bias:
            raise ValueError("ModelNew custom kernel supports bias=False only")
        if not (isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2):
            raise ValueError("kernel_size must be a 2-tuple")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32):
            return self.conv_transpose2d(x)
        return self.custom_ops.conv_transposed2d_square_input_asymmetric_cuda(
            x, self.conv_transpose2d.weight
        )