import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CUDA_CHECK(err) do {                             \
  cudaError_t err__ = (err);                             \
  if (err__ != cudaSuccess) {                            \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
           cudaGetErrorString(err__));                   \
  }                                                      \
} while(0)

static inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

// Specialized constant weights for Cin=Cout=32, K=3
__constant__ float W_CONST_32_32_K3[32 * 32 * 3 * 3];

static inline void copy_w_to_const_32_32_k3(torch::Tensor w_c) {
    const size_t bytes = 32ull * 32ull * 3ull * 3ull * sizeof(float);
    CUDA_CHECK(cudaMemcpyToSymbol(W_CONST_32_32_K3, w_c.data_ptr<float>(), bytes, 0, cudaMemcpyDeviceToDevice));
}

// Kernel: keep 16x16 tile, full Cin slab in shared once.
// Key change vs baseline:
// - keep effective OC_BLOCK=8 per blockIdx.z like baseline, BUT compute in two sequential OC4 passes
//   to reduce per-thread live accumulator registers (4 instead of 8 in hot loop).
// - add modest shared-memory prefetching to increase ILP.
// - improve slab load with float4 when possible (falls back to scalar on tail).
__global__ __launch_bounds__(256, 2)
void convT2d_k3_c32_slab_oc8_twopass4(
    const float* __restrict__ x,  // [N,32,Hin,Win]
    float* __restrict__ y,        // [N,32,Hout,Wout]
    int Hin, int Win,
    int Hout, int Wout
) {
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int K = 3;
    constexpr int HALO = K - 1; // 2
    constexpr int SH_W = TILE_W + HALO; // 18
    constexpr int SH_H = TILE_H + HALO; // 18
    constexpr int SH_HW = SH_H * SH_W;  // 324
    constexpr int CIN = 32;
    constexpr int OC_BLOCK = 8;
    constexpr int OC_PASS = 4;

    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int lane = ty * TILE_W + tx; // 0..255

    const int tile_ow0 = (int)blockIdx.x * TILE_W;
    const int tile_oh0 = (int)blockIdx.y * TILE_H;

    // blockIdx.z packs (n, oc_block_of_8)
    const int oc_blocks = div_up(32, OC_BLOCK); // =4
    const int z = (int)blockIdx.z;
    const int n = z / oc_blocks;
    const int ocb = z - n * oc_blocks;
    const int oc0 = ocb * OC_BLOCK;

    extern __shared__ float smem[];
    float* sx = smem;

    const int oh = tile_oh0 + ty;
    const int ow = tile_ow0 + tx;

    const int x_n_base = n * CIN * Hin * Win;
    const int ih_base = tile_oh0 - HALO;
    const int iw_base = tile_ow0 - HALO;

    // -------- Load full slab once into shared --------
    // Flattened slab size
    const int slab_elems = CIN * SH_HW;   // 10368
    const int slab_elems4 = slab_elems & ~3;

    // Vectorized float4 path when aligned
    // We'll interpret sx and global x as scalar; vectorize only in shared linear index domain.
    for (int s = lane * 4; s < slab_elems4; s += 256 * 4) {
        // load 4 consecutive slab entries; map each to (ic,shy,shx)
        float vals[4];

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int ss = s + v;
            const int ic = ss / SH_HW;
            const int rem = ss - ic * SH_HW;
            const int shy = rem / SH_W;
            const int shx = rem - shy * SH_W;
            const int ih = ih_base + shy;
            const int iw = iw_base + shx;

            float val = 0.0f;
            if ((unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
                val = __ldg(x + x_n_base + ic * Hin * Win + ih * Win + iw);
            }
            vals[v] = val;
        }

        // Store 4 consecutive to shared; alignment is OK for sx as float array for any index,
        // but float4 store requires address aligned to 16 bytes. We guard it.
        float* dst = sx + s;
        if ((((uintptr_t)dst) & 0xF) == 0) {
            *reinterpret_cast<float4*>(dst) = make_float4(vals[0], vals[1], vals[2], vals[3]);
        } else {
            dst[0] = vals[0]; dst[1] = vals[1]; dst[2] = vals[2]; dst[3] = vals[3];
        }
    }
    // Tail (at most 3 elements)
    for (int s = slab_elems4 + lane; s < slab_elems; s += 256) {
        const int ic = s / SH_HW;
        const int rem = s - ic * SH_HW;
        const int shy = rem / SH_W;
        const int shx = rem - shy * SH_W;
        const int ih = ih_base + shy;
        const int iw = iw_base + shx;

        float val = 0.0f;
        if ((unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
            val = __ldg(x + x_n_base + ic * Hin * Win + ih * Win + iw);
        }
        sx[s] = val;
    }

    __syncthreads();

    if (!((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout)) return;

    const int sh_oy = (oh - ih_base); // ty+2
    const int sh_ox = (ow - iw_base); // tx+2
    const int spatial = oh * Wout + ow;
    const int y_n_base = n * 32 * Hout * Wout;

    // Compute in two passes of 4 output channels each, to reduce registers.
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int oc_pass0 = oc0 + pass * OC_PASS;

        float acc[OC_PASS];
        #pragma unroll
        for (int i = 0; i < OC_PASS; ++i) acc[i] = 0.0f;

        // Iterate input channels
        #pragma unroll
        for (int ic = 0; ic < CIN; ++ic) {
            const float* sxi = sx + ic * SH_HW;

            // Prefetch for kh=0
            int sh_row0 = (sh_oy - 0) * SH_W;
            float x0_0 = sxi[sh_row0 + (sh_ox - 0)];
            float x0_1 = sxi[sh_row0 + (sh_ox - 1)];
            float x0_2 = sxi[sh_row0 + (sh_ox - 2)];

            // kh=0
            {
                const int w_kh_base = 0;
                #pragma unroll
                for (int oci = 0; oci < OC_PASS; ++oci) {
                    const int oc = oc_pass0 + oci;
                    const int w_base = (ic * 32 + oc) * 9 + w_kh_base;
                    float sum = acc[oci];
                    sum = fmaf(x0_0, W_CONST_32_32_K3[w_base + 0], sum);
                    sum = fmaf(x0_1, W_CONST_32_32_K3[w_base + 1], sum);
                    sum = fmaf(x0_2, W_CONST_32_32_K3[w_base + 2], sum);
                    acc[oci] = sum;
                }
            }

            // kh=1 prefetch then compute
            int sh_row1 = (sh_oy - 1) * SH_W;
            float x1_0 = sxi[sh_row1 + (sh_ox - 0)];
            float x1_1 = sxi[sh_row1 + (sh_ox - 1)];
            float x1_2 = sxi[sh_row1 + (sh_ox - 2)];
            {
                const int w_kh_base = 3;
                #pragma unroll
                for (int oci = 0; oci < OC_PASS; ++oci) {
                    const int oc = oc_pass0 + oci;
                    const int w_base = (ic * 32 + oc) * 9 + w_kh_base;
                    float sum = acc[oci];
                    sum = fmaf(x1_0, W_CONST_32_32_K3[w_base + 0], sum);
                    sum = fmaf(x1_1, W_CONST_32_32_K3[w_base + 1], sum);
                    sum = fmaf(x1_2, W_CONST_32_32_K3[w_base + 2], sum);
                    acc[oci] = sum;
                }
            }

            // kh=2 prefetch then compute
            int sh_row2 = (sh_oy - 2) * SH_W;
            float x2_0 = sxi[sh_row2 + (sh_ox - 0)];
            float x2_1 = sxi[sh_row2 + (sh_ox - 1)];
            float x2_2 = sxi[sh_row2 + (sh_ox - 2)];
            {
                const int w_kh_base = 6;
                #pragma unroll
                for (int oci = 0; oci < OC_PASS; ++oci) {
                    const int oc = oc_pass0 + oci;
                    const int w_base = (ic * 32 + oc) * 9 + w_kh_base;
                    float sum = acc[oci];
                    sum = fmaf(x2_0, W_CONST_32_32_K3[w_base + 0], sum);
                    sum = fmaf(x2_1, W_CONST_32_32_K3[w_base + 1], sum);
                    sum = fmaf(x2_2, W_CONST_32_32_K3[w_base + 2], sum);
                    acc[oci] = sum;
                }
            }
        }

        // Store this pass results
        #pragma unroll
        for (int oci = 0; oci < OC_PASS; ++oci) {
            const int oc = oc_pass0 + oci;
            y[y_n_base + oc * Hout * Wout + spatial] = acc[oci];
        }
    }
}

// Generic fallback (unchanged)
__global__ __launch_bounds__(256, 2)
void convT2d_generic_1d(
    const float* __restrict__ x,   // [N, Cin, Hin, Win]
    const float* __restrict__ w,   // [Cin, Cout, K, K]
    float* __restrict__ y,         // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int K,
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
    const int x_n_base = n * Cin * Hin * Win;
    const int w_ic_stride = Cout * K * K;

    for (int ic = 0; ic < Cin; ++ic) {
        const int x_ic_base = x_n_base + ic * Hin * Win;
        const int w_ic_base = ic * w_ic_stride + oc * (K * K);

        #pragma unroll 1
        for (int kh = 0; kh < K; ++kh) {
            int ih = oh - kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;
            const int x_h_base = x_ic_base + ih * Win;

            #pragma unroll 1
            for (int kw = 0; kw < K; ++kw) {
                int iw = ow - kw;
                if ((unsigned)iw >= (unsigned)Win) continue;

                float xv = __ldg(x + x_h_base + iw);
                float wv = w[w_ic_base + kh * K + kw];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = acc;
}

// Cached constant-weight update
static uintptr_t g_last_w_ptr = 0;
static int64_t   g_last_w_version = -1;

torch::Tensor conv_transposed2d_asymmetric_input_square_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be 4D [Cin,Cout,K,K]");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int N   = (int)x_c.size(0);
    const int Cin = (int)x_c.size(1);
    const int Hin = (int)x_c.size(2);
    const int Win = (int)x_c.size(3);

    const int wCin = (int)w_c.size(0);
    const int Cout = (int)w_c.size(1);
    const int K1   = (int)w_c.size(2);
    const int K2   = (int)w_c.size(3);

    TORCH_CHECK(wCin == Cin, "w.size(0) (Cin) must match x.size(1) (Cin)");
    TORCH_CHECK(K1 == K2, "Kernel must be square");
    TORCH_CHECK(K1 > 0, "Kernel size must be > 0");

    const int Hout = Hin + K1 - 1;
    const int Wout = Win + K1 - 1;

    auto y = torch::empty({N, Cout, Hout, Wout}, x_c.options());

    if (Cin == 32 && Cout == 32 && K1 == 3) {
        // Update constant weights only when storage ptr or version changes.
        uintptr_t wptr = (uintptr_t)w_c.storage().data();
        int64_t wver = w_c._version();
        if (wptr != g_last_w_ptr || wver != g_last_w_version) {
            copy_w_to_const_32_32_k3(w_c);
            g_last_w_ptr = wptr;
            g_last_w_version = wver;
        }

        constexpr int TILE_W = 16;
        constexpr int TILE_H = 16;
        constexpr int OC_BLOCK = 8;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid(div_up(Wout, TILE_W), div_up(Hout, TILE_H), N * div_up(32, OC_BLOCK));

        // shared memory: 32*(18*18) floats
        const size_t shmem_bytes = (size_t)32 * (size_t)(TILE_W + 2) * (size_t)(TILE_H + 2) * sizeof(float);

        convT2d_k3_c32_slab_oc8_twopass4<<<grid, block, shmem_bytes>>>(
            x_c.data_ptr<float>(),
            y.data_ptr<float>(),
            Hin, Win, Hout, Wout
        );
        CUDA_CHECK(cudaGetLastError());
        return y;
    }

    // Generic fallback
    int64_t total64 = (int64_t)N * Cout * Hout * Wout;
    TORCH_CHECK(total64 <= (int64_t)INT_MAX, "Output too large for 1D indexing in generic kernel");
    int total = (int)total64;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    convT2d_generic_1d<<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, K1,
        Hout, Wout
    );
    CUDA_CHECK(cudaGetLastError());
    return y;
}
"""

cpp_source = r"""
torch::Tensor conv_transposed2d_asymmetric_input_square_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transposed2d_asymmetric_input_square_v8",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transposed2d_asymmetric_input_square_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom ConvTranspose2d forward specialized for:
      - NCHW float32
      - stride=1, padding=0, output_padding=0
      - groups=1, bias=False
      - square kernels

    Fast path: Cin=Cout=32, K=3 using full-Cin shared-memory slab + constant weights.
    Optimized to reduce register pressure by computing OC_BLOCK=8 in two sequential OC_PASS=4 passes
    without increasing grid.z (avoids repeating slab loads / barriers).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
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

        if stride != 1 or padding != 0 or output_padding != 0 or groups != 1 or bias:
            raise ValueError(
                "ModelNew custom kernel supports only stride=1, padding=0, output_padding=0, groups=1, bias=False"
            )
        if isinstance(kernel_size, (tuple, list)):
            if kernel_size[0] != kernel_size[1]:
                raise ValueError("ModelNew custom kernel supports only square kernels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.conv_transposed2d_asymmetric_input_square_cuda(
            x, self.conv_transpose2d.weight
        )