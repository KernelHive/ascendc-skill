import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: matmul_for_symmetric_matrices (BK=32 + padded linear smem + perfect-balanced float4 staging fastpath) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(C10_CUDA_KERNEL_LAUNCH_CHECK)
  #define KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#else
  #define KERNEL_LAUNCH_CHECK() do {                                  \
      cudaError_t err = cudaGetLastError();                            \
      TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err)); \
    } while (0)
#endif

__device__ __forceinline__ float ro_load(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float4 ld_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

template<int BM, int BN, int BK, int TM, int TN, int PAD_K, int PAD_N>
__global__ void gemm_blocked_f32_bk32_padded_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    const int block_row = (int)blockIdx.y;
    const int block_col = (int)blockIdx.x;

    const int tx = (int)threadIdx.x; // [0, BN/TN)
    const int ty = (int)threadIdx.y; // [0, BM/TM)

    const int row0 = block_row * BM + ty * TM;
    const int col0 = block_col * BN + tx * TN;

    // linear shared memory with padding
    constexpr int As_stride = BK + PAD_K;
    constexpr int Bs_stride = BN + PAD_N;

    __shared__ float As[BM * As_stride];
    __shared__ float Bs[BK * Bs_stride];

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    const int threads_x = BN / TN; // 16
    const int threads_y = BM / TM; // 16
    const int tid = ty * threads_x + tx;
    const int nthreads = threads_x * threads_y; // 256

    // Fastpath conditions (true for provided benchmark N=4096, contiguous tensors).
    const bool alignedA = (((uintptr_t)A & 15ull) == 0ull);
    const bool alignedB = (((uintptr_t)B & 15ull) == 0ull);
    const bool n_ok = ((N & 3) == 0);          // float4 along contiguous dimension
    const bool tile_ok = ((N & (BN - 1)) == 0); // multiple of 64
    const bool vec_fast = alignedA && alignedB && n_ok && tile_ok;

    // For BK=32:
    // A tile float4 loads: BM*(BK/4) = 64*8 = 512 => 2 float4/thread
    // B tile float4 loads: BK*(BN/4) = 32*16 = 512 => 2 float4/thread
    for (int kb = 0; kb < N; kb += BK) {
        if (vec_fast) {
            // ---- Load A tile: 2 float4 per thread ----
            // idx in [0, 512)
            int idxA = tid;
#pragma unroll
            for (int rep = 0; rep < 2; ++rep, idxA += 256) {
                const int vec_cols = BK / 4; // 8
                const int r = idxA / vec_cols;     // 0..63
                const int vc = idxA - r * vec_cols; // 0..7
                const int gr = block_row * BM + r;
                const int gc = kb + vc * 4;
                const float4 v4 = ld_float4(A + gr * N + gc);
                const int base = r * As_stride + vc * 4;
                As[base + 0] = v4.x;
                As[base + 1] = v4.y;
                As[base + 2] = v4.z;
                As[base + 3] = v4.w;
            }

            // ---- Load B tile: 2 float4 per thread ----
            int idxB = tid;
#pragma unroll
            for (int rep = 0; rep < 2; ++rep, idxB += 256) {
                const int vec_cols = BN / 4; // 16
                const int r = idxB / vec_cols;      // 0..31 (k within tile)
                const int vc = idxB - r * vec_cols; // 0..15
                const int gr = kb + r;
                const int gc = block_col * BN + vc * 4;
                const float4 v4 = ld_float4(B + gr * N + gc);
                const int base = r * Bs_stride + vc * 4;
                Bs[base + 0] = v4.x;
                Bs[base + 1] = v4.y;
                Bs[base + 2] = v4.z;
                Bs[base + 3] = v4.w;
            }
        } else {
            // Safe fallback (handles edges, non-multiple sizes, non-alignment)
            for (int idx = tid; idx < BM * BK; idx += nthreads) {
                const int r = idx / BK;
                const int c = idx - r * BK;
                const int gr = block_row * BM + r;
                const int gc = kb + c;
                float v = 0.0f;
                if (gr < N && gc < N) v = ro_load(A + gr * N + gc);
                As[r * As_stride + c] = v;
            }
            for (int idx = tid; idx < BK * BN; idx += nthreads) {
                const int r = idx / BN;
                const int c = idx - r * BN;
                const int gr = kb + r;
                const int gc = block_col * BN + c;
                float v = 0.0f;
                if (gr < N && gc < N) v = ro_load(B + gr * N + gc);
                Bs[r * Bs_stride + c] = v;
            }
        }

        __syncthreads();

        // ---- Compute ----
        // Moderate unroll: BK=32, keep full unroll for compute but avoid extra staging overhead elsewhere.
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int r = ty * TM + i;
                a_frag[i] = As[r * As_stride + k];
            }

            const int b_row = k * Bs_stride + tx * TN;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const float b = Bs[b_row + j];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    acc[i][j] = fmaf(a_frag[i], b, acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + i;
        if (r < N) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int c = col0 + j;
                if (c < N) C[r * N + c] = acc[i][j];
            }
        }
    }
}

torch::Tensor matmul_for_symmetric_matrices_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D (N,N)");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square (N,N)");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square (N,N)");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same N");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int N = (int)A.size(0);
    auto C = torch::empty({N, N}, A.options());

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int PAD_K = 1; // mitigate bank conflicts when many threads read same k across rows
    constexpr int PAD_N = 1; // mitigate bank conflicts when many threads read across columns

    dim3 block(BN / TN, BM / TM);           // (16,16) => 256 threads
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    gemm_blocked_f32_bk32_padded_kernel<BM, BN, BK, TM, TN, PAD_K, PAD_N><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        N
    );
    KERNEL_LAUNCH_CHECK();

    return C;
}
"""

cpp_src = r"""
torch::Tensor matmul_for_symmetric_matrices_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_symm_opt_bk32_padded_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_for_symmetric_matrices_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Performs C = A @ B using an optimized custom CUDA kernel (FP32).
    Symmetry of A,B is not used to change semantics (full GEMM result computed).
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.matmul_for_symmetric_matrices_cuda(A, B)


# Keep original input helpers for compatibility with the provided scaffold.
N = 4096

def get_inputs():
    A = torch.rand(N, N, device="cuda", dtype=torch.float32)
    A = (A + A.t()) * 0.5
    B = torch.rand(N, N, device="cuda", dtype=torch.float32)
    B = (B + B.t()) * 0.5
    return [A, B]

def get_init_inputs():
    return []