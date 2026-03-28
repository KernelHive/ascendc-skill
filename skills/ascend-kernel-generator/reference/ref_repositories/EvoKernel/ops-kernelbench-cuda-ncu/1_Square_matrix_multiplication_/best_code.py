import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: square_matrix_multiplication (optimized square GEMM) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

static inline __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Blocked FP32 square GEMM:
// CTA computes BM x BN tile of C.
// Cooperative load A and B tiles into shared memory (vectorized float4 when possible).
// A is stored transposed in shared to improve access patterns for per-thread micro-tiles.
template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256, 2)
void square_gemm_blocked_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // 256 threads in (16,16)
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    // Shared memory double buffer.
    // A stored transposed: As[buf][k][m] to make k-major loads contiguous for warp when fetching a_frag.
    __shared__ float As[2][BK][BM + 1];  // +1 padding to reduce bank conflicts
    __shared__ float Bs[2][BK][BN + 1];  // +1 padding

    // Each thread computes TM x TN micro-tile.
    const int out_m0 = block_m + ty * TM;
    const int out_n0 = block_n + tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    const uintptr_t A_addr = (uintptr_t)A;
    const uintptr_t B_addr = (uintptr_t)B;
    const uintptr_t C_addr = (uintptr_t)C;
    const bool A_aligned16 = (A_addr & 0xF) == 0;
    const bool B_aligned16 = (B_addr & 0xF) == 0;
    const bool C_aligned16 = (C_addr & 0xF) == 0;

    const int num_k_tiles = (N + BK - 1) / BK;

    auto load_tile = [&](int buf, int k0) {
        // Load A tile: BM x BK. We vectorize along M dimension by storing transposed.
        // We'll load A rows (m) contiguous along K; but for transpose fill we write As[k][m].
        // Use float4 along K when possible.
        const int A_vecs_per_row = BK / 4;  // BK assumed multiple of 4 (we set BK=16)
        const int total_A_vecs = BM * A_vecs_per_row;

        for (int idx = tid; idx < total_A_vecs; idx += 256) {
            const int mi = idx / A_vecs_per_row;               // 0..BM-1
            const int kk4 = (idx - mi * A_vecs_per_row) * 4;   // 0,4,8,12
            const int g_m = block_m + mi;
            const int g_k = k0 + kk4;

            if (A_aligned16 && g_m < N && (g_k + 3) < N) {
                const float4* p4 = reinterpret_cast<const float4*>(A + (int64_t)g_m * N + g_k);
                float4 v = *p4;
                As[buf][kk4 + 0][mi] = v.x;
                As[buf][kk4 + 1][mi] = v.y;
                As[buf][kk4 + 2][mi] = v.z;
                As[buf][kk4 + 3][mi] = v.w;
            } else {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int kk = kk4 + d;
                    const int gk = k0 + kk;
                    As[buf][kk][mi] = (g_m < N && gk < N) ? ldg_f32(A + (int64_t)g_m * N + gk) : 0.0f;
                }
            }
        }

        // Load B tile: BK x BN. Vectorize along N (columns) with float4.
        const int B_vecs_per_row = BN / 4;  // BN assumed multiple of 4 (we set BN=128)
        const int total_B_vecs = BK * B_vecs_per_row;

        for (int idx = tid; idx < total_B_vecs; idx += 256) {
            const int kk = idx / B_vecs_per_row;              // 0..BK-1
            const int nj4 = (idx - kk * B_vecs_per_row) * 4;  // 0..BN-4
            const int g_k = k0 + kk;
            const int g_n = block_n + nj4;

            if (B_aligned16 && g_k < N && (g_n + 3) < N) {
                const float4* p4 = reinterpret_cast<const float4*>(B + (int64_t)g_k * N + g_n);
                float4 v = *p4;
                Bs[buf][kk][nj4 + 0] = v.x;
                Bs[buf][kk][nj4 + 1] = v.y;
                Bs[buf][kk][nj4 + 2] = v.z;
                Bs[buf][kk][nj4 + 3] = v.w;
            } else {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int nj = nj4 + d;
                    const int gn = block_n + nj;
                    Bs[buf][kk][nj] = (g_k < N && gn < N) ? ldg_f32(B + (int64_t)g_k * N + gn) : 0.0f;
                }
            }
        }
    };

    int buf = 0;

    // Prologue: load first tile
    load_tile(buf, 0);
    __syncthreads();

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        const int next_tile = tile + 1;
        const int next_buf = buf ^ 1;

        // Prefetch next tile into the other buffer before compute ends (software pipelining).
        // We must ensure current compute doesn't race with writes to the other buffer; that's safe.
        if (next_tile < num_k_tiles) {
            const int next_k0 = next_tile * BK;
            load_tile(next_buf, next_k0);
        }

        // Compute on current buffer
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int mi = ty * TM + i; // 0..BM-1
                a_frag[i] = As[buf][kk][mi];
            }

            float b_frag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int nj = tx * TN + j; // 0..BN-1
                b_frag[j] = Bs[buf][kk][nj];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        // If we prefetched, wait for it to be visible before swapping buffers.
        if (next_tile < num_k_tiles) {
            __syncthreads();
            buf = next_buf;
        }
    }

    // Store C: vectorize along N when possible
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_m = out_m0 + i;
        if (g_m >= N) continue;

        if (TN % 4 == 0 && C_aligned16) {
#pragma unroll
            for (int j4 = 0; j4 < TN; j4 += 4) {
                const int g_n = out_n0 + j4;
                if ((g_n + 3) < N) {
                    float4 v;
                    v.x = acc[i][j4 + 0];
                    v.y = acc[i][j4 + 1];
                    v.z = acc[i][j4 + 2];
                    v.w = acc[i][j4 + 3];
                    float4* p4 = reinterpret_cast<float4*>(C + (int64_t)g_m * N + g_n);
                    *p4 = v;
                } else {
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const int gn = g_n + d;
                        if (gn < N) C[(int64_t)g_m * N + gn] = acc[i][j4 + d];
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int g_n = out_n0 + j;
                if (g_n < N) C[(int64_t)g_m * N + g_n] = acc[i][j];
            }
        }
    }
}

torch::Tensor square_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D (N,N)");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square (N,N)");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square (N,N)");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same N");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    const int N = (int)A_c.size(0);

    auto out = torch::empty({N, N}, A_c.options());

    // Tuned parameters (square GEMM).
    // 128x128 tile, BK=16, 16x16 threads, 8x8 micro-tile => each CTA computes 128x128.
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    square_gemm_blocked_f32_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        (const float*)A_c.data_ptr<float>(),
        (const float*)B_c.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        N
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor square_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_square_mm_opt2_pipelined_noptx",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["square_matrix_multiplication_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Performs a single square matrix multiplication (C = A * B) using an optimized custom CUDA kernel.
    Expects CUDA float32 inputs.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.square_matrix_multiplication_cuda(A, B)


N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, N, device="cuda", dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []