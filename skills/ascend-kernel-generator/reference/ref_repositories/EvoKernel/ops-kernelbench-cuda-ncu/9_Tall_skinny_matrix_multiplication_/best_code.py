import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: tall_skinny_matrix_multiplication ---------
# Semantics: C = A @ B for 2D tensors A[M,K], B[K,N] -> C[M,N]
# Fast path specialized for K==32 using warp-per-row streaming with float4 loads/stores (no __syncthreads()).
# Fallback path for K<=64 uses the prior shared-memory staging kernel.

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

// -------------------- K==32 fast path: warp-per-row, float4 columns --------------------
// Block: 128 threads = 4 warps.
// Each warp computes one row (row = blockIdx.y * 4 + warp_id).
// Each lane computes a float4 at columns: col4 = (blockIdx.x * (32*4)) + lane*4.
// So each warp covers 128 columns, each block covers 4 rows x 128 cols.
//
// Access pattern:
// - B loads are perfectly coalesced: for fixed k, lanes load consecutive float4 -> 512B/warp, contiguous.
// - A loads: each lane loads A[row,k] scalar; duplicated across lanes but tiny (K=32).
// - No shared memory, no barriers.

__device__ __forceinline__ float4 ld_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void st_float4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

__global__ __launch_bounds__(128, 4)
void gemm_k32_warp4x128cols_f32(
    const float* __restrict__ A,  // [M,32]
    const float* __restrict__ B,  // [32,N]
    float* __restrict__ C,        // [M,N]
    int M, int N
) {
    const int tid = (int)threadIdx.x;     // 0..127
    const int warp = tid >> 5;            // 0..3
    const int lane = tid & 31;            // 0..31

    const int row = (int)blockIdx.y * 4 + warp;
    const int col4 = ((int)blockIdx.x * 128) + lane * 4;  // starting column for float4

    if (row >= M) return;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    // If within bounds for vector store, use vector loads. Otherwise scalar tail.
    if (col4 + 3 < N) {
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            float a = A[(int64_t)row * 32 + k];
            float4 b4 = ld_float4(&B[(int64_t)k * N + col4]);
            acc0 = fmaf(a, b4.x, acc0);
            acc1 = fmaf(a, b4.y, acc1);
            acc2 = fmaf(a, b4.z, acc2);
            acc3 = fmaf(a, b4.w, acc3);
        }
        float4 out;
        out.x = acc0; out.y = acc1; out.z = acc2; out.w = acc3;
        st_float4(&C[(int64_t)row * N + col4], out);
    } else {
        // Tail: N not multiple of 4 or last tile.
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            float a = A[(int64_t)row * 32 + k];
            int base = (int64_t)k * N + col4;
            float b0 = (col4 + 0 < N) ? B[base + 0] : 0.f;
            float b1 = (col4 + 1 < N) ? B[base + 1] : 0.f;
            float b2 = (col4 + 2 < N) ? B[base + 2] : 0.f;
            float b3 = (col4 + 3 < N) ? B[base + 3] : 0.f;
            acc0 = fmaf(a, b0, acc0);
            acc1 = fmaf(a, b1, acc1);
            acc2 = fmaf(a, b2, acc2);
            acc3 = fmaf(a, b3, acc3);
        }
        int out_base = (int64_t)row * N + col4;
        if (col4 + 0 < N) C[out_base + 0] = acc0;
        if (col4 + 1 < N) C[out_base + 1] = acc1;
        if (col4 + 2 < N) C[out_base + 2] = acc2;
        if (col4 + 3 < N) C[out_base + 3] = acc3;
    }
}

// -------------------- Fallback: prior shared-memory kernel for K<=64 --------------------
template<int BM, int BN, int TN, int BK_MAX>
__global__ __launch_bounds__(256, 3)
void gemm_k_small_b_shared_f32(
    const float* __restrict__ A,  // [M,K] row-major
    const float* __restrict__ B,  // [K,N] row-major
    float* __restrict__ C,        // [M,N] row-major
    int M, int K, int N
) {
    __shared__ float Bs[BK_MAX][BN];

    const int tx = (int)threadIdx.x; // 0..31
    const int ty = (int)threadIdx.y; // 0..BM-1

    const int row = (int)blockIdx.y * BM + ty;
    const int col0 = (int)blockIdx.x * BN + tx * TN;

    float acc[TN];
    #pragma unroll
    for (int i = 0; i < TN; ++i) acc[i] = 0.0f;

    for (int k = 0; k < K; ++k) {
        int b_col = (int)blockIdx.x * BN + ty * 32 + tx; // covers BN=BM*32 when BN=128,BM=8
        float bv = 0.0f;
        if (b_col < N) {
            bv = B[(int64_t)k * N + b_col];
        }
        Bs[k][ty * 32 + tx] = bv;

        __syncthreads();

        if (row < M) {
            float a = A[(int64_t)row * K + k];
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                int tcol = tx * TN + i;
                float b = Bs[k][tcol];
                acc[i] = fmaf(a, b, acc[i]);
            }
        }

        __syncthreads();
    }

    if (row < M) {
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            int col = col0 + i;
            if (col < N) {
                C[(int64_t)row * N + col] = acc[i];
            }
        }
    }
}

torch::Tensor tall_skinny_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A(M,K), B(K,N)");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // Fast path: K == 32
    if (K == 32) {
        // Each block: 4 rows x 128 cols
        dim3 block(128, 1, 1);
        dim3 grid(div_up_int(N, 128), div_up_int(M, 4), 1);
        gemm_k32_warp4x128cols_f32<<<grid, block>>>(
            (const float*)A.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            (float*)C.data_ptr<float>(),
            M, N
        );
        return C;
    }

    // Fallback for K small (<=64): baseline shared-memory kernel
    TORCH_CHECK(K <= 64, "This custom kernel supports K <= 64; got K=", K);

    constexpr int BK_MAX = 64;
    constexpr int TN = 4;
    constexpr int BN = 32 * TN;  // 128
    constexpr int BM = 8;

    dim3 block(32, BM, 1);
    dim3 grid(div_up_int(N, BN), div_up_int(M, BM), 1);

    gemm_k_small_b_shared_f32<BM, BN, TN, BK_MAX><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor tall_skinny_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_tall_skinny_mm_v6_k32_warp4x128_float4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["tall_skinny_matrix_multiplication_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Performs a single matrix multiplication using a custom CUDA kernel.
    Semantics match torch.matmul for 2D inputs: (M,K) @ (K,N) -> (M,N).
    Optimized for float32 CUDA, contiguous, with a fast path for K==32 and fallback for K<=64.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if (
            A.is_cuda and B.is_cuda and
            A.dtype == torch.float32 and B.dtype == torch.float32 and
            A.dim() == 2 and B.dim() == 2 and
            A.size(1) == B.size(0)
        ):
            K = int(A.size(1))
            if K <= 64:
                return self.custom_ops_lib.tall_skinny_matrix_multiplication_cuda(A, B)
        return torch.matmul(A, B)


# Keep original input helpers for compatibility with the provided scaffold.
M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, M, device="cuda", dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []