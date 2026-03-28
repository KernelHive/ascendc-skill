import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

// ---- type conversion ----
template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
    return __half2float(reinterpret_cast<const __half&>(v));
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(v));
}
#endif

// Vectorized load helpers
__device__ __forceinline__ float4 ld_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ uint2 ld_u32x2(const uint32_t* p) {
    return *reinterpret_cast<const uint2*>(p);
}

template <typename scalar_t>
__device__ __forceinline__ bool is_aligned_16(const scalar_t* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// ---- Kernel ----
// Block: 16x16 threads.
// Output tile: (BM x BN) where BM=32, BN=32 via per-thread micro-tile TMxTN = 2x2.
// K tile BK=32.
// Shared: As[BM][BK+1], Bs[BK][BN+1] (padding to reduce bank conflicts).
template <typename scalar_t, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_largek_rblock_kernel(
    const scalar_t* __restrict__ A, // [M,K]
    const scalar_t* __restrict__ B, // [K,N]
    float* __restrict__ C,          // [M,N]
    int M, int N, int K
) {
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREADS_X = BN / TN; // 32/2 = 16
    constexpr int THREADS_Y = BM / TM; // 32/2 = 16

    const int tx = threadIdx.x; // [0, THREADS_X)
    const int ty = threadIdx.y; // [0, THREADS_Y)

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // Each thread computes TM x TN outputs
    const int row0 = block_m + ty * TM;
    const int col0 = block_n + tx * TN;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Decide vectorization based on dtype + alignment and contiguous dimensions.
    const bool vecA = (std::is_same<scalar_t, float>::value) && is_aligned_16(reinterpret_cast<const float*>(A));
    const bool vecB = (std::is_same<scalar_t, float>::value) && is_aligned_16(reinterpret_cast<const float*>(B));

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load A tile: As[BM][BK]
        // We have 256 threads; need BM*BK = 1024 floats => 4 floats per thread.
        // Map linear load index across As.
        int t = ty * THREADS_X + tx; // 0..255
        int a_linear = t * 4;        // 0..1020 step 4

        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx = a_linear + it;
            int a_r = idx / BK;  // 0..BM-1
            int a_c = idx - a_r * BK; // 0..BK-1
            int gr = block_m + a_r;
            int gc = k0 + a_c;
            float v = 0.0f;
            if (gr < M && gc < K) v = to_float<scalar_t>(A[gr * K + gc]);
            As[a_r][a_c] = v;
        }

        // Cooperative load B tile: Bs[BK][BN]
        // Need BK*BN = 1024 floats => 4 floats per thread.
        int b_linear = t * 4;

        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx = b_linear + it;
            int b_r = idx / BN;      // 0..BK-1
            int b_c = idx - b_r * BN; // 0..BN-1
            int gr = k0 + b_r;
            int gc = block_n + b_c;
            float v = 0.0f;
            if (gr < K && gc < N) v = to_float<scalar_t>(B[gr * N + gc]);
            Bs[b_r][b_c] = v;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                a_reg[i] = As[r][kk];
            }

            float b_reg[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j;
                b_reg[j] = Bs[kk][c];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col0 + j;
                if (c < N) {
                    C[r * N + c] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_largek_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match (A: MxK, B: KxN)");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have same dtype");

    const int64_t M64 = A.size(0);
    const int64_t K64 = A.size(1);
    const int64_t N64 = B.size(1);
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "Dimensions too large for int");

    const int M = (int)M64;
    const int N = (int)N64;
    const int K = (int)K64;

    auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

    constexpr int BM = 32, BN = 32, BK = 32;
    constexpr int TM = 2, TN = 2;
    dim3 block(BN / TN, BM / TM, 1); // 16x16
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, A.scalar_type(), "matmul_largek_cuda", [&] {
        const scalar_t* Ap = (const scalar_t*)A.data_ptr<scalar_t>();
        const scalar_t* Bp = (const scalar_t*)B.data_ptr<scalar_t>();
        float* Cp = (float*)C.data_ptr<float>();
        matmul_largek_rblock_kernel<scalar_t, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(Ap, Bp, Cp, M, N, K);
    });

    return C;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor matmul_largek_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_largek_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_largek_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return self.custom_ops_lib.matmul_largek_cuda(A, B)