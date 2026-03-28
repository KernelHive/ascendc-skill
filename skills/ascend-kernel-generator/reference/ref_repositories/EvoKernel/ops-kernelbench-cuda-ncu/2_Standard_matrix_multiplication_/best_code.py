import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: standard_matrix_multiplication (opt: lower regs + full BN via 2-phase cols) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ bool is_aligned_16_device(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}
__device__ __forceinline__ bool is_aligned_8_device(const void* p) {
    return (((uintptr_t)p) & 0x7) == 0;
}

// 64x64x16 tiling, 16x16 threads.
// Each thread computes TM x TN outputs for TWO column groups to cover full BN:
// group0: cols (tx*TN + 0/1)
// group1: cols (32 + tx*TN + 0/1)
template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256, 3)
void gemm_64x64_f32_kernel_tn2_2phase(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;    // 0..255

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Accumulators: [TM][TN] for left half and right half
    float acc0[TM][TN];
    float acc1[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc0[i][j] = 0.0f;
            acc1[i][j] = 0.0f;
        }
    }

    const int out_m0 = block_m + ty * TM;

    // column bases for two phases
    const int col_base0 = block_n + tx * TN;          // 0..30 step 2 within tile
    const int col_base1 = block_n + 32 + tx * TN;     // 32..62 step 2 within tile

    const bool A_aligned = is_aligned_16_device(A);
    const bool B_aligned = is_aligned_16_device(B);

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BK;

        // Load A tile: BM*(BK/4) = 64*4 = 256 float4 segments (when BK=16)
        if (BK % 4 == 0) {
            int a4 = tid;
            int am = a4 / (BK / 4);
            int ak4 = (a4 - am * (BK / 4)) * 4;

            int g_m = block_m + am;
            int g_k = k0 + ak4;

            if (A_aligned && g_m < M && (g_k + 3) < K) {
                const float4* p4 = reinterpret_cast<const float4*>(A + (int64_t)g_m * K + g_k);
                float4 v = *p4;
                As[am][ak4 + 0] = v.x;
                As[am][ak4 + 1] = v.y;
                As[am][ak4 + 2] = v.z;
                As[am][ak4 + 3] = v.w;
            } else {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int gk = g_k + d;
                    As[am][ak4 + d] = (g_m < M && gk < K) ? ldg_f32(A + (int64_t)g_m * K + gk) : 0.0f;
                }
            }
        }

        // Load B tile: BK*(BN/4) = 16*16 = 256 float4 segments (when BN=64)
        if (BN % 4 == 0) {
            int b4 = tid;
            int bk = b4 / (BN / 4);                 // 0..BK-1
            int bn4 = (b4 - bk * (BN / 4)) * 4;     // 0..BN-4

            int g_k = k0 + bk;
            int g_n = block_n + bn4;

            if (B_aligned && g_k < K && (g_n + 3) < N) {
                const float4* p4 = reinterpret_cast<const float4*>(B + (int64_t)g_k * N + g_n);
                float4 v = *p4;
                Bs[bk][bn4 + 0] = v.x;
                Bs[bk][bn4 + 1] = v.y;
                Bs[bk][bn4 + 2] = v.z;
                Bs[bk][bn4 + 3] = v.w;
            } else {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int gn = g_n + d;
                    Bs[bk][bn4 + d] = (g_k < K && gn < N) ? ldg_f32(B + (int64_t)g_k * N + gn) : 0.0f;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                int am = ty * TM + i;
                a_frag[i] = As[am][kk];
            }

            // Load B fragments for left and right halves (2 columns each)
            float b0[TN];
            float b1[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int bn0 = (tx * TN) + j;        // 0..31
                int bn1 = 32 + (tx * TN) + j;   // 32..63
                b0[j] = Bs[kk][bn0];
                b1[j] = Bs[kk][bn1];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc0[i][j] = fmaf(a_frag[i], b0[j], acc0[i][j]);
                    acc1[i][j] = fmaf(a_frag[i], b1[j], acc1[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store: vectorize float2 when safe
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_m = out_m0 + i;
        if (g_m >= M) continue;

        // Left half
        int g_n0 = col_base0;
        if (g_n0 < N) {
            int64_t out_base0 = (int64_t)g_m * N + g_n0;
            if ((g_n0 + 1) < N && is_aligned_8_device(C + out_base0)) {
                float2 v;
                v.x = acc0[i][0];
                v.y = acc0[i][1];
                *reinterpret_cast<float2*>(C + out_base0) = v;
            } else {
                C[out_base0] = acc0[i][0];
                if ((g_n0 + 1) < N) C[out_base0 + 1] = acc0[i][1];
            }
        }

        // Right half
        int g_n1 = col_base1;
        if (g_n1 < N) {
            int64_t out_base1 = (int64_t)g_m * N + g_n1;
            if ((g_n1 + 1) < N && is_aligned_8_device(C + out_base1)) {
                float2 v;
                v.x = acc1[i][0];
                v.y = acc1[i][1];
                *reinterpret_cast<float2*>(C + out_base1) = v;
            } else {
                C[out_base1] = acc1[i][0];
                if ((g_n1 + 1) < N) C[out_base1 + 1] = acc1[i][1];
            }
        }
    }
}

torch::Tensor standard_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M,K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,N)");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A(M,K), B(K,N)");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    auto out = torch::empty({M, N}, A.options());

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 2;

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_64x64_f32_kernel_tn2_2phase<BM, BN, BK, TM, TN><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        M, K, N
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor standard_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_standard_mm_opt6_tn2_2phase",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["standard_matrix_multiplication_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    using an optimized custom CUDA kernel.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.standard_matrix_multiplication_cuda(A, B)


# Keep original input helpers for compatibility with the provided scaffold.
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K, device="cuda", dtype=torch.float32)
    B = torch.rand(K, N, device="cuda", dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []