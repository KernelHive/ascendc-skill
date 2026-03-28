import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

__host__ __device__ __forceinline__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

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

template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(128, 3)
void at_bt_gemm_tiled_64x64_kernel(
    const float* __restrict__ A, // (K, M) row-major, lda = M
    const float* __restrict__ B, // (N, K) row-major, ldb = K
    float* __restrict__ C,       // (M, N) row-major, ldc = N
    int M, int N, int K
){
    // blockDim = (16,8) => 128 threads
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..7
    const int tid = ty * 16 + tx;    // 0..127

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    // Output mapping:
    // BN=64, TN=4 => 16 thread columns
    // BM=64, TM=2 => 32 thread rows, but we have ty in 0..7, so 4 waves over M.
    constexpr int OUT_TC = BN / TN; // 16
    constexpr int OUT_TR = BM / TM; // 32
    static_assert(OUT_TC == 16, "Expected BN/TN == 16");
    static_assert(OUT_TR == 32, "Expected BM/TM == 32");

    const int logical_col = tx;              // 0..15
    const int logical_row0 = ty;             // 0..7
    const int logical_row_stride = 8;        // waves=4

    float acc[4][TM][TN];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[w][i][j] = 0.0f;
        }
    }

    // Dynamic shared memory:
    // As: BK x (BM+1) (pad to reduce bank conflicts for A broadcast-like access)
    // BsT: BK x (BN+1) where BsT[kk][nj] = B[block_n+nj, k0+kk] (transposed in smem)
    extern __shared__ float smem[];
    float* As  = smem;                               // size BK*(BM+1)
    float* BsT = As + (BK * (BM + 1));               // size BK*(BN+1)

    const bool A_aligned16 = ((((uintptr_t)A) & 0xF) == 0);
    const bool B_aligned16 = ((((uintptr_t)B) & 0xF) == 0);
    const bool C_aligned16 = ((((uintptr_t)C) & 0xF) == 0);

    // Helper lambdas for smem indexing
    auto As_idx = [&](int kk, int mi) -> int { return kk * (BM + 1) + mi; };
    auto BsT_idx = [&](int kk, int nj) -> int { return kk * (BN + 1) + nj; };

    // K tile loop
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load As: BK x BM floats, vectorize along M as float4 (mi multiple of 4)
        // Total float4: BK * (BM/4) = BK*16
        constexpr int VEC = 4;
        constexpr int AV4_PER_K = BM / VEC; // 16
        const int total_A_v4 = BK * AV4_PER_K;

        for (int vid = tid; vid < total_A_v4; vid += 128) {
            const int kk = vid / AV4_PER_K;      // 0..BK-1
            const int vm = vid - kk * AV4_PER_K; // 0..15
            const int mi0 = vm * VEC;

            const int gk = k0 + kk;
            const int gm = block_m + mi0;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gk < K) {
                const float* base = A + (int64_t)gk * M + gm;
                if (A_aligned16 && ((gm & 3) == 0) && (gm + 3) < M) {
                    v = *reinterpret_cast<const float4*>(base);
                } else {
                    v.x = (gm + 0 < M) ? ldg_f32(base + 0) : 0.f;
                    v.y = (gm + 1 < M) ? ldg_f32(base + 1) : 0.f;
                    v.z = (gm + 2 < M) ? ldg_f32(base + 2) : 0.f;
                    v.w = (gm + 3 < M) ? ldg_f32(base + 3) : 0.f;
                }
            }

            As[As_idx(kk, mi0 + 0)] = v.x;
            As[As_idx(kk, mi0 + 1)] = v.y;
            As[As_idx(kk, mi0 + 2)] = v.z;
            As[As_idx(kk, mi0 + 3)] = v.w;
        }

        // Load B into BsT transposed: for each kk, load BN values across nj
        // Vectorize along K in global memory as float4 (k multiple of 4) by loading 4 ks for fixed nj,
        // then scatter into BsT for kk..kk+3. Total float4: BN*(BK/4)=64*4=256
        constexpr int BV4_PER_N = BK / VEC; // 4
        const int total_B_v4 = BN * BV4_PER_N; // 256

        for (int vid = tid; vid < total_B_v4; vid += 128) {
            const int nj = vid / BV4_PER_N;      // 0..BN-1
            const int vk = vid - nj * BV4_PER_N; // 0..3
            const int kk0 = vk * VEC;

            const int gn = block_n + nj;
            const int gk = k0 + kk0;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gn < N) {
                const float* base = B + (int64_t)gn * K + gk;
                if (B_aligned16 && ((gk & 3) == 0) && (gk + 3) < K) {
                    v = *reinterpret_cast<const float4*>(base);
                } else {
                    v.x = (gk + 0 < K) ? ldg_f32(base + 0) : 0.f;
                    v.y = (gk + 1 < K) ? ldg_f32(base + 1) : 0.f;
                    v.z = (gk + 2 < K) ? ldg_f32(base + 2) : 0.f;
                    v.w = (gk + 3 < K) ? ldg_f32(base + 3) : 0.f;
                }
            }

            // Transpose into BsT: BsT[kk][nj]
            BsT[BsT_idx(kk0 + 0, nj)] = v.x;
            BsT[BsT_idx(kk0 + 1, nj)] = v.y;
            BsT[BsT_idx(kk0 + 2, nj)] = v.z;
            BsT[BsT_idx(kk0 + 3, nj)] = v.w;
        }

        __syncthreads();

        // Compute
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // B fragment (TN=4)
            float b_frag[TN];
            const int n0 = logical_col * TN; // 0..60
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int nj = n0 + j;
                b_frag[j] = BsT[BsT_idx(kk, nj)];
            }

#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int logical_row = logical_row0 + w * logical_row_stride; // 0..31
                const int m0 = logical_row * TM; // 0..62

                float a_frag[TM];
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int mi = m0 + i;
                    a_frag[i] = As[As_idx(kk, mi)];
                }

#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[w][i][j] = fmaf(a_frag[i], b_frag[j], acc[w][i][j]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Store
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int logical_row = logical_row0 + w * logical_row_stride;
        const int m0 = logical_row * TM;
        const int n0 = logical_col * TN;

#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int gm = block_m + m0 + i;
            if (gm < M) {
                const int gn_base = block_n + n0;
                int64_t out_base = (int64_t)gm * N + gn_base;

                if (gn_base + (TN - 1) < N && C_aligned16 && ((gn_base & 3) == 0)) {
                    float4 v;
                    v.x = acc[w][i][0];
                    v.y = acc[w][i][1];
                    v.z = acc[w][i][2];
                    v.w = acc[w][i][3];
                    *reinterpret_cast<float4*>(C + out_base) = v;
                } else {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int gn = gn_base + j;
                        if (gn < N) C[out_base + j] = acc[w][i][j];
                    }
                }
            }
        }
    }
}

torch::Tensor matmul_with_transposed_both_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2, "A must be 2D (K, M)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N, K)");
    const int K = (int)A.size(0);
    const int M = (int)A.size(1);
    TORCH_CHECK((int)B.size(1) == K, "B shape must be (N, K) with same K as A.size(0)");
    const int N = (int)B.size(0);

    auto C = torch::empty({M, N}, A.options());

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 2;
    constexpr int TN = 4;

    dim3 block(16, 8);
    dim3 grid(div_up_int(N, BN), div_up_int(M, BM));

    // smem: As BK*(BM+1) + BsT BK*(BN+1)
    size_t smem_bytes = (size_t)BK * (size_t)((BM + 1) + (BN + 1)) * sizeof(float);

    at_bt_gemm_tiled_64x64_kernel<BM, BN, BK, TM, TN><<<grid, block, smem_bytes>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        M, N, K
    );
    return C;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor matmul_with_transposed_both_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_with_transposed_both_opt4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_with_transposed_both_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization", "-lineinfo"],
    extra_cflags=["-O3"],
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
        return self.custom_ops_lib.matmul_with_transposed_both_cuda(A, B)