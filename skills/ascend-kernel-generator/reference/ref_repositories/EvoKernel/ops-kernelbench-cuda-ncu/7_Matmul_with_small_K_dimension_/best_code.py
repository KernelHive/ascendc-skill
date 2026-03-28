import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------
# Custom CUDA matmul (small K=64) extension - occupancy-tuned 128-thread CTA (16x8),
# 2x4 per-thread micro-tile with 4 waves in M, vectorized float4 loads/stores.
# -----------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ bool aligned16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

__host__ __device__ __forceinline__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

// Specialized for K=64 (BK=64). Tile: 64x64 in C.
// CTA: 16x8 = 128 threads.
// Each thread computes TN=4 cols, TM=2 rows, repeated over WAVES_M=4 to cover 64 rows.
template<int BM=64, int BN=64, int BK=64, int TM=2, int TN=4, int WAVES_M=4>
__global__ __launch_bounds__(128, 3)
void gemm_smallk64_f32_kernel_occ(
    const float* __restrict__ A, // [M,K]
    const float* __restrict__ B, // [K,N]
    float* __restrict__ C,       // [M,N]
    int M, int N
) {
    // blockDim = (16,8)
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..7
    const int tid = ty * 16 + tx;    // 0..127

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    // Shared memory (1D) to reduce index math overhead.
    // A tile: [BM][BK] (row-major by K), pad K by +1 to mitigate bank conflicts.
    // B tile: [BK][BN] (row-major by N), pad N by +1 to mitigate bank conflicts.
    __shared__ float As[BM * (BK + 1)];
    __shared__ float Bs[BK * (BN + 1)];

    const bool A_aligned = aligned16(A);
    const bool B_aligned = aligned16(B);
    const bool C_aligned = aligned16(C);

    // Cooperative loads:
    // A: BM*(BK/4) float4 = 64*16=1024 float4 -> 8 per thread (128 threads)
    // B: BK*(BN/4) float4 = 64*16=1024 float4 -> 8 per thread
#pragma unroll
    for (int it = 0; it < 8; ++it) {
        // Load A float4 segments
        int a4 = tid + it * 128;          // 0..1023
        int am = a4 >> 4;                 // /16 => 0..63
        int ak4 = (a4 & 15) << 2;         // (mod16)*4 => 0..60
        int g_m = block_m + am;
        int g_k = ak4; // K starts at 0, BK=K=64

        float4 va = {0.f, 0.f, 0.f, 0.f};
        if (g_m < M) {
            const float* gp = A + (int64_t)g_m * BK + g_k;
            if (A_aligned) {
                va = *reinterpret_cast<const float4*>(gp);
            } else {
                va.x = ldg_f32(gp + 0);
                va.y = ldg_f32(gp + 1);
                va.z = ldg_f32(gp + 2);
                va.w = ldg_f32(gp + 3);
            }
        }
        // store to shared: As[am][ak4..ak4+3]
        float* ap = As + am * (BK + 1) + ak4;
        ap[0] = va.x; ap[1] = va.y; ap[2] = va.z; ap[3] = va.w;

        // Load B float4 segments
        int b4 = tid + it * 128;          // 0..1023
        int bk = b4 >> 4;                 // /16 => 0..63
        int bn4 = (b4 & 15) << 2;         // 0..60
        int g_n = block_n + bn4;

        float4 vb = {0.f, 0.f, 0.f, 0.f};
        if (g_n + 3 < N) {
            const float* gp = B + (int64_t)bk * N + g_n;
            if (B_aligned) {
                vb = *reinterpret_cast<const float4*>(gp);
            } else {
                vb.x = ldg_f32(gp + 0);
                vb.y = ldg_f32(gp + 1);
                vb.z = ldg_f32(gp + 2);
                vb.w = ldg_f32(gp + 3);
            }
        } else {
            // tail-safe
            const float* gp0 = B + (int64_t)bk * N + g_n;
            vb.x = (g_n + 0 < N) ? ldg_f32(gp0 + 0) : 0.f;
            vb.y = (g_n + 1 < N) ? ldg_f32(gp0 + 1) : 0.f;
            vb.z = (g_n + 2 < N) ? ldg_f32(gp0 + 2) : 0.f;
            vb.w = (g_n + 3 < N) ? ldg_f32(gp0 + 3) : 0.f;
        }
        float* bp = Bs + bk * (BN + 1) + bn4;
        bp[0] = vb.x; bp[1] = vb.y; bp[2] = vb.z; bp[3] = vb.w;
    }

    __syncthreads();

    // Accumulators: 4 waves * (TM*TN)=8 => 32 floats per thread.
    float acc[WAVES_M][TM][TN];
#pragma unroll
    for (int w = 0; w < WAVES_M; ++w) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[w][i][j] = 0.0f;
        }
    }

    const int n0 = tx * TN; // 0..60

    // Compute K=64. Slightly reduced unroll to avoid excessive register pressure.
#pragma unroll 4
    for (int kk = 0; kk < BK; ++kk) {
        // Load B fragment once per kk
        float b0 = Bs[kk * (BN + 1) + (n0 + 0)];
        float b1 = Bs[kk * (BN + 1) + (n0 + 1)];
        float b2 = Bs[kk * (BN + 1) + (n0 + 2)];
        float b3 = Bs[kk * (BN + 1) + (n0 + 3)];

#pragma unroll
        for (int w = 0; w < WAVES_M; ++w) {
            // Cover BM rows: ty(0..7) + w*8 => 0..31 micro-rows; each micro-row has TM=2 => 64 rows
            const int micro_row = ty + w * 8;  // 0..31
            const int m0 = micro_row * TM;     // 0..62
            float a0 = As[(m0 + 0) * (BK + 1) + kk];
            float a1 = As[(m0 + 1) * (BK + 1) + kk];

            // FMA
            acc[w][0][0] = fmaf(a0, b0, acc[w][0][0]);
            acc[w][0][1] = fmaf(a0, b1, acc[w][0][1]);
            acc[w][0][2] = fmaf(a0, b2, acc[w][0][2]);
            acc[w][0][3] = fmaf(a0, b3, acc[w][0][3]);

            acc[w][1][0] = fmaf(a1, b0, acc[w][1][0]);
            acc[w][1][1] = fmaf(a1, b1, acc[w][1][1]);
            acc[w][1][2] = fmaf(a1, b2, acc[w][1][2]);
            acc[w][1][3] = fmaf(a1, b3, acc[w][1][3]);
        }
    }

    // Store results (vectorize along N)
#pragma unroll
    for (int w = 0; w < WAVES_M; ++w) {
        const int micro_row = ty + w * 8;
        const int m0 = micro_row * TM;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            int g_m = block_m + m0 + i;
            if (g_m < M) {
                int g_n = block_n + n0;
                float* cp = C + (int64_t)g_m * N + g_n;

                if (g_n + 3 < N && C_aligned && ((((uintptr_t)cp) & 0xF) == 0)) {
                    float4 out;
                    out.x = acc[w][i][0];
                    out.y = acc[w][i][1];
                    out.z = acc[w][i][2];
                    out.w = acc[w][i][3];
                    *reinterpret_cast<float4*>(cp) = out;
                } else {
                    if (g_n + 0 < N) cp[0] = acc[w][i][0];
                    if (g_n + 1 < N) cp[1] = acc[w][i][1];
                    if (g_n + 2 < N) cp[2] = acc[w][i][2];
                    if (g_n + 3 < N) cp[3] = acc[w][i][3];
                }
            }
        }
    }
}

torch::Tensor matmul_with_small_k_dimension_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2, "A must be 2D (M,K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,N)");
    int64_t M64 = A.size(0);
    int64_t K64 = A.size(1);
    TORCH_CHECK(B.size(0) == K64, "B.size(0) must equal A.size(1)");
    int64_t N64 = B.size(1);

    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "Sizes must fit in int32");
    TORCH_CHECK(K64 == 64, "This optimized kernel is specialized for K=64");

    const int M = (int)M64;
    const int N = (int)N64;

    auto C = torch::empty({M64, N64}, A.options());

    constexpr int BM = 64, BN = 64, BK = 64, TM = 2, TN = 4, WAVES_M = 4;
    dim3 block(16, 8);
    dim3 grid(div_up_int(N, BN), div_up_int(M, BM));

    gemm_smallk64_f32_kernel_occ<BM, BN, BK, TM, TN, WAVES_M><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        M, N
    );

    return C;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor matmul_with_small_k_dimension_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_smallk_k64_occ_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_with_small_k_dimension_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replaces torch.matmul(A, B) with a custom CUDA kernel specialized for K=64 (float32, contiguous, CUDA).
    Falls back to torch.matmul otherwise.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not (A.is_cuda and B.is_cuda):
            return torch.matmul(A, B)
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            return torch.matmul(A, B)
        if A.dim() != 2 or B.dim() != 2:
            return torch.matmul(A, B)
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        if A.size(1) != 64:
            return torch.matmul(A, B)
        return self.custom_ops_lib.matmul_with_small_k_dimension_cuda(A, B)