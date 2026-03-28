import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: higher-occupancy tiled batched bmm (float32, CUDA) ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

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

__device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// 64x64x16 tile, 16x16 threads (256), each thread computes 4x4 outputs.
// Shared memory padded on fastest dim to reduce bank conflicts.
// Designed to reduce barrier overhead and improve latency hiding vs smaller tiles.
template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256, 2)
void bmm_64x64x16_f32_kernel(
    const float* __restrict__ A, // [B,M,K]
    const float* __restrict__ B, // [B,K,N]
    float* __restrict__ C,       // [B,M,N]
    int batch, int M, int K, int N
) {
    const int b = (int)blockIdx.z;

    const float* __restrict__ Ab = A + (size_t)b * (size_t)M * (size_t)K;
    const float* __restrict__ Bb = B + (size_t)b * (size_t)K * (size_t)N;
    float* __restrict__ Cb = C + (size_t)b * (size_t)M * (size_t)N;

    // 16x16 threads
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;    // 0..255

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    // Each thread computes a 4x4 output tile
    const int out_m0 = block_m + ty * TM;
    const int out_n0 = block_n + tx * TN;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    const bool A_aligned = is_aligned_16(Ab);
    const bool B_aligned = is_aligned_16(Bb);

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BK;

        // Load A tile: BM x BK, vectorize along K (BK=16 -> 4 float4 per row)
        // float4 segments = BM*(BK/4) = 64*4=256 => one per thread
        {
            const int a4 = tid;                // 0..255
            const int am = a4 / (BK / 4);      // 0..63
            const int ak4 = (a4 - am * (BK/4)) * 4; // 0,4,8,12

            const int g_m = block_m + am;
            const int g_k = k0 + ak4;
            const float* base = Ab + (size_t)g_m * (size_t)K + (size_t)g_k;

            if (A_aligned && g_m < M && (g_k + 3) < K) {
                const float4 v = *reinterpret_cast<const float4*>(base);
                As[am][ak4 + 0] = v.x;
                As[am][ak4 + 1] = v.y;
                As[am][ak4 + 2] = v.z;
                As[am][ak4 + 3] = v.w;
            } else {
                As[am][ak4 + 0] = (g_m < M && (g_k + 0) < K) ? ldg_f32(base + 0) : 0.0f;
                As[am][ak4 + 1] = (g_m < M && (g_k + 1) < K) ? ldg_f32(base + 1) : 0.0f;
                As[am][ak4 + 2] = (g_m < M && (g_k + 2) < K) ? ldg_f32(base + 2) : 0.0f;
                As[am][ak4 + 3] = (g_m < M && (g_k + 3) < K) ? ldg_f32(base + 3) : 0.0f;
            }
        }

        // Load B tile: BK x BN, vectorize along N (BN=64 -> 16 float4 per row)
        // float4 segments = BK*(BN/4) = 16*16=256 => one per thread
        {
            const int b4 = tid;                 // 0..255
            const int bk = b4 / (BN / 4);       // 0..15
            const int bn4 = (b4 - bk * (BN/4)) * 4; // 0..60 step4

            const int g_k = k0 + bk;
            const int g_n = block_n + bn4;
            const float* base = Bb + (size_t)g_k * (size_t)N + (size_t)g_n;

            if (B_aligned && g_k < K && (g_n + 3) < N) {
                const float4 v = *reinterpret_cast<const float4*>(base);
                Bs[bk][bn4 + 0] = v.x;
                Bs[bk][bn4 + 1] = v.y;
                Bs[bk][bn4 + 2] = v.z;
                Bs[bk][bn4 + 3] = v.w;
            } else {
                Bs[bk][bn4 + 0] = (g_k < K && (g_n + 0) < N) ? ldg_f32(base + 0) : 0.0f;
                Bs[bk][bn4 + 1] = (g_k < K && (g_n + 1) < N) ? ldg_f32(base + 1) : 0.0f;
                Bs[bk][bn4 + 2] = (g_k < K && (g_n + 2) < N) ? ldg_f32(base + 2) : 0.0f;
                Bs[bk][bn4 + 3] = (g_k < K && (g_n + 3) < N) ? ldg_f32(base + 3) : 0.0f;
            }
        }

        __syncthreads();

        // Compute BK=16
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int am = ty * TM + i;
                a_frag[i] = As[am][kk];
            }

            float b_frag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int bn = tx * TN + j;
                b_frag[j] = Bs[kk][bn];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_m = out_m0 + i;
        if (g_m < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int g_n = out_n0 + j;
                if (g_n < N) {
                    Cb[(size_t)g_m * (size_t)N + (size_t)g_n] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.dim() == 3, "A must be 3D [batch, M, K]");
    TORCH_CHECK(B.dim() == 3, "B must be 3D [batch, K, N]");
    TORCH_CHECK(A.size(0) == B.size(0), "batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "A.size(2) (K) must equal B.size(1) (K)");

    const int batch = (int)A.size(0);
    const int M = (int)A.size(1);
    const int K = (int)A.size(2);
    const int N = (int)B.size(2);

    auto C = torch::empty({batch, M, N}, A.options());

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(16, 16, 1); // 256 threads
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM,
              batch);

    bmm_64x64x16_f32_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        batch, M, K, N
    );

    return C;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_bmm_opt64x64x16",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["bmm_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) using a custom CUDA kernel.
    Expects float32 CUDA contiguous inputs.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors for A and B.")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise RuntimeError("ModelNew currently supports float32 only.")
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return self.custom_ops_lib.bmm_cuda(A, B)