import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
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

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ld_ro_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ld_ro_f32(const float* p) { return *p; }
#endif

// Warp maps to one (n,m) row.
// Each lane computes VEC outputs; warp covers WARP_TILE = 32*VEC columns of L.
template<int VEC, int WARP_TILE, bool L_IS_768>
__global__ __launch_bounds__(256, 2)
void t3d_matmul_warp_row_kernel(
    const float* __restrict__ A, // [N,M,K]
    const float* __restrict__ B, // [K,L]
    float* __restrict__ C,       // [N,M,L]
    int N, int M, int K, int L
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;      // 0..(blockDim/32-1)
    const int lane = tid & 31;

    const int n = (int)blockIdx.z;
    const int m = (int)blockIdx.y * (int)(blockDim.x >> 5) + warp; // one row per warp
    const int l_tile = (int)blockIdx.x;

    if (n >= N || m >= M) return;

    const int l_base = l_tile * WARP_TILE + lane * VEC;

    const float* Ap = A + ((int64_t)n * M + m) * (int64_t)K;
    float* Cp = C + ((int64_t)n * M + m) * (int64_t)L + l_base;

    float acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    const bool B_aligned16 = (((uintptr_t)B & 0xF) == 0);

    // Fast path: L==768, l_base always in range because grid.x = 768 / 128 = 6, and WARP_TILE=128
    if constexpr (L_IS_768) {
        // Use float4 loads for B and unroll K by 4 to reduce loop overhead.
        // Assumes VEC==4.
        const int64_t l_off = (int64_t)l_base;
        #pragma unroll 1
        for (int k = 0; k < K; k += 4) {
            float a0 = ld_ro_f32(Ap + (k + 0));
            float a1 = ld_ro_f32(Ap + (k + 1));
            float a2 = ld_ro_f32(Ap + (k + 2));
            float a3 = ld_ro_f32(Ap + (k + 3));

            // B rows are contiguous in L; load float4 at [k, l_base:l_base+3]
            const float4* b0p = (const float4*)(B + (int64_t)(k + 0) * 768 + l_off);
            const float4* b1p = (const float4*)(B + (int64_t)(k + 1) * 768 + l_off);
            const float4* b2p = (const float4*)(B + (int64_t)(k + 2) * 768 + l_off);
            const float4* b3p = (const float4*)(B + (int64_t)(k + 3) * 768 + l_off);

            float4 b0 = B_aligned16 ? *b0p : make_float4(
                ld_ro_f32(B + (int64_t)(k + 0) * 768 + l_off + 0),
                ld_ro_f32(B + (int64_t)(k + 0) * 768 + l_off + 1),
                ld_ro_f32(B + (int64_t)(k + 0) * 768 + l_off + 2),
                ld_ro_f32(B + (int64_t)(k + 0) * 768 + l_off + 3)
            );
            float4 b1 = B_aligned16 ? *b1p : make_float4(
                ld_ro_f32(B + (int64_t)(k + 1) * 768 + l_off + 0),
                ld_ro_f32(B + (int64_t)(k + 1) * 768 + l_off + 1),
                ld_ro_f32(B + (int64_t)(k + 1) * 768 + l_off + 2),
                ld_ro_f32(B + (int64_t)(k + 1) * 768 + l_off + 3)
            );
            float4 b2 = B_aligned16 ? *b2p : make_float4(
                ld_ro_f32(B + (int64_t)(k + 2) * 768 + l_off + 0),
                ld_ro_f32(B + (int64_t)(k + 2) * 768 + l_off + 1),
                ld_ro_f32(B + (int64_t)(k + 2) * 768 + l_off + 2),
                ld_ro_f32(B + (int64_t)(k + 2) * 768 + l_off + 3)
            );
            float4 b3 = B_aligned16 ? *b3p : make_float4(
                ld_ro_f32(B + (int64_t)(k + 3) * 768 + l_off + 0),
                ld_ro_f32(B + (int64_t)(k + 3) * 768 + l_off + 1),
                ld_ro_f32(B + (int64_t)(k + 3) * 768 + l_off + 2),
                ld_ro_f32(B + (int64_t)(k + 3) * 768 + l_off + 3)
            );

            acc[0] = fmaf(a0, b0.x, acc[0]); acc[1] = fmaf(a0, b0.y, acc[1]); acc[2] = fmaf(a0, b0.z, acc[2]); acc[3] = fmaf(a0, b0.w, acc[3]);
            acc[0] = fmaf(a1, b1.x, acc[0]); acc[1] = fmaf(a1, b1.y, acc[1]); acc[2] = fmaf(a1, b1.z, acc[2]); acc[3] = fmaf(a1, b1.w, acc[3]);
            acc[0] = fmaf(a2, b2.x, acc[0]); acc[1] = fmaf(a2, b2.y, acc[1]); acc[2] = fmaf(a2, b2.z, acc[2]); acc[3] = fmaf(a2, b2.w, acc[3]);
            acc[0] = fmaf(a3, b3.x, acc[0]); acc[1] = fmaf(a3, b3.y, acc[1]); acc[2] = fmaf(a3, b3.z, acc[2]); acc[3] = fmaf(a3, b3.w, acc[3]);
        }

        // Store: always in-bounds for L==768
        float4 out = make_float4(acc[0], acc[1], acc[2], acc[3]);
        *(float4*)Cp = out;
        return;
    }

    // Generic path: handle any L, any K. Use float4 loads when possible.
    // Loop K, accumulate.
    #pragma unroll 1
    for (int k = 0; k < K; ++k) {
        float a = ld_ro_f32(Ap + k);

        int l0 = l_base;
        if ((l0 + (VEC - 1)) < L) {
            // in bounds
            if constexpr (VEC == 4) {
                if (B_aligned16) {
                    const float4* p4 = (const float4*)(B + (int64_t)k * L + l0);
                    float4 b = *p4;
                    acc[0] = fmaf(a, b.x, acc[0]);
                    acc[1] = fmaf(a, b.y, acc[1]);
                    acc[2] = fmaf(a, b.z, acc[2]);
                    acc[3] = fmaf(a, b.w, acc[3]);
                } else {
                    acc[0] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + (l0 + 0)), acc[0]);
                    acc[1] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + (l0 + 1)), acc[1]);
                    acc[2] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + (l0 + 2)), acc[2]);
                    acc[3] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + (l0 + 3)), acc[3]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    acc[v] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + (l0 + v)), acc[v]);
                }
            }
        } else {
            // tail
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                int li = l0 + v;
                if (li < L) acc[v] = fmaf(a, ld_ro_f32(B + (int64_t)k * L + li), acc[v]);
            }
        }
    }

    // Store generic
    #pragma unroll
    for (int v = 0; v < VEC; ++v) {
        int li = l_base + v;
        if (li < L) Cp[v] = acc[v];
    }
}

torch::Tensor t3d_tensor_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.dim() == 3, "A must be 3D (N, M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K, L)");
    TORCH_CHECK(A.size(2) == B.size(0), "Inner dimension mismatch: A.size(2) must equal B.size(0)");

    int64_t N64 = A.size(0);
    int64_t M64 = A.size(1);
    int64_t K64 = A.size(2);
    int64_t L64 = B.size(1);

    TORCH_CHECK(N64 <= INT_MAX && M64 <= INT_MAX && K64 <= INT_MAX && L64 <= INT_MAX,
                "Tensor sizes must fit in int32 for this kernel");

    int N = (int)N64;
    int M = (int)M64;
    int K = (int)K64;
    int L = (int)L64;

    auto C = torch::empty({N64, M64, L64}, A.options());

    // Mapping:
    // - 256-thread CTA = 8 warps => computes 8 rows (m) per block
    // - each warp computes WARP_TILE=128 columns of L
    // - grid.x tiles L by 128
    constexpr int VEC = 4;
    constexpr int WARP_TILE = 128;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    dim3 block(THREADS, 1, 1);

    // grid.y is blocks over M, each block covers WARPS_PER_BLOCK rows
    dim3 grid((unsigned int)((L + WARP_TILE - 1) / WARP_TILE),
              (unsigned int)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
              (unsigned int)N);

    // Specialized fast path for the common shape L==768 with K multiple-of-4.
    if (L == 768 && (K % 4) == 0) {
        t3d_matmul_warp_row_kernel<VEC, WARP_TILE, true><<<grid, block>>>(
            (const float*)A.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            (float*)C.data_ptr<float>(),
            N, M, K, L
        );
    } else {
        t3d_matmul_warp_row_kernel<VEC, WARP_TILE, false><<<grid, block>>>(
            (const float*)A.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            (float*)C.data_ptr<float>(),
            N, M, K, L
        );
    }

    return C;
}
"""

cpp_source = r"""
torch::Tensor t3d_tensor_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_t3d_v5_warp_row",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["t3d_tensor_matrix_multiplication_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication via an optimized custom CUDA kernel:
        C[n, m, l] = sum_k A[n, m, k] * B[k, l]
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensors")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise RuntimeError("ModelNew custom kernel supports float32 only")
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return self.custom_ops_lib.t3d_tensor_matrix_multiplication_cuda(A, B)