import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAStream.h>

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

// One warp computes one flattened p=(i*J+j) row for a fixed batch b,
// and a tile of K columns. Each lane computes VEC contiguous k outputs.
// Warp tile in K: WARP_K = 32*VEC.
// grid:
//   x: tiles of K by WARP_K
//   y: warps over P=I*J (warps_per_block per block)
//   z: batch B
template<int VEC, int WARP_K, bool FAST_256_768>
__global__ __launch_bounds__(256, 2)
void t4d_bijl_lk_bijk_warp_kernel(
    const float* __restrict__ A,   // [B,I,J,L]
    const float* __restrict__ Bm,  // [L,K]
    float* __restrict__ C,         // [B,I,J,K]
    int B, int I, int J, int L, int K
) {
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;     // 0..WARPS_PER_BLOCK-1
    const int lane = tid & 31;

    const int b = (int)blockIdx.z;

    const int warps_per_block = (int)(blockDim.x >> 5);
    const int p = (int)blockIdx.y * warps_per_block + warp; // flattened (i,j)
    const int P = I * J;
    if (b >= B || p >= P) return;

    const int k_tile = (int)blockIdx.x;
    const int k_base = k_tile * WARP_K + lane * VEC;
    if (k_base >= K) return;

    const int ii = p / J;
    const int jj = p - ii * J;

    const float* __restrict__ Ap = A + (((int64_t)b * I + ii) * (int64_t)J + jj) * (int64_t)L;
    float* __restrict__ Cp = C + ((((int64_t)b * I + ii) * (int64_t)J + jj) * (int64_t)K + (int64_t)k_base);

    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    const bool B_aligned16 = (((uintptr_t)Bm & 0xF) == 0) && (((uintptr_t)(k_base * (int)sizeof(float)) & 0xF) == 0);

    if constexpr (FAST_256_768) {
        // L==256, K==768. K tiles are exactly 6 tiles when WARP_K=128.
        // k_base always valid due to launch grid.x = 6 and k_base < 768.
        // We still handle lane/VEC mapping; store is always in-bounds.
        const int64_t k_off = (int64_t)k_base;
        #pragma unroll 1
        for (int l0 = 0; l0 < 256; l0 += 4) {
            float a0 = ld_ro_f32(Ap + (l0 + 0));
            float a1 = ld_ro_f32(Ap + (l0 + 1));
            float a2 = ld_ro_f32(Ap + (l0 + 2));
            float a3 = ld_ro_f32(Ap + (l0 + 3));

            // Load B rows at [l, k_base:k_base+3] as float4 if aligned; else scalar.
            const float4* b0p = (const float4*)(Bm + (int64_t)(l0 + 0) * 768 + k_off);
            const float4* b1p = (const float4*)(Bm + (int64_t)(l0 + 1) * 768 + k_off);
            const float4* b2p = (const float4*)(Bm + (int64_t)(l0 + 2) * 768 + k_off);
            const float4* b3p = (const float4*)(Bm + (int64_t)(l0 + 3) * 768 + k_off);

            float4 b0, b1, b2, b3;
            if (B_aligned16) {
                b0 = *b0p; b1 = *b1p; b2 = *b2p; b3 = *b3p;
            } else {
                b0 = make_float4(
                    ld_ro_f32(Bm + (int64_t)(l0 + 0) * 768 + k_off + 0),
                    ld_ro_f32(Bm + (int64_t)(l0 + 0) * 768 + k_off + 1),
                    ld_ro_f32(Bm + (int64_t)(l0 + 0) * 768 + k_off + 2),
                    ld_ro_f32(Bm + (int64_t)(l0 + 0) * 768 + k_off + 3)
                );
                b1 = make_float4(
                    ld_ro_f32(Bm + (int64_t)(l0 + 1) * 768 + k_off + 0),
                    ld_ro_f32(Bm + (int64_t)(l0 + 1) * 768 + k_off + 1),
                    ld_ro_f32(Bm + (int64_t)(l0 + 1) * 768 + k_off + 2),
                    ld_ro_f32(Bm + (int64_t)(l0 + 1) * 768 + k_off + 3)
                );
                b2 = make_float4(
                    ld_ro_f32(Bm + (int64_t)(l0 + 2) * 768 + k_off + 0),
                    ld_ro_f32(Bm + (int64_t)(l0 + 2) * 768 + k_off + 1),
                    ld_ro_f32(Bm + (int64_t)(l0 + 2) * 768 + k_off + 2),
                    ld_ro_f32(Bm + (int64_t)(l0 + 2) * 768 + k_off + 3)
                );
                b3 = make_float4(
                    ld_ro_f32(Bm + (int64_t)(l0 + 3) * 768 + k_off + 0),
                    ld_ro_f32(Bm + (int64_t)(l0 + 3) * 768 + k_off + 1),
                    ld_ro_f32(Bm + (int64_t)(l0 + 3) * 768 + k_off + 2),
                    ld_ro_f32(Bm + (int64_t)(l0 + 3) * 768 + k_off + 3)
                );
            }

            // VEC is expected 4 in this specialization.
            acc[0] = fmaf(a0, b0.x, acc[0]); acc[1] = fmaf(a0, b0.y, acc[1]); acc[2] = fmaf(a0, b0.z, acc[2]); acc[3] = fmaf(a0, b0.w, acc[3]);
            acc[0] = fmaf(a1, b1.x, acc[0]); acc[1] = fmaf(a1, b1.y, acc[1]); acc[2] = fmaf(a1, b1.z, acc[2]); acc[3] = fmaf(a1, b1.w, acc[3]);
            acc[0] = fmaf(a2, b2.x, acc[0]); acc[1] = fmaf(a2, b2.y, acc[1]); acc[2] = fmaf(a2, b2.z, acc[2]); acc[3] = fmaf(a2, b2.w, acc[3]);
            acc[0] = fmaf(a3, b3.x, acc[0]); acc[1] = fmaf(a3, b3.y, acc[1]); acc[2] = fmaf(a3, b3.z, acc[2]); acc[3] = fmaf(a3, b3.w, acc[3]);
        }

        *(float4*)Cp = make_float4(acc[0], acc[1], acc[2], acc[3]);
        return;
    }

    // Generic path: any L,K. Iterate over L, load B in vector when possible.
    #pragma unroll 1
    for (int l = 0; l < L; ++l) {
        float a = ld_ro_f32(Ap + l);

        // If full vector in-bounds, do vectorized load.
        if ((k_base + (VEC - 1)) < K) {
            if constexpr (VEC == 4) {
                if (B_aligned16) {
                    const float4* p4 = (const float4*)(Bm + (int64_t)l * K + (int64_t)k_base);
                    float4 b4 = *p4;
                    acc[0] = fmaf(a, b4.x, acc[0]);
                    acc[1] = fmaf(a, b4.y, acc[1]);
                    acc[2] = fmaf(a, b4.z, acc[2]);
                    acc[3] = fmaf(a, b4.w, acc[3]);
                } else {
                    acc[0] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)k_base + 0), acc[0]);
                    acc[1] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)k_base + 1), acc[1]);
                    acc[2] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)k_base + 2), acc[2]);
                    acc[3] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)k_base + 3), acc[3]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    acc[v] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)(k_base + v)), acc[v]);
                }
            }
        } else {
            // tail
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                int kk = k_base + v;
                if (kk < K) acc[v] = fmaf(a, ld_ro_f32(Bm + (int64_t)l * K + (int64_t)kk), acc[v]);
            }
        }
    }

    // Store generic
    #pragma unroll
    for (int v = 0; v < VEC; ++v) {
        int kk = k_base + v;
        if (kk < K) Cp[v] = acc[v];
    }
}

torch::Tensor t4d_tensor_matrix_multiplication_cuda(torch::Tensor A4, torch::Tensor Bm) {
    CHECK_INPUT(A4);
    CHECK_INPUT(Bm);

    TORCH_CHECK(A4.dim() == 4, "A must be 4D (b, i, j, l)");
    TORCH_CHECK(Bm.dim() == 2, "B must be 2D (l, k)");
    TORCH_CHECK(A4.size(3) == Bm.size(0), "Inner dimension mismatch");

    int64_t B64 = A4.size(0);
    int64_t I64 = A4.size(1);
    int64_t J64 = A4.size(2);
    int64_t L64 = A4.size(3);
    int64_t K64 = Bm.size(1);

    TORCH_CHECK(B64 <= INT_MAX && I64 <= INT_MAX && J64 <= INT_MAX && L64 <= INT_MAX && K64 <= INT_MAX,
                "Tensor sizes must fit in int32");

    int B = (int)B64, I = (int)I64, J = (int)J64, L = (int)L64, K = (int)K64;

    auto C4 = torch::empty({B64, I64, J64, K64}, A4.options());

    // Warp-tiled K
    constexpr int VEC = 4;
    constexpr int WARP_K = 128;          // 32 lanes * 4 = 128 cols per warp tile
    constexpr int WARPS_PER_BLOCK = 8;   // 256 threads
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    dim3 block(THREADS, 1, 1);
    int P = I * J;

    dim3 grid((unsigned int)((K + WARP_K - 1) / WARP_K),
              (unsigned int)((P + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
              (unsigned int)B);

    auto stream = at::cuda::getDefaultCUDAStream();

    // Specialize for the dominant shape (L==256, K==768), and require VEC==4 and WARP_K==128.
    if (L == 256 && K == 768) {
        t4d_bijl_lk_bijk_warp_kernel<VEC, WARP_K, true><<<grid, block, 0, stream>>>(
            (const float*)A4.data_ptr<float>(),
            (const float*)Bm.data_ptr<float>(),
            (float*)C4.data_ptr<float>(),
            B, I, J, L, K
        );
    } else {
        t4d_bijl_lk_bijk_warp_kernel<VEC, WARP_K, false><<<grid, block, 0, stream>>>(
            (const float*)A4.data_ptr<float>(),
            (const float*)Bm.data_ptr<float>(),
            (float*)C4.data_ptr<float>(),
            B, I, J, L, K
        );
    }

    return C4;
}
"""

cpp_source = r"""
torch::Tensor t4d_tensor_matrix_multiplication_cuda(torch::Tensor A4, torch::Tensor Bm);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_t4d_warp_row_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["t4d_tensor_matrix_multiplication_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)

class ModelNew(nn.Module):
    """
    Custom CUDA implementation of:
        torch.einsum("bijl,lk->bijk", A, B)
    float32 CUDA-only.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("ModelNew expects CUDA tensors")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise RuntimeError("ModelNew custom kernel supports float32 only")
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return self.custom_ops_lib.t4d_tensor_matrix_multiplication_cuda(A, B)