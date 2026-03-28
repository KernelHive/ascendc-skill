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
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static __device__ __forceinline__ bool aligned16(const void* p) {
    return ((((uintptr_t)p) & 0xFu) == 0u);
}

// Block computes one row; multiple warps collaborate over K.
// B is tiled into shared memory for reuse across warps.
template<int WARPS, int BTILE>
__global__ __launch_bounds__(WARPS * 32, 3)
void matvec_blockreduce_btile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K
) {
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int row = (int)blockIdx.x;
    if (row >= M) return;

    const float* __restrict__ Arow = A + (int64_t)row * (int64_t)K;

    __shared__ float Bs[BTILE];        // B tile staged in shared
    __shared__ float warp_sums[WARPS]; // warp partial sums

    const bool a_al16 = aligned16(Arow);
    const bool b_al16 = aligned16(B);

    float total = 0.0f;

    // Iterate over B tiles
    for (int k0 = 0; k0 < K; k0 += BTILE) {
        const int tile_elems = (k0 + BTILE <= K) ? BTILE : (K - k0);

        // Cooperative load of B tile into shared.
        // Prefer float4 vectorized loads when possible.
        if (b_al16 && (tile_elems == BTILE) && ((BTILE & 3) == 0)) {
            // Total float4 = BTILE/4; distribute across threads.
            const int n4 = BTILE >> 2;
            for (int i4 = tid; i4 < n4; i4 += WARPS * 32) {
                const float4* p4 = reinterpret_cast<const float4*>(B + k0);
                float4 v = p4[i4];
                int o = i4 << 2;
                Bs[o + 0] = v.x;
                Bs[o + 1] = v.y;
                Bs[o + 2] = v.z;
                Bs[o + 3] = v.w;
            }
        } else {
            for (int i = tid; i < tile_elems; i += WARPS * 32) {
                Bs[i] = ldg_f32(B + k0 + i);
            }
        }
        __syncthreads();

        // Each warp processes a strided subset of the tile to increase parallelism.
        float sum = 0.0f;

        if (a_al16 && (tile_elems == BTILE) && ((BTILE & 3) == 0)) {
            // Vectorize A loads too, aligned and full tile.
            const float4* A4 = reinterpret_cast<const float4*>(Arow + k0);
            const float4* B4s = reinterpret_cast<const float4*>(Bs);
            const int n4 = BTILE >> 2;

            // Each warp covers i4 = warp + t*WARPS; lanes stride by 32 within warp.
            for (int i4 = warp + lane; i4 < n4; i4 += WARPS * 32) {
                float4 av = A4[i4];
                float4 bv = B4s[i4];
                sum = fmaf(av.x, bv.x, sum);
                sum = fmaf(av.y, bv.y, sum);
                sum = fmaf(av.z, bv.z, sum);
                sum = fmaf(av.w, bv.w, sum);
            }
        } else {
            // Scalar path
            for (int i = warp * 32 + lane; i < tile_elems; i += WARPS * 32) {
                sum = fmaf(Arow[k0 + i], Bs[i], sum);
            }
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();

        // Reduce warps within the block
        if (warp == 0) {
            float w = (lane < WARPS) ? warp_sums[lane] : 0.0f;
            w = warp_reduce_sum(w);
            if (lane == 0) total += w;
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row] = total;
    }
}

torch::Tensor matrix_vector_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.dim() == 2, "A must be 2D (M,K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,1)");
    TORCH_CHECK(B.size(1) == 1, "B must have shape (K,1)");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);

    auto C = torch::empty({M, 1}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    // One block per row. With M=2048 this is fine; for extremely large M, a 2D grid
    // and/or multi-block reduction would be needed.
    constexpr int WARPS = 8;              // 256 threads, more parallelism over K
    constexpr int THREADS = WARPS * 32;
    constexpr int BTILE = 4096;           // 16KB shared for B tile (float), good reuse

    dim3 block(THREADS);
    dim3 grid(M);

    matvec_blockreduce_btile_kernel<WARPS, BTILE><<<grid, block>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        M, K
    );

    return C;
}
"""

cpp_src = r"""
torch::Tensor matrix_vector_multiplication_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matvec_btile_blockreduce_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matrix_vector_multiplication_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Matrix-vector multiplication (C = A * B) using an optimized custom CUDA kernel.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        A = A.contiguous()
        B = B.contiguous()
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
        return self.custom_ops_lib.matrix_vector_multiplication_cuda(A, B)

# Keep helpers compatible with the provided scaffold
M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K, device="cuda", dtype=torch.float32)
    B = torch.rand(K, 1, device="cuda", dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []