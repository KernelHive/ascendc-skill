import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA: fused diag(A) @ B == row-wise scaling of B by A
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// One-CTA-per-row tiled over columns. Keeps every warp on the same row stream.
// Per-thread ILP via small fixed unroll (UNROLL=2).
template<int THREADS, int UNROLL>
__global__ __launch_bounds__(THREADS, 4) void diag_scale_row_vec4_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M // elements, multiple of 4
) {
    const int row = (int)blockIdx.y;
    if (row >= N) return;

    const float a = ro_load_f32(A + row);

    const int vecM = M >> 2; // float4 per row
    const int tile = THREADS * UNROLL;

    const int vbase = (int)blockIdx.x * tile + (int)threadIdx.x;

    const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B + (int64_t)row * M);
    float4* __restrict__ C4 = reinterpret_cast<float4*>(C + (int64_t)row * M);

#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int v = vbase + u * THREADS;
        if (v < vecM) {
            float4 x = B4[v];
            x.x *= a; x.y *= a; x.z *= a; x.w *= a;
            C4[v] = x;
        }
    }
}

template<int THREADS, int UNROLL>
__global__ __launch_bounds__(THREADS, 4) void diag_scale_row_vec2_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M // elements, multiple of 2
) {
    const int row = (int)blockIdx.y;
    if (row >= N) return;

    const float a = ro_load_f32(A + row);

    const int vecM = M >> 1; // float2 per row
    const int tile = THREADS * UNROLL;

    const int vbase = (int)blockIdx.x * tile + (int)threadIdx.x;

    const float2* __restrict__ B2 = reinterpret_cast<const float2*>(B + (int64_t)row * M);
    float2* __restrict__ C2 = reinterpret_cast<float2*>(C + (int64_t)row * M);

#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int v = vbase + u * THREADS;
        if (v < vecM) {
            float2 x = B2[v];
            x.x *= a; x.y *= a;
            C2[v] = x;
        }
    }
}

template<int THREADS, int UNROLL>
__global__ __launch_bounds__(THREADS, 4) void diag_scale_row_scalar_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    const int row = (int)blockIdx.y;
    if (row >= N) return;

    const float a = ro_load_f32(A + row);

    const int tile = THREADS * UNROLL;
    const int jbase = (int)blockIdx.x * tile + (int)threadIdx.x;

    const int64_t base = (int64_t)row * (int64_t)M;

#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const int j = jbase + u * THREADS;
        if (j < M) {
            C[base + j] = a * B[base + j];
        }
    }
}

torch::Tensor matmul_with_diagonal_matrices_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 1, "A must be 1D of shape (N,)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D of shape (N, M)");
    TORCH_CHECK(B.size(0) == A.size(0), "B.size(0) must equal A.size(0)");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    const int64_t N64 = A_c.size(0);
    const int64_t M64 = B_c.size(1);
    TORCH_CHECK(N64 <= INT32_MAX && M64 <= INT32_MAX, "N and M must fit in int32 for this kernel");

    const int N = (int)N64;
    const int M = (int)M64;

    auto C = torch::empty({N64, M64}, B_c.options());

    const uintptr_t Bp = reinterpret_cast<uintptr_t>(B_c.data_ptr<float>());
    const uintptr_t Cp = reinterpret_cast<uintptr_t>(C.data_ptr<float>());
    const bool aligned16 = ((Bp | Cp) & 0xF) == 0;
    const bool aligned8  = ((Bp | Cp) & 0x7) == 0;

    // Occupancy/register-pressure tuned defaults:
    // 128-thread CTAs often allow more residency for simple memory kernels.
    constexpr int THREADS = 128;
    constexpr int UNROLL = 2; // modest ILP without bloating registers
    const int tile = THREADS * UNROLL;

    // grid.y = rows, grid.x = tiles of columns for that row
    dim3 block(THREADS, 1, 1);
    dim3 grid(1, (unsigned)N, 1);

    if (aligned16 && ((M & 3) == 0)) {
        const int vecM = M >> 2;
        const int grid_x = (int)ceil_div_i64((int64_t)vecM, (int64_t)tile);
        grid.x = (unsigned)grid_x;
        diag_scale_row_vec4_f32<THREADS, UNROLL><<<grid, block>>>(
            A_c.data_ptr<float>(),
            B_c.data_ptr<float>(),
            C.data_ptr<float>(),
            N, M
        );
    } else if (aligned8 && ((M & 1) == 0)) {
        const int vecM = M >> 1;
        const int grid_x = (int)ceil_div_i64((int64_t)vecM, (int64_t)tile);
        grid.x = (unsigned)grid_x;
        diag_scale_row_vec2_f32<THREADS, UNROLL><<<grid, block>>>(
            A_c.data_ptr<float>(),
            B_c.data_ptr<float>(),
            C.data_ptr<float>(),
            N, M
        );
    } else {
        const int grid_x = (int)ceil_div_i64((int64_t)M, (int64_t)tile);
        grid.x = (unsigned)grid_x;
        diag_scale_row_scalar_f32<THREADS, UNROLL><<<grid, block>>>(
            A_c.data_ptr<float>(),
            B_c.data_ptr<float>(),
            C.data_ptr<float>(),
            N, M
        );
    }

    return C;
}
"""

cpp_source = r"""
torch::Tensor matmul_with_diagonal_matrices_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_with_diagonal_matrices_opt6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_with_diagonal_matrices_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, A, B):
        return self.custom_ops_lib.matmul_with_diagonal_matrices_cuda(A, B)