import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA source: mean reduction over dimension 1 for a 3D tensor [B, D1, D2] -> [B, D2]
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

template<int BLOCK_X, int BLOCK_Y, int UNROLL_I>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y, 2)
void mean_dim1_tiled_f32_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int B, int D1, int D2) {
    // 2D block:
    // - x dimension: columns j in a tile (coalesced along D2)
    // - y dimension: cooperative reduction over D1
    constexpr int THREADS = BLOCK_X * BLOCK_Y;
    int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    int b = (int)blockIdx.y;
    int j0 = ((int)blockIdx.x) * BLOCK_X + (int)threadIdx.x;
    if (b >= B || j0 >= D2) return;

    // Each thread accumulates partial sum for its column j0 over i = threadIdx.y + k*BLOCK_Y
    float sum = 0.0f;

    // Base pointer to x[b, 0, j0]
    // x index: ((b * D1 + i) * D2 + j)
    int base = (b * D1) * D2 + j0;

    // Unrolled loop over i
    int i = (int)threadIdx.y;
    int stride = BLOCK_Y;

    // Process main unrolled chunks
    int i_end_unroll = D1 - (UNROLL_I * stride);
    for (; i <= i_end_unroll; i += UNROLL_I * stride) {
#pragma unroll
        for (int u = 0; u < UNROLL_I; ++u) {
            int iu = i + u * stride;
            sum += x[base + iu * D2];
        }
    }
    // Tail
    for (; i < D1; i += stride) {
        sum += x[base + i * D2];
    }

    // Reduce across BLOCK_Y threads that share same j0 (same threadIdx.x)
    // Do reduction by first reducing within each warp, then across warps using shared memory.
    // Layout of threads: tid = y*BLOCK_X + x; consecutive tid increments in x then y.
    // We'll reduce over all threads with same x; this is a "striped" reduction.
    // Approach: each thread writes its sum to shared, then do tree over y (BLOCK_Y is small).
    __shared__ float shmem[THREADS];
    shmem[tid] = sum;
    __syncthreads();

    // Tree reduce over y dimension for each x
    // BLOCK_Y is compile-time constant (e.g., 8), so unroll it.
#pragma unroll
    for (int offset = BLOCK_Y / 2; offset > 0; offset >>= 1) {
        if ((int)threadIdx.y < offset) {
            shmem[tid] += shmem[tid + offset * BLOCK_X];
        }
        __syncthreads();
    }

    if ((int)threadIdx.y == 0) {
        out[b * D2 + j0] = shmem[threadIdx.x] * (1.0f / (float)D1);
    }
}

// Vectorized along D2: each thread computes 2 columns (j, j+1) when possible.
// This helps for odd D2 like 4095: most of the row is covered by pairs, last col handled by scalar kernel.
template<int BLOCK_X, int BLOCK_Y, int UNROLL_I>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y, 2)
void mean_dim1_tiled_f32x2_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int B, int D1, int D2) {
    constexpr int THREADS = BLOCK_X * BLOCK_Y;
    int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    int b = (int)blockIdx.y;
    int j_pair0 = ((int)blockIdx.x) * (BLOCK_X * 2) + ((int)threadIdx.x * 2);
    if (b >= B || j_pair0 >= D2) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int base0 = (b * D1) * D2 + j_pair0;

    int i = (int)threadIdx.y;
    int stride = BLOCK_Y;

    int i_end_unroll = D1 - (UNROLL_I * stride);
    for (; i <= i_end_unroll; i += UNROLL_I * stride) {
#pragma unroll
        for (int u = 0; u < UNROLL_I; ++u) {
            int iu = i + u * stride;
            const float* ptr = x + base0 + iu * D2;
            // Try vector-like paired loads, safe with bounds checks
            sum0 += ptr[0];
            if (j_pair0 + 1 < D2) sum1 += ptr[1];
        }
    }
    for (; i < D1; i += stride) {
        const float* ptr = x + base0 + i * D2;
        sum0 += ptr[0];
        if (j_pair0 + 1 < D2) sum1 += ptr[1];
    }

    __shared__ float sh0[THREADS];
    __shared__ float sh1[THREADS];
    sh0[tid] = sum0;
    sh1[tid] = sum1;
    __syncthreads();

#pragma unroll
    for (int offset = BLOCK_Y / 2; offset > 0; offset >>= 1) {
        if ((int)threadIdx.y < offset) {
            sh0[tid] += sh0[tid + offset * BLOCK_X];
            sh1[tid] += sh1[tid + offset * BLOCK_X];
        }
        __syncthreads();
    }

    if ((int)threadIdx.y == 0) {
        float inv = 1.0f / (float)D1;
        int out_base = b * D2 + j_pair0;
        out[out_base] = sh0[threadIdx.x] * inv;
        if (j_pair0 + 1 < D2) out[out_base + 1] = sh1[threadIdx.x] * inv;
    }
}

torch::Tensor mean_dim1_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 3, "mean_dim1_cuda: x must be 3D [B, D1, D2]");
    CHECK_CONTIGUOUS(x);

    const int B = (int)x.size(0);
    const int D1 = (int)x.size(1);
    const int D2 = (int)x.size(2);

    auto out = torch::empty({B, D2}, x.options());

    // Tune: BLOCK_X controls j-tile width (coalescing), BLOCK_Y controls reduction parallelism over D1.
    // Keep shared memory modest and occupancy decent.
    constexpr int BLOCK_X = 128; // 4 warps across x-dim
    constexpr int BLOCK_Y = 8;   // 8-way split over D1
    constexpr int UNROLL_I = 4;  // ILP in inner loop

    dim3 threads(BLOCK_X, BLOCK_Y, 1);

    // Prefer x2 path to cut loop overhead and improve memory transaction efficiency without requiring D2%4==0.
    // Grid x covers pairs (2 columns per thread).
    {
        int tiles_x2 = (D2 + (BLOCK_X * 2 - 1)) / (BLOCK_X * 2);
        dim3 blocks(tiles_x2, B, 1);
        mean_dim1_tiled_f32x2_kernel<BLOCK_X, BLOCK_Y, UNROLL_I>
            <<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, D1, D2);
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor mean_dim1_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_mean_reduction_opt_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["mean_dim1_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Uses an optimized custom CUDA kernel to compute mean reduction over dimension=1.
    Specialized for 3D contiguous float32 CUDA tensors: [B, D1, D2] -> [B, D2].
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim != 1:
            raise ValueError("ModelNew currently provides a custom CUDA kernel specialized for dim=1.")
        self.dim = dim
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor input.")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.mean_dim1_cuda(x)