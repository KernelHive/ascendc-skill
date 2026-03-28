import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ extension: optimized sum reduction over dim=1 for 3D float32 CUDA tensors, keepdim=True
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// Block processes one (b, d2_tile). threadIdx.x spans columns (D2), threadIdx.y spans warps splitting D1.
// Cross-warp reduction is done via shared memory (WARPS x COLS) and one __syncthreads().
template<int COLS, int WARPS, int UNROLL_D1, bool VEC4>
__global__ __launch_bounds__(COLS*WARPS, 2)
void sum_dim1_keepdim_3d_f32_tiled(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int D1, int D2
) {
    const int b = (int)blockIdx.y;

    // grid-stride over tiles in x dimension for better scheduling flexibility
    const int tiles_per_grid = (int)gridDim.x;
    for (int tile = (int)blockIdx.x; tile < (int)((D2 + (VEC4 ? (COLS*4 - 1) : (COLS - 1))) / (VEC4 ? (COLS*4) : COLS)); tile += tiles_per_grid) {

        const int lane = (int)threadIdx.x;   // 0..COLS-1
        const int warp_id = (int)threadIdx.y; // 0..WARPS-1

        const int d2_base = tile * (VEC4 ? (COLS * 4) : COLS);

        if constexpr (!VEC4) {
            const int d2 = d2_base + lane;
            if (d2 >= D2) continue;

            float acc = 0.0f;

            // Each warp reduces D1 in a strided way; unroll to increase ILP
            int d1 = warp_id;
            const int stride = WARPS * UNROLL_D1;
            for (; d1 + (UNROLL_D1 - 1) * WARPS < D1; d1 += stride) {
#pragma unroll
                for (int u = 0; u < UNROLL_D1; ++u) {
                    int d1u = d1 + u * WARPS;
                    const int idx = (b * D1 + d1u) * D2 + d2;
                    acc += x[idx];
                }
            }
            // tail
            for (; d1 < D1; d1 += WARPS) {
                const int idx = (b * D1 + d1) * D2 + d2;
                acc += x[idx];
            }

            // Shared staging: one value per warp per column
            __shared__ float smem[WARPS][COLS];
            smem[warp_id][lane] = acc;
            __syncthreads();

            if (warp_id == 0) {
                float sumv = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) sumv += smem[w][lane];
                out[b * D2 + d2] = sumv;
            }
        } else {
            // Each lane handles 4 consecutive columns.
            const int d2 = d2_base + lane * 4;
            if (d2 >= D2) continue;

            // Since VEC4 is only enabled when D2 % 4 == 0, all row starts preserve 16B alignment
            // if the base pointer is 16B aligned. We still guard base alignment.
            float4 acc4; acc4.x = acc4.y = acc4.z = acc4.w = 0.0f;

            int d1 = warp_id;
            const int stride = WARPS * UNROLL_D1;
            for (; d1 + (UNROLL_D1 - 1) * WARPS < D1; d1 += stride) {
#pragma unroll
                for (int u = 0; u < UNROLL_D1; ++u) {
                    int d1u = d1 + u * WARPS;
                    const float* row = x + (b * D1 + d1u) * D2 + d2;

                    // base-alignment guard (safe because D2%4==0 => row alignment invariant across d1u)
                    uintptr_t addr = (uintptr_t)row;
                    if ((addr & 0xF) == 0 && (d2 + 3) < D2) {
                        float4 v = *reinterpret_cast<const float4*>(row);
                        acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                    } else {
                        // conservative fallback (also handles tail, though VEC4 usually implies no tail)
                        if (d2 + 0 < D2) acc4.x += row[0];
                        if (d2 + 1 < D2) acc4.y += row[1];
                        if (d2 + 2 < D2) acc4.z += row[2];
                        if (d2 + 3 < D2) acc4.w += row[3];
                    }
                }
            }
            for (; d1 < D1; d1 += WARPS) {
                const float* row = x + (b * D1 + d1) * D2 + d2;
                uintptr_t addr = (uintptr_t)row;
                if ((addr & 0xF) == 0 && (d2 + 3) < D2) {
                    float4 v = *reinterpret_cast<const float4*>(row);
                    acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                } else {
                    if (d2 + 0 < D2) acc4.x += row[0];
                    if (d2 + 1 < D2) acc4.y += row[1];
                    if (d2 + 2 < D2) acc4.z += row[2];
                    if (d2 + 3 < D2) acc4.w += row[3];
                }
            }

            __shared__ float4 smem4[WARPS][COLS];
            smem4[warp_id][lane] = acc4;
            __syncthreads();

            if (warp_id == 0) {
                float4 sum4; sum4.x = sum4.y = sum4.z = sum4.w = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    float4 v = smem4[w][lane];
                    sum4.x += v.x; sum4.y += v.y; sum4.z += v.z; sum4.w += v.w;
                }
                float* o = out + b * D2 + d2;
                if (d2 + 3 < D2) {
                    o[0] = sum4.x; o[1] = sum4.y; o[2] = sum4.z; o[3] = sum4.w;
                } else {
                    if (d2 + 0 < D2) o[0] = sum4.x;
                    if (d2 + 1 < D2) o[1] = sum4.y;
                    if (d2 + 2 < D2) o[2] = sum4.z;
                    if (d2 + 3 < D2) o[3] = sum4.w;
                }
            }
        }
    }
}

torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor [B, D1, D2]");

    const int B = (int)x.size(0);
    const int D1 = (int)x.size(1);
    const int D2 = (int)x.size(2);

    auto out3d = torch::empty({B, 1, D2}, x.options());
    // flatten output for simpler indexing: [B, D2]
    auto out2d = out3d.view({B, D2});

    constexpr int COLS = 128;     // columns per block (scalar) or lanes per block (vec4)
    constexpr int WARPS = 4;      // warps per block to split D1 and hide latency
    constexpr int UNROLL = 4;     // ILP

    // Only enable vec4 when stride preserves alignment for all rows: requires D2 % 4 == 0.
    bool use_vec4 = (D2 % 4 == 0);

    dim3 block(COLS, WARPS, 1);

    int tiles = 0;
    if (use_vec4) tiles = (D2 + (COLS * 4) - 1) / (COLS * 4);
    else          tiles = (D2 + COLS - 1) / COLS;

    // Use a modest grid.x to allow grid-stride tiling and better residency across SMs.
    // Cap to avoid excessive launch overhead.
    int grid_x = tiles;
    if (grid_x > 4096) grid_x = 4096;
    dim3 grid(grid_x, B, 1);

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out2d.data_ptr<float>();

    if (use_vec4) {
        sum_dim1_keepdim_3d_f32_tiled<COLS, WARPS, UNROLL, true><<<grid, block, 0>>>(
            xp, op, B, D1, D2
        );
    } else {
        sum_dim1_keepdim_3d_f32_tiled<COLS, WARPS, UNROLL, false><<<grid, block, 0>>>(
            xp, op, B, D1, D2
        );
    }

    return out3d;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sum_dim1_keepdim_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_sum_reduction_dim1_keepdim_ext_opt3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sum_dim1_keepdim_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized replacement model using a custom CUDA kernel for sum reduction over dim=1 with keepdim=True.
    Specialized for input shape [B, D1, D2] float32 CUDA contiguous tensors.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.sum(x, dim=self.dim, keepdim=True)
        if (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 3 or (not x.is_contiguous()):
            return torch.sum(x, dim=self.dim, keepdim=True)
        return self.custom_ops_lib.sum_dim1_keepdim_cuda(x)