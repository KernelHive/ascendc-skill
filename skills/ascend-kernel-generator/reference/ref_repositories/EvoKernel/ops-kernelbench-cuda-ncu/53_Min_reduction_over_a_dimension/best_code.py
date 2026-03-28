import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: min_reduction_over_a_dimension ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>
#include <limits>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_min(float v) {
    unsigned mask = 0xffffffffu;
    v = fminf(v, __shfl_down_sync(mask, v, 16));
    v = fminf(v, __shfl_down_sync(mask, v, 8));
    v = fminf(v, __shfl_down_sync(mask, v, 4));
    v = fminf(v, __shfl_down_sync(mask, v, 2));
    v = fminf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

// --------------------------------------
// dim=2 (contiguous reduction):
// x [D0,D1,D2] -> out [D0,D1]
// 1 warp -> 1 output element (i,j).
// --------------------------------------
template<int WARPS_PER_BLOCK, int UNROLL>
__global__ __launch_bounds__(WARPS_PER_BLOCK*32, 2)
void min_reduce_dim2_warp_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int D0, int D1, int D2
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_total_x = gridDim.x * WARPS_PER_BLOCK;

    int warp_global = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int out_linear = warp_global + (int)blockIdx.y * warps_total_x;

    const int total_out = D0 * D1;
    if (out_linear >= total_out) return;

    const int i = out_linear / D1;
    const int j = out_linear - i * D1;

    const int64_t base = ((int64_t)i * D1 + j) * (int64_t)D2;

    float vmin = INFINITY;
    int k0 = lane;
    int stride = 32 * UNROLL;

    for (int k = k0; k < D2; k += stride) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u * 32;
            if (kk < D2) vmin = fminf(vmin, ldg_f32(x + base + kk));
        }
    }

    float r = warp_min(vmin);
    if (lane == 0) out[(int64_t)i * D1 + j] = r;
}

// --------------------------------------
// dim=0 (reduce over i, stride D1*D2):
// x [D0,D1,D2] -> out [D1,D2]
// 1 warp -> 1 output element (j,k).
// --------------------------------------
template<int WARPS_PER_BLOCK, int UNROLL>
__global__ __launch_bounds__(WARPS_PER_BLOCK*32, 2)
void min_reduce_dim0_warp_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int D0, int D1, int D2
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_total_x = gridDim.x * WARPS_PER_BLOCK;

    int warp_global = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int out_linear = warp_global + (int)blockIdx.y * warps_total_x;

    const int total_out = D1 * D2;
    if (out_linear >= total_out) return;

    const int j = out_linear / D2;
    const int k = out_linear - j * D2;

    const int64_t step = (int64_t)D1 * (int64_t)D2;
    const int64_t base = (int64_t)j * (int64_t)D2 + k;

    float vmin = INFINITY;
    int i0 = lane;
    int stride = 32 * UNROLL;

    for (int i = i0; i < D0; i += stride) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int ii = i + u * 32;
            if (ii < D0) vmin = fminf(vmin, ldg_f32(x + base + (int64_t)ii * step));
        }
    }

    float r = warp_min(vmin);
    if (lane == 0) out[(int64_t)j * D2 + k] = r;
}

// --------------------------------------
// dim=1 fast path (hot path):
// x [B, D1, D2] -> out [B, D2]
//
// Each thread computes min over j for 4 consecutive k values (k4).
// This removes shared memory + syncthreads and improves coalescing/overhead.
// Grid: blockIdx.y = b, blockIdx.x covers k4 vectors.
// --------------------------------------
__global__ __launch_bounds__(256, 2)
void min_reduce_dim1_k4_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int D1, int D2
) {
    const int b = (int)blockIdx.y;
    if (b >= B) return;

    const int tid = (int)threadIdx.x;
    const int k4 = (((int)blockIdx.x * (int)blockDim.x) + tid) * 4;
    if (k4 >= D2) return;

    const int valid = (k4 + 3 < D2) ? 4 : (D2 - k4);

    const int64_t base_b = (int64_t)b * (int64_t)D1 * (int64_t)D2;
    int64_t ptr = base_b + (int64_t)k4;     // points to x[b, 0, k4]
    const int64_t step = (int64_t)D2;       // advance j

    float best0 = INFINITY, best1 = INFINITY, best2 = INFINITY, best3 = INFINITY;

    int j = 0;
    // Unroll by 4 for ILP
    for (; j + 3 < D1; j += 4, ptr += 4 * step) {
        const float* p0 = x + ptr;
        float v00 = ldg_f32(p0 + 0);
        float v01 = (valid > 1) ? ldg_f32(p0 + 1) : INFINITY;
        float v02 = (valid > 2) ? ldg_f32(p0 + 2) : INFINITY;
        float v03 = (valid > 3) ? ldg_f32(p0 + 3) : INFINITY;
        best0 = fminf(best0, v00);
        if (valid > 1) best1 = fminf(best1, v01);
        if (valid > 2) best2 = fminf(best2, v02);
        if (valid > 3) best3 = fminf(best3, v03);

        const float* p1 = p0 + step;
        float v10 = ldg_f32(p1 + 0);
        float v11 = (valid > 1) ? ldg_f32(p1 + 1) : INFINITY;
        float v12 = (valid > 2) ? ldg_f32(p1 + 2) : INFINITY;
        float v13 = (valid > 3) ? ldg_f32(p1 + 3) : INFINITY;
        best0 = fminf(best0, v10);
        if (valid > 1) best1 = fminf(best1, v11);
        if (valid > 2) best2 = fminf(best2, v12);
        if (valid > 3) best3 = fminf(best3, v13);

        const float* p2 = p1 + step;
        float v20 = ldg_f32(p2 + 0);
        float v21 = (valid > 1) ? ldg_f32(p2 + 1) : INFINITY;
        float v22 = (valid > 2) ? ldg_f32(p2 + 2) : INFINITY;
        float v23 = (valid > 3) ? ldg_f32(p2 + 3) : INFINITY;
        best0 = fminf(best0, v20);
        if (valid > 1) best1 = fminf(best1, v21);
        if (valid > 2) best2 = fminf(best2, v22);
        if (valid > 3) best3 = fminf(best3, v23);

        const float* p3 = p2 + step;
        float v30 = ldg_f32(p3 + 0);
        float v31 = (valid > 1) ? ldg_f32(p3 + 1) : INFINITY;
        float v32 = (valid > 2) ? ldg_f32(p3 + 2) : INFINITY;
        float v33 = (valid > 3) ? ldg_f32(p3 + 3) : INFINITY;
        best0 = fminf(best0, v30);
        if (valid > 1) best1 = fminf(best1, v31);
        if (valid > 2) best2 = fminf(best2, v32);
        if (valid > 3) best3 = fminf(best3, v33);
    }

    for (; j < D1; ++j, ptr += step) {
        const float* p = x + ptr;
        float v0 = ldg_f32(p + 0);
        float v1 = (valid > 1) ? ldg_f32(p + 1) : INFINITY;
        float v2 = (valid > 2) ? ldg_f32(p + 2) : INFINITY;
        float v3 = (valid > 3) ? ldg_f32(p + 3) : INFINITY;
        best0 = fminf(best0, v0);
        if (valid > 1) best1 = fminf(best1, v1);
        if (valid > 2) best2 = fminf(best2, v2);
        if (valid > 3) best3 = fminf(best3, v3);
    }

    const int64_t out_base = (int64_t)b * (int64_t)D2 + (int64_t)k4;
    out[out_base + 0] = best0;
    if (valid > 1) out[out_base + 1] = best1;
    if (valid > 2) out[out_base + 2] = best2;
    if (valid > 3) out[out_base + 3] = best3;
}

torch::Tensor min_reduction_over_a_dimension_cuda(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor [D0, D1, D2]");
    TORCH_CHECK(dim >= -3 && dim <= 2, "dim must be in [-3, 2] for a 3D tensor");

    int rdim = (int)dim;
    if (rdim < 0) rdim += 3;
    TORCH_CHECK(rdim >= 0 && rdim < 3, "normalized dim must be in [0,2]");

    if (!x.is_contiguous()) x = x.contiguous();

    const int D0 = (int)x.size(0);
    const int D1 = (int)x.size(1);
    const int D2 = (int)x.size(2);

    torch::Tensor out;
    if (rdim == 0) out = torch::empty({D1, D2}, x.options());
    else if (rdim == 1) out = torch::empty({D0, D2}, x.options());
    else out = torch::empty({D0, D1}, x.options());

    at::cuda::CUDAGuard device_guard(x.device());

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

    if (rdim == 1) {
        // 2D launch: y=b, x=k4 tiles
        constexpr int THREADS = 256;
        int k4_vecs = (D2 + 4 - 1) / 4;
        int blocks_x = ceil_div(k4_vecs, THREADS);
        if (blocks_x < 1) blocks_x = 1;
        if (blocks_x > 65535) blocks_x = 65535;
        dim3 block((unsigned)THREADS, 1, 1);
        dim3 grid((unsigned)blocks_x, (unsigned)D0, 1);
        min_reduce_dim1_k4_f32<<<grid, block>>>(xp, op, D0, D1, D2);
        return out;
    }

    // Warp-per-output kernels for dim=0 and dim=2
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    constexpr int UNROLL = 4;

    if (rdim == 2) {
        const int total_out = D0 * D1;
        int blocks_x = 256;
        int warps_per_row = blocks_x * WARPS_PER_BLOCK;
        int blocks_y = ceil_div(total_out, warps_per_row);
        if (blocks_y == 1) {
            blocks_x = ceil_div(total_out, WARPS_PER_BLOCK);
            if (blocks_x < 1) blocks_x = 1;
        }
        dim3 block((unsigned)THREADS, 1, 1);
        dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);
        min_reduce_dim2_warp_f32<WARPS_PER_BLOCK, UNROLL><<<grid, block>>>(xp, op, D0, D1, D2);
    } else { // rdim == 0
        const int total_out = D1 * D2;
        int blocks_x = 256;
        int warps_per_row = blocks_x * WARPS_PER_BLOCK;
        int blocks_y = ceil_div(total_out, warps_per_row);
        if (blocks_y == 1) {
            blocks_x = ceil_div(total_out, WARPS_PER_BLOCK);
            if (blocks_x < 1) blocks_x = 1;
        }
        dim3 block((unsigned)THREADS, 1, 1);
        dim3 grid((unsigned)blocks_x, (unsigned)blocks_y, 1);
        min_reduce_dim0_warp_f32<WARPS_PER_BLOCK, UNROLL><<<grid, block>>>(xp, op, D0, D1, D2);
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor min_reduction_over_a_dimension_cuda(torch::Tensor x, int64_t dim);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_min_reduce_dim_opt6_k4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["min_reduction_over_a_dimension_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Model that performs min reduction over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.min_reduction_over_a_dimension_cuda(x, self.dim)


# Keep original input helpers for compatibility with the provided scaffold.
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [1]