import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#endif

static __device__ __forceinline__ float neg_inf_f32() {
    return -__int_as_float(0x7f800000); // -inf
}

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float4 ldg_f4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    float4 v;
    v.x = __ldg(&p->x);
    v.y = __ldg(&p->y);
    v.z = __ldg(&p->z);
    v.w = __ldg(&p->w);
    return v;
#else
    return *p;
#endif
}

static __device__ __forceinline__ float warp_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

// -------------------------------------------
// dim=2 (last-dim contiguous): x2d [outer, reduce] -> out [outer]
// 1 warp reduces 1 row.
// -------------------------------------------
template<int WARPS_PER_BLOCK, int UNROLL>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void max_reduce_lastdim_warp_rows_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int outer,
    int reduce
) {
    constexpr int WARP = 32;
    const int lane = (int)(threadIdx.x & (WARP - 1));
    const int warp_in_block = (int)(threadIdx.x >> 5);
    const int warps_in_grid = (int)gridDim.x * WARPS_PER_BLOCK;

    int warp_global = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    for (int row = warp_global; row < outer; row += warps_in_grid) {
        const float* row_ptr = x + (int64_t)row * (int64_t)reduce;

        float tmax = neg_inf_f32();

        const uintptr_t addr = (uintptr_t)row_ptr;
        const bool aligned16 = ((addr & 0xF) == 0);

        if (aligned16) {
            const int n4 = reduce >> 2;
            const int tail = reduce & 3;
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(row_ptr);

            int i = lane;
            for (; i + (UNROLL - 1) * 32 < n4; i += UNROLL * 32) {
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    float4 v = ldg_f4(p4 + (i + u * 32));
                    tmax = fmaxf(tmax, v.x);
                    tmax = fmaxf(tmax, v.y);
                    tmax = fmaxf(tmax, v.z);
                    tmax = fmaxf(tmax, v.w);
                }
            }
            for (; i < n4; i += 32) {
                float4 v = ldg_f4(p4 + i);
                tmax = fmaxf(tmax, v.x);
                tmax = fmaxf(tmax, v.y);
                tmax = fmaxf(tmax, v.z);
                tmax = fmaxf(tmax, v.w);
            }
            if (tail) {
                const float* tail_ptr = row_ptr + ((int64_t)n4 << 2);
                for (int t = lane; t < tail; t += 32) {
                    tmax = fmaxf(tmax, ldg_f32(tail_ptr + t));
                }
            }
        } else {
            for (int col = lane; col < reduce; col += 32) {
                tmax = fmaxf(tmax, ldg_f32(row_ptr + col));
            }
        }

        tmax = warp_max(tmax);
        if (lane == 0) out[row] = tmax;
    }
}

// -------------------------------------------
// dim=1 specialization for 3D contiguous x [B, D1, D2] -> out [B, D2]
// Each thread computes max over j for 4 consecutive k (k4).
// Avoids permute/contiguous and avoids cross-thread reduction.
// Uses float4 loads when k4 aligned and valid==4.
// -------------------------------------------
__global__ __launch_bounds__(256, 2)
void max_reduce_dim1_k4_f32(
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
    int64_t ptr = base_b + (int64_t)k4;   // x[b, 0, k4]
    const int64_t step = (int64_t)D2;     // advance j

    float best0 = neg_inf_f32();
    float best1 = neg_inf_f32();
    float best2 = neg_inf_f32();
    float best3 = neg_inf_f32();

    // If we have full 4-valid outputs and the pointer is 16B aligned, use float4
    const bool can_vec4 = (valid == 4) && (((uintptr_t)(x + ptr) & 0xF) == 0);

    int j = 0;
    if (can_vec4) {
        const float4* p4 = reinterpret_cast<const float4*>(x + ptr);
        // step in float4 units is D2/4? Not necessarily integer. So we must use byte addressing.
        // Instead, keep scalar pointer arithmetic but load float4 from (x+ptr) each iteration.
        for (; j + 3 < D1; j += 4, ptr += 4 * step) {
            float4 v0 = ldg_f4(reinterpret_cast<const float4*>(x + ptr));
            best0 = fmaxf(best0, v0.x); best1 = fmaxf(best1, v0.y);
            best2 = fmaxf(best2, v0.z); best3 = fmaxf(best3, v0.w);

            float4 v1 = ldg_f4(reinterpret_cast<const float4*>(x + ptr + step));
            best0 = fmaxf(best0, v1.x); best1 = fmaxf(best1, v1.y);
            best2 = fmaxf(best2, v1.z); best3 = fmaxf(best3, v1.w);

            float4 v2 = ldg_f4(reinterpret_cast<const float4*>(x + ptr + 2 * step));
            best0 = fmaxf(best0, v2.x); best1 = fmaxf(best1, v2.y);
            best2 = fmaxf(best2, v2.z); best3 = fmaxf(best3, v2.w);

            float4 v3 = ldg_f4(reinterpret_cast<const float4*>(x + ptr + 3 * step));
            best0 = fmaxf(best0, v3.x); best1 = fmaxf(best1, v3.y);
            best2 = fmaxf(best2, v3.z); best3 = fmaxf(best3, v3.w);
        }
        for (; j < D1; ++j, ptr += step) {
            float4 v = ldg_f4(reinterpret_cast<const float4*>(x + ptr));
            best0 = fmaxf(best0, v.x); best1 = fmaxf(best1, v.y);
            best2 = fmaxf(best2, v.z); best3 = fmaxf(best3, v.w);
        }
    } else {
        for (; j + 3 < D1; j += 4, ptr += 4 * step) {
            const float* p0 = x + ptr;
            float v00 = ldg_f32(p0 + 0);
            float v01 = (valid > 1) ? ldg_f32(p0 + 1) : neg_inf_f32();
            float v02 = (valid > 2) ? ldg_f32(p0 + 2) : neg_inf_f32();
            float v03 = (valid > 3) ? ldg_f32(p0 + 3) : neg_inf_f32();
            best0 = fmaxf(best0, v00);
            if (valid > 1) best1 = fmaxf(best1, v01);
            if (valid > 2) best2 = fmaxf(best2, v02);
            if (valid > 3) best3 = fmaxf(best3, v03);

            const float* p1 = p0 + step;
            float v10 = ldg_f32(p1 + 0);
            float v11 = (valid > 1) ? ldg_f32(p1 + 1) : neg_inf_f32();
            float v12 = (valid > 2) ? ldg_f32(p1 + 2) : neg_inf_f32();
            float v13 = (valid > 3) ? ldg_f32(p1 + 3) : neg_inf_f32();
            best0 = fmaxf(best0, v10);
            if (valid > 1) best1 = fmaxf(best1, v11);
            if (valid > 2) best2 = fmaxf(best2, v12);
            if (valid > 3) best3 = fmaxf(best3, v13);

            const float* p2 = p1 + step;
            float v20 = ldg_f32(p2 + 0);
            float v21 = (valid > 1) ? ldg_f32(p2 + 1) : neg_inf_f32();
            float v22 = (valid > 2) ? ldg_f32(p2 + 2) : neg_inf_f32();
            float v23 = (valid > 3) ? ldg_f32(p2 + 3) : neg_inf_f32();
            best0 = fmaxf(best0, v20);
            if (valid > 1) best1 = fmaxf(best1, v21);
            if (valid > 2) best2 = fmaxf(best2, v22);
            if (valid > 3) best3 = fmaxf(best3, v23);

            const float* p3 = p2 + step;
            float v30 = ldg_f32(p3 + 0);
            float v31 = (valid > 1) ? ldg_f32(p3 + 1) : neg_inf_f32();
            float v32 = (valid > 2) ? ldg_f32(p3 + 2) : neg_inf_f32();
            float v33 = (valid > 3) ? ldg_f32(p3 + 3) : neg_inf_f32();
            best0 = fmaxf(best0, v30);
            if (valid > 1) best1 = fmaxf(best1, v31);
            if (valid > 2) best2 = fmaxf(best2, v32);
            if (valid > 3) best3 = fmaxf(best3, v33);
        }

        for (; j < D1; ++j, ptr += step) {
            const float* p = x + ptr;
            float v0 = ldg_f32(p + 0);
            float v1 = (valid > 1) ? ldg_f32(p + 1) : neg_inf_f32();
            float v2 = (valid > 2) ? ldg_f32(p + 2) : neg_inf_f32();
            float v3 = (valid > 3) ? ldg_f32(p + 3) : neg_inf_f32();
            best0 = fmaxf(best0, v0);
            if (valid > 1) best1 = fmaxf(best1, v1);
            if (valid > 2) best2 = fmaxf(best2, v2);
            if (valid > 3) best3 = fmaxf(best3, v3);
        }
    }

    const int64_t out_base = (int64_t)b * (int64_t)D2 + (int64_t)k4;
    out[out_base + 0] = best0;
    if (valid > 1) out[out_base + 1] = best1;
    if (valid > 2) out[out_base + 2] = best2;
    if (valid > 3) out[out_base + 3] = best3;
}

// API 1: lastdim reduction on 2D [outer, reduce]
torch::Tensor max_reduce_lastdim_cuda_f32(torch::Tensor x2d) {
    CHECK_CUDA(x2d);
    CHECK_CONTIGUOUS(x2d);
    CHECK_FLOAT(x2d);
    TORCH_CHECK(x2d.dim() == 2, "x2d must be a 2D tensor [outer, reduce]");

    const int outer = (int)x2d.size(0);
    const int reduce = (int)x2d.size(1);

    auto out = torch::empty({outer}, x2d.options());

    const at::cuda::CUDAGuard device_guard(x2d.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    constexpr int UNROLL = 4;

    int grid = (outer + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (grid > 32768) grid = 32768;
    if (grid < 1) grid = 1;

    max_reduce_lastdim_warp_rows_f32<WARPS_PER_BLOCK, UNROLL>
        <<<grid, THREADS, 0, stream>>>(
            (const float*)x2d.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            outer,
            reduce
        );

    return out;
}

// API 2: dim=1 reduction for 3D contiguous [B,D1,D2] -> [B,D2]
torch::Tensor max_reduce_dim1_cuda_f32(torch::Tensor x3d) {
    CHECK_CUDA(x3d);
    CHECK_CONTIGUOUS(x3d);
    CHECK_FLOAT(x3d);
    TORCH_CHECK(x3d.dim() == 3, "x3d must be a 3D tensor [B, D1, D2]");

    const int B = (int)x3d.size(0);
    const int D1 = (int)x3d.size(1);
    const int D2 = (int)x3d.size(2);

    auto out = torch::empty({B, D2}, x3d.options());

    const at::cuda::CUDAGuard device_guard(x3d.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

    constexpr int THREADS = 256;
    const int k4_vecs = (D2 + 4 - 1) / 4;
    int blocks_x = ceil_div(k4_vecs, THREADS);
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 65535) blocks_x = 65535;

    dim3 block((unsigned)THREADS, 1, 1);
    dim3 grid((unsigned)blocks_x, (unsigned)B, 1);

    max_reduce_dim1_k4_f32<<<grid, block, 0, stream>>>(
        (const float*)x3d.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, D1, D2
    );

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor max_reduce_lastdim_cuda_f32(torch::Tensor x2d);
torch::Tensor max_reduce_dim1_cuda_f32(torch::Tensor x3d);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_max_reduce_opt7_dim1_k4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["max_reduce_lastdim_cuda_f32", "max_reduce_dim1_cuda_f32"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Max reduction over a specific dimension using optimized custom CUDA kernels.
    Fast-paths:
      - dim==last: warp-per-row reduction (2D view)
      - dim==1 for 3D contiguous: direct k4 streaming reduction (avoids permute)
    Falls back to torch.max otherwise.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if dim < 0:
            dim = x.dim() + dim

        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() < 1):
            return torch.max(x, dim=dim)[0]

        # Specialize the common benchmark shape: 3D reduce over dim==1 without permute
        if x.dim() == 3 and dim == 1:
            x_contig = x if x.is_contiguous() else x.contiguous()
            return custom_ops_lib.max_reduce_dim1_cuda_f32(x_contig)

        x_contig = x if x.is_contiguous() else x.contiguous()

        # Reduce last dim only; permute otherwise
        if dim != x_contig.dim() - 1:
            perm = [d for d in range(x_contig.dim()) if d != dim] + [dim]
            x_perm = x_contig.permute(perm).contiguous()

            out_shape = [x_contig.size(d) for d in range(x_contig.dim()) if d != dim]
            outer = 1
            for s in out_shape:
                outer *= int(s)
            reduce = int(x_contig.size(dim))

            x2d = x_perm.view(outer, reduce)
            out_flat = custom_ops_lib.max_reduce_lastdim_cuda_f32(x2d)
            return out_flat.view(out_shape)

        out_shape = list(x_contig.shape[:-1])
        outer = 1
        for s in out_shape:
            outer *= int(s)
        reduce = int(x_contig.size(-1))

        x2d = x_contig.view(outer, reduce)
        out_flat = custom_ops_lib.max_reduce_lastdim_cuda_f32(x2d)
        return out_flat.view(out_shape)