import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads => 32 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int nwarps = (blockDim.x + 31) >> 5;
    float out = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum(out);

    __shared__ float total;
    if (threadIdx.x == 0) total = out;
    __syncthreads();
    return total;
}

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Fused row-wise L2 norm + normalization for 2D float32 CUDA tensors.
// y[n, d] = x[n, d] * rsqrt(sum_d x^2 + eps)
// eps is expected tiny; this is numerically close to x / (sqrt(sum)+eps) and avoids a divide.
__global__ void l2_norm_fused_vec2_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int64_t N, int64_t D,
                                         float eps) {
    for (int64_t row = (int64_t)blockIdx.x; row < N; row += (int64_t)gridDim.x) {
        const int64_t base = row * D;
        const float* __restrict__ xr = x + base;
        float* __restrict__ yr = y + base;

        // Check 8B alignment for float2 vectorization.
        uintptr_t ax = (uintptr_t)xr;
        uintptr_t ay = (uintptr_t)yr;
        const bool can_vec2 = ((ax & 0x7) == 0) && ((ay & 0x7) == 0);

        float sum = 0.0f;

        if (!can_vec2) {
            // Scalar path with ILP unroll.
            int64_t j = (int64_t)threadIdx.x;
            const int64_t stride = (int64_t)blockDim.x;

            for (; j + 3 * stride < D; j += 4 * stride) {
                float v0 = ldg_f32(xr + j);
                float v1 = ldg_f32(xr + j + stride);
                float v2 = ldg_f32(xr + j + 2 * stride);
                float v3 = ldg_f32(xr + j + 3 * stride);
                sum = fmaf(v0, v0, sum);
                sum = fmaf(v1, v1, sum);
                sum = fmaf(v2, v2, sum);
                sum = fmaf(v3, v3, sum);
            }
            for (; j < D; j += stride) {
                float v = ldg_f32(xr + j);
                sum = fmaf(v, v, sum);
            }

            float total = block_reduce_sum(sum);

            __shared__ float inv;
            if (threadIdx.x == 0) {
                // use rsqrt for speed; eps folded into sum domain
                inv = rsqrtf(total + eps);
            }
            __syncthreads();
            float s = inv;

            // Writeback scalar, ILP unroll
            j = (int64_t)threadIdx.x;
            for (; j + 3 * stride < D; j += 4 * stride) {
                float v0 = ldg_f32(xr + j);
                float v1 = ldg_f32(xr + j + stride);
                float v2 = ldg_f32(xr + j + 2 * stride);
                float v3 = ldg_f32(xr + j + 3 * stride);
                yr[j] = v0 * s;
                yr[j + stride] = v1 * s;
                yr[j + 2 * stride] = v2 * s;
                yr[j + 3 * stride] = v3 * s;
            }
            for (; j < D; j += stride) {
                yr[j] = ldg_f32(xr + j) * s;
            }
        } else {
            // Vec2 path: reduce over float2 pairs + odd tail.
            const int64_t D2 = D >> 1;
            const float2* __restrict__ x2 = reinterpret_cast<const float2*>(xr);

            int64_t j2 = (int64_t)threadIdx.x;
            const int64_t stride2 = (int64_t)blockDim.x;

            for (; j2 + 3 * stride2 < D2; j2 += 4 * stride2) {
                float2 a0 = x2[j2];
                float2 a1 = x2[j2 + stride2];
                float2 a2 = x2[j2 + 2 * stride2];
                float2 a3 = x2[j2 + 3 * stride2];
                sum = fmaf(a0.x, a0.x, sum); sum = fmaf(a0.y, a0.y, sum);
                sum = fmaf(a1.x, a1.x, sum); sum = fmaf(a1.y, a1.y, sum);
                sum = fmaf(a2.x, a2.x, sum); sum = fmaf(a2.y, a2.y, sum);
                sum = fmaf(a3.x, a3.x, sum); sum = fmaf(a3.y, a3.y, sum);
            }
            for (; j2 < D2; j2 += stride2) {
                float2 a = x2[j2];
                sum = fmaf(a.x, a.x, sum);
                sum = fmaf(a.y, a.y, sum);
            }

            if ((D & 1) && threadIdx.x == 0) {
                float v = ldg_f32(xr + (D2 << 1));
                sum = fmaf(v, v, sum);
            }

            float total = block_reduce_sum(sum);

            __shared__ float inv;
            if (threadIdx.x == 0) {
                inv = rsqrtf(total + eps);
            }
            __syncthreads();
            float s = inv;

            // Writeback vec2
            float2* __restrict__ y2 = reinterpret_cast<float2*>(yr);
            j2 = (int64_t)threadIdx.x;

            for (; j2 + 3 * stride2 < D2; j2 += 4 * stride2) {
                float2 a0 = x2[j2];
                float2 a1 = x2[j2 + stride2];
                float2 a2 = x2[j2 + 2 * stride2];
                float2 a3 = x2[j2 + 3 * stride2];

                y2[j2] = make_float2(a0.x * s, a0.y * s);
                y2[j2 + stride2] = make_float2(a1.x * s, a1.y * s);
                y2[j2 + 2 * stride2] = make_float2(a2.x * s, a2.y * s);
                y2[j2 + 3 * stride2] = make_float2(a3.x * s, a3.y * s);
            }
            for (; j2 < D2; j2 += stride2) {
                float2 a = x2[j2];
                y2[j2] = make_float2(a.x * s, a.y * s);
            }

            if ((D & 1) && threadIdx.x == 0) {
                yr[(D2 << 1)] = ldg_f32(xr + (D2 << 1)) * s;
            }
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "l2_norm_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "l2_norm_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 2, "l2_norm_cuda: only 2D tensors [N, D] are supported");
    TORCH_CHECK(x.is_contiguous(), "l2_norm_cuda: x must be contiguous");

    auto N = (int64_t)x.size(0);
    auto D = (int64_t)x.size(1);
    auto y = torch::empty_like(x);

    // More threads helps memory-level parallelism for very wide rows.
    const int threads = 512;

    // Cap grid to limit launch overhead and use persistent grid-stride.
    int blocks = (int)N;
    if (blocks > 8192) blocks = 8192;
    if (blocks < 1) blocks = 1;

    l2_norm_fused_vec2_kernel<<<(unsigned int)blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (int64_t)N, (int64_t)D,
        (float)eps
    );

    return y;
}
"""

cpp_source = r"""
torch::Tensor l2_norm_cuda(torch::Tensor x, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_l2_norm_opt_vec2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    L2 normalization using an optimized fused CUDA kernel.
    Assumes input is a contiguous 2D CUDA float32 tensor shaped [N, D].
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.l2_norm_cuda(x, self.eps)