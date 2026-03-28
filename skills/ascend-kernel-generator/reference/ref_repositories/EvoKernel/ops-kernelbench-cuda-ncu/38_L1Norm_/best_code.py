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

// Returns the block sum in all threads (via one shared broadcast).
static __forceinline__ __device__ float block_reduce_sum_1sync(float v) {
    __shared__ float smem[8];  // up to 256 threads -> 8 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        // only first warp loads partial sums
        out = (lane < ((blockDim.x + 31) >> 5)) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__global__ __launch_bounds__(256, 4)
void l1_norm_fused_vec_kernel(const float* __restrict__ x,
                              float* __restrict__ y,
                              int64_t N, int64_t D,
                              float eps) {
    int64_t row = (int64_t)blockIdx.x;
    if (row >= N) return;

    const int64_t base = row * D;
    const float* __restrict__ xr = x + base;
    float* __restrict__ yr = y + base;

    uintptr_t ax = (uintptr_t)xr;
    uintptr_t ay = (uintptr_t)yr;

    const bool can_vec4 = ((ax & 0xF) == 0) && ((ay & 0xF) == 0);
    const bool can_vec2 = ((ax & 0x7) == 0) && ((ay & 0x7) == 0);

    float sum = 0.0f;

    if (can_vec4) {
        int64_t D4 = D >> 2; // float4 count
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xr);

        int64_t j4 = (int64_t)threadIdx.x;
        int64_t stride4 = (int64_t)blockDim.x;

        // modest ILP
        for (; j4 + stride4 < D4; j4 += 2 * stride4) {
            float4 a0 = x4[j4];
            float4 a1 = x4[j4 + stride4];
            sum += fabsf(a0.x) + fabsf(a0.y) + fabsf(a0.z) + fabsf(a0.w);
            sum += fabsf(a1.x) + fabsf(a1.y) + fabsf(a1.z) + fabsf(a1.w);
        }
        for (; j4 < D4; j4 += stride4) {
            float4 a = x4[j4];
            sum += fabsf(a.x) + fabsf(a.y) + fabsf(a.z) + fabsf(a.w);
        }

        // tail 0..3
        int64_t tail = D & 3LL;
        if (tail) {
            int64_t tbase = (D4 << 2);
            int t = (int)threadIdx.x;
            if (t < (int)tail) sum += fabsf(ldg_f32(xr + tbase + t));
        }
    } else if (can_vec2) {
        int64_t D2 = D >> 1; // float2 count
        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(xr);

        int64_t j2 = (int64_t)threadIdx.x;
        int64_t stride2 = (int64_t)blockDim.x;

        for (; j2 + 3 * stride2 < D2; j2 += 4 * stride2) {
            float2 a0 = x2[j2];
            float2 a1 = x2[j2 + stride2];
            float2 a2 = x2[j2 + 2 * stride2];
            float2 a3 = x2[j2 + 3 * stride2];
            sum += fabsf(a0.x) + fabsf(a0.y)
                 + fabsf(a1.x) + fabsf(a1.y)
                 + fabsf(a2.x) + fabsf(a2.y)
                 + fabsf(a3.x) + fabsf(a3.y);
        }
        for (; j2 < D2; j2 += stride2) {
            float2 a = x2[j2];
            sum += fabsf(a.x) + fabsf(a.y);
        }

        if ((D & 1LL) && threadIdx.x == 0) {
            sum += fabsf(ldg_f32(xr + (D2 << 1)));
        }
    } else {
        int64_t j = (int64_t)threadIdx.x;
        int64_t stride = (int64_t)blockDim.x;

        for (; j + 3 * stride < D; j += 4 * stride) {
            float v0 = ldg_f32(xr + j);
            float v1 = ldg_f32(xr + j + stride);
            float v2 = ldg_f32(xr + j + 2 * stride);
            float v3 = ldg_f32(xr + j + 3 * stride);
            sum += fabsf(v0) + fabsf(v1) + fabsf(v2) + fabsf(v3);
        }
        for (; j < D; j += stride) {
            sum += fabsf(ldg_f32(xr + j));
        }
    }

    float total = block_reduce_sum_1sync(sum);

    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        float mean_abs = total / (float)D;
        if (mean_abs < eps) mean_abs = eps;
        inv_mean = 1.0f / mean_abs;
    }
    __syncthreads();
    float s = inv_mean;

    if (can_vec4) {
        int64_t D4 = D >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xr);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(yr);

        int64_t j4 = (int64_t)threadIdx.x;
        int64_t stride4 = (int64_t)blockDim.x;

        for (; j4 + stride4 < D4; j4 += 2 * stride4) {
            float4 a0 = x4[j4];
            float4 a1 = x4[j4 + stride4];

            a0.x *= s; a0.y *= s; a0.z *= s; a0.w *= s;
            a1.x *= s; a1.y *= s; a1.z *= s; a1.w *= s;

            y4[j4] = a0;
            y4[j4 + stride4] = a1;
        }
        for (; j4 < D4; j4 += stride4) {
            float4 a = x4[j4];
            a.x *= s; a.y *= s; a.z *= s; a.w *= s;
            y4[j4] = a;
        }

        int64_t tail = D & 3LL;
        if (tail) {
            int64_t tbase = (D4 << 2);
            int t = (int)threadIdx.x;
            if (t < (int)tail) yr[tbase + t] = ldg_f32(xr + tbase + t) * s;
        }
    } else if (can_vec2) {
        int64_t D2 = D >> 1;
        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(xr);
        float2* __restrict__ y2 = reinterpret_cast<float2*>(yr);

        int64_t j2 = (int64_t)threadIdx.x;
        int64_t stride2 = (int64_t)blockDim.x;

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

        if ((D & 1LL) && threadIdx.x == 0) {
            yr[(D2 << 1)] = ldg_f32(xr + (D2 << 1)) * s;
        }
    } else {
        int64_t j = (int64_t)threadIdx.x;
        int64_t stride = (int64_t)blockDim.x;

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
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "l1_norm_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "l1_norm_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 2, "l1_norm_cuda: only 2D tensors [N, D] are supported");
    TORCH_CHECK(x.is_contiguous(), "l1_norm_cuda: x must be contiguous");

    auto N = (int64_t)x.size(0);
    auto D = (int64_t)x.size(1);
    auto y = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (int)N;  // one CTA per row; do not cap (avoid prior failure mode)

    l1_norm_fused_vec_kernel<<<(unsigned int)blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, D,
        (float)eps
    );

    return y;
}
"""

cpp_source = r"""
torch::Tensor l1_norm_cuda(torch::Tensor x, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_l1_norm_opt3_vec4_vec2_onecta",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["l1_norm_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)

class ModelNew(nn.Module):
    """
    L1 normalization using a fused CUDA kernel:
      y = x / mean(abs(x), dim=1, keepdim=True)

    Fast path: contiguous 2D CUDA float32 tensor [N, D].
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.l1_norm_cuda(x, self.eps)