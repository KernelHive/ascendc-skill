import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# InstanceNorm2d forward (no affine, no running stats) - v7
# Improvements vs baseline:
#  - Reduce (sum, sumsq) together using warp shuffles + tiny cross-warp shared reduction
#    (avoids two separate CUB BlockReduce temp storages)
#  - Slight ILP via unrolled float4 loop
#  - Autotune threads: 128 for better occupancy when register-limited, 256 when HW huge
#  - Keep vectorized float4 IO when 16B-aligned
# ------------------------------------------------------------

instance_norm_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float2 warp_reduce_sum_float2(float2 v) {
    // Reduce within warp using shfl
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(0xffffffff, v.x, offset);
        v.y += __shfl_down_sync(0xffffffff, v.y, offset);
    }
    return v;
}

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void instancenorm2d_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps
) {
    int nc = (int)blockIdx.x;
    int n = nc / C;
    int c = nc - n * C;
    int HW = H * W;
    int base = ((n * C + c) * HW);

    const float* __restrict__ xb = x + base;
    float* __restrict__ yb = y + base;

    // vectorization eligibility
    bool aligned16 = ((((uintptr_t)xb) & 0xF) == 0) && ((((uintptr_t)yb) & 0xF) == 0);

    float2 acc;
    acc.x = 0.0f; // sum
    acc.y = 0.0f; // sumsq

    if (aligned16) {
        int hw4 = HW >> 2;
        int tail = HW & 3;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);

        // modest unroll for ILP
        int tid = (int)threadIdx.x;
        int stride = BLOCK_THREADS;

        int i = tid;
        for (; i + 1 * stride < hw4; i += 2 * stride) {
            float4 v0 = x4[i];
            float4 v1 = x4[i + stride];

            acc.x += (v0.x + v0.y) + (v0.z + v0.w);
            acc.y = fmaf(v0.x, v0.x, acc.y);
            acc.y = fmaf(v0.y, v0.y, acc.y);
            acc.y = fmaf(v0.z, v0.z, acc.y);
            acc.y = fmaf(v0.w, v0.w, acc.y);

            acc.x += (v1.x + v1.y) + (v1.z + v1.w);
            acc.y = fmaf(v1.x, v1.x, acc.y);
            acc.y = fmaf(v1.y, v1.y, acc.y);
            acc.y = fmaf(v1.z, v1.z, acc.y);
            acc.y = fmaf(v1.w, v1.w, acc.y);
        }
        for (; i < hw4; i += stride) {
            float4 v = x4[i];
            acc.x += (v.x + v.y) + (v.z + v.w);
            acc.y = fmaf(v.x, v.x, acc.y);
            acc.y = fmaf(v.y, v.y, acc.y);
            acc.y = fmaf(v.z, v.z, acc.y);
            acc.y = fmaf(v.w, v.w, acc.y);
        }

        if (tail) {
            int start = hw4 * 4;
            for (int t = (int)threadIdx.x; t < tail; t += BLOCK_THREADS) {
                float v = xb[start + t];
                acc.x += v;
                acc.y = fmaf(v, v, acc.y);
            }
        }
    } else {
        for (int i = (int)threadIdx.x; i < HW; i += BLOCK_THREADS) {
            float v = xb[i];
            acc.x += v;
            acc.y = fmaf(v, v, acc.y);
        }
    }

    // block reduction:
    // 1) reduce within each warp
    float2 w = warp_reduce_sum_float2(acc);

    // 2) cross-warp reduction via shared
    __shared__ float2 warp_partials[8]; // supports up to 256 threads (8 warps)
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;

    if (lane == 0) {
        warp_partials[warp] = w;
    }
    __syncthreads();

    float2 total;
    total.x = 0.0f;
    total.y = 0.0f;

    // first warp loads all warp partials
    if (warp == 0) {
        int num_warps = (BLOCK_THREADS + 31) >> 5;
        float2 v;
        v.x = (lane < num_warps) ? warp_partials[lane].x : 0.0f;
        v.y = (lane < num_warps) ? warp_partials[lane].y : 0.0f;
        v = warp_reduce_sum_float2(v);
        if (lane == 0) {
            warp_partials[0] = v;
        }
    }
    __syncthreads();
    total = warp_partials[0];

    float inv_denom = 1.0f / (float)HW;
    float mean = total.x * inv_denom;
    float ex2  = total.y * inv_denom;
    float var = ex2 - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    float invstd = rsqrtf(var + eps);

    // normalize
    if (aligned16) {
        int hw4 = HW >> 2;
        int tail = HW & 3;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(yb);

        for (int i4 = (int)threadIdx.x; i4 < hw4; i4 += BLOCK_THREADS) {
            float4 v = x4[i4];
            float4 o;
            o.x = (v.x - mean) * invstd;
            o.y = (v.y - mean) * invstd;
            o.z = (v.z - mean) * invstd;
            o.w = (v.w - mean) * invstd;
            y4[i4] = o;
        }
        if (tail) {
            int start = hw4 * 4;
            for (int t = (int)threadIdx.x; t < tail; t += BLOCK_THREADS) {
                float v = xb[start + t];
                yb[start + t] = (v - mean) * invstd;
            }
        }
    } else {
        for (int i = (int)threadIdx.x; i < HW; i += BLOCK_THREADS) {
            float v = xb[i];
            yb[i] = (v - mean) * invstd;
        }
    }
}

torch::Tensor instance_norm2d_forward_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW contiguous)");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto y = torch::empty_like(x);

    const int blocks = N * C;
    const int HW = H * W;

    // simple heuristic: prefer 256 threads for very large planes to maximize MLP,
    // but drop to 128 to improve occupancy when register-limited.
    if (HW >= (1 << 18)) { // ~262k elements
        constexpr int THREADS = 256;
        instancenorm2d_fwd_kernel<THREADS><<<blocks, THREADS>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W,
            (float)eps
        );
    } else {
        constexpr int THREADS = 128;
        instancenorm2d_fwd_kernel<THREADS><<<blocks, THREADS>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W,
            (float)eps
        );
    }

    return y;
}
"""

instance_norm_cpp_decl = r"""
torch::Tensor instance_norm2d_forward_cuda(torch::Tensor x, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_instance_norm_ops_v7",
    cpp_sources=instance_norm_cpp_decl,
    cuda_sources=instance_norm_cuda_src,
    functions=["instance_norm2d_forward_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized InstanceNorm2d forward for the common case:
    affine=False, track_running_stats=False, input float32 CUDA contiguous NCHW.
    """
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 4:
            raise RuntimeError("ModelNew expects a CUDA float32 4D NCHW tensor.")
        if x.size(1) != self.num_features:
            raise RuntimeError(f"Expected C={self.num_features}, got C={x.size(1)}.")
        x_contig = x.contiguous()
        return self.custom_ops_lib.instance_norm2d_forward_cuda(x_contig, self.eps)