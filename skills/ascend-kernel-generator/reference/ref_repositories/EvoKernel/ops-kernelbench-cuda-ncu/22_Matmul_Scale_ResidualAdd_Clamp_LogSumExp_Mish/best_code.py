import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension ---------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// ----------------- helpers -----------------
__device__ __forceinline__ float clamp_f32(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ float softplus_f32(float x) {
    // stable softplus
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish_f32(float x) {
    float sp = softplus_f32(x);
    return x * tanhf(sp);
}

// Read-only load helper (works for global memory pointers)
__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

// ----------------- kernel -----------------
template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 3)
void fused_2pass_lse_mish_kernel(
    const float* __restrict__ X,  // [B,H]
    float* __restrict__ Y,        // [B,1] contiguous
    int B, int H,
    float scale,
    float clamp_min,
    float clamp_max
) {
    int row = (int)blockIdx.x;
    if (row >= B) return;

    const float s = 2.0f * scale; // scale then residual add (x+x)
    const float* __restrict__ row_ptr = X + (int64_t)row * (int64_t)H;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int WARP = 32;
    int num_warps = BLOCK / WARP;

    // ---------------- pass 1: max ----------------
    float tmax = -INFINITY;

    // vectorized load path if row_ptr is 16B aligned
    bool aligned16 = (((uintptr_t)row_ptr) & 0xF) == 0;
    int H4 = H & ~3;

    if (aligned16) {
        const float4* __restrict__ row4 = reinterpret_cast<const float4*>(row_ptr);
        int n4 = H4 >> 2;
        for (int idx4 = tid; idx4 < n4; idx4 += BLOCK) {
            float4 v4 = row4[idx4]; // true vector load
            float v0 = clamp_f32(v4.x * s, clamp_min, clamp_max);
            float v1 = clamp_f32(v4.y * s, clamp_min, clamp_max);
            float v2 = clamp_f32(v4.z * s, clamp_min, clamp_max);
            float v3 = clamp_f32(v4.w * s, clamp_min, clamp_max);
            tmax = fmaxf(tmax, v0);
            tmax = fmaxf(tmax, v1);
            tmax = fmaxf(tmax, v2);
            tmax = fmaxf(tmax, v3);
        }
    } else {
        for (int col = tid; col < H4; col += BLOCK) {
            float v = clamp_f32(ldg_f32(row_ptr + col) * s, clamp_min, clamp_max);
            tmax = fmaxf(tmax, v);
        }
    }

    // tail
    for (int col = H4 + tid; col < H; col += BLOCK) {
        float v = clamp_f32(ldg_f32(row_ptr + col) * s, clamp_min, clamp_max);
        tmax = fmaxf(tmax, v);
    }

    // reduce max within warp then across warps
    float wmax = warp_reduce_max(tmax);

    __shared__ float sh_max[32];
    if (lane == 0) sh_max[warp] = wmax;
    __syncthreads();

    float bmax = -INFINITY;
    if (warp == 0) {
        float v = (lane < num_warps) ? sh_max[lane] : -INFINITY;
        bmax = warp_reduce_max(v);
        if (lane == 0) sh_max[0] = bmax;
    }
    __syncthreads();
    bmax = sh_max[0];

    // ---------------- pass 2: sumexp ----------------
    // Use exp2/log2 to leverage fast-math paths.
    // exp(x) = exp2(x * log2(e)), log(x) = log2(x) * ln(2)
    const float LOG2E = 1.4426950408889634f;   // log2(e)
    const float LN2   = 0.6931471805599453f;   // ln(2)

    float tsum = 0.0f;

    if (aligned16) {
        const float4* __restrict__ row4 = reinterpret_cast<const float4*>(row_ptr);
        int n4 = H4 >> 2;
        for (int idx4 = tid; idx4 < n4; idx4 += BLOCK) {
            float4 v4 = row4[idx4];
            float v0 = clamp_f32(v4.x * s, clamp_min, clamp_max);
            float v1 = clamp_f32(v4.y * s, clamp_min, clamp_max);
            float v2 = clamp_f32(v4.z * s, clamp_min, clamp_max);
            float v3 = clamp_f32(v4.w * s, clamp_min, clamp_max);

            // exp(v - bmax)
            tsum += exp2f((v0 - bmax) * LOG2E);
            tsum += exp2f((v1 - bmax) * LOG2E);
            tsum += exp2f((v2 - bmax) * LOG2E);
            tsum += exp2f((v3 - bmax) * LOG2E);
        }
    } else {
        for (int col = tid; col < H4; col += BLOCK) {
            float v = clamp_f32(ldg_f32(row_ptr + col) * s, clamp_min, clamp_max);
            tsum += exp2f((v - bmax) * LOG2E);
        }
    }

    for (int col = H4 + tid; col < H; col += BLOCK) {
        float v = clamp_f32(ldg_f32(row_ptr + col) * s, clamp_min, clamp_max);
        tsum += exp2f((v - bmax) * LOG2E);
    }

    float wsum = warp_reduce_sum(tsum);

    __shared__ float sh_sum[32];
    if (lane == 0) sh_sum[warp] = wsum;
    __syncthreads();

    float bsum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? sh_sum[lane] : 0.0f;
        bsum = warp_reduce_sum(v);
        if (lane == 0) sh_sum[0] = bsum;
    }
    __syncthreads();
    bsum = sh_sum[0];

    if (tid == 0) {
        // lse = bmax + log(bsum)
        float lse = bmax + (log2f(bsum) * LN2);
        float mish = mish_f32(lse);
        Y[row] = lse * mish;
    }
}

torch::Tensor matmul_scale_residual_add_clamp_log_sum_exp_mish_cuda(
    torch::Tensor X,
    double scale_factor,
    double clamp_min,
    double clamp_max
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [B,H]");
    if (!X.is_contiguous()) X = X.contiguous();

    int B = (int)X.size(0);
    int H = (int)X.size(1);

    auto Y = torch::empty({B, 1}, X.options());

    // Heuristic: 256 for typical; 512 for very wide to reduce iterations.
    // 8192-wide benefits from 256 or 512 depending on GPU; keep both and branch on width.
    if (H >= 16384) {
        constexpr int BLOCK = 512;
        fused_2pass_lse_mish_kernel<BLOCK><<<dim3(B), dim3(BLOCK)>>>(
            (const float*)X.data_ptr<float>(),
            (float*)Y.data_ptr<float>(),
            B, H,
            (float)scale_factor,
            (float)clamp_min,
            (float)clamp_max
        );
    } else {
        constexpr int BLOCK = 256;
        fused_2pass_lse_mish_kernel<BLOCK><<<dim3(B), dim3(BLOCK)>>>(
            (const float*)X.data_ptr<float>(),
            (float*)Y.data_ptr<float>(),
            B, H,
            (float)scale_factor,
            (float)clamp_min,
            (float)clamp_max
        );
    }

    return Y;
}
"""

cpp_src = r"""
torch::Tensor matmul_scale_residual_add_clamp_log_sum_exp_mish_cuda(
    torch::Tensor X,
    double scale_factor,
    double clamp_min,
    double clamp_max
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_scale_residual_add_clamp_log_sum_exp_mish_v7_2pass_max_sumexp_exp2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_scale_residual_add_clamp_log_sum_exp_mish_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Linear (cuBLAS) + fused post-op on CUDA:
      scale + residual add (x+x) + clamp + logsumexp(dim=1, keepdim=True) + (x * mish(x))
    Output: [B, 1]
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)
        if not x.is_cuda:
            x = x * self.scale_factor
            x = x + x
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = x * torch.nn.functional.mish(x)
            return x
        return self.custom_ops_lib.matmul_scale_residual_add_clamp_log_sum_exp_mish_cuda(
            x, self.scale_factor, self.clamp_min, self.clamp_max
        )