import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension
# Fused: GroupNorm (NCDHW) + min + clamp + dropout
# float32, contiguous, CUDA only
#
# v5 optimizations over current baseline (v3):
# - Fast-path specialization when Cg == 2 (common for out_channels=16, groups=8):
#     * remove gamma/beta shared staging (less smem + less bookkeeping)
#     * avoid per-element division/mod by DHW for channel index
#     * keep gamma/beta scalars in registers and select via compare
# - Generic path remains essentially v3 (shared staged gamma/beta)
# - No persistent-grid loop (avoid failed pattern)
# - Keep float4 vectorized IO and xorshift RNG (avoid heavy 64-bit hashing)
# ----------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// xorshift32 RNG
__device__ __forceinline__ uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
__device__ __forceinline__ float u01_from_u32(uint32_t x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void gn3d_fused_min_clamp_dropout_kernel_v5(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, int DHW, int G,
    float eps,
    float min_value, float max_value,
    float p, uint64_t seed)
{
    int idx = (int)blockIdx.x; // (n,g)
    int n = idx / G;
    int g = idx - n * G;
    if (n >= N) return;

    int Cg = C / G;
    int D = Cg * DHW;

    const float* x_ptr = x + ((n * C + g * Cg) * DHW);
    float* out_ptr = out + ((n * C + g * Cg) * DHW);

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory layout depends on path:
    // - Generic: gamma[Cg] + beta[Cg] + sum[NUM_WARPS] + sumsq[NUM_WARPS]
    // - Cg==2 fast-path: sum[NUM_WARPS] + sumsq[NUM_WARPS] only
    extern __shared__ float smem[];
    float* sh_sum;
    float* sh_sumsq;
    float* s_gamma = nullptr;
    float* s_beta  = nullptr;

    bool fast2 = (Cg == 2);

    if (!fast2) {
        s_gamma = smem;            // [Cg]
        s_beta  = smem + Cg;       // [Cg]
        sh_sum  = smem + 2 * Cg;   // [NUM_WARPS]
        sh_sumsq= sh_sum + NUM_WARPS;
        for (int ci = tid; ci < Cg; ci += THREADS) {
            int c = g * Cg + ci;
            s_gamma[ci] = ldgf(gamma + c);
            s_beta[ci]  = ldgf(beta + c);
        }
        __syncthreads();
    } else {
        sh_sum   = smem;                 // [NUM_WARPS]
        sh_sumsq = smem + NUM_WARPS;     // [NUM_WARPS]
        // no staging; still need synchronization later for reduction scratch
    }

    // reduction
    float lsum = 0.0f;
    float lsq  = 0.0f;

    uintptr_t addr = (uintptr_t)(x_ptr);
    bool vec_ok = ((addr & 0xF) == 0) && ((D & 3) == 0);

    if (vec_ok) {
        const float4* xp4 = (const float4*)x_ptr;
        int D4 = D >> 2;
        for (int i4 = tid; i4 < D4; i4 += THREADS) {
            float4 v = xp4[i4];
            lsum += (v.x + v.y + v.z + v.w);
            lsq  = fmaf(v.x, v.x, lsq);
            lsq  = fmaf(v.y, v.y, lsq);
            lsq  = fmaf(v.z, v.z, lsq);
            lsq  = fmaf(v.w, v.w, lsq);
        }
    } else {
        for (int i = tid; i < D; i += THREADS) {
            float v = x_ptr[i];
            lsum += v;
            lsq = fmaf(v, v, lsq);
        }
    }

    float wsum = warp_sum(lsum);
    float wsq  = warp_sum(lsq);

    if (lane == 0) {
        sh_sum[warp] = wsum;
        sh_sumsq[warp] = wsq;
    }
    __syncthreads();

    __shared__ float sh_mean;
    __shared__ float sh_rstd;

    if (warp == 0) {
        float v_sum = (lane < NUM_WARPS) ? sh_sum[lane] : 0.0f;
        float v_sq  = (lane < NUM_WARPS) ? sh_sumsq[lane] : 0.0f;
        float bsum = warp_sum(v_sum);
        float bsq  = warp_sum(v_sq);
        if (lane == 0) {
            float invD = 1.0f / (float)D;
            float m = bsum * invD;
            float var = bsq * invD - m * m;
            var = var < 0.0f ? 0.0f : var;
            sh_mean = m;
            sh_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float m = sh_mean;
    float inv = sh_rstd;

    float scale = (p >= 1.0f) ? 0.0f : (1.0f / (1.0f - p));

    // per-thread RNG state
    uint32_t s = (uint32_t)(seed) ^ (uint32_t)(seed >> 32);
    s ^= (uint32_t)idx * 0x9E3779B9u;
    s ^= (uint32_t)tid * 0xBB67AE85u;
    s |= 1u;

    if (fast2) {
        // load gamma/beta scalars once
        int c0 = g * 2 + 0;
        int c1 = g * 2 + 1;
        float g0 = ldgf(gamma + c0);
        float g1 = ldgf(gamma + c1);
        float b0 = ldgf(beta + c0);
        float b1 = ldgf(beta + c1);

        if (vec_ok) {
            const float4* xp4 = (const float4*)x_ptr;
            float4* op4 = (float4*)out_ptr;
            int D4 = D >> 2;

            for (int i4 = tid; i4 < D4; i4 += THREADS) {
                int base = i4 << 2; // element index in [0, D)
                float4 v = xp4[i4];

                // For Cg==2, channel is 0 if idx<DHW else 1
                bool ch0_0 = (base + 0) < DHW;
                bool ch0_1 = (base + 1) < DHW;
                bool ch0_2 = (base + 2) < DHW;
                bool ch0_3 = (base + 3) < DHW;

                float gg0 = ch0_0 ? g0 : g1; float bb0 = ch0_0 ? b0 : b1;
                float gg1 = ch0_1 ? g0 : g1; float bb1 = ch0_1 ? b0 : b1;
                float gg2 = ch0_2 ? g0 : g1; float bb2 = ch0_2 ? b0 : b1;
                float gg3 = ch0_3 ? g0 : g1; float bb3 = ch0_3 ? b0 : b1;

                float y0 = (v.x - m) * inv; y0 = fmaf(y0, gg0, bb0);
                float y1 = (v.y - m) * inv; y1 = fmaf(y1, gg1, bb1);
                float y2 = (v.z - m) * inv; y2 = fmaf(y2, gg2, bb2);
                float y3 = (v.w - m) * inv; y3 = fmaf(y3, gg3, bb3);

                y0 = fminf(y0, min_value); y0 = clampf(y0, min_value, max_value);
                y1 = fminf(y1, min_value); y1 = clampf(y1, min_value, max_value);
                y2 = fminf(y2, min_value); y2 = clampf(y2, min_value, max_value);
                y3 = fminf(y3, min_value); y3 = clampf(y3, min_value, max_value);

                uint32_t r0 = xorshift32(s);
                uint32_t r1 = xorshift32(s);
                uint32_t r2 = xorshift32(s);
                uint32_t r3 = xorshift32(s);

                float u0 = u01_from_u32(r0);
                float u1 = u01_from_u32(r1);
                float u2 = u01_from_u32(r2);
                float u3 = u01_from_u32(r3);

                float k0 = (u0 >= p) ? scale : 0.0f;
                float k1 = (u1 >= p) ? scale : 0.0f;
                float k2 = (u2 >= p) ? scale : 0.0f;
                float k3 = (u3 >= p) ? scale : 0.0f;

                float4 o;
                o.x = y0 * k0;
                o.y = y1 * k1;
                o.z = y2 * k2;
                o.w = y3 * k3;
                op4[i4] = o;

                // decorrelate
                s ^= (uint32_t)(i4 + 1) * 0xA341316Cu;
                s = xorshift32(s);
            }
        } else {
            for (int i = tid; i < D; i += THREADS) {
                bool ch0 = (i < DHW);
                float gg = ch0 ? g0 : g1;
                float bb = ch0 ? b0 : b1;

                float v = x_ptr[i];
                float y = (v - m) * inv;
                y = fmaf(y, gg, bb);
                y = fminf(y, min_value);
                y = clampf(y, min_value, max_value);

                uint32_t r = xorshift32(s);
                float u = u01_from_u32(r);
                float keep = (u >= p) ? scale : 0.0f;
                out_ptr[i] = y * keep;

                s ^= (uint32_t)(i + 1) * 0xA341316Cu;
            }
        }
        return;
    }

    // generic path (staged gamma/beta)
    if (vec_ok) {
        const float4* xp4 = (const float4*)x_ptr;
        float4* op4 = (float4*)out_ptr;
        int D4 = D >> 2;

        for (int i4 = tid; i4 < D4; i4 += THREADS) {
            int base = i4 << 2;
            int ci0 = (base + 0) / DHW;
            int ci1 = (base + 1) / DHW;
            int ci2 = (base + 2) / DHW;
            int ci3 = (base + 3) / DHW;

            float4 v = xp4[i4];

            float y0 = (v.x - m) * inv; y0 = fmaf(y0, s_gamma[ci0], s_beta[ci0]);
            float y1 = (v.y - m) * inv; y1 = fmaf(y1, s_gamma[ci1], s_beta[ci1]);
            float y2 = (v.z - m) * inv; y2 = fmaf(y2, s_gamma[ci2], s_beta[ci2]);
            float y3 = (v.w - m) * inv; y3 = fmaf(y3, s_gamma[ci3], s_beta[ci3]);

            y0 = fminf(y0, min_value); y0 = clampf(y0, min_value, max_value);
            y1 = fminf(y1, min_value); y1 = clampf(y1, min_value, max_value);
            y2 = fminf(y2, min_value); y2 = clampf(y2, min_value, max_value);
            y3 = fminf(y3, min_value); y3 = clampf(y3, min_value, max_value);

            uint32_t r0 = xorshift32(s);
            uint32_t r1 = xorshift32(s);
            uint32_t r2 = xorshift32(s);
            uint32_t r3 = xorshift32(s);

            float u0 = u01_from_u32(r0);
            float u1 = u01_from_u32(r1);
            float u2 = u01_from_u32(r2);
            float u3 = u01_from_u32(r3);

            float k0 = (u0 >= p) ? scale : 0.0f;
            float k1 = (u1 >= p) ? scale : 0.0f;
            float k2 = (u2 >= p) ? scale : 0.0f;
            float k3 = (u3 >= p) ? scale : 0.0f;

            float4 o;
            o.x = y0 * k0;
            o.y = y1 * k1;
            o.z = y2 * k2;
            o.w = y3 * k3;
            op4[i4] = o;

            s ^= (uint32_t)(i4 + 1) * 0xA341316Cu;
            s = xorshift32(s);
        }
    } else {
        for (int i = tid; i < D; i += THREADS) {
            int ci = i / DHW;
            float v = x_ptr[i];
            float y = (v - m) * inv;
            y = fmaf(y, s_gamma[ci], s_beta[ci]);
            y = fminf(y, min_value);
            y = clampf(y, min_value, max_value);

            uint32_t r = xorshift32(s);
            float u = u01_from_u32(r);
            float keep = (u >= p) ? scale : 0.0f;
            out_ptr[i] = y * keep;

            s ^= (uint32_t)(i + 1) * 0xA341316Cu;
        }
    }
}

torch::Tensor gn3d_forward_min_clamp_dropout_cuda_v5(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    double min_value,
    double max_value,
    double dropout_p,
    uint64_t seed)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported for x");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32, "only float32 supported for gamma/beta");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() == 5, "expected x as NCDHW");
    TORCH_CHECK(num_groups > 0, "num_groups must be > 0");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t Dd = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    TORCH_CHECK(dropout_p >= 0.0 && dropout_p <= 1.0, "dropout_p must be in [0,1]");
    TORCH_CHECK((float)min_value <= (float)max_value, "min_value must be <= max_value");

    int64_t DHW = Dd * H * W;
    auto out = torch::empty_like(x);

    int blocks = (int)(N * num_groups);
    constexpr int THREADS = 256;
    int Cg = (int)(C / num_groups);

    // Shared memory:
    // - generic: (2*Cg + 2*NUM_WARPS) floats
    // - fast2:   (2*NUM_WARPS) floats
    int NUM_WARPS = THREADS / 32;
    size_t shmem_generic = (size_t)(2 * Cg + 2 * NUM_WARPS) * sizeof(float);
    size_t shmem_fast2   = (size_t)(2 * NUM_WARPS) * sizeof(float);
    size_t shmem = (Cg == 2) ? shmem_fast2 : shmem_generic;

    gn3d_fused_min_clamp_dropout_kernel_v5<THREADS><<<blocks, THREADS, shmem, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)N, (int)C, (int)DHW, (int)num_groups,
        (float)eps,
        (float)min_value, (float)max_value,
        (float)dropout_p, seed
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gn3d_forward_min_clamp_dropout_cuda_v5(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    double min_value,
    double max_value,
    double dropout_p,
    uint64_t seed);
"""

custom_ops_lib = load_inline(
    name="custom_conv3d_gn_min_clamp_dropout_ops_opt5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gn3d_forward_min_clamp_dropout_cuda_v5"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d -> (custom) GroupNorm -> min -> clamp -> dropout
    Fused CUDA kernel for float32 contiguous NCDHW.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.num_groups = int(groups)
        self.eps = 1e-5

        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.dropout_p = float(dropout_p)

        self.weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        b = self.bias
        if x.is_cuda and (not w.is_cuda or w.device != x.device):
            w = w.to(device=x.device)
            b = b.to(device=x.device)
        w = w.contiguous()
        b = b.contiguous()

        p = self.dropout_p if self.training else 0.0
        seed = int(torch.seed()) & 0xFFFFFFFFFFFFFFFF

        return self.custom_ops_lib.gn3d_forward_min_clamp_dropout_cuda_v5(
            x, w, b,
            self.num_groups,
            float(self.eps),
            float(self.min_value),
            float(self.max_value),
            float(p),
            seed
        )