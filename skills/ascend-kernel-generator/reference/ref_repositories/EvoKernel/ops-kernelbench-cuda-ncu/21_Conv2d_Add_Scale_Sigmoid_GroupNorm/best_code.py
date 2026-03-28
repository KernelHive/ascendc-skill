import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused add+bias, scale, sigmoid, GroupNorm forward
# v5 improvements over v4 baseline:
#  - Hot-shape kernel (Cg=4, HxW=65536) keeps plane-specialized addressing
#    but reduces param load pressure by broadcasting params via shared memory.
#  - Switch sigmoid_fast from tanh-based to expf-based (fast-math __expf) to
#    reduce SFU pressure and improve throughput on common GPUs.
#  - Add simple software pipelining/prefetch of float4 loads to increase ILP
#    and hide memory latency in the hot kernel.
#  - Provide both 256-thread and 128-thread hot kernels; select at runtime to
#    mitigate register-pressure/occupancy issues across architectures.
#  - Generic kernel remains essentially v4 to avoid regressions.
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// exp-based fast sigmoid: 1/(1+exp(-x)). Under --use_fast_math, __expf is fast.
static inline __device__ float sigmoid_fast(float x) {
    float z = __expf(-x);
    return 1.0f / (1.0f + z);
}

static inline __device__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ void block_reduce_sum_sumsq(float sum, float sumsq,
                                                       float* sh_sum, float* sh_sumsq,
                                                       float &out_sum, float &out_sumsq) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;

    float wsum = warp_sum(sum);
    float wsq  = warp_sum(sumsq);

    if (lane == 0) { sh_sum[warp] = wsum; sh_sumsq[warp] = wsq; }
    __syncthreads();

    out_sum = 0.0f;
    out_sumsq = 0.0f;
    if (warp == 0) {
        float v1 = (lane < num_warps) ? sh_sum[lane] : 0.0f;
        float v2 = (lane < num_warps) ? sh_sumsq[lane] : 0.0f;
        out_sum = warp_sum(v1);
        out_sumsq = warp_sum(v2);
    }
}

__global__ __launch_bounds__(256, 2)
void fused_kernel_generic_v5(
    const float* __restrict__ x,     // [N,C,H,W]
    float* __restrict__ y,           // [N,C,H,W]
    const float* __restrict__ params,// [4*C] (bias, scale, gamma, beta)
    int N, int C, int HxW, int G, float eps)
{
    int tid = threadIdx.x;
    int num_warps = blockDim.x >> 5;

    extern __shared__ float shmem[];
    float* sh_sum   = shmem;
    float* sh_sumsq = shmem + num_warps;

    __shared__ float s_mean;
    __shared__ float s_rstd;

    int total = N * G;
    int Cg = C / G;

    const float* __restrict__ bias  = params + 0 * C;
    const float* __restrict__ scale = params + 1 * C;
    const float* __restrict__ gamma = params + 2 * C;
    const float* __restrict__ beta  = params + 3 * C;

    for (int idx = (int)blockIdx.x; idx < total; idx += (int)gridDim.x) {
        int n = idx / G;
        int g = idx - n * G;

        const float* x_ptr = x + ((n * C + g * Cg) * HxW);
        float* y_ptr = y + ((n * C + g * Cg) * HxW);

        int D = Cg * HxW;

        float sum = 0.0f;
        float sumsq = 0.0f;

        uintptr_t xaddr = (uintptr_t)x_ptr;
        bool vec_ok = ((xaddr & 0xF) == 0);
        int D4 = D >> 2;

        if (vec_ok) {
            const float4* x4 = reinterpret_cast<const float4*>(x_ptr);
            for (int i4 = tid; i4 < D4; i4 += blockDim.x) {
                int base = i4 << 2;

                int c0 = base / HxW;
                int c1 = (base + 1) / HxW;
                int c2 = (base + 2) / HxW;
                int c3 = (base + 3) / HxW;

                int ch0 = g * Cg + c0;
                int ch1 = g * Cg + c1;
                int ch2 = g * Cg + c2;
                int ch3 = g * Cg + c3;

#if __CUDA_ARCH__ >= 350
                float b0 = __ldg(bias + ch0),  s0 = __ldg(scale + ch0);
                float b1 = __ldg(bias + ch1),  s1 = __ldg(scale + ch1);
                float b2 = __ldg(bias + ch2),  s2 = __ldg(scale + ch2);
                float b3 = __ldg(bias + ch3),  s3 = __ldg(scale + ch3);
#else
                float b0 = bias[ch0],  s0 = scale[ch0];
                float b1 = bias[ch1],  s1 = scale[ch1];
                float b2 = bias[ch2],  s2 = scale[ch2];
                float b3 = bias[ch3],  s3 = scale[ch3];
#endif
                float4 v = x4[i4];
                float t0 = sigmoid_fast((v.x + b0) * s0);
                float t1 = sigmoid_fast((v.y + b1) * s1);
                float t2 = sigmoid_fast((v.z + b2) * s2);
                float t3 = sigmoid_fast((v.w + b3) * s3);
                sum   += (t0 + t1 + t2 + t3);
                sumsq += (t0*t0 + t1*t1 + t2*t2 + t3*t3);
            }
            for (int i = (D4<<2) + tid; i < D; i += blockDim.x) {
                int c_in_group = i / HxW;
                int ch = g * Cg + c_in_group;
#if __CUDA_ARCH__ >= 350
                float b = __ldg(bias + ch);
                float s = __ldg(scale + ch);
#else
                float b = bias[ch];
                float s = scale[ch];
#endif
                float t = sigmoid_fast((x_ptr[i] + b) * s);
                sum += t;
                sumsq += t*t;
            }
        } else {
            for (int i = tid; i < D; i += blockDim.x) {
                int c_in_group = i / HxW;
                int ch = g * Cg + c_in_group;
#if __CUDA_ARCH__ >= 350
                float b = __ldg(bias + ch);
                float s = __ldg(scale + ch);
#else
                float b = bias[ch];
                float s = scale[ch];
#endif
                float t = sigmoid_fast((x_ptr[i] + b) * s);
                sum += t;
                sumsq += t*t;
            }
        }

        float bsum, bsumsq;
        block_reduce_sum_sumsq(sum, sumsq, sh_sum, sh_sumsq, bsum, bsumsq);

        if (tid == 0) {
            float invD = 1.0f / (float)D;
            float mean = bsum * invD;
            float var = bsumsq * invD - mean * mean;
            var = var < 0.0f ? 0.0f : var;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
        __syncthreads();

        float m = s_mean;
        float inv = s_rstd;

        uintptr_t yaddr = (uintptr_t)y_ptr;
        bool vec_ok2 = vec_ok && ((yaddr & 0xF) == 0);

        if (vec_ok2) {
            const float4* x4 = reinterpret_cast<const float4*>(x_ptr);
            float4* y4 = reinterpret_cast<float4*>(y_ptr);
            for (int i4 = tid; i4 < D4; i4 += blockDim.x) {
                int base = i4 << 2;

                int c0 = base / HxW;
                int c1 = (base + 1) / HxW;
                int c2 = (base + 2) / HxW;
                int c3 = (base + 3) / HxW;

                int ch0 = g * Cg + c0;
                int ch1 = g * Cg + c1;
                int ch2 = g * Cg + c2;
                int ch3 = g * Cg + c3;

#if __CUDA_ARCH__ >= 350
                float b0 = __ldg(bias + ch0),  s0 = __ldg(scale + ch0);
                float b1 = __ldg(bias + ch1),  s1 = __ldg(scale + ch1);
                float b2 = __ldg(bias + ch2),  s2 = __ldg(scale + ch2);
                float b3 = __ldg(bias + ch3),  s3 = __ldg(scale + ch3);

                float ga0 = __ldg(gamma + ch0), be0 = __ldg(beta + ch0);
                float ga1 = __ldg(gamma + ch1), be1 = __ldg(beta + ch1);
                float ga2 = __ldg(gamma + ch2), be2 = __ldg(beta + ch2);
                float ga3 = __ldg(gamma + ch3), be3 = __ldg(beta + ch3);
#else
                float b0 = bias[ch0],  s0 = scale[ch0];
                float b1 = bias[ch1],  s1 = scale[ch1];
                float b2 = bias[ch2],  s2 = scale[ch2];
                float b3 = bias[ch3],  s3 = scale[ch3];

                float ga0 = gamma[ch0], be0 = beta[ch0];
                float ga1 = gamma[ch1], be1 = beta[ch1];
                float ga2 = gamma[ch2], be2 = beta[ch2];
                float ga3 = gamma[ch3], be3 = beta[ch3];
#endif
                float4 v = x4[i4];
                float t0 = sigmoid_fast((v.x + b0) * s0);
                float t1 = sigmoid_fast((v.y + b1) * s1);
                float t2 = sigmoid_fast((v.z + b2) * s2);
                float t3 = sigmoid_fast((v.w + b3) * s3);

                float4 o;
                o.x = ((t0 - m) * inv) * ga0 + be0;
                o.y = ((t1 - m) * inv) * ga1 + be1;
                o.z = ((t2 - m) * inv) * ga2 + be2;
                o.w = ((t3 - m) * inv) * ga3 + be3;
                y4[i4] = o;
            }
            for (int i = (D4<<2) + tid; i < D; i += blockDim.x) {
                int c_in_group = i / HxW;
                int ch = g * Cg + c_in_group;
#if __CUDA_ARCH__ >= 350
                float b = __ldg(bias + ch);
                float s = __ldg(scale + ch);
                float ga = __ldg(gamma + ch);
                float be = __ldg(beta + ch);
#else
                float b = bias[ch];
                float s = scale[ch];
                float ga = gamma[ch];
                float be = beta[ch];
#endif
                float t = sigmoid_fast((x_ptr[i] + b) * s);
                y_ptr[i] = ((t - m) * inv) * ga + be;
            }
        } else {
            for (int i = tid; i < D; i += blockDim.x) {
                int c_in_group = i / HxW;
                int ch = g * Cg + c_in_group;
#if __CUDA_ARCH__ >= 350
                float b = __ldg(bias + ch);
                float s = __ldg(scale + ch);
                float ga = __ldg(gamma + ch);
                float be = __ldg(beta + ch);
#else
                float b = bias[ch];
                float s = scale[ch];
                float ga = gamma[ch];
                float be = beta[ch];
#endif
                float t = sigmoid_fast((x_ptr[i] + b) * s);
                y_ptr[i] = ((t - m) * inv) * ga + be;
            }
        }

        __syncthreads();
    }
}

// Hot-shape kernels: Cg==4 and HxW==65536.
// Two variants for occupancy/reg tuning: threads=256 and threads=128.
// Use shared param broadcast to reduce param traffic and per-thread regs.

template<int THREADS, int MINBLOCKS>
__global__ __launch_bounds__(THREADS, MINBLOCKS)
void fused_kernel_cg4_hw65536_v5(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ params,
    int N, int C, int G, float eps)
{
    const int tid = threadIdx.x;
    const int HxW = 65536;
    const int Cg = 4;
    const int D = Cg * HxW;     // 262144
    const int V = HxW >> 2;     // 16384 float4 per plane

    const float* __restrict__ bias  = params + 0 * C;
    const float* __restrict__ scale = params + 1 * C;
    const float* __restrict__ gamma = params + 2 * C;
    const float* __restrict__ beta  = params + 3 * C;

    // reduction shared
    const int num_warps = THREADS / 32;
    __shared__ float sh_sum[8];    // max 8 warps for 256 threads
    __shared__ float sh_sumsq[8];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    // shared params (broadcast)
    __shared__ float sb[4], ss[4], sga[4], sbe[4];

    int total = N * G;

    for (int idx = (int)blockIdx.x; idx < total; idx += (int)gridDim.x) {
        int n = idx / G;
        int g = idx - n * G;
        int base_c = g * Cg;

        if (tid < 4) {
#if __CUDA_ARCH__ >= 350
            sb[tid]  = __ldg(bias  + base_c + tid);
            ss[tid]  = __ldg(scale + base_c + tid);
            sga[tid] = __ldg(gamma + base_c + tid);
            sbe[tid] = __ldg(beta  + base_c + tid);
#else
            sb[tid]  = bias[base_c + tid];
            ss[tid]  = scale[base_c + tid];
            sga[tid] = gamma[base_c + tid];
            sbe[tid] = beta[base_c + tid];
#endif
        }
        __syncthreads();

        const float* xg = x + (n * C + base_c) * HxW;
        float* yg = y + (n * C + base_c) * HxW;

        const float* x0 = xg + 0 * HxW;
        const float* x1 = xg + 1 * HxW;
        const float* x2 = xg + 2 * HxW;
        const float* x3 = xg + 3 * HxW;

        float* y0 = yg + 0 * HxW;
        float* y1 = yg + 1 * HxW;
        float* y2 = yg + 2 * HxW;
        float* y3 = yg + 3 * HxW;

        bool vec_ok = ((((uintptr_t)x0) & 0xF) == 0) &&
                      ((((uintptr_t)x1) & 0xF) == 0) &&
                      ((((uintptr_t)x2) & 0xF) == 0) &&
                      ((((uintptr_t)x3) & 0xF) == 0) &&
                      ((((uintptr_t)y0) & 0xF) == 0) &&
                      ((((uintptr_t)y1) & 0xF) == 0) &&
                      ((((uintptr_t)y2) & 0xF) == 0) &&
                      ((((uintptr_t)y3) & 0xF) == 0);

        float sum = 0.0f, sumsq = 0.0f;

        if (vec_ok) {
            const float4* a0 = reinterpret_cast<const float4*>(x0);
            const float4* a1 = reinterpret_cast<const float4*>(x1);
            const float4* a2 = reinterpret_cast<const float4*>(x2);
            const float4* a3 = reinterpret_cast<const float4*>(x3);

            // software pipeline: preload next vector
            int i = tid;
            float4 v0n, v1n, v2n, v3n;
            bool has_next = (i < V);
            if (has_next) {
                v0n = a0[i];
                v1n = a1[i];
                v2n = a2[i];
                v3n = a3[i];
            }

            for (; i < V; i += THREADS) {
                float4 v0 = v0n, v1 = v1n, v2 = v2n, v3 = v3n;
                int inext = i + THREADS;
                if (inext < V) {
                    v0n = a0[inext];
                    v1n = a1[inext];
                    v2n = a2[inext];
                    v3n = a3[inext];
                }

                float b0 = sb[0], b1 = sb[1], b2 = sb[2], b3 = sb[3];
                float s0 = ss[0], s1 = ss[1], s2 = ss[2], s3 = ss[3];

                float t00 = sigmoid_fast((v0.x + b0) * s0);
                float t01 = sigmoid_fast((v0.y + b0) * s0);
                float t02 = sigmoid_fast((v0.z + b0) * s0);
                float t03 = sigmoid_fast((v0.w + b0) * s0);

                float t10 = sigmoid_fast((v1.x + b1) * s1);
                float t11 = sigmoid_fast((v1.y + b1) * s1);
                float t12 = sigmoid_fast((v1.z + b1) * s1);
                float t13 = sigmoid_fast((v1.w + b1) * s1);

                float t20 = sigmoid_fast((v2.x + b2) * s2);
                float t21 = sigmoid_fast((v2.y + b2) * s2);
                float t22 = sigmoid_fast((v2.z + b2) * s2);
                float t23 = sigmoid_fast((v2.w + b2) * s2);

                float t30 = sigmoid_fast((v3.x + b3) * s3);
                float t31 = sigmoid_fast((v3.y + b3) * s3);
                float t32 = sigmoid_fast((v3.z + b3) * s3);
                float t33 = sigmoid_fast((v3.w + b3) * s3);

                float sA = (t00+t01+t02+t03) + (t10+t11+t12+t13) + (t20+t21+t22+t23) + (t30+t31+t32+t33);
                sum += sA;

                sumsq += (t00*t00 + t01*t01 + t02*t02 + t03*t03) +
                         (t10*t10 + t11*t11 + t12*t12 + t13*t13) +
                         (t20*t20 + t21*t21 + t22*t22 + t23*t23) +
                         (t30*t30 + t31*t31 + t32*t32 + t33*t33);
            }
        } else {
            for (int i = tid; i < HxW; i += THREADS) {
                float t0 = sigmoid_fast((x0[i] + sb[0]) * ss[0]);
                float t1 = sigmoid_fast((x1[i] + sb[1]) * ss[1]);
                float t2 = sigmoid_fast((x2[i] + sb[2]) * ss[2]);
                float t3 = sigmoid_fast((x3[i] + sb[3]) * ss[3]);
                sum += (t0 + t1 + t2 + t3);
                sumsq += (t0*t0 + t1*t1 + t2*t2 + t3*t3);
            }
        }

        float bsum, bsumsq;
        block_reduce_sum_sumsq(sum, sumsq, sh_sum, sh_sumsq, bsum, bsumsq);

        if (tid == 0) {
            float invD = 1.0f / (float)D;
            float mean = bsum * invD;
            float var = bsumsq * invD - mean * mean;
            var = var < 0.0f ? 0.0f : var;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
        __syncthreads();

        float m = s_mean;
        float inv = s_rstd;

        if (vec_ok) {
            const float4* a0 = reinterpret_cast<const float4*>(x0);
            const float4* a1 = reinterpret_cast<const float4*>(x1);
            const float4* a2 = reinterpret_cast<const float4*>(x2);
            const float4* a3 = reinterpret_cast<const float4*>(x3);

            float4* o0 = reinterpret_cast<float4*>(y0);
            float4* o1 = reinterpret_cast<float4*>(y1);
            float4* o2 = reinterpret_cast<float4*>(y2);
            float4* o3 = reinterpret_cast<float4*>(y3);

            int i = tid;
            float4 v0n, v1n, v2n, v3n;
            bool has_next = (i < V);
            if (has_next) {
                v0n = a0[i];
                v1n = a1[i];
                v2n = a2[i];
                v3n = a3[i];
            }

            float b0 = sb[0], b1 = sb[1], b2 = sb[2], b3 = sb[3];
            float s0 = ss[0], s1 = ss[1], s2 = ss[2], s3 = ss[3];
            float ga0 = sga[0], ga1 = sga[1], ga2 = sga[2], ga3 = sga[3];
            float be0 = sbe[0], be1 = sbe[1], be2 = sbe[2], be3 = sbe[3];

            for (; i < V; i += THREADS) {
                float4 v0 = v0n, v1 = v1n, v2 = v2n, v3 = v3n;
                int inext = i + THREADS;
                if (inext < V) {
                    v0n = a0[inext];
                    v1n = a1[inext];
                    v2n = a2[inext];
                    v3n = a3[inext];
                }

                float4 r0, r1, r2, r3;

                float t00 = sigmoid_fast((v0.x + b0) * s0);
                float t01 = sigmoid_fast((v0.y + b0) * s0);
                float t02 = sigmoid_fast((v0.z + b0) * s0);
                float t03 = sigmoid_fast((v0.w + b0) * s0);

                r0.x = ((t00 - m) * inv) * ga0 + be0;
                r0.y = ((t01 - m) * inv) * ga0 + be0;
                r0.z = ((t02 - m) * inv) * ga0 + be0;
                r0.w = ((t03 - m) * inv) * ga0 + be0;

                float t10 = sigmoid_fast((v1.x + b1) * s1);
                float t11 = sigmoid_fast((v1.y + b1) * s1);
                float t12 = sigmoid_fast((v1.z + b1) * s1);
                float t13 = sigmoid_fast((v1.w + b1) * s1);

                r1.x = ((t10 - m) * inv) * ga1 + be1;
                r1.y = ((t11 - m) * inv) * ga1 + be1;
                r1.z = ((t12 - m) * inv) * ga1 + be1;
                r1.w = ((t13 - m) * inv) * ga1 + be1;

                float t20 = sigmoid_fast((v2.x + b2) * s2);
                float t21 = sigmoid_fast((v2.y + b2) * s2);
                float t22 = sigmoid_fast((v2.z + b2) * s2);
                float t23 = sigmoid_fast((v2.w + b2) * s2);

                r2.x = ((t20 - m) * inv) * ga2 + be2;
                r2.y = ((t21 - m) * inv) * ga2 + be2;
                r2.z = ((t22 - m) * inv) * ga2 + be2;
                r2.w = ((t23 - m) * inv) * ga2 + be2;

                float t30 = sigmoid_fast((v3.x + b3) * s3);
                float t31 = sigmoid_fast((v3.y + b3) * s3);
                float t32 = sigmoid_fast((v3.z + b3) * s3);
                float t33 = sigmoid_fast((v3.w + b3) * s3);

                r3.x = ((t30 - m) * inv) * ga3 + be3;
                r3.y = ((t31 - m) * inv) * ga3 + be3;
                r3.z = ((t32 - m) * inv) * ga3 + be3;
                r3.w = ((t33 - m) * inv) * ga3 + be3;

                o0[i] = r0;
                o1[i] = r1;
                o2[i] = r2;
                o3[i] = r3;
            }
        } else {
            for (int i = tid; i < HxW; i += THREADS) {
                float t0 = sigmoid_fast((x0[i] + sb[0]) * ss[0]);
                float t1 = sigmoid_fast((x1[i] + sb[1]) * ss[1]);
                float t2 = sigmoid_fast((x2[i] + sb[2]) * ss[2]);
                float t3 = sigmoid_fast((x3[i] + sb[3]) * ss[3]);

                y0[i] = ((t0 - m) * inv) * sga[0] + sbe[0];
                y1[i] = ((t1 - m) * inv) * sga[1] + sbe[1];
                y2[i] = ((t2 - m) * inv) * sga[2] + sbe[2];
                y3[i] = ((t3 - m) * inv) * sga[3] + sbe[3];
            }
        }

        __syncthreads();
    }
}

torch::Tensor fused_forward_cuda_v5(torch::Tensor x,
                                   torch::Tensor params,
                                   int64_t num_groups,
                                   double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");

    TORCH_CHECK(params.is_cuda(), "params must be CUDA");
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.dim() == 1, "params must be flat [4*C]");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HxW = H * W;

    TORCH_CHECK((int)params.numel() == 4 * C, "params must have 4*C elements");
    int G = (int)num_groups;
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    auto y = torch::empty_like(x);

    int total = N * G;
    int blocks = total;
    int maxBlocks = 4096;
    if (blocks > maxBlocks) blocks = maxBlocks;
    if (blocks < 1) blocks = 1;

    int Cg = C / G;

    // Hot shape specialization
    if (Cg == 4 && HxW == 65536) {
        // Heuristic: 128 threads can reduce reg pressure and improve occupancy on some builds.
        // Use 256 by default; fall back to 128 when total blocks is small (less parallelism) to increase residency.
        // (Still safe; correctness identical.)
        if (blocks < 1024) {
            fused_kernel_cg4_hw65536_v5<128, 4><<<blocks, 128, 0>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)params.data_ptr<float>(),
                N, C, G, (float)eps
            );
        } else {
            fused_kernel_cg4_hw65536_v5<256, 2><<<blocks, 256, 0>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float*)params.data_ptr<float>(),
                N, C, G, (float)eps
            );
        }
    } else {
        int threads = 256;
        int num_warps = threads / 32;
        size_t shmem = (size_t)(2 * num_warps) * sizeof(float);

        fused_kernel_generic_v5<<<blocks, threads, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)params.data_ptr<float>(),
            N, C, HxW, G, (float)eps
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor fused_forward_cuda_v5(torch::Tensor x,
                                   torch::Tensor params,
                                   int64_t num_groups,
                                   double eps);
"""

custom_ops_lib = load_inline(
    name="custom_conv2d_add_scale_sigmoid_gn_ops_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_forward_cuda_v5"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Convolution kept as nn.Conv2d; fuse add+bias, scale, sigmoid, and GroupNorm
    into a single CUDA kernel (float32, contiguous NCHW).

    v5: hot-shape kernel uses shared param broadcast + exp-based fast sigmoid +
        simple prefetch + 128/256-thread variant selection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scale = nn.Parameter(torch.randn(scale_shape, dtype=torch.float32))

        self.num_groups = int(num_groups)
        self.eps = 1e-5

        self.gn_weight = nn.Parameter(torch.ones(out_channels, dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

        self.custom_ops_lib = custom_ops_lib
        self._packed_params = None
        self._packed_device = None
        self._packed_C = None

    def _get_packed_params(self, device: torch.device):
        C = self.gn_weight.numel()
        if (self._packed_params is not None and
            self._packed_device == device and
            self._packed_C == C and
            self._packed_params.is_cuda and
            self._packed_params.device == device):
            return self._packed_params

        bias_c = self.bias.contiguous().view(-1).to(device=device, dtype=torch.float32)
        scale_c = self.scale.contiguous().view(-1).to(device=device, dtype=torch.float32)
        gn_w = self.gn_weight.contiguous().to(device=device, dtype=torch.float32)
        gn_b = self.gn_bias.contiguous().to(device=device, dtype=torch.float32)

        packed = torch.cat([bias_c, scale_c, gn_w, gn_b], dim=0).contiguous()
        self._packed_params = packed
        self._packed_device = device
        self._packed_C = C
        return packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        if not x.is_cuda:
            b = self.bias.float()
            s = self.scale.float()
            y = torch.sigmoid((x + b) * s)
            return nn.functional.group_norm(y, self.num_groups, self.gn_weight, self.gn_bias, self.eps)

        params = self._get_packed_params(x.device)
        return self.custom_ops_lib.fused_forward_cuda_v5(x, params, int(self.num_groups), float(self.eps))