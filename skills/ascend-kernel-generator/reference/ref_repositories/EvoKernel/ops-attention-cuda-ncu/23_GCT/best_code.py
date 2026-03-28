import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------
# Custom CUDA extension: gct_fused_fastpath_v4 + generic fallback
# ---------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() TORCH_CHECK(cudaGetLastError() == cudaSuccess, cudaGetErrorString(cudaGetLastError()))
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    // Standard two-level reduction: warp then block.
    __shared__ float warp_sums[8]; // enough for up to 256 threads (8 warps)
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        out = (tid < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            out += __shfl_down_sync(0xffffffff, out, offset);
        }
    }
    return out;
}

// ---------------------------
// Generic baseline (3-stage) kernels (kept for generality)
// ---------------------------

__global__ void mean_hw_kernel(const float* __restrict__ x, float* __restrict__ y, int N, int C, int H, int W) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int NC = N * C;
    if (idx >= NC) return;

    int HW = H * W;
    const float* base = x + (int64_t)idx * (int64_t)HW;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        if (i >= HW) break;
        sum += base[i];
    }
    for (int i = 64; i < HW; ++i) sum += base[i];

    y[idx] = sum / (float)HW;
}

__global__ void per_n_stats_from_y_kernel(const float* __restrict__ y,
                                         float* __restrict__ mean,
                                         float* __restrict__ mean_x2,
                                         int N, int C) {
    int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N) return;

    const float* yn = y + (int64_t)n * (int64_t)C;

    float s1 = 0.0f;
    float s2 = 0.0f;
#pragma unroll 8
    for (int c = 0; c < C; ++c) {
        float v = yn[c];
        s1 += v;
        s2 += v * v;
    }
    float invC = 1.0f / (float)C;
    mean[n] = s1 * invC;
    mean_x2[n] = s2 * invC;
}

__global__ void gate_apply_fused_kernel(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ mean_x2,
                                       float* __restrict__ out,
                                       int NC, int C, int H, int W,
                                       float c_param, float eps) {
    int HW = H * W;
    int tid = (int)threadIdx.x;

    for (int nc = (int)blockIdx.x; nc < NC; nc += (int)gridDim.x) {
        int n = nc / C;

        float m   = ldg_f32(mean + n);
        float mx2 = ldg_f32(mean_x2 + n);
        float var = mx2 - m * m;
        var = var < 0.0f ? 0.0f : var;
        float inv_std = rsqrtf(var + eps);

        float ync = ldg_f32(y + nc);
        float y_norm = (ync - m) * inv_std;
        float t = -((y_norm * y_norm) * (0.5f * c_param));
        float g = __expf(t);

        const float* xbase = x + (int64_t)nc * (int64_t)HW;
        float* obase = out + (int64_t)nc * (int64_t)HW;

        uintptr_t xb = (uintptr_t)xbase;
        uintptr_t ob = (uintptr_t)obase;
        bool aligned16 = ((xb | ob) & 0xF) == 0;

        if (aligned16 && HW >= 4) {
            int vec4 = HW >> 2;
            int tail = HW & 3;

            const float4* x4 = reinterpret_cast<const float4*>(xbase);
            float4* o4 = reinterpret_cast<float4*>(obase);

            for (int i = tid; i < vec4; i += (int)blockDim.x) {
                float4 v = x4[i];
                v.x *= g; v.y *= g; v.z *= g; v.w *= g;
                o4[i] = v;
            }

            if (tail) {
                int base = vec4 << 2;
                for (int i = tid; i < tail; i += (int)blockDim.x) {
                    obase[base + i] = xbase[base + i] * g;
                }
            }
        } else {
            for (int i = tid; i < HW; i += (int)blockDim.x) {
                obase[i] = xbase[i] * g;
            }
        }
    }
}

// ---------------------------
// Fast path fused kernel for C=512, H=W=7
// One block per n.
// ---------------------------

__global__ __launch_bounds__(256, 2)
void gct_fused_c512_hw49_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int N,
                               float c_param,
                               float eps) {
    constexpr int C = 512;
    constexpr int HW = 49;
    constexpr float invHW = 1.0f / 49.0f;

    int n = (int)blockIdx.x;
    if (n >= N) return;

    int tid = (int)threadIdx.x;

    extern __shared__ float smem[];
    float* y_sh = smem;            // 512 floats
    float* g_sh = smem + C;        // 512 floats

    // 1) Compute y[n,c] = mean over HW
    // Parallelize over channels: each thread processes multiple channels.
    for (int c = tid; c < C; c += blockDim.x) {
        const float* xbase = x + ((int64_t)n * C + c) * (int64_t)HW;

        // 49 scalar loads (HW small). Keep simple; compiler unrolls.
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < HW; ++i) {
            s += xbase[i];
        }
        y_sh[c] = s * invHW;
    }
    __syncthreads();

    // 2) Reduce across channels for mean and mean_x2.
    float local_s1 = 0.0f;
    float local_s2 = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float v = y_sh[c];
        local_s1 += v;
        local_s2 += v * v;
    }

    float sum1 = block_reduce_sum(local_s1);
    float sum2 = block_reduce_sum(local_s2);

    __shared__ float mean_sh;
    __shared__ float inv_std_sh;
    if (tid == 0) {
        float invC = 1.0f / 512.0f;
        float m = sum1 * invC;
        float mx2 = sum2 * invC;
        float var = mx2 - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean_sh = m;
        inv_std_sh = rsqrtf(var + eps);
    }
    __syncthreads();

    float m = mean_sh;
    float inv_std = inv_std_sh;

    // 3) Compute gate per channel once and cache.
    for (int c = tid; c < C; c += blockDim.x) {
        float yn = (y_sh[c] - m) * inv_std;
        g_sh[c] = __expf(-((yn * yn) * (0.5f * c_param)));
    }
    __syncthreads();

    // 4) Apply: parallelize over (c, hw) with a linear index.
    // total = 512*49 = 25088 elements. 256 threads -> ~98 iters.
    // Use float4 for hw in [0..47] and scalar for tail hw=48.
    int total = C * HW;
    int linear_stride = blockDim.x;

    // Vectorized path for the first 48 positions per channel (12 float4s).
    // Map a "vec-linear" index over C*12 float4s.
    int total_vec4 = C * 12; // 512*12 = 6144 float4s (covers hw 0..47)
    for (int idx = tid; idx < total_vec4; idx += linear_stride) {
        int c = idx / 12;
        int j = idx - c * 12; // 0..11
        int hw4 = j;          // float4 index within channel
        float g = g_sh[c];

        const float* xbase = x + ((int64_t)n * C + c) * (int64_t)HW;
        float* obase = out + ((int64_t)n * C + c) * (int64_t)HW;

        const float* xptr = xbase + (hw4 * 4);
        float* optr = obase + (hw4 * 4);

        uintptr_t xb = (uintptr_t)xptr;
        uintptr_t ob = (uintptr_t)optr;
        if (((xb | ob) & 0xF) == 0) {
            float4 v = *reinterpret_cast<const float4*>(xptr);
            v.x *= g; v.y *= g; v.z *= g; v.w *= g;
            *reinterpret_cast<float4*>(optr) = v;
        } else {
            // Safe fallback (rare if contiguous)
            optr[0] = xptr[0] * g;
            optr[1] = xptr[1] * g;
            optr[2] = xptr[2] * g;
            optr[3] = xptr[3] * g;
        }
    }

    // Tail element hw=48 for each channel: 512 elements total.
    for (int c = tid; c < C; c += blockDim.x) {
        float g = g_sh[c];
        const float* xbase = x + ((int64_t)n * C + c) * (int64_t)HW;
        float* obase = out + ((int64_t)n * C + c) * (int64_t)HW;
        obase[48] = xbase[48] * g;
    }
}

torch::Tensor gct_forward_cuda(torch::Tensor x, double c, double eps) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor (N,C,H,W)");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    auto out = torch::empty_like(x);

    // Specialized fast path
    if (C == 512 && H == 7 && W == 7) {
        int threads = 256;
        int blocks = N;
        size_t shmem = (size_t)(512 + 512) * sizeof(float); // y + gate
        gct_fused_c512_hw49_kernel<<<blocks, threads, shmem>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            N,
            (float)c,
            (float)eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // Generic fallback (baseline 3-stage)
    auto y = torch::empty({N, C}, x.options());
    auto mean = torch::empty({N}, x.options());
    auto mean_x2 = torch::empty({N}, x.options());

    const int threads1 = 256;
    const int threads2 = 256;
    const int threads3 = 128;

    int NC = N * C;

    {
        int blocks = (NC + threads1 - 1) / threads1;
        mean_hw_kernel<<<blocks, threads1>>>(x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    {
        int blocks = (N + threads2 - 1) / threads2;
        per_n_stats_from_y_kernel<<<blocks, threads2>>>(y.data_ptr<float>(), mean.data_ptr<float>(), mean_x2.data_ptr<float>(), N, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    {
        int maxBlocks = 4096;
        int blocks = NC;
        if (blocks > maxBlocks) blocks = maxBlocks;
        gate_apply_fused_kernel<<<blocks, threads3>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            mean.data_ptr<float>(),
            mean_x2.data_ptr<float>(),
            out.data_ptr<float>(),
            NC, C, H, W,
            (float)c, (float)eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gct_forward_cuda(torch::Tensor x, double c, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gct_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gct_forward_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """GCT module using an optimized custom CUDA kernel."""

    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.c = float(c)
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor input")
        if x.dtype != torch.float32:
            raise RuntimeError("ModelNew requires float32 input")
        if not x.is_contiguous():
            x = x.contiguous()
        return custom_ops_lib.gct_forward_cuda(x, self.c, self.eps)