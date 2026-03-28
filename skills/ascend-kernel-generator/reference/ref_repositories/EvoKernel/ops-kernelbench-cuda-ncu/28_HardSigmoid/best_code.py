import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------
# Custom CUDA op: HardSigmoid
# y = clamp(x * 1/6 + 0.5, 0, 1)
# Optimizations:
# - float4/float2 vectorization
# - per-thread ILP (multiple vectors per loop iteration)
# - optional half/half2 path to cut bandwidth for FP16
# - simple, low-overhead launch sizing (no 2D indexing, no occupancy API calls)
# ---------------------------------------------------------

hardsigmoid_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

static __forceinline__ __device__ float clamp01_f32(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

static __forceinline__ __device__ float hsigmoid_f32(float v) {
    // y = clamp(v/6 + 0.5, 0, 1)
    return clamp01_f32(fmaf(v, (1.0f / 6.0f), 0.5f));
}

static __forceinline__ __device__ half clamp01_f16(half v) {
    // clamp in f32 for correctness, return f16
    float f = __half2float(v);
    f = clamp01_f32(f);
    return __float2half_rn(f);
}

static __forceinline__ __device__ half hsigmoid_f16(half v) {
    float f = __half2float(v);
    f = hsigmoid_f32(f);
    return __float2half_rn(f);
}

static __forceinline__ __device__ half2 clamp01_f16x2(half2 v) {
    // clamp each lane in f32 (avoids relying on half min/max availability/perf)
    float2 f = __half22float2(v);
    f.x = clamp01_f32(f.x);
    f.y = clamp01_f32(f.y);
    return __floats2half2_rn(f.x, f.y);
}

static __forceinline__ __device__ half2 hsigmoid_f16x2(half2 v) {
    float2 f = __half22float2(v);
    f.x = hsigmoid_f32(f.x);
    f.y = hsigmoid_f32(f.y);
    return __floats2half2_rn(f.x, f.y);
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void hardsigmoid_fwd_f32x4_ilp(const float4* __restrict__ x,
                              float4* __restrict__ out,
                              int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // ILP: each thread processes VEC_PER_THREAD float4s per outer iteration
    int64_t base = tid;

    for (int64_t i = base; i < n4; i += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n4) {
                float4 v = x[j];
                float4 o;
                o.x = hsigmoid_f32(v.x);
                o.y = hsigmoid_f32(v.y);
                o.z = hsigmoid_f32(v.z);
                o.w = hsigmoid_f32(v.w);
                out[j] = o;
            }
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void hardsigmoid_fwd_f32x2_ilp(const float2* __restrict__ x,
                              float2* __restrict__ out,
                              int64_t n2) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    int64_t base = tid;
    for (int64_t i = base; i < n2; i += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n2) {
                float2 v = x[j];
                float2 o;
                o.x = hsigmoid_f32(v.x);
                o.y = hsigmoid_f32(v.y);
                out[j] = o;
            }
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void hardsigmoid_fwd_f32_scalar_ilp(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    int64_t base = tid;
    for (int64_t i = base; i < n; i += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n) out[j] = hsigmoid_f32(x[j]);
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void hardsigmoid_fwd_f16x2_ilp(const half2* __restrict__ x,
                              half2* __restrict__ out,
                              int64_t n2) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    int64_t base = tid;
    for (int64_t i = base; i < n2; i += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n2) {
                half2 v = x[j];
                out[j] = hsigmoid_f16x2(v);
            }
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void hardsigmoid_fwd_f16_scalar_ilp(const half* __restrict__ x,
                                   half* __restrict__ out,
                                   int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    int64_t base = tid;
    for (int64_t i = base; i < n; i += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n) out[j] = hsigmoid_f16(x[j]);
        }
    }
}

static inline int pick_blocks_1d(int64_t n, int threads) {
    // Enough blocks to cover work, but cap to 1D limit.
    // Also oversubscribe a bit (up to ~4096 blocks) for latency hiding.
    int64_t blocks_work = (n + threads - 1) / threads;
    int64_t blocks = blocks_work;

    // Oversubscribe moderately for very large n; keep simple to avoid overhead.
    if (blocks < 2048) blocks = 2048;
    if (blocks > blocks_work) blocks = blocks_work;

    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    return (int)blocks;
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
                "hardsigmoid_cuda: only float32/float16 supported");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    const int threads = 256;

    // FP16 path: keep output in FP16 and reduce bandwidth when caller uses FP16
    if (x.scalar_type() == at::kHalf) {
        const uintptr_t xp = (uintptr_t)x.data_ptr<at::Half>();
        const uintptr_t yp = (uintptr_t)out.data_ptr<at::Half>();
        // Prefer half2 if 4B aligned and even length
        if (((xp | yp) & 0x3) == 0 && (n % 2 == 0)) {
            int64_t n2 = n / 2;
            int blocks = pick_blocks_1d(n2, threads);
            constexpr int VPT = 2; // modest ILP for half2
            hardsigmoid_fwd_f16x2_ilp<VPT><<<blocks, threads, 0, stream>>>(
                (const half2*)x.data_ptr<at::Half>(),
                (half2*)out.data_ptr<at::Half>(),
                n2
            );
        } else {
            int blocks = pick_blocks_1d(n, threads);
            constexpr int VPT = 2;
            hardsigmoid_fwd_f16_scalar_ilp<VPT><<<blocks, threads, 0, stream>>>(
                (const half*)x.data_ptr<at::Half>(),
                (half*)out.data_ptr<at::Half>(),
                n
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // FP32 path
    const uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    const uintptr_t yp = (uintptr_t)out.data_ptr<float>();

    // Prefer float4 if 16B aligned and length multiple of 4
    if (((xp | yp) & 0xF) == 0 && (n % 4 == 0)) {
        int64_t n4 = n / 4;
        int blocks = pick_blocks_1d(n4, threads);
        constexpr int VPT = 2; // modest ILP; avoids too much reg pressure
        hardsigmoid_fwd_f32x4_ilp<VPT><<<blocks, threads, 0, stream>>>(
            (const float4*)x.data_ptr<float>(),
            (float4*)out.data_ptr<float>(),
            n4
        );
    } else if (((xp | yp) & 0x7) == 0 && (n % 2 == 0)) {
        int64_t n2 = n / 2;
        int blocks = pick_blocks_1d(n2, threads);
        constexpr int VPT = 2;
        hardsigmoid_fwd_f32x2_ilp<VPT><<<blocks, threads, 0, stream>>>(
            (const float2*)x.data_ptr<float>(),
            (float2*)out.data_ptr<float>(),
            n2
        );
    } else {
        int blocks = pick_blocks_1d(n, threads);
        constexpr int VPT = 2;
        hardsigmoid_fwd_f32_scalar_ilp<VPT><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            n
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

hardsigmoid_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor hardsigmoid_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_hardsigmoid_opt3",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_cuda_source,
    functions=["hardsigmoid_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
)


class ModelNew(nn.Module):
    """
    HardSigmoid activation using an optimized custom CUDA kernel.
    Supports float16 and float32 on CUDA; falls back to PyTorch otherwise.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.hardsigmoid(x)
        if x.dtype not in (torch.float16, torch.float32):
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.hardsigmoid_cuda(x)