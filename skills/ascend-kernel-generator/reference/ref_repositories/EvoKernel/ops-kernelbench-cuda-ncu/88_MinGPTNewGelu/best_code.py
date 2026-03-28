import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA kernel: minGPT "new GELU" (tanh approximation)
# v5: add fast FP16/half2 I/O path (FP32 math), wider vectorization + ILP
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ float mingpt_new_gelu_fwd_f32(float x) {
    // 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    // inner = kAlpha * (x + kBeta*x^3)
    float inner = kAlpha * fmaf(kBeta, x3, x);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ half2 ldg_h2(const half2* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// --------------------------- FP32 kernel ---------------------------

__global__ __launch_bounds__(256, 2)
void mingpt_new_gelu_fwd_f32_vec4(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n
) {
    // float4 vectorization when 16B aligned; scalar tail within same kernel
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    int64_t n4 = n >> 2;
    int64_t vec_end = n4 << 2;

    for (int64_t i4 = (int64_t)tid; i4 < n4; i4 += (int64_t)stride) {
        int64_t e = i4 << 2;
        float4 v = *reinterpret_cast<const float4*>(x + e);
        v.x = mingpt_new_gelu_fwd_f32(v.x);
        v.y = mingpt_new_gelu_fwd_f32(v.y);
        v.z = mingpt_new_gelu_fwd_f32(v.z);
        v.w = mingpt_new_gelu_fwd_f32(v.w);
        *reinterpret_cast<float4*>(out + e) = v;
    }

    for (int64_t i = vec_end + (int64_t)tid; i < n; i += (int64_t)stride) {
        float xv = ldg_f32(x + i);
        out[i] = mingpt_new_gelu_fwd_f32(xv);
    }
}

__global__ __launch_bounds__(256, 2)
void mingpt_new_gelu_fwd_f32_scalar(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float xv = ldg_f32(x + i);
        out[i] = mingpt_new_gelu_fwd_f32(xv);
    }
}

// --------------------------- FP16 kernel (half2 I/O, FP32 math) ---------------------------

__device__ __forceinline__ half2 gelu_half2(half2 hx) {
    float2 xf = __half22float2(hx);
    xf.x = mingpt_new_gelu_fwd_f32(xf.x);
    xf.y = mingpt_new_gelu_fwd_f32(xf.y);
    return __floats2half2_rn(xf.x, xf.y);
}

__global__ __launch_bounds__(256, 2)
void mingpt_new_gelu_fwd_f16_h2_vec_ilp(
    const half* __restrict__ x,
    half* __restrict__ out,
    int64_t n
) {
    // Process half2 (2 elems). Unroll to increase ILP.
    // We handle remaining element if n is odd in a small scalar tail.
    constexpr int UNROLL_H2 = 4; // 4 half2 = 8 half per thread/iter

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    int64_t n2 = n >> 1;          // number of half2
    int64_t vec_end = n2 << 1;    // covered half elements

    const half2* x2 = reinterpret_cast<const half2*>(x);
    half2* o2 = reinterpret_cast<half2*>(out);

    int64_t i2 = (int64_t)tid;
    int64_t step2 = (int64_t)stride;

    for (; i2 < n2; i2 += (int64_t)UNROLL_H2 * step2) {
        #pragma unroll
        for (int u = 0; u < UNROLL_H2; ++u) {
            int64_t j2 = i2 + (int64_t)u * step2;
            if (j2 < n2) {
                half2 v = ldg_h2(x2 + j2);
                v = gelu_half2(v);
                o2[j2] = v;
            }
        }
    }

    // If odd tail element exists
    if ((n & 1) && (int64_t)tid == 0) {
        int64_t last = n - 1;
        float xv = __half2float(x[last]);
        float y = mingpt_new_gelu_fwd_f32(xv);
        out[last] = __float2half_rn(y);
    }
}

// --------------------------- Dispatcher ---------------------------

torch::Tensor mingpt_new_gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mingpt_new_gelu_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "mingpt_new_gelu_cuda: input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "mingpt_new_gelu_cuda: only float32/float16 supported");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;

    int device = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm_count = prop.multiProcessorCount;

    // More blocks to increase memory-level parallelism for latency-bound elementwise kernels.
    int max_blocks = sm_count * 48;
    int blocks_from_n = (int)((n + threads - 1) / threads);
    int blocks = blocks_from_n < max_blocks ? blocks_from_n : max_blocks;
    if (blocks < 1) blocks = 1;

    if (x.scalar_type() == torch::kFloat16) {
        // half2 path requires 4B alignment for half2; additionally prefer 16B alignment (not required).
        uintptr_t ax = (uintptr_t)x.data_ptr<at::Half>();
        uintptr_t ao = (uintptr_t)out.data_ptr<at::Half>();
        bool aligned4 = ((ax | ao) & 0x3) == 0;

        if (aligned4) {
            mingpt_new_gelu_fwd_f16_h2_vec_ilp<<<blocks, threads>>>(
                (const half*)x.data_ptr<at::Half>(),
                (half*)out.data_ptr<at::Half>(),
                n
            );
        } else {
            // Fallback: do FP32 kernel via temporary (rare). Keep it simple and correct.
            auto xf = x.to(torch::kFloat32);
            auto yf = torch::empty_like(xf);
            int64_t nf = xf.numel();

            uintptr_t addr_x = (uintptr_t)xf.data_ptr<float>();
            uintptr_t addr_o = (uintptr_t)yf.data_ptr<float>();
            bool aligned16 = ((addr_x | addr_o) & 0xF) == 0;

            if (aligned16) {
                mingpt_new_gelu_fwd_f32_vec4<<<blocks, threads>>>(
                    (const float*)xf.data_ptr<float>(),
                    (float*)yf.data_ptr<float>(),
                    nf
                );
            } else {
                mingpt_new_gelu_fwd_f32_scalar<<<blocks, threads>>>(
                    (const float*)xf.data_ptr<float>(),
                    (float*)yf.data_ptr<float>(),
                    nf
                );
            }
            out.copy_(yf.to(torch::kFloat16));
        }
        return out;
    }

    // FP32 path
    uintptr_t addr_x = (uintptr_t)x.data_ptr<float>();
    uintptr_t addr_o = (uintptr_t)out.data_ptr<float>();
    bool aligned16 = ((addr_x | addr_o) & 0xF) == 0;

    if (aligned16) {
        mingpt_new_gelu_fwd_f32_vec4<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            n
        );
    } else {
        mingpt_new_gelu_fwd_f32_scalar<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            n
        );
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor mingpt_new_gelu_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mingpt_new_gelu_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mingpt_new_gelu_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)

# -----------------------------------------------------------------------------
# Model using the custom CUDA op
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    minGPT new GELU activation implemented with an optimized custom CUDA kernel.
    Supports float16/float32 inputs; prefers float16 fast path when provided.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype not in (torch.float16, torch.float32):
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.mingpt_new_gelu_cuda(x)


# Integration helpers (kept for compatibility)
batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return []