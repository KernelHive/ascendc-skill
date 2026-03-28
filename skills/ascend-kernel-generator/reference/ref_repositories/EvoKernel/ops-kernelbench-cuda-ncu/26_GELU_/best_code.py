import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Faster GELU: split aligned fast path (pure float4 grid-stride, unroll=4)
# + generic safe path. Keep coverage-driven grid sizing and add error checking.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __forceinline__
#define __forceinline__ __attribute__((forceinline))
#endif

// ----------------------------- Math -----------------------------------------
// GELU(tanh approximation):
// gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
__device__ __forceinline__ float gelu_tanh_fwd(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;
    float inner = fmaf(kBeta, x3, x);
    inner *= kAlpha;
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float4 gelu_tanh_fwd4(float4 v) {
    v.x = gelu_tanh_fwd(v.x);
    v.y = gelu_tanh_fwd(v.y);
    v.z = gelu_tanh_fwd(v.z);
    v.w = gelu_tanh_fwd(v.w);
    return v;
}

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if defined(__CUDA_ARCH__)
  #if __CUDA_ARCH__ >= 350
    return __ldg(p);
  #else
    return *p;
  #endif
#else
  return *p;
#endif
}

// Bitcast helpers using uint4 to encourage 128-bit LD/ST.
__device__ __forceinline__ float4 ld128_as_f4(const float4* p) {
    uint4 u = *reinterpret_cast<const uint4*>(p);
    return *reinterpret_cast<float4*>(&u);
}
__device__ __forceinline__ void st128_from_f4(float4* p, float4 v) {
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<uint4*>(&v);
}

// ----------------------------- Kernels --------------------------------------

// Generic, alignment-safe kernel (scalar grid-stride).
__global__ __launch_bounds__(256, 2)
void gelu_fwd_generic(const float* __restrict__ x,
                      float* __restrict__ out,
                      int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        out[i] = gelu_tanh_fwd(ro_load_f32(x + i));
    }
}

// Aligned fast kernel: x and out are 16B-aligned, process in float4 units.
template<int UNROLL>
__global__ __launch_bounds__(256, 2)
void gelu_fwd_aligned_f4(const float* __restrict__ x,
                         float* __restrict__ out,
                         int64_t n) {
    // n_vec: number of float4 items
    int64_t n_vec = n >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Vector grid-stride with modest unroll to increase MLP
    for (int64_t vi = tid; vi < n_vec; vi += (int64_t)UNROLL * stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = vi + (int64_t)u * stride;
            if (j < n_vec) {
                // Explicit 128-bit load/store
                float4 v = ld128_as_f4(x4 + j);
                v = gelu_tanh_fwd4(v);
                st128_from_f4(o4 + j, v);
            }
        }
    }

    // Tail (0..3 floats)
    int64_t tail_base = n_vec << 2;
    for (int64_t i = tail_base + tid; i < n; i += stride) {
        out[i] = gelu_tanh_fwd(ro_load_f32(x + i));
    }
}

// ----------------------------- Launcher -------------------------------------

torch::Tensor gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "gelu_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "gelu_cuda: input must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);

    int64_t n = x.numel();
    if (n == 0) return out;

    const int threads = 256;

    // Coverage-driven like baseline; clamp only to avoid absurd grids.
    int64_t blocks64 = (n + threads - 1) / threads;
    const int64_t max_blocks = 262144;
    if (blocks64 > max_blocks) blocks64 = max_blocks;
    if (blocks64 < 1) blocks64 = 1;
    int blocks = (int)blocks64;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    bool aligned16 = ((((uintptr_t)xp | (uintptr_t)op) & 0xF) == 0);

    if (aligned16) {
        gelu_fwd_aligned_f4<4><<<blocks, threads, 0, stream>>>(xp, op, n);
    } else {
        gelu_fwd_generic<<<blocks, threads, 0, stream>>>(xp, op, n);
    }

    // Lightweight error check (no device sync)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gelu_cuda kernel launch failed: ", cudaGetErrorString(err));

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gelu_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gelu_opt_f4_unroll4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gelu_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a GELU activation using an optimized custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.gelu_cuda(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []