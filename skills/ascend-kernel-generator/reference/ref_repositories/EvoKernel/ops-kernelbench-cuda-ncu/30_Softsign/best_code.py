import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ extension: optimized fused softsign forward (float32, CUDA)
# softsign(x) = x / (1 + abs(x))

softsign_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

// Use __ldg for scalar loads (on newer arch this may be equivalent, but harmless)
static __forceinline__ __device__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__global__ void softsign_fwd_scalar_gs(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = ld_ro(x + i);
        float av = fabsf(v);
        out[i] = v * __fdividef(1.0f, (1.0f + av));
    }
}

__global__ void softsign_fwd_float2_gs(const float2* __restrict__ x2,
                                      float2* __restrict__ out2,
                                      int64_t n2) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = idx; i < n2; i += stride) {
        float2 v = x2[i];
        float a0 = fabsf(v.x);
        float a1 = fabsf(v.y);
        float2 o;
        o.x = v.x * __fdividef(1.0f, (1.0f + a0));
        o.y = v.y * __fdividef(1.0f, (1.0f + a1));
        out2[i] = o;
    }
}

__global__ void softsign_fwd_float4_gs(const float4* __restrict__ x4,
                                      float4* __restrict__ out4,
                                      int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = idx; i < n4; i += stride) {
        float4 v = x4[i];
        float4 o;
        float a0 = fabsf(v.x);
        float a1 = fabsf(v.y);
        float a2 = fabsf(v.z);
        float a3 = fabsf(v.w);
        o.x = v.x * __fdividef(1.0f, (1.0f + a0));
        o.y = v.y * __fdividef(1.0f, (1.0f + a1));
        o.z = v.z * __fdividef(1.0f, (1.0f + a2));
        o.w = v.w * __fdividef(1.0f, (1.0f + a3));
        out4[i] = o;
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "softsign_cuda: only float32 is supported");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    const int threads = 256;
    // Cap blocks to avoid excessive launch overhead while still providing enough parallelism
    int64_t maxBlocks = 65535;
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = (int)((blocks64 > maxBlocks) ? maxBlocks : blocks64);
    if (blocks < 1) blocks = 1;

    const uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    const uintptr_t yp = (uintptr_t)out.data_ptr<float>();

    // Prefer float4 when both pointers are 16B aligned and length is multiple of 4
    if (((xp | yp) & 0xF) == 0 && (n % 4 == 0)) {
        int64_t n4 = n / 4;
        softsign_fwd_float4_gs<<<blocks, threads, 0, stream>>>(
            (const float4*)x.data_ptr<float>(),
            (float4*)out.data_ptr<float>(),
            n4
        );
    } else if (((xp | yp) & 0x7) == 0 && (n % 2 == 0)) {
        int64_t n2 = n / 2;
        softsign_fwd_float2_gs<<<blocks, threads, 0, stream>>>(
            (const float2*)x.data_ptr<float>(),
            (float2*)out.data_ptr<float>(),
            n2
        );
    } else {
        softsign_fwd_scalar_gs<<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            n
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

softsign_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor softsign_cuda(torch::Tensor x);
"""

# Compile and load into a module handle named custom_ops_lib
custom_ops_lib = load_inline(
    name="custom_ops_lib_softsign_opt2",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_cuda_source,
    functions=["softsign_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Softsign activation via a custom CUDA kernel.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return x / (1 + torch.abs(x))
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.softsign_cuda(x)

# Keep the same input helper signatures as provided
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []