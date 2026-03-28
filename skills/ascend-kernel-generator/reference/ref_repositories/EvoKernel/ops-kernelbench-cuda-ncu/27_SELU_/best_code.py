import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized SELU: single-kernel grid-stride, alignment-safe float4 vectorization,
# ILP via processing two float4 per iteration, fast exp, branch-minimized select.
# -----------------------------------------------------------------------------

selu_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __forceinline__
#define __forceinline__ __attribute__((forceinline))
#endif

__device__ __forceinline__ float selu_fwd(float v) {
    // PyTorch SELU constants
    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;

    // branch-minimized: compute negative branch then select
    float ev = __expf(v) - 1.0f;          // fast exp under --use_fast_math
    float neg = alpha * ev;
    float y = (v > 0.0f) ? v : neg;
    return scale * y;
}

__device__ __forceinline__ float4 selu_fwd4(float4 a) {
    a.x = selu_fwd(a.x);
    a.y = selu_fwd(a.y);
    a.z = selu_fwd(a.z);
    a.w = selu_fwd(a.w);
    return a;
}

template<int UNROLL_VEC4>
__global__ __launch_bounds__(256, 2)
void selu_fwd_vec4_align_ilp(const float* __restrict__ x,
                            float* __restrict__ out,
                            int64_t n) {
    int64_t gtid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)gridDim.x * blockDim.x;
    if (n <= 0) return;

    // Uniform prologue to make both pointers 16B aligned (for float4)
    uintptr_t xa = (uintptr_t)x;
    uintptr_t oa = (uintptr_t)out;

    int64_t pro = 0;
    if (((xa | oa) & 0xF) != 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uintptr_t xai = xa + (uintptr_t)(i * sizeof(float));
            uintptr_t oai = oa + (uintptr_t)(i * sizeof(float));
            if (((xai | oai) & 0xF) == 0) { pro = i; break; }
        }
        // If still misaligned (shouldn't happen), force scalar-only.
        if ((((xa + (uintptr_t)(pro * sizeof(float))) | (oa + (uintptr_t)(pro * sizeof(float)))) & 0xF) != 0) {
            pro = n;
        }
    }

    // Scalar prologue
    for (int64_t i = gtid; i < pro && i < n; i += gstride) {
        out[i] = selu_fwd(x[i]);
    }

    // Vector body
    int64_t base = pro;
    int64_t n_rem = n - base;
    int64_t n_vec = n_rem >> 2;              // float4 count
    int64_t vec_tail_base = base + (n_vec << 2);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out + base);

    // Process UNROLL_VEC4 float4 chunks per thread per outer iteration.
    // UNROLL_VEC4=2 means two float4 (8 floats) -> better ILP for memory+SFU.
    for (int64_t vi = gtid; vi < n_vec; vi += (int64_t)UNROLL_VEC4 * gstride) {
        #pragma unroll
        for (int u = 0; u < UNROLL_VEC4; ++u) {
            int64_t j = vi + (int64_t)u * gstride;
            if (j < n_vec) {
                float4 v = x4[j];
                o4[j] = selu_fwd4(v);
            }
        }
    }

    // Scalar epilogue
    for (int64_t i = vec_tail_base + gtid; i < n; i += gstride) {
        out[i] = selu_fwd(x[i]);
    }
}

torch::Tensor selu_forward_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "selu_forward_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "selu_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "selu_forward_cuda: input must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    const int threads = 256;

    // Plenty of CTAs for bandwidth-bound op; clamp grid to keep launch overhead sane.
    int64_t blocks64 = (n + threads - 1) / threads;
    const int64_t max_blocks = 262144;
    if (blocks64 > max_blocks) blocks64 = max_blocks;
    if (blocks64 < 1) blocks64 = 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // UNROLL_VEC4=2 is a good balance of ILP vs registers for exp-heavy elementwise ops.
    selu_fwd_vec4_align_ilp<2><<< (int)blocks64, threads, 0, stream >>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n
    );

    return out;
}
"""

selu_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor selu_forward_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_selu_opt_vec4_ilp",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_cuda_source,
    functions=["selu_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA SELU kernel.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.selu(x)
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.selu_forward_cuda(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []