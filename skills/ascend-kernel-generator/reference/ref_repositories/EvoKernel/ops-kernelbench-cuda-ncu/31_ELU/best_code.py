import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA ELU (forward): alignment-safe vec4 streaming + ILP unroll,
# no serialized tail, plus alpha specializations.
# -----------------------------------------------------------------------------

elu_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __forceinline__
#define __forceinline__ __attribute__((forceinline))
#endif

// ELU forward helpers
__device__ __forceinline__ float elu_fwd_alpha1(float x) {
    // x>0 ? x : exp(x)-1
    float neg = __expf(x) - 1.0f;
    return (x > 0.0f) ? x : neg;
}

__device__ __forceinline__ float elu_fwd(float x, float alpha) {
    float neg = alpha * (__expf(x) - 1.0f);
    return (x > 0.0f) ? x : neg;
}

__device__ __forceinline__ float4 elu_fwd4_alpha1(float4 v) {
    v.x = elu_fwd_alpha1(v.x);
    v.y = elu_fwd_alpha1(v.y);
    v.z = elu_fwd_alpha1(v.z);
    v.w = elu_fwd_alpha1(v.w);
    return v;
}

__device__ __forceinline__ float4 elu_fwd4(float4 v, float alpha) {
    v.x = elu_fwd(v.x, alpha);
    v.y = elu_fwd(v.y, alpha);
    v.z = elu_fwd(v.z, alpha);
    v.w = elu_fwd(v.w, alpha);
    return v;
}

__device__ __forceinline__ float relu_fwd(float x) { return x > 0.0f ? x : 0.0f; }

template<int UNROLL, int MODE>
__global__ __launch_bounds__(256, 2)
void elu_fwd_vec4_align_kernel(const float* __restrict__ x,
                              float* __restrict__ out,
                              int64_t n,
                              float alpha) {
    int64_t gtid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)gridDim.x * blockDim.x;
    if (n <= 0) return;

    // Compute a small global prologue [0..3] so that (x+pro) and (out+pro) are 16B aligned.
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
        // If still misaligned after shifting by up to 3 floats, disable vector path.
        if ((((xa + (uintptr_t)(pro * sizeof(float))) |
              (oa + (uintptr_t)(pro * sizeof(float)))) & 0xF) != 0) {
            pro = n; // scalar-only
        }
    }

    // Scalar prologue: first 'pro' elements
    for (int64_t i = gtid; i < pro && i < n; i += gstride) {
        float vx = x[i];
        float vy;
        if constexpr (MODE == 0)      vy = elu_fwd(vx, alpha);
        else if constexpr (MODE == 1) vy = elu_fwd_alpha1(vx);
        else                          vy = relu_fwd(vx); // MODE==2 alpha==0
        out[i] = vy;
    }

    // Vector body
    int64_t base = pro;
    int64_t n_rem = n - base;
    int64_t n_vec = n_rem >> 2;           // float4 packs
    int64_t vec_tail_base = base + (n_vec << 2);

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out + base);

    for (int64_t vi = gtid; vi < n_vec; vi += (int64_t)UNROLL * gstride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = vi + (int64_t)u * gstride;
            if (j < n_vec) {
                float4 v = x4[j];
                float4 o;
                if constexpr (MODE == 0)      o = elu_fwd4(v, alpha);
                else if constexpr (MODE == 1) o = elu_fwd4_alpha1(v);
                else {
                    // ReLU vector
                    o.x = relu_fwd(v.x);
                    o.y = relu_fwd(v.y);
                    o.z = relu_fwd(v.z);
                    o.w = relu_fwd(v.w);
                }
                o4[j] = o;
            }
        }
    }

    // Scalar epilogue: leftover elements after float4 body
    for (int64_t i = vec_tail_base + gtid; i < n; i += gstride) {
        float vx = x[i];
        float vy;
        if constexpr (MODE == 0)      vy = elu_fwd(vx, alpha);
        else if constexpr (MODE == 1) vy = elu_fwd_alpha1(vx);
        else                          vy = relu_fwd(vx);
        out[i] = vy;
    }
}

torch::Tensor elu_forward_cuda(torch::Tensor x, double alpha_d) {
    TORCH_CHECK(x.is_cuda(), "elu_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "elu_forward_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "elu_forward_cuda: x must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    const float* xptr = (const float*)x.data_ptr<float>();
    float* optr = (float*)out.data_ptr<float>();
    float alpha = (float)alpha_d;

    // Launch: simple streaming heuristic (avoid per-call device queries).
    // Enough CTAs for latency hiding, clamp to avoid excessive launch overhead.
    const int threads = 256;
    int64_t blocks64 = (n + threads - 1) / threads;
    // Since we use vec4/unroll, we don't need blocks64 huge; clamp to a practical range.
    const int64_t min_blocks = 80;        // usually >= SM count on common GPUs; ok if overshoots
    const int64_t max_blocks = 262144;    // safety clamp

    if (blocks64 < min_blocks) blocks64 = min_blocks;
    if (blocks64 > max_blocks) blocks64 = max_blocks;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Unroll=2 is a good ILP tradeoff for exp-heavy elementwise ops.
    if (alpha == 1.0f) {
        elu_fwd_vec4_align_kernel<2, 1><<< (int)blocks64, threads, 0, stream >>>(xptr, optr, n, alpha);
    } else if (alpha == 0.0f) {
        elu_fwd_vec4_align_kernel<2, 2><<< (int)blocks64, threads, 0, stream >>>(xptr, optr, n, alpha);
    } else {
        elu_fwd_vec4_align_kernel<2, 0><<< (int)blocks64, threads, 0, stream >>>(xptr, optr, n, alpha);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

elu_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor elu_forward_cuda(torch::Tensor x, double alpha);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_elu_v5_vec4_align",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=["elu_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--extra-device-vectorization",
        "-lineinfo",
    ],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs an ELU activation using a custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.elu(x, alpha=self.alpha)
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.elu_forward_cuda(x, self.alpha)