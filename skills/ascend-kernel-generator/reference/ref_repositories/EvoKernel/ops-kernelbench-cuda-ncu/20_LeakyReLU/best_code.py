import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: LeakyReLU forward (opt: single-kernel vec4 + pro/epi + unroll + optional streaming stores) ----

leaky_relu_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef LEAKY_RELU_USE_STREAMING_STORES
#define LEAKY_RELU_USE_STREAMING_STORES 1
#endif

__device__ __forceinline__ float lrelu_f(float v, float neg) {
    // Predication is typical; avoid extra function calls.
    return (v >= 0.0f) ? v : (v * neg);
}

__device__ __forceinline__ float4 lrelu_f4(float4 v, float neg) {
    v.x = lrelu_f(v.x, neg);
    v.y = lrelu_f(v.y, neg);
    v.z = lrelu_f(v.z, neg);
    v.w = lrelu_f(v.w, neg);
    return v;
}

// Use int4 stores to allow streaming store emission.
__device__ __forceinline__ void store_int4_streaming(int4* p, int4 v) {
#if LEAKY_RELU_USE_STREAMING_STORES
    // Write-combining store; bypasses (most) cache hierarchy on supported arch.
    // If not supported by compiler/arch, it will typically fall back to a regular store.
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#else
    *p = v;
#endif
}

template<int UNROLL>
__global__ __launch_bounds__(256, 2)
void leaky_relu_fwd_vec4_pro_epi(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int64_t n,
                                 float negative_slope) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Compute scalar prologue length (0..3) to align both pointers to 16B.
    uintptr_t xa = (uintptr_t)x;
    uintptr_t oa = (uintptr_t)out;
    int64_t prologue = 0;
    if (((xa | oa) & 0xF) != 0) {
        // advance i floats so that (x+i) and (out+i) are both 16B aligned
        // worst-case < 4
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uintptr_t xai = xa + (uintptr_t)(i * sizeof(float));
            uintptr_t oai = oa + (uintptr_t)(i * sizeof(float));
            if (((xai | oai) & 0xF) == 0) { prologue = i; break; }
        }
        // If no i found (can happen only with very odd alignment mismatch), keep prologue=0 and rely on scalar path below.
        // But in practice, i in [0,3] always works for 16B alignment with float pointers.
    }

    // Scalar prefix: only first `prologue` elements.
    for (int64_t i = tid; i < prologue && i < n; i += stride) {
        float v = x[i];
        out[i] = lrelu_f(v, negative_slope);
    }

    int64_t base = prologue;
    int64_t n_rem = n - base;
    int64_t n4 = n_rem >> 2;  // number of float4

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
#if LEAKY_RELU_USE_STREAMING_STORES
    int4* __restrict__ o4i = reinterpret_cast<int4*>(out + base);
#else
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out + base);
#endif

    // Vector body with ILP unroll over float4 elements.
    for (int64_t i = tid; i < n4; i += (int64_t)UNROLL * stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n4) {
                float4 v = x4[j];
                float4 r = lrelu_f4(v, negative_slope);
#if LEAKY_RELU_USE_STREAMING_STORES
                int4 ri = *reinterpret_cast<int4*>(&r);
                store_int4_streaming(o4i + j, ri);
#else
                o4[j] = r;
#endif
            }
        }
    }

    // Scalar epilogue for remaining elements (0..3).
    int64_t tail_base = base + (n4 << 2);
    for (int64_t i = tail_base + tid; i < n; i += stride) {
        float v = x[i];
        out[i] = lrelu_f(v, negative_slope);
    }
}

torch::Tensor leaky_relu_forward_cuda(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "leaky_relu_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "leaky_relu_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "leaky_relu_forward_cuda: x must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    const int threads = 256;
    // Conventional grid sizing; clamp to avoid pathological launch sizes.
    // Large tensors will saturate anyway; this avoids host-side device queries.
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = (int)(blocks64 > 131072 ? 131072 : blocks64);
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Unroll=4 is typically a good balance for bandwidth kernels without too much register pressure.
    leaky_relu_fwd_vec4_pro_epi<4><<<blocks, threads, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n,
        (float)negative_slope
    );

    return out;
}
"""

leaky_relu_cpp_source = r"""
torch::Tensor leaky_relu_forward_cuda(torch::Tensor x, double negative_slope);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_leaky_relu_opt4_single_kernel",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_cuda_source,
    functions=["leaky_relu_forward_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        # Enable streaming stores by default; can be toggled by editing macro.
        "-DLEAKY_RELU_USE_STREAMING_STORES=1",
    ],
    verbose=False,
)

# ---- Model wrapper using the custom op ----

class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation using an optimized custom CUDA kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.leaky_relu_forward_cuda(x, self.negative_slope)


# Keep the same input helpers for compatibility with the original harness.
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []