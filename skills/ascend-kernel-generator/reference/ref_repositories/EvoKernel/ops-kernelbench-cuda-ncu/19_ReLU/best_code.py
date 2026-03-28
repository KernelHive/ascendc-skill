import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Further-optimized custom CUDA ReLU compiled as a PyTorch extension
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float relu_f(float v) {
    return fmaxf(v, 0.0f);
}

__device__ __forceinline__ float4 relu_f4(float4 v) {
    v.x = relu_f(v.x);
    v.y = relu_f(v.y);
    v.z = relu_f(v.z);
    v.w = relu_f(v.w);
    return v;
}

template<int THREADS, int UNROLL>
__global__ __launch_bounds__(THREADS, 4)
void relu_fwd_vec4_or_scalar(const float* __restrict__ x,
                            float* __restrict__ out,
                            int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * THREADS + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * THREADS;

    if (n <= 0) return;

    // Fast path: both pointers 16B-aligned and n large enough.
    const uintptr_t xa = (uintptr_t)x;
    const uintptr_t oa = (uintptr_t)out;
    const bool aligned16 = (((xa | oa) & 0xF) == 0);

    if (aligned16) {
        int64_t n4 = n >> 2;           // number of float4 elements
        int64_t tail = n & 3;          // remaining scalars

        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

        // Process full UNROLL tiles without any per-lane bounds checks.
        // i counts float4 indices.
        int64_t i0 = tid;
        int64_t tile_stride = (int64_t)UNROLL * stride;
        int64_t n4_even = n4 - (n4 % ((int64_t)UNROLL * stride));

        for (int64_t i = i0; i < n4_even; i += tile_stride) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int64_t j = i + (int64_t)u * stride;
                float4 v = x4[j];
                o4[j] = relu_f4(v);
            }
        }

        // Cleanup for remaining float4s (at most UNROLL*stride - 1 items total).
        for (int64_t i = n4_even + i0; i < n4; i += stride) {
            float4 v = x4[i];
            o4[i] = relu_f4(v);
        }

        // Scalar tail for n % 4 (only a few elements; negligible)
        if (tail) {
            int64_t base = n4 << 2; // first tail scalar index
            for (int64_t k = base + tid; k < n; k += stride) {
                out[k] = relu_f(x[k]);
            }
        }
        return;
    }

    // Scalar fallback: simple grid-stride loop with modest unrolling.
    for (int64_t i = tid; i < n; i += (int64_t)UNROLL * stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j < n) out[j] = relu_f(x[j]);
        }
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "relu_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "relu_cuda: input must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    constexpr int threads = 256;
    constexpr int unroll = 4;

    // Keep simple sizing to avoid per-call device queries; enough blocks to saturate BW.
    // Cap to avoid excessive launch size.
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = (int)(blocks64 > 65535 ? 65535 : blocks64);
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    relu_fwd_vec4_or_scalar<threads, unroll><<<blocks, threads, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n
    );
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor relu_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_relu_vec4_even_tiles_unroll4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# -----------------------------------------------------------------------------
# New model using the optimized custom CUDA op
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation using an optimized custom CUDA kernel.
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
        return self.custom_ops.relu_cuda(x)


# Keep the same helper functions/signatures for integration
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []