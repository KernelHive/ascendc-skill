import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__)
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

__device__ __forceinline__ float sigmoid_f32_fast(float x) {
    // --use_fast_math enables a faster exp approximation; acceptable for inference-like use.
    return 1.0f / (1.0f + expf(-x));
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void sigmoid_f32x4_aligned_divisible_ilp(const float4* __restrict__ x4,
                                        float4* __restrict__ o4,
                                        int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    for (int64_t base = tid; base < n4; base += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t i = base + (int64_t)u * stride;
            if (i < n4) {
                float4 v = LDG(x4 + i);
                v.x = sigmoid_f32_fast(v.x);
                v.y = sigmoid_f32_fast(v.y);
                v.z = sigmoid_f32_fast(v.z);
                v.w = sigmoid_f32_fast(v.w);
                o4[i] = v;
            }
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void sigmoid_f32x2_aligned_divisible_ilp(const float2* __restrict__ x2,
                                        float2* __restrict__ o2,
                                        int64_t n2) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    for (int64_t base = tid; base < n2; base += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t i = base + (int64_t)u * stride;
            if (i < n2) {
                float2 v = LDG(x2 + i);
                v.x = sigmoid_f32_fast(v.x);
                v.y = sigmoid_f32_fast(v.y);
                o2[i] = v;
            }
        }
    }
}

template<int VEC_PER_THREAD>
__global__ __launch_bounds__(256, 2)
void sigmoid_f32_scalar_ilp(const float* __restrict__ x,
                           float* __restrict__ out,
                           int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    for (int64_t base = tid; base < n; base += stride * VEC_PER_THREAD) {
        #pragma unroll
        for (int u = 0; u < VEC_PER_THREAD; ++u) {
            int64_t i = base + (int64_t)u * stride;
            if (i < n) out[i] = sigmoid_f32_fast(LDG(x + i));
        }
    }
}

static inline int pick_blocks_1d(int64_t iters, int threads) {
    // iters is number of vector items (n4/n2) or scalar items (n).
    // Oversubscribe aggressively to hide memory latency, but cap for 1D grid.
    int64_t blocks_work = (iters + threads - 1) / threads;
    int64_t blocks = blocks_work;

    // Heuristic oversubscription for large tensors; keep simple/no device queries.
    const int64_t min_blocks = 4096;
    if (blocks < min_blocks) blocks = min_blocks;
    if (blocks > blocks_work) blocks = blocks_work;

    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    return (int)blocks;
}

static inline void maybe_set_l2_persist_window(cudaStream_t stream,
                                              const void* base_ptr,
                                              size_t num_bytes) {
#if CUDART_VERSION >= 11020
    // Best-effort: on supported GPUs, encourage some of the read working set to persist in L2.
    // If unsupported, cudaStreamSetAttribute returns an error; ignore to avoid overhead.
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(base_ptr);
    // Cap window to a modest size to avoid hurting other kernels.
    const size_t cap = (size_t)32 * 1024 * 1024; // 32MB
    attr.accessPolicyWindow.num_bytes = (num_bytes < cap) ? num_bytes : cap;
    attr.accessPolicyWindow.hitRatio  = 0.6f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
#else
    (void)stream; (void)base_ptr; (void)num_bytes;
#endif
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sigmoid_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "sigmoid_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "sigmoid_cuda: input must be contiguous");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int threads = 256;

    const float* xp = x.data_ptr<float>();
    float* op = out.data_ptr<float>();

    // L2 persist hint (best-effort). Window on input only; output is streaming writes.
    maybe_set_l2_persist_window(stream, xp, (size_t)n * sizeof(float));

    const uintptr_t xa = (uintptr_t)xp;
    const uintptr_t oa = (uintptr_t)op;

    const bool aligned16 = ((xa | oa) & 0xF) == 0;
    const bool aligned8  = ((xa | oa) & 0x7) == 0;

    // Increase ILP modestly; for pure memory-bound kernels this can reduce latency stalls.
    // Keep small enough to avoid big reg blow-ups.
    constexpr int VPT4 = 4;
    constexpr int VPT2 = 4;
    constexpr int VPTS = 2;

    if (aligned16 && (n % 4 == 0)) {
        int64_t n4 = n / 4;
        int blocks = pick_blocks_1d(n4, threads);
        sigmoid_f32x4_aligned_divisible_ilp<VPT4><<<blocks, threads, 0, stream>>>(
            (const float4*)xp, (float4*)op, n4
        );
    } else if (aligned8 && (n % 2 == 0)) {
        int64_t n2 = n / 2;
        int blocks = pick_blocks_1d(n2, threads);
        sigmoid_f32x2_aligned_divisible_ilp<VPT2><<<blocks, threads, 0, stream>>>(
            (const float2*)xp, (float2*)op, n2
        );
    } else {
        int blocks = pick_blocks_1d(n, threads);
        sigmoid_f32_scalar_ilp<VPTS><<<blocks, threads, 0, stream>>>(xp, op, n);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor sigmoid_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_sigmoid_opt_v8",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sigmoid_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA sigmoid kernel.
    Expects CUDA float32 contiguous input for best performance.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.float32:
            return torch.sigmoid(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.sigmoid_cuda(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []