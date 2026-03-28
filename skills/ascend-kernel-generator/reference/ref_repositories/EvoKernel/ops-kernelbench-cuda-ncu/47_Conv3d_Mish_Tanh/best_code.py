import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA: mish(x) -> tanh
# Focus: memory-latency/bandwidth bound elementwise epilogue
#  - float4/float2/scalar paths (guarded by alignment & numel)
#  - modest unroll for ILP (2)
#  - guarded __ldg read-only loads (where supported)
#  - avoid fragile __tanhf intrinsic; use tanhf under --use_fast_math
#  - keep simple static launch sizing (no occupancy API overhead)
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

static inline bool is_aligned_16_host(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xFULL) == 0ULL);
}
static inline bool is_aligned_8_host(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0x7ULL) == 0ULL);
}

template <typename T>
__device__ __forceinline__ T ldg(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float softplus_stable(float x) {
    // stable softplus
    // For large positive x, softplus ~ x; for large negative x, ~ exp(x)
    // Using expf/log1pf; under --use_fast_math expf may become __expf-like.
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish_f32(float x) {
    // mish(x) = x * tanh(softplus(x))
    float sp = softplus_stable(x);
    return x * tanhf(sp);
}

__device__ __forceinline__ float fused_mish_tanh_f32(float x) {
    // tanh(mish(x))
    return tanhf(mish_f32(x));
}

template <int VEC> // 1, 2, 4
__global__ __launch_bounds__(256, 2) void fused_mish_tanh_kernel_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t numel
) {
    const int tid = (int)threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)tid;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    constexpr int UNROLL = 2;

    if constexpr (VEC == 4) {
        int64_t n4 = numel >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

        int64_t i = idx;
        int64_t step = stride * UNROLL;

        // Unrolled main loop: two independent vectors per iteration (if available)
        for (; i + stride < n4; i += step) {
            float4 v0 = ldg(x4 + i);
            float4 v1 = ldg(x4 + (i + stride));

            v0.x = fused_mish_tanh_f32(v0.x);
            v0.y = fused_mish_tanh_f32(v0.y);
            v0.z = fused_mish_tanh_f32(v0.z);
            v0.w = fused_mish_tanh_f32(v0.w);

            v1.x = fused_mish_tanh_f32(v1.x);
            v1.y = fused_mish_tanh_f32(v1.y);
            v1.z = fused_mish_tanh_f32(v1.z);
            v1.w = fused_mish_tanh_f32(v1.w);

            out4[i] = v0;
            out4[i + stride] = v1;
        }

        // Tail
        for (; i < n4; i += stride) {
            float4 v = ldg(x4 + i);
            v.x = fused_mish_tanh_f32(v.x);
            v.y = fused_mish_tanh_f32(v.y);
            v.z = fused_mish_tanh_f32(v.z);
            v.w = fused_mish_tanh_f32(v.w);
            out4[i] = v;
        }
    } else if constexpr (VEC == 2) {
        int64_t n2 = numel >> 1;
        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
        float2* __restrict__ out2 = reinterpret_cast<float2*>(out);

        int64_t i = idx;
        int64_t step = stride * UNROLL;

        for (; i + stride < n2; i += step) {
            float2 v0 = ldg(x2 + i);
            float2 v1 = ldg(x2 + (i + stride));

            v0.x = fused_mish_tanh_f32(v0.x);
            v0.y = fused_mish_tanh_f32(v0.y);
            v1.x = fused_mish_tanh_f32(v1.x);
            v1.y = fused_mish_tanh_f32(v1.y);

            out2[i] = v0;
            out2[i + stride] = v1;
        }

        for (; i < n2; i += stride) {
            float2 v = ldg(x2 + i);
            v.x = fused_mish_tanh_f32(v.x);
            v.y = fused_mish_tanh_f32(v.y);
            out2[i] = v;
        }

        // Handle odd last element (rare; only when numel is odd, but VEC==2 chosen implies even)
    } else { // VEC == 1
        int64_t i = idx;
        int64_t step = stride * UNROLL;

        for (; i + stride < numel; i += step) {
            float v0 = ldg(x + i);
            float v1 = ldg(x + (i + stride));
            out[i] = fused_mish_tanh_f32(v0);
            out[i + stride] = fused_mish_tanh_f32(v1);
        }

        for (; i < numel; i += stride) {
            out[i] = fused_mish_tanh_f32(ldg(x + i));
        }
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) return out;

    const float* xptr = x.data_ptr<float>();
    float* outptr = out.data_ptr<float>();

    // Bandwidth kernels usually do well with 256 threads; keep simple.
    constexpr int threads = 256;

    // Static cap avoids per-call occupancy overhead; keep enough CTAs to cover large tensors.
    int64_t blocks64 = (numel + threads - 1) / threads;
    int blocks = (int)(blocks64 > 8192 ? 8192 : (blocks64 < 1 ? 1 : blocks64));

    dim3 block(threads, 1, 1);
    dim3 grid((unsigned)blocks, 1, 1);

    bool can_vec4 = ((numel & 3LL) == 0LL) && is_aligned_16_host(xptr) && is_aligned_16_host(outptr);
    bool can_vec2 = ((numel & 1LL) == 0LL) && is_aligned_8_host(xptr) && is_aligned_8_host(outptr);

    if (can_vec4) {
        fused_mish_tanh_kernel_f32<4><<<grid, block>>>(xptr, outptr, numel);
    } else if (can_vec2) {
        fused_mish_tanh_kernel_f32<2><<<grid, block>>>(xptr, outptr, numel);
    } else {
        fused_mish_tanh_kernel_f32<1><<<grid, block>>>(xptr, outptr, numel);
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor fused_mish_tanh_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_fused_mish_tanh_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_mish_tanh_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d computed by PyTorch/cuDNN, then a fused CUDA kernel applies:
      mish -> tanh

    Fast-path: CUDA float32 contiguous tensors.
    Otherwise falls back to PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            return torch.tanh(F.mish(x))

        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.fused_mish_tanh_cuda(x)


# Keep original helper signatures for integration consistency
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]