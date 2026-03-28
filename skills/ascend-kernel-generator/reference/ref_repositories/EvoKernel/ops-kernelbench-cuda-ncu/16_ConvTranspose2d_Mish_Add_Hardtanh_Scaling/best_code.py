import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused mish + add + hardtanh + scale (v3)
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// ------------------------------
// Fast math helpers (approx, stable enough for typical activations)
// ------------------------------
static __forceinline__ __device__ float softplus_fast(float x) {
    // softplus(x) = log(1+exp(x))
    // Use piecewise for stability; exp via exp2 for speed.
    // For large x, softplus ~ x; for small x, softplus ~ exp(x).
    if (x > 20.0f) return x;
    if (x < -20.0f) {
        // exp(x) is tiny; log1p(exp(x)) ~ exp(x)
        // expf is fine here since it's rarely on hot path, but exp2f is OK too.
        return __expf(x);
    }
    // log1p(exp(x)) computed as log(1 + 2^(x / ln2))
    // log1pf + exp2f is typically faster under --use_fast_math.
    const float inv_ln2 = 1.4426950408889634f; // 1/ln(2)
    float ex = exp2f(x * inv_ln2);
    return log1pf(ex);
}

static __forceinline__ __device__ float mish_f(float x) {
    float sp = softplus_fast(x);
    // tanhf is fast under --use_fast_math
    return x * tanhf(sp);
}

static __forceinline__ __device__ float hardtanh_f(float x) {
    return fminf(1.0f, fmaxf(-1.0f, x));
}

// ------------------------------
// Vectorized kernel (true float4 load/store)
// ------------------------------
template<int ILP>
__global__ __launch_bounds__(256, 2)
void mish_add_hardtanh_scale_vec4_kernel_v3(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t total,
    float add_value,
    float scale
) {
    // total is element count (floats)
    const int64_t total4 = total >> 2; // number of float4 elements
    const int64_t stride4 = (int64_t)blockDim.x * gridDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    int64_t idx4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes ILP float4s per outer iteration
    while (idx4 < total4) {
#pragma unroll
        for (int k = 0; k < ILP; ++k) {
            int64_t j = idx4 + (int64_t)k * stride4;
            if (j < total4) {
                // True vector load/store
                float4 v = x4[j];

                v.x = hardtanh_f(mish_f(v.x) + add_value) * scale;
                v.y = hardtanh_f(mish_f(v.y) + add_value) * scale;
                v.z = hardtanh_f(mish_f(v.z) + add_value) * scale;
                v.w = hardtanh_f(mish_f(v.w) + add_value) * scale;

                y4[j] = v;
            }
        }
        idx4 += (int64_t)ILP * stride4;
    }

    // Tail elements (0..3), scalar cleanup
    int64_t base = (total4 << 2);
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t t = base + tid; t < total; t += stride) {
        float v = x[t];
        v = hardtanh_f(mish_f(v) + add_value);
        y[t] = v * scale;
    }
}

__global__ __launch_bounds__(256, 2)
void mish_add_hardtanh_scale_scalar_kernel_v3(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t total,
    float add_value,
    float scale
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        float v = x[i];
        v = hardtanh_f(mish_f(v) + add_value);
        y[i] = v * scale;
    }
}

// Host launcher
torch::Tensor mish_add_hardtanh_scale_cuda(torch::Tensor x, double add_value, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");

    auto y = torch::empty_like(x);
    int64_t total = x.numel();

    constexpr int threads = 256;

    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sms = prop.multiProcessorCount;

    // Allow more blocks/SM to increase scheduling flexibility for a memory-bound kernel
    int max_blocks = sms * 12;

    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    const float* xp = (const float*)x.data_ptr<float>();
    float* yp = (float*)y.data_ptr<float>();

    // float4 fast-path: require 16B alignment; handle any total via tail
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(xp);
    uintptr_t y_addr = reinterpret_cast<uintptr_t>(yp);
    bool aligned16 = ((x_addr | y_addr) & 0xF) == 0;

    // Use ILP=2 by default; ILP=4 can increase regs and hurt occupancy on some GPUs.
    if (aligned16 && total >= 4096) {
        mish_add_hardtanh_scale_vec4_kernel_v3<2><<<blocks, threads>>>(
            xp, yp, total, (float)add_value, (float)scale
        );
    } else {
        mish_add_hardtanh_scale_scalar_kernel_v3<<<blocks, threads>>>(
            xp, yp, total, (float)add_value, (float)scale
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mish_add_hardtanh_scale_cuda(torch::Tensor x, double add_value, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_conv_transpose2d_mish_add_hardtanh_scale_ops_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mish_add_hardtanh_scale_cuda"],
    verbose=False,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        "-lineinfo",
    ],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keep ConvTranspose2d (cuDNN), replace post-ops with one fused CUDA kernel:
      mish -> +add_value -> hardtanh(-1,1) -> *scale
    Includes CPU fallback to eager ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.add_value = float(add_value)
        self.scale = float(scale)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        if not x.is_cuda:
            y = torch.nn.functional.mish(x)
            y = y + self.add_value
            y = torch.nn.functional.hardtanh(y, min_val=-1.0, max_val=1.0)
            y = y * self.scale
            return y

        return self.custom_ops_lib.mish_add_hardtanh_scale_cuda(x, float(self.add_value), float(self.scale))