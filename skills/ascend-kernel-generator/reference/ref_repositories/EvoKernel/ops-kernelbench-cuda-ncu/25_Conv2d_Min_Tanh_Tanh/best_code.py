import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: channel-min reduction + tanh(tanh(.)) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <math.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

static __device__ __forceinline__ float ld_ro_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float tanh2_fast(float x) {
    float y = tanhf(x);      // --use_fast_math makes this fast approx
    return tanhf(y);
}

static __device__ __forceinline__ float4 ld_ro_f4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    // __ldg supports vector types on modern toolchains; if not, compiler will lower.
    return __ldg(p);
#else
    return *p;
#endif
}

// Vectorized kernel: each thread computes 4 adjacent W elements (w4..w4+3) for fixed (n,h)
// Only valid when W4 = W/4 and base pointer is 16B aligned and w4 < W4.
template<bool C64>
__global__ __launch_bounds__(256, 2)
void channel_min_tanh2_vec4_f32_kernel(
    const float* __restrict__ x,  // [N,C,H,W] contiguous
    float* __restrict__ out,      // [N,1,H,W] contiguous
    int N, int C, int H, int W, int W4
) {
    int n = blockIdx.z;
    int h = blockIdx.y;

    int w4 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (w4 >= W4) return;

    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t base_n = (int64_t)n * (int64_t)C * HW;
    int64_t base_hw4 = (int64_t)h * (int64_t)W4 + (int64_t)w4; // index in (H, W4) for float4

    float4 m4;
    m4.x = INFINITY; m4.y = INFINITY; m4.z = INFINITY; m4.w = INFINITY;

    if constexpr (C64) {
        #pragma unroll 8
        for (int c = 0; c < 64; ++c) {
            const float4* x4 = (const float4*)(x + base_n + (int64_t)c * HW);
            float4 v = ld_ro_f4(x4 + base_hw4);
            m4.x = fminf(m4.x, v.x);
            m4.y = fminf(m4.y, v.y);
            m4.z = fminf(m4.z, v.z);
            m4.w = fminf(m4.w, v.w);
        }
    } else {
        for (int c = 0; c < C; ++c) {
            const float4* x4 = (const float4*)(x + base_n + (int64_t)c * HW);
            float4 v = ld_ro_f4(x4 + base_hw4);
            m4.x = fminf(m4.x, v.x);
            m4.y = fminf(m4.y, v.y);
            m4.z = fminf(m4.z, v.z);
            m4.w = fminf(m4.w, v.w);
        }
    }

    // write 4 outputs
    int64_t out_base = ((int64_t)n * (int64_t)H + (int64_t)h) * (int64_t)W + (int64_t)w4 * 4;
    out[out_base + 0] = tanh2_fast(m4.x);
    out[out_base + 1] = tanh2_fast(m4.y);
    out[out_base + 2] = tanh2_fast(m4.z);
    out[out_base + 3] = tanh2_fast(m4.w);
}

// Scalar tail / generic kernel (handles any W, any alignment)
__global__ __launch_bounds__(256, 2)
void channel_min_tanh2_scalar_f32_kernel(
    const float* __restrict__ x,  // [N,C,H,W] contiguous
    float* __restrict__ out,      // [N,1,H,W] contiguous
    int N, int C, int H, int W
) {
    int64_t HW = (int64_t)H * (int64_t)W;
    int64_t NHW = (int64_t)N * HW;

    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    for (int64_t idx = tid; idx < NHW; idx += stride) {
        int64_t n = idx / HW;
        int64_t hw = idx - n * HW;

        const float* xp = x + n * ((int64_t)C * HW) + hw;

        float m = INFINITY;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            m = fminf(m, ld_ro_f32(xp + (int64_t)c * HW));
        }
        out[idx] = tanh2_fast(m);
    }
}

torch::Tensor conv2d_min_tanh_tanh_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "conv2d_min_tanh_tanh_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "conv2d_min_tanh_tanh_cuda: x must be float32");
    TORCH_CHECK(x.dim() == 4, "conv2d_min_tanh_tanh_cuda: x must be 4D NCHW");
    TORCH_CHECK(x.is_contiguous(), "conv2d_min_tanh_tanh_cuda: x must be contiguous (NCHW)");

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    auto out = torch::empty({N, 1, H, W}, x.options());

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    // Vectorized path requirements:
    // - W divisible by 4
    // - x pointer is 16-byte aligned (so float4 loads are aligned)
    // NOTE: out is freshly allocated and aligned; only x alignment is checked.
    bool can_vec4 = ((W & 3) == 0) && ((((uintptr_t)xp) & 15) == 0);

    if (can_vec4) {
        int W4 = W >> 2;
        dim3 block(256, 1, 1);
        dim3 grid((W4 + block.x - 1) / block.x, H, N);

        if (C == 64) {
            channel_min_tanh2_vec4_f32_kernel<true><<<grid, block, 0, stream>>>(xp, op, N, C, H, W, W4);
        } else {
            channel_min_tanh2_vec4_f32_kernel<false><<<grid, block, 0, stream>>>(xp, op, N, C, H, W, W4);
        }
    } else {
        // Scalar fallback
        constexpr int THREADS = 256;
        int64_t NHW = (int64_t)N * (int64_t)H * (int64_t)W;
        int blocks = (int)((NHW + THREADS - 1) / THREADS);
        if (blocks < 120) blocks = 120;
        if (blocks > 8192) blocks = 8192;

        channel_min_tanh2_scalar_f32_kernel<<<blocks, THREADS, 0, stream>>>(xp, op, N, C, H, W);
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor conv2d_min_tanh_tanh_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_min_tanh_tanh_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_min_tanh_tanh_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Convolution (cuDNN) followed by fused:
      y = tanh(tanh(min_c conv(x)[n,c,h,w]))  with keepdim=True -> [N,1,H,W]

    CUDA kernel supports: CUDA, float32, contiguous NCHW, 4D.
    Fallback path handles other cases for correctness.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4):
            x = torch.min(x, dim=1, keepdim=True)[0]
            x = torch.tanh(x)
            x = torch.tanh(x)
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.conv2d_min_tanh_tanh_cuda(x)