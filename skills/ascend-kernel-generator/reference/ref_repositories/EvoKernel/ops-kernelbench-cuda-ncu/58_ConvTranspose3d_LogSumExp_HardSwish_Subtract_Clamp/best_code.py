import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// Constant-memory scalar bias (this model uses a single scalar bias)
__device__ __constant__ float g_bias_scalar;

static inline __device__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float clamp1(float v) {
    return fminf(1.0f, fmaxf(-1.0f, v));
}

// Online (single-pass) logsumexp over channels
__device__ __forceinline__ float lse_over_c_online(
    const float* __restrict__ x, int64_t base, int64_t strideC, int C
) {
    float m = -INFINITY;
    float s = 0.0f;
    #pragma unroll 1
    for (int c = 0; c < C; ++c) {
        float v = ldg_f(x + base + (int64_t)c * strideC);
        if (v <= m) {
            s += __expf(v - m);
        } else {
            s = s * __expf(m - v) + 1.0f;
            m = v;
        }
    }
    return m + __logf(s);
}

template<int C_FIXED>
__device__ __forceinline__ float lse_over_c_online_fixed(
    const float* __restrict__ x, int64_t base, int64_t strideC
) {
    float m = -INFINITY;
    float s = 0.0f;
    #pragma unroll
    for (int c = 0; c < C_FIXED; ++c) {
        float v = ldg_f(x + base + (int64_t)c * strideC);
        if (v <= m) {
            s += __expf(v - m);
        } else {
            s = s * __expf(m - v) + 1.0f;
            m = v;
        }
    }
    return m + __logf(s);
}

__device__ __forceinline__ float postop(float lse, float b) {
    // hard-swish: x * sigmoid(x+3)/6
    float hs = lse * sigmoidf_fast(lse + 3.0f) * (1.0f / 6.0f);
    return clamp1(hs - b);
}

// 2D launch: grid.y spans tiles of DHW; grid.x spans N.
// Each thread computes 2 consecutive dhw indices to raise ILP and reduce loop overhead.
template<int C_FIXED>
__global__ __launch_bounds__(256, 2)
void lse_hswish_sub_clamp_2d_fixedC(
    const float* __restrict__ x, // [N,C,DHW] contiguous in DHW
    float* __restrict__ y,       // [N,1,DHW]
    int DHW
) {
    const int n = (int)blockIdx.x;
    const int64_t strideC = (int64_t)DHW;
    const float b = g_bias_scalar;

    // tile base within this n
    const int tile = (int)blockIdx.y;
    const int t = (int)threadIdx.x;

    // Two outputs per thread
    int dhw0 = (tile * (int)blockDim.x * 2) + (t * 2);
    int64_t n_base_x = ((int64_t)n * (int64_t)C_FIXED) * (int64_t)DHW;
    int64_t n_base_y = (int64_t)n * (int64_t)DHW;

    if (dhw0 + 1 < DHW) {
        int64_t base0 = n_base_x + (int64_t)dhw0;
        float lse0 = lse_over_c_online_fixed<C_FIXED>(x, base0, strideC);
        float out0 = postop(lse0, b);

        int64_t base1 = base0 + 1;
        float lse1 = lse_over_c_online_fixed<C_FIXED>(x, base1, strideC);
        float out1 = postop(lse1, b);

        // Try vectorized float2 store when aligned
        // y pointer alignment is typically 256B; ensure offset is aligned for float2 (8B)
        float* yptr = y + n_base_y + (int64_t)dhw0;
        if ((((uintptr_t)yptr) & 0x7) == 0) {
            reinterpret_cast<float2*>(yptr)[0] = make_float2(out0, out1);
        } else {
            yptr[0] = out0;
            yptr[1] = out1;
        }
    } else if (dhw0 < DHW) {
        int64_t base0 = n_base_x + (int64_t)dhw0;
        float lse0 = lse_over_c_online_fixed<C_FIXED>(x, base0, strideC);
        y[n_base_y + (int64_t)dhw0] = postop(lse0, b);
    }
}

__global__ __launch_bounds__(256, 2)
void lse_hswish_sub_clamp_2d_generic(
    const float* __restrict__ x, // [N,C,DHW]
    float* __restrict__ y,       // [N,1,DHW]
    int C, int DHW
) {
    const int n = (int)blockIdx.x;
    const int64_t strideC = (int64_t)DHW;
    const float b = g_bias_scalar;

    const int tile = (int)blockIdx.y;
    const int t = (int)threadIdx.x;
    int dhw0 = (tile * (int)blockDim.x * 2) + (t * 2);

    int64_t n_base_x = ((int64_t)n * (int64_t)C) * (int64_t)DHW;
    int64_t n_base_y = (int64_t)n * (int64_t)DHW;

    if (dhw0 + 1 < DHW) {
        int64_t base0 = n_base_x + (int64_t)dhw0;
        float lse0 = lse_over_c_online(x, base0, strideC, C);
        float out0 = postop(lse0, b);

        int64_t base1 = base0 + 1;
        float lse1 = lse_over_c_online(x, base1, strideC, C);
        float out1 = postop(lse1, b);

        float* yptr = y + n_base_y + (int64_t)dhw0;
        if ((((uintptr_t)yptr) & 0x7) == 0) {
            reinterpret_cast<float2*>(yptr)[0] = make_float2(out0, out1);
        } else {
            yptr[0] = out0;
            yptr[1] = out1;
        }
    } else if (dhw0 < DHW) {
        int64_t base0 = n_base_x + (int64_t)dhw0;
        float lse0 = lse_over_c_online(x, base0, strideC, C);
        y[n_base_y + (int64_t)dhw0] = postop(lse0, b);
    }
}

static inline void set_bias_scalar_cuda(torch::Tensor bias) {
    // Copy first element of bias tensor into constant memory scalar (device-to-device).
    // bias must be contiguous CUDA float32, numel>=1.
    const float* bptr = bias.data_ptr<float>();
    cudaError_t st = cudaMemcpyToSymbol(g_bias_scalar, bptr, sizeof(float), 0, cudaMemcpyDeviceToDevice);
    if (st != cudaSuccess) {
        // Clear sticky error and fallback handled by throwing
        cudaGetLastError();
        TORCH_CHECK(false, "cudaMemcpyToSymbol(g_bias_scalar) failed");
    }
}

torch::Tensor lse_hardswish_sub_clamp_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");
    TORCH_CHECK(bias.numel() >= 1, "bias must have at least 1 element");

    auto x_c = x.contiguous();
    auto b_c = bias.contiguous();

    int N = (int)x_c.size(0);
    int C = (int)x_c.size(1);
    int D = (int)x_c.size(2);
    int H = (int)x_c.size(3);
    int W = (int)x_c.size(4);
    int DHW = D * H * W;

    auto y = torch::empty({N, 1, D, H, W}, x_c.options());

    // Update constant-memory bias scalar
    set_bias_scalar_cuda(b_c);

    const float* xptr = (const float*)x_c.data_ptr<float>();
    float* yptr = (float*)y.data_ptr<float>();

    const int threads = 256;

    // 2D grid: grid.x = N, grid.y = tiles over DHW where each block covers blockDim.x*2 elements
    int elems_per_block = threads * 2;
    int tiles = (DHW + elems_per_block - 1) / elems_per_block;
    dim3 grid((unsigned int)N, (unsigned int)tiles, 1);

    if (C == 16) {
        lse_hswish_sub_clamp_2d_fixedC<16><<<grid, threads>>>(
            xptr, yptr, DHW
        );
    } else {
        lse_hswish_sub_clamp_2d_generic<<<grid, threads>>>(
            xptr, yptr, C, DHW
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor lse_hardswish_sub_clamp_cuda(torch::Tensor x, torch::Tensor bias);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_lse_hswish_sub_clamp_v6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lse_hardswish_sub_clamp_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Keeps ConvTranspose3d in PyTorch (cuDNN), fuses:
      logsumexp(dim=1, keepdim=True) -> hard-swish -> subtract bias -> clamp[-1,1]
    into a single optimized custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=torch.float32))
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        if (not x.is_cuda) or x.dtype != torch.float32:
            t = torch.logsumexp(x, dim=1, keepdim=True)
            t = t * torch.sigmoid(t + 3) / 6
            t = t - self.bias
            return torch.clamp(t, min=-1, max=1)
        return self.custom_ops.lse_hardswish_sub_clamp_cuda(x, self.bias)