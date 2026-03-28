import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused bias + clamp + scale + clamp + inv_scale
# Optimizations vs baseline:
#  - Change mapping: one block handles one (n, c) plane, iterates over HW contiguously
#    -> eliminates per-element div/mod, improves bias reuse + cache behavior.
#  - Optional float4 vectorized path along W when W%4==0 and pointers aligned.
#  - Bias loaded once per block using read-only cache load (__ldg where available).
#  - Conservative __launch_bounds__ to curb register growth.
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if (__CUDA_ARCH__ >= 350)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float clamp01(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

// Scalar kernel: block <-> (n,c), threads cover HW
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_plane_scalar(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ bias_c,
    float scale,
    float inv_scale,
    int N, int C, int H, int W, int HW)
{
    int plane = (int)blockIdx.x;            // [0, N*C)
    int n = plane / C;
    int c = plane - n * C;

    float b = ldg_f32(bias_c + c);
    int base = (n * C + c) * HW;

    for (int idx = (int)threadIdx.x; idx < HW; idx += THREADS) {
        float v = x[base + idx] + b;
        v = clamp01(v);
        v = clamp01(v * scale);
        y[base + idx] = v * inv_scale;
    }
}

// Vector kernel: assumes W divisible by 4, and x/y pointers 16B aligned.
// We vectorize along width so each float4 is contiguous in memory.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_plane_vec4_w(
    const float4* __restrict__ x4,
    float4* __restrict__ y4,
    const float* __restrict__ bias_c,
    float scale,
    float inv_scale,
    int N, int C, int H, int W, int HW)
{
    int plane = (int)blockIdx.x;            // [0, N*C)
    int n = plane / C;
    int c = plane - n * C;

    float b = ldg_f32(bias_c + c);

    int W4 = W >> 2;                        // W/4
    int row4 = W4;
    int HW4 = H * row4;

    // base in float4 units
    int base4 = (n * C + c) * HW4;

    for (int idx4 = (int)threadIdx.x; idx4 < HW4; idx4 += THREADS) {
        float4 v = x4[base4 + idx4];

        float o0 = clamp01(v.x + b);
        float o1 = clamp01(v.y + b);
        float o2 = clamp01(v.z + b);
        float o3 = clamp01(v.w + b);

        o0 = clamp01(o0 * scale) * inv_scale;
        o1 = clamp01(o1 * scale) * inv_scale;
        o2 = clamp01(o2 * scale) * inv_scale;
        o3 = clamp01(o3 * scale) * inv_scale;

        y4[base4 + idx4] = make_float4(o0, o1, o2, o3);
    }
}

torch::Tensor fused_bias_clamp_scale_clamp_invscale_cuda(torch::Tensor x,
                                                        torch::Tensor bias_c,
                                                        double scale)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias_c.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(bias_c.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(bias_c.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(bias_c.dim() == 1, "bias must be 1D [C]");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK((int)bias_c.numel() == C, "bias must have C elements");

    auto y = torch::empty_like(x);

    int HW = H * W;
    float fscale = (float)scale;
    float inv_scale = 1.0f / fscale;

    // Grid: one block per (n,c) plane
    int blocks = N * C;

    // Threads: tune for bandwidth; 256 is a good default for pure mem kernels
    constexpr int THREADS = 256;

    // Vectorized eligibility: W%4==0 and pointers 16B aligned
    bool vec_ok = ((W & 3) == 0);
    uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    uintptr_t yp = (uintptr_t)y.data_ptr<float>();
    vec_ok = vec_ok && ((xp & 15u) == 0u) && ((yp & 15u) == 0u);

    if (vec_ok) {
        const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y4 = reinterpret_cast<float4*>(y.data_ptr<float>());
        fused_plane_vec4_w<THREADS><<<blocks, THREADS>>>(
            x4, y4,
            (const float*)bias_c.data_ptr<float>(),
            fscale, inv_scale,
            N, C, H, W, HW
        );
    } else {
        fused_plane_scalar<THREADS><<<blocks, THREADS>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias_c.data_ptr<float>(),
            fscale, inv_scale,
            N, C, H, W, HW
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor fused_bias_clamp_scale_clamp_invscale_cuda(torch::Tensor x,
                                                        torch::Tensor bias_c,
                                                        double scale);
"""

custom_ops_lib = load_inline(
    name="custom_conv_transpose2d_bias_add_clamp_scaling_clamp_divide_ops_v2_plane",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_bias_clamp_scale_clamp_invscale_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keep ConvTranspose2d; fuse:
      x = x + bias
      x = clamp(x, 0, 1)
      x = x * scale
      x = clamp(x, 0, 1)
      x = x / scale
    into one custom CUDA kernel (float32, contiguous NCHW).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            int(in_channels), int(out_channels), int(kernel_size),
            stride=int(stride), padding=int(padding), output_padding=int(output_padding)
        )
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib

    @staticmethod
    def _bias_to_c(bias: torch.Tensor):
        # Convert per-channel bias to contiguous [C]
        if bias.dim() == 3:
            # typical [C,1,1]
            return bias.contiguous().view(bias.size(0), -1)[:, 0].contiguous()
        if bias.dim() == 1:
            return bias.contiguous()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        bias = self.bias
        if bias.dtype != torch.float32:
            bias = bias.float()

        bias_c = self._bias_to_c(bias)

        # CPU or odd bias shape fallback
        if (not x.is_cuda) or (bias_c is None):
            y = x + self.bias.to(dtype=x.dtype, device=x.device)
            y = torch.clamp(y, min=0.0, max=1.0)
            y = y * self.scaling_factor
            y = torch.clamp(y, min=0.0, max=1.0)
            y = y / self.scaling_factor
            return y

        if (not bias_c.is_cuda) or (bias_c.device != x.device):
            bias_c = bias_c.to(device=x.device)

        return self.custom_ops_lib.fused_bias_clamp_scale_clamp_invscale_cuda(
            x, bias_c, float(self.scaling_factor)
        )