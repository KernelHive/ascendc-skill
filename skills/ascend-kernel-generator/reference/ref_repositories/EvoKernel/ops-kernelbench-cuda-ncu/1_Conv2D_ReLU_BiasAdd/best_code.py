import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA post-ops for Conv2d output (NCHW):
#   out = relu(x) + bias[C,1,1] (broadcast over N,H,W)
#
# Improvements vs current baseline:
# - Flattened grid-stride streaming kernel (better latency hiding)
# - Vectorized float4/float2 paths with strict alignment/size gating
# - Bias cached in __constant__ memory when C <= 4096 (fast broadcast)
# - Warp-uniform "same-channel" fastpath for vec loads to avoid per-lane div/mod
# - Light unroll for ILP; __launch_bounds__ to constrain registers
# - Optional out-inplace kernel interface (not used here; safe out-of-place)
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

static __forceinline__ __device__ float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
static __forceinline__ __device__ float ldg_f32(const float* p) { return __ldg(p); }
#else
static __forceinline__ __device__ float ldg_f32(const float* p) { return *p; }
#endif

// Constant memory bias cache (enough for typical channel sizes; 128 here)
#ifndef BIAS_CONST_MAX
#define BIAS_CONST_MAX 4096
#endif
__device__ __constant__ float g_bias_const[BIAS_CONST_MAX];

static __forceinline__ __device__ float load_bias(int c, const float* __restrict__ bias_c, bool use_const) {
    if (use_const) return g_bias_const[c];
    return ldg_f32(bias_c + c);
}

template<int UNROLL>
__global__ __launch_bounds__(256, 3)
void relu_biasadd_f32_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias_c, // [C]
    float* __restrict__ out,
    int64_t total, // N*C*H*W
    int C,
    int HW,
    bool use_const_bias
) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    int64_t total4 = total >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < total4; i += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j >= total4) break;

            // element base index for this float4
            int64_t base = j << 2;

            // Determine whether all 4 lanes are in same channel:
            // channel index changes only every HW elements.
            int64_t t0 = base / (int64_t)HW;
            int64_t t3 = (base + 3) / (int64_t)HW;

            float4 v = x4[j];

            if (t0 == t3) {
                int c = (int)(t0 % (int64_t)C);
                float b = load_bias(c, bias_c, use_const_bias);
                v.x = relu_f32(v.x) + b;
                v.y = relu_f32(v.y) + b;
                v.z = relu_f32(v.z) + b;
                v.w = relu_f32(v.w) + b;
            } else {
                // Rare boundary crossing: compute per-lane channel
                int c0 = (int)(t0 % (int64_t)C);
                int c1 = (int)(((base + 1) / (int64_t)HW) % (int64_t)C);
                int c2 = (int)(((base + 2) / (int64_t)HW) % (int64_t)C);
                int c3 = (int)(t3 % (int64_t)C);

                float b0 = load_bias(c0, bias_c, use_const_bias);
                float b1 = load_bias(c1, bias_c, use_const_bias);
                float b2 = load_bias(c2, bias_c, use_const_bias);
                float b3 = load_bias(c3, bias_c, use_const_bias);

                v.x = relu_f32(v.x) + b0;
                v.y = relu_f32(v.y) + b1;
                v.z = relu_f32(v.z) + b2;
                v.w = relu_f32(v.w) + b3;
            }

            out4[j] = v;
        }
    }
}

template<int UNROLL>
__global__ __launch_bounds__(256, 3)
void relu_biasadd_f32_vec2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias_c, // [C]
    float* __restrict__ out,
    int64_t total, // N*C*H*W
    int C,
    int HW,
    bool use_const_bias
) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    int64_t total2 = total >> 1;
    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    float2* __restrict__ out2 = reinterpret_cast<float2*>(out);

    for (int64_t i = tid; i < total2; i += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j >= total2) break;

            int64_t base = j << 1;
            int64_t t0 = base / (int64_t)HW;
            int64_t t1 = (base + 1) / (int64_t)HW;

            float2 v = x2[j];

            if (t0 == t1) {
                int c = (int)(t0 % (int64_t)C);
                float b = load_bias(c, bias_c, use_const_bias);
                v.x = relu_f32(v.x) + b;
                v.y = relu_f32(v.y) + b;
            } else {
                int c0 = (int)(t0 % (int64_t)C);
                int c1 = (int)(t1 % (int64_t)C);
                float b0 = load_bias(c0, bias_c, use_const_bias);
                float b1 = load_bias(c1, bias_c, use_const_bias);
                v.x = relu_f32(v.x) + b0;
                v.y = relu_f32(v.y) + b1;
            }

            out2[j] = v;
        }
    }
}

template<int UNROLL>
__global__ __launch_bounds__(256, 3)
void relu_biasadd_f32_scalar_kernel2(
    const float* __restrict__ x,
    const float* __restrict__ bias_c, // [C]
    float* __restrict__ out,
    int64_t total,
    int C,
    int HW,
    bool use_const_bias
) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < total; i += stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t j = i + (int64_t)u * stride;
            if (j >= total) break;
            int64_t t = j / (int64_t)HW;
            int c = (int)(t % (int64_t)C);
            float b = load_bias(c, bias_c, use_const_bias);
            float v = x[j];
            out[j] = relu_f32(v) + b;
        }
    }
}

static inline bool is_aligned(uintptr_t p, uintptr_t a) { return (p & (a - 1)) == 0; }

torch::Tensor fused_relu_bias_add_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "fused_relu_bias_add_cuda: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "fused_relu_bias_add_cuda: bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fused_relu_bias_add_cuda: only float32 supported");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "fused_relu_bias_add_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "fused_relu_bias_add_cuda: x must be contiguous (NCHW)");
    TORCH_CHECK(bias.is_contiguous(), "fused_relu_bias_add_cuda: bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "fused_relu_bias_add_cuda: x must be 4D [N,C,H,W]");
    TORCH_CHECK(bias.dim() == 3, "fused_relu_bias_add_cuda: bias must be 3D [C,1,1]");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    TORCH_CHECK(bias.size(0) == C, "fused_relu_bias_add_cuda: bias.size(0) must match x.size(1)");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1,
                "fused_relu_bias_add_cuda: bias must have shape [C,1,1]");

    auto bias_c = bias.view({C});
    auto out = torch::empty_like(x);

    const int64_t total = (int64_t)N * (int64_t)C * (int64_t)H * (int64_t)W;
    const int HW = H * W;

    const float* xp = (const float*)x.data_ptr<float>();
    const float* bp = (const float*)bias_c.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    bool use_const_bias = (C <= BIAS_CONST_MAX);
    if (use_const_bias) {
        // Copy bias to constant memory (very small; overhead is negligible relative to conv)
        cudaError_t err = cudaMemcpyToSymbol(g_bias_const, bp, (size_t)C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol(g_bias_const) failed");
    }

    // Launch config tuned for streaming kernels
    const int threads = 256;
    const int max_blocks = 4096;
    int blocks = (int)((total + threads - 1) / threads);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > max_blocks ? max_blocks : blocks;

    const uintptr_t xaddr = (uintptr_t)xp;
    const uintptr_t oaddr = (uintptr_t)op;

    const bool can_vec4 = ((total & 3LL) == 0LL) && is_aligned(xaddr, 16) && is_aligned(oaddr, 16);
    const bool can_vec2 = ((total & 1LL) == 0LL) && is_aligned(xaddr, 8) && is_aligned(oaddr, 8);

    if (can_vec4) {
        relu_biasadd_f32_vec4_kernel<2><<<blocks, threads>>>(
            xp, bp, op, total, C, HW, use_const_bias
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    } else if (can_vec2) {
        relu_biasadd_f32_vec2_kernel<2><<<blocks, threads>>>(
            xp, bp, op, total, C, HW, use_const_bias
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    } else {
        relu_biasadd_f32_scalar_kernel2<2><<<blocks, threads>>>(
            xp, bp, op, total, C, HW, use_const_bias
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
}
"""

cpp_source = r"""
torch::Tensor fused_relu_bias_add_cuda(torch::Tensor x, torch::Tensor bias);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_relu_bias_add_opt4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_relu_bias_add_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps Conv2d in PyTorch/cuDNN, fuses post-ops (ReLU + BiasAdd) in a custom CUDA kernel.
    Fast-path gated on CUDA + float32 + contiguous + expected bias shape [C,1,1],
    otherwise falls back to PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            x = torch.relu(x)
            return x + self.bias

        if not x.is_contiguous():
            x = x.contiguous()

        bias = self.bias
        if not bias.is_cuda:
            bias = bias.to(device=x.device)
        if bias.dtype != torch.float32:
            bias = bias.float()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        if bias.dim() != 3 or bias.size(0) != x.size(1) or bias.size(1) != 1 or bias.size(2) != 1:
            x = torch.relu(x)
            return x + self.bias

        return self.custom_ops_lib.fused_relu_bias_add_cuda(x, bias)


# Keep helper signatures consistent with the original
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]