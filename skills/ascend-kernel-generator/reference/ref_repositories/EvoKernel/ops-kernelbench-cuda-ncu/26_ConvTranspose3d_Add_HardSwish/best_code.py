import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized fused post-op for:
#   t = x + add_in
#   y = t * hardswish(t)
# where hardswish(t) = t * clamp(t+3,0,6)/6
# => y = t^2 * clamp(t+3,0,6)/6
#
# Key improvements vs baseline:
# - 2D launch over (nc = N*C) blocks, each processes contiguous inner = D*H*W
#   segment -> better coalescing + cache behavior, simpler addressing.
# - Stronger vectorization: float4 for fp32; half4 (packed) for fp16.
# - More ILP: each thread processes multiple vector chunks per loop iteration.
# - __launch_bounds__ to help keep occupancy healthy.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
#include <cuda_fp16.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// -------------------- math helpers --------------------
__device__ __forceinline__ float clamp_relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

__device__ __forceinline__ float fused_op_f(float x, float a) {
    float t = x + a;
    float r6 = clamp_relu6(t + 3.0f);
    // y = t^2 * r6 / 6
    return (t * t) * (r6 * (1.0f / 6.0f));
}

__device__ __forceinline__ __half fused_op_h(__half xh, __half ah) {
    float x = __half2float(xh);
    float a = __half2float(ah);
    return __float2half_rn(fused_op_f(x, a));
}

// -------------------- fp32 kernels (nc-inner mapping) --------------------
__global__ __launch_bounds__(256, 2)
void add_hardswish_mul_f32_nc_vec4(
    const float* __restrict__ x,
    const float* __restrict__ add_in,
    float* __restrict__ y,
    int64_t inner, // D*H*W
    int C
) {
    int64_t nc = (int64_t)blockIdx.x;
    int64_t base = nc * inner;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
    const float4* __restrict__ a4 = reinterpret_cast<const float4*>(add_in + base);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y + base);

    int64_t inner4 = inner >> 2;
    int64_t t = (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x;

    // Process 2 vectors per iteration for more ILP
    for (int64_t i = t; i < inner4; i += stride * 2) {
        // v0
        float4 vx0 = x4[i];
        float4 va0 = a4[i];
        float4 out0;
        out0.x = fused_op_f(vx0.x, va0.x);
        out0.y = fused_op_f(vx0.y, va0.y);
        out0.z = fused_op_f(vx0.z, va0.z);
        out0.w = fused_op_f(vx0.w, va0.w);
        y4[i] = out0;

        // v1
        int64_t j = i + stride;
        if (j < inner4) {
            float4 vx1 = x4[j];
            float4 va1 = a4[j];
            float4 out1;
            out1.x = fused_op_f(vx1.x, va1.x);
            out1.y = fused_op_f(vx1.y, va1.y);
            out1.z = fused_op_f(vx1.z, va1.z);
            out1.w = fused_op_f(vx1.w, va1.w);
            y4[j] = out1;
        }
    }
}

__global__ __launch_bounds__(256, 2)
void add_hardswish_mul_f32_nc_scalar(
    const float* __restrict__ x,
    const float* __restrict__ add_in,
    float* __restrict__ y,
    int64_t inner
) {
    int64_t nc = (int64_t)blockIdx.x;
    int64_t base = nc * inner;

    int64_t t = (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x;

    for (int64_t i = t; i < inner; i += stride) {
        float xv = x[base + i];
        float av = add_in[base + i];
        y[base + i] = fused_op_f(xv, av);
    }
}

// -------------------- fp16 kernels (nc-inner mapping) --------------------
// half4 implemented as two half2 packed in int2 (8 bytes)
__device__ __forceinline__ __half2 load_half2(const __half* p) {
    return *reinterpret_cast<const __half2*>(p);
}
__device__ __forceinline__ void store_half2(__half* p, __half2 v) {
    *reinterpret_cast<__half2*>(p) = v;
}

__global__ __launch_bounds__(256, 2)
void add_hardswish_mul_f16_nc_half4(
    const __half* __restrict__ x,
    const __half* __restrict__ add_in,
    __half* __restrict__ y,
    int64_t inner
) {
    int64_t nc = (int64_t)blockIdx.x;
    int64_t base = nc * inner;

    int64_t inner4 = inner >> 2; // groups of 4 halfs
    int64_t t = (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x;

    const __half* xp = x + base;
    const __half* ap = add_in + base;
    __half* yp = y + base;

    for (int64_t i4 = t; i4 < inner4; i4 += stride * 2) {
        int64_t off = i4 << 2;

        // group 0: 4 halfs as two half2
        __half2 x0a = load_half2(xp + off);
        __half2 a0a = load_half2(ap + off);
        __half2 x0b = load_half2(xp + off + 2);
        __half2 a0b = load_half2(ap + off + 2);

        float2 xf0 = __half22float2(x0a);
        float2 af0 = __half22float2(a0a);
        float2 xf1 = __half22float2(x0b);
        float2 af1 = __half22float2(a0b);

        __half2 y0a = __floats2half2_rn(fused_op_f(xf0.x, af0.x), fused_op_f(xf0.y, af0.y));
        __half2 y0b = __floats2half2_rn(fused_op_f(xf1.x, af1.x), fused_op_f(xf1.y, af1.y));

        store_half2(yp + off, y0a);
        store_half2(yp + off + 2, y0b);

        // group 1 (ILP)
        int64_t j4 = i4 + stride;
        if (j4 < inner4) {
            int64_t off2 = j4 << 2;

            __half2 x1a = load_half2(xp + off2);
            __half2 a1a = load_half2(ap + off2);
            __half2 x1b = load_half2(xp + off2 + 2);
            __half2 a1b = load_half2(ap + off2 + 2);

            float2 xg0 = __half22float2(x1a);
            float2 ag0 = __half22float2(a1a);
            float2 xg1 = __half22float2(x1b);
            float2 ag1 = __half22float2(a1b);

            __half2 y1a = __floats2half2_rn(fused_op_f(xg0.x, ag0.x), fused_op_f(xg0.y, ag0.y));
            __half2 y1b = __floats2half2_rn(fused_op_f(xg1.x, ag1.x), fused_op_f(xg1.y, ag1.y));

            store_half2(yp + off2, y1a);
            store_half2(yp + off2 + 2, y1b);
        }
    }
}

__global__ __launch_bounds__(256, 2)
void add_hardswish_mul_f16_nc_scalar(
    const __half* __restrict__ x,
    const __half* __restrict__ add_in,
    __half* __restrict__ y,
    int64_t inner
) {
    int64_t nc = (int64_t)blockIdx.x;
    int64_t base = nc * inner;

    int64_t t = (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x;

    for (int64_t i = t; i < inner; i += stride) {
        __half xv = x[base + i];
        __half av = add_in[base + i];
        y[base + i] = fused_op_h(xv, av);
    }
}

// -------------------- launcher --------------------
torch::Tensor add_hardswish_mul_cuda(torch::Tensor x, torch::Tensor add_in) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(add_in.is_cuda(), "add_in must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(add_in.is_contiguous(), "add_in must be contiguous");
    TORCH_CHECK(x.sizes() == add_in.sizes(), "x and add_in must have identical shapes");
    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");

    auto y = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t inner = x.size(2) * x.size(3) * x.size(4);
    TORCH_CHECK(inner > 0, "inner must be > 0");

    int64_t nc_total = N * C;
    TORCH_CHECK(nc_total > 0, "N*C must be > 0");

    const int threads = 256;
    dim3 grid((unsigned int)nc_total, 1, 1);

    if (x.dtype() == torch::kFloat32) {
        TORCH_CHECK(add_in.dtype() == torch::kFloat32, "add_in must match x dtype");
        const float* xp = x.data_ptr<float>();
        const float* ap = add_in.data_ptr<float>();
        float* yp = y.data_ptr<float>();

        // vec4 safe if:
        // - base pointers 16B aligned
        // - inner multiple of 4 (so every (nc*inner) base stays aligned)
        uintptr_t xa = (uintptr_t)xp;
        uintptr_t aa = (uintptr_t)ap;
        uintptr_t ya = (uintptr_t)yp;
        bool aligned16 = ((xa | aa | ya) & 0xF) == 0;
        bool inner_mult4 = (inner & 3LL) == 0;

        if (aligned16 && inner_mult4) {
            add_hardswish_mul_f32_nc_vec4<<<grid, threads>>>(xp, ap, yp, inner, (int)C);
        } else {
            add_hardswish_mul_f32_nc_scalar<<<grid, threads>>>(xp, ap, yp, inner);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    } else if (x.dtype() == torch::kFloat16) {
        TORCH_CHECK(add_in.dtype() == torch::kFloat16, "add_in must match x dtype");
        const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
        const __half* ap = reinterpret_cast<const __half*>(add_in.data_ptr<at::Half>());
        __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

        // half4 path requires:
        // - 8B alignment for base pointers
        // - inner multiple of 4 (half elements)
        uintptr_t xa = (uintptr_t)xp;
        uintptr_t aa = (uintptr_t)ap;
        uintptr_t ya = (uintptr_t)yp;
        bool aligned8 = ((xa | aa | ya) & 0x7) == 0;
        bool inner_mult4 = (inner & 3LL) == 0;

        if (aligned8 && inner_mult4) {
            add_hardswish_mul_f16_nc_half4<<<grid, threads>>>(xp, ap, yp, inner);
        } else {
            add_hardswish_mul_f16_nc_scalar<<<grid, threads>>>(xp, ap, yp, inner);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    } else {
        TORCH_CHECK(false, "Unsupported dtype: expected float32 or float16");
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor add_hardswish_mul_cuda(torch::Tensor x, torch::Tensor add_in);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_add_hard_swish_v3_ncinner",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_hardswish_mul_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d in cuDNN + optimized fused CUDA post-op:
      t = x + add_input
      y = t * hardswish(t)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Kept for signature/state compatibility; not used in forward.
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=torch.float32))
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
        o = self.conv_transpose(x)
        # Fast path expects contiguous; fall back to contiguous copies (still typically cheap vs conv).
        o = o.contiguous()
        add_c = add_input.contiguous()
        return self.custom_ops.add_hardswish_mul_cuda(o, add_c)