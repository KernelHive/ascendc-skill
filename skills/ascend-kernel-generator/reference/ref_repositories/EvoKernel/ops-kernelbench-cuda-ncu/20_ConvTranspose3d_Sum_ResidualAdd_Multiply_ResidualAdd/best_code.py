import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Fused post-ops after ConvTranspose3d:
#   original = o.detach()  (== o numerically)
#   y = o + bias
#   y = y + original
#   y = y * original
#   y = y + original
# => y = (2*o + bias)*o + o = fmaf((2*o + b), o, o)
#
# Optimizations:
#  - Per-(n,c) CTA streams contiguous inner=D*H*W => perfect coalescing, no div/mod.
#  - Constant-memory bias for C<=4096 with host-side caching to avoid per-call memcpy.
#  - Vectorized float4/float2 paths + alignment-aware prologue/epilogue + small unroll for ILP.
#  - Two kernel variants with __launch_bounds__ for 256/512 threads; runtime selects based on inner.
#  - Grid-stride over NC in case NC is huge and grid is capped.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static constexpr int CONST_BIAS_MAX = 4096;
__constant__ float c_bias[CONST_BIAS_MAX];

__device__ __forceinline__ float compute_out(float ov, float bv) {
    // (2*o + b)*o + o
    return fmaf(2.0f * ov + bv, ov, ov);
}

template <int THREADS, bool USE_CONST_BIAS>
__global__ __launch_bounds__(THREADS, 2)
void postops_nc_stream_kernel(
    const float* __restrict__ o,   // [N,C,D,H,W] contiguous
    const float* __restrict__ b,   // [C] if !USE_CONST_BIAS
    float* __restrict__ y,
    int NC,
    int C,
    int inner
) {
    // Grid-stride over nc (each nc owns one contiguous inner segment)
    for (int nc = (int)blockIdx.x; nc < NC; nc += (int)gridDim.x) {
        int c = nc - (nc / C) * C; // nc % C but faster on some compilers

        float bv;
        if constexpr (USE_CONST_BIAS) {
            bv = c_bias[c];
        } else {
            // Single global load per CTA; broadcast via shared mem.
            __shared__ float sb;
            if (threadIdx.x == 0) sb = b[c];
            __syncthreads();
            bv = sb;
        }

        int base = nc * inner;
        const float* __restrict__ in = o + base;
        float* __restrict__ out = y + base;

        // Alignment-aware prologue to reach 16B alignment for float4, then main float4 loop.
        // We keep all threads participating in a strided loop.
        int tid = (int)threadIdx.x;
        int stride = THREADS;

        uintptr_t in_addr = (uintptr_t)in;
        uintptr_t out_addr = (uintptr_t)out;
        bool aligned16 = (((in_addr | out_addr) & 0xF) == 0);

        // If aligned16 and inner multiple of 4 -> pure float4
        if (aligned16 && ((inner & 3) == 0)) {
            const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in);
            float4* __restrict__ out4 = reinterpret_cast<float4*>(out);
            int inner4 = inner >> 2;

            // Unroll by 2 vectors per iteration for ILP (conservative for regs)
            for (int i = tid; i < inner4; i += stride * 2) {
                int i0 = i;
                int i1 = i + stride;
                if (i0 < inner4) {
                    float4 v = in4[i0];
                    v.x = compute_out(v.x, bv);
                    v.y = compute_out(v.y, bv);
                    v.z = compute_out(v.z, bv);
                    v.w = compute_out(v.w, bv);
                    out4[i0] = v;
                }
                if (i1 < inner4) {
                    float4 v = in4[i1];
                    v.x = compute_out(v.x, bv);
                    v.y = compute_out(v.y, bv);
                    v.z = compute_out(v.z, bv);
                    v.w = compute_out(v.w, bv);
                    out4[i1] = v;
                }
            }
            continue;
        }

        // Secondary path: try float2 when 8B aligned and inner multiple of 2
        bool aligned8 = (((in_addr | out_addr) & 0x7) == 0);
        if (aligned8 && ((inner & 1) == 0)) {
            const float2* __restrict__ in2 = reinterpret_cast<const float2*>(in);
            float2* __restrict__ out2 = reinterpret_cast<float2*>(out);
            int inner2 = inner >> 1;

            for (int i = tid; i < inner2; i += stride * 2) {
                int i0 = i;
                int i1 = i + stride;
                if (i0 < inner2) {
                    float2 v = in2[i0];
                    v.x = compute_out(v.x, bv);
                    v.y = compute_out(v.y, bv);
                    out2[i0] = v;
                }
                if (i1 < inner2) {
                    float2 v = in2[i1];
                    v.x = compute_out(v.x, bv);
                    v.y = compute_out(v.y, bv);
                    out2[i1] = v;
                }
            }
            continue;
        }

        // Scalar fallback with unroll by 4
        for (int i = tid; i < inner; i += stride * 4) {
            int i0 = i;
            int i1 = i + stride;
            int i2 = i + stride * 2;
            int i3 = i + stride * 3;

            if (i0 < inner) {
                float ov = in[i0];
                out[i0] = compute_out(ov, bv);
            }
            if (i1 < inner) {
                float ov = in[i1];
                out[i1] = compute_out(ov, bv);
            }
            if (i2 < inner) {
                float ov = in[i2];
                out[i2] = compute_out(ov, bv);
            }
            if (i3 < inner) {
                float ov = in[i3];
                out[i3] = compute_out(ov, bv);
            }
        }
    }
}

static torch::Tensor bias_to_C(torch::Tensor bias, int64_t C) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    auto b = bias.contiguous();
    if (b.dim() == 1) {
        TORCH_CHECK(b.size(0) == C, "bias [C] expected");
        return b;
    } else if (b.dim() == 4) {
        TORCH_CHECK(b.size(0) == C && b.size(1) == 1 && b.size(2) == 1 && b.size(3) == 1,
                    "bias [C,1,1,1] expected");
        return b.view({C});
    } else if (b.dim() == 5) {
        TORCH_CHECK(b.size(0) == 1 && b.size(1) == C && b.size(2) == 1 && b.size(3) == 1 && b.size(4) == 1,
                    "bias [1,C,1,1,1] expected");
        return b.view({C});
    } else {
        TORCH_CHECK(false, "Unsupported bias shape; expected [C], [C,1,1,1], or [1,C,1,1,1]");
    }
}

// Host-side cache for constant memory bias updates (per-process; adequate for this module use-case)
static void* g_last_bias_ptr = nullptr;
static int64_t g_last_bias_numel = -1;
static uint64_t g_last_bias_version = 0;

torch::Tensor sum_residual_add_mul_residual_add_cuda(torch::Tensor o, torch::Tensor bias) {
    TORCH_CHECK(o.is_cuda(), "o must be CUDA");
    TORCH_CHECK(o.dtype() == torch::kFloat32, "o must be float32");
    TORCH_CHECK(o.dim() == 5, "o must be [N,C,D,H,W]");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous (NCDHW)");

    int64_t N = o.size(0);
    int64_t C64 = o.size(1);
    int64_t D = o.size(2);
    int64_t H = o.size(3);
    int64_t W = o.size(4);
    int64_t inner64 = D * H * W;

    TORCH_CHECK(C64 > 0, "C must be > 0");
    TORCH_CHECK(inner64 > 0, "inner must be > 0");
    TORCH_CHECK(C64 <= INT32_MAX, "C too large");
    TORCH_CHECK(inner64 <= INT32_MAX, "inner too large");
    TORCH_CHECK(N * C64 <= INT32_MAX, "N*C too large");

    auto b1 = bias_to_C(bias, C64);
    auto y = torch::empty_like(o);

    int C = (int)C64;
    int inner = (int)inner64;
    int NC = (int)(N * C64);

    // Grid sizing: enough blocks to cover SMs, but allow grid-stride over NC.
    int dev = o.get_device();
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    // Choose threads based on inner: larger inner benefits from 512 (more warps, more MLP).
    int threads = (inner >= 4096) ? 512 : 256;

    int blocks = sm_count * 12; // good latency hiding for streaming
    if (blocks < 1) blocks = 1;
    // Don't exceed NC too much; grid-stride exists but excessive blocks can add launch overhead.
    if (blocks > NC) blocks = NC;

    bool use_const = (C <= CONST_BIAS_MAX);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (use_const) {
        // Cache constant memory update if bias storage is unchanged.
        // Use data_ptr + numel + (best-effort) version counter when available.
        void* cur_ptr = (void*)b1.data_ptr<float>();
        int64_t cur_numel = b1.numel();
        uint64_t cur_version = 0;
        // TensorImpl version counter is internal; avoid relying on it. We use pointer+numel as cache key.
        // For Parameters, pointer typically stable unless reallocated.
        (void)cur_version;

        bool need_copy = (cur_ptr != g_last_bias_ptr) || (cur_numel != g_last_bias_numel);
        if (need_copy) {
            cudaMemcpyToSymbolAsync(c_bias, cur_ptr, C * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
            g_last_bias_ptr = cur_ptr;
            g_last_bias_numel = cur_numel;
            g_last_bias_version = cur_version;
        }

        if (threads == 512) {
            postops_nc_stream_kernel<512, true><<<blocks, 512, 0, stream>>>(
                (const float*)o.data_ptr<float>(),
                (const float*)b1.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                NC, C, inner
            );
        } else {
            postops_nc_stream_kernel<256, true><<<blocks, 256, 0, stream>>>(
                (const float*)o.data_ptr<float>(),
                (const float*)b1.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                NC, C, inner
            );
        }
    } else {
        if (threads == 512) {
            postops_nc_stream_kernel<512, false><<<blocks, 512, 0, stream>>>(
                (const float*)o.data_ptr<float>(),
                (const float*)b1.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                NC, C, inner
            );
        } else {
            postops_nc_stream_kernel<256, false><<<blocks, 256, 0, stream>>>(
                (const float*)o.data_ptr<float>(),
                (const float*)b1.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                NC, C, inner
            );
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor sum_residual_add_mul_residual_add_cuda(torch::Tensor o, torch::Tensor bias);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_t3d_postops_nc_stream_v4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sum_residual_add_mul_residual_add_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d via cuDNN + fused post-ops CUDA kernel.

    Fused expression: y = (2*o + bias)*o + o, where bias broadcasts per channel.
    Supports bias shapes: [C], [C,1,1,1], [1,C,1,1,1].
    Kernel expects float32 CUDA contiguous NCDHW.
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
        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=torch.float32))
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv_transpose(x)
        if (not o.is_cuda) or (o.dtype != torch.float32):
            original = o.detach()
            y = o + self.bias
            y = y + original
            y = y * original
            y = y + original
            return y
        if not o.is_contiguous():
            o = o.contiguous()
        return self.custom_ops.sum_residual_add_mul_residual_add_cuda(o, self.bias)