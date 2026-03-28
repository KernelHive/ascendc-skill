import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA op: dy_quantize_per_token_group_fwd_cuda
# Improvements over baseline:
#   - Specialized single-pass kernel for group_size == 128:
#       * load each element exactly once (cache per-lane 4 values in registers)
#       * bf16x2 vector loads via __nv_bfloat162 when 4B-aligned (guarded)
#       * warp-shuffle reduction for amax (no __syncthreads)
#       * tuned launch: 128 threads (4 warps) + __launch_bounds__ to control regs
#   - Generic fallback kernel for other group sizes (unchanged structure).
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, o);
    }
    return v;
}

__device__ __forceinline__ float clamp_fp8_range(float x, float fp8_max) {
    x = fminf(fp8_max, x);
    x = fmaxf(-fp8_max, x);
    return x;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 v) {
#if (__CUDA_ARCH__ >= 800)
    return __bfloat1622float2(v);
#else
    // Should not happen for bf16 CUDA builds, but keep safe.
    float2 o; o.x = 0.f; o.y = 0.f; return o;
#endif
}

// Single-pass specialized kernel for G=128.
// 1 warp per (token, group), each lane owns 4 elements (total 128).
// Launch tuned to 128 threads/block (4 warps). launch_bounds helps cap regs.
__global__ __launch_bounds__(128, 2)
void dy_quantize_per_token_group_g128_singlepass_kernel(
    const __nv_bfloat16* __restrict__ x, // [M, N]
    float* __restrict__ y,               // [M, N]
    float* __restrict__ scale_out,       // [M, Ng]
    int M, int N,
    float fp8_max,
    int aligned_4b // 1 if x pointer is 4-byte aligned
) {
    constexpr int G = 128;
    int Ng = N / G;

    int warp_in_block = (int)(threadIdx.x >> 5);  // 0..3
    int lane = (int)(threadIdx.x & 31);
    int global_warp = (int)(blockIdx.x * (blockDim.x >> 5) + warp_in_block);

    int total_groups = M * Ng;
    if (global_warp >= total_groups) return;

    int m = global_warp / Ng;
    int gg = global_warp - m * Ng;
    int base = m * N + gg * G;

    // Cache 4 values in registers
    float v0, v1, v2, v3;

    // Load path: prefer bf16x2 vector loads if aligned; else scalar.
    // Access pattern is fully coalesced either way.
    if (aligned_4b) {
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + base);
        // each lane covers 4 elems -> two bf16x2 loads at indices lane*2 and lane*2+1
        __nv_bfloat162 a = __ldg(x2 + (lane * 2 + 0));
        __nv_bfloat162 b = __ldg(x2 + (lane * 2 + 1));
        float2 fa = bf162_to_float2(a);
        float2 fb = bf162_to_float2(b);
        v0 = fa.x; v1 = fa.y; v2 = fb.x; v3 = fb.y;
    } else {
        int off = lane * 4;
        v0 = __bfloat162float(x[base + off + 0]);
        v1 = __bfloat162float(x[base + off + 1]);
        v2 = __bfloat162float(x[base + off + 2]);
        v3 = __bfloat162float(x[base + off + 3]);
    }

    // local amax
    float local_max = 0.0f;
    local_max = fmaxf(local_max, fabsf(v0));
    local_max = fmaxf(local_max, fabsf(v1));
    local_max = fmaxf(local_max, fabsf(v2));
    local_max = fmaxf(local_max, fabsf(v3));

    float amax = warp_reduce_max(local_max);
    amax = __shfl_sync(0xffffffff, amax, 0);
    amax = fmaxf(amax, 1.0e-4f);

    float scale = amax / fp8_max;
    float inv_scale = 1.0f / scale;

    if (lane == 0) {
        scale_out[m * Ng + gg] = scale;
    }

    // quantize and store from cached values
    float q0 = clamp_fp8_range(v0 * inv_scale, fp8_max);
    float q1 = clamp_fp8_range(v1 * inv_scale, fp8_max);
    float q2 = clamp_fp8_range(v2 * inv_scale, fp8_max);
    float q3 = clamp_fp8_range(v3 * inv_scale, fp8_max);

    int off = lane * 4;
    y[base + off + 0] = q0;
    y[base + off + 1] = q1;
    y[base + off + 2] = q2;
    y[base + off + 3] = q3;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float sh[32]; // up to 1024 threads -> 32 warps
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        float t = (tid < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
        t = warp_reduce_max(t);
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    out = sh[0];
    return out;
}

__global__ void dy_quantize_per_token_group_generic_kernel(
    const __nv_bfloat16* __restrict__ x, // [M, N]
    float* __restrict__ y,               // [M, N]
    float* __restrict__ scale_out,       // [M, Ng]
    int M, int N, int G,
    float fp8_max
) {
    int Ng = N / G;
    int idx = (int)blockIdx.x; // 0 .. M*Ng-1
    int m = idx / Ng;
    int gg = idx - m * Ng;
    if (m >= M) return;

    int base = m * N + gg * G;

    float local_max = 0.0f;
    for (int i = (int)threadIdx.x; i < G; i += (int)blockDim.x) {
        float v = __bfloat162float(x[base + i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float amax = block_reduce_max(local_max);
    amax = fmaxf(amax, 1.0e-4f);
    float scale = amax / fp8_max;

    if (threadIdx.x == 0) {
        scale_out[m * Ng + gg] = scale;
    }
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int i = (int)threadIdx.x; i < G; i += (int)blockDim.x) {
        float v = __bfloat162float(x[base + i]);
        float q = clamp_fp8_range(v * inv_scale, fp8_max);
        y[base + i] = q;
    }
}

std::vector<torch::Tensor> dy_quantize_per_token_group_fwd_cuda(
    torch::Tensor x, int64_t group_size, double fp8_max_d
) {
    CHECK_INPUT_BF16(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,N]");
    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    TORCH_CHECK((x.size(1) % group_size) == 0, "hidden_size must be divisible by group_size");

    int64_t M64 = x.size(0);
    int64_t N64 = x.size(1);
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX, "sizes too large for int32 indexing");
    int M = (int)M64;
    int N = (int)N64;
    int G = (int)group_size;
    int Ng = N / G;

    auto y = torch::empty({M, N}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
    auto scale = torch::empty({M, Ng}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    float fp8_max = (float)fp8_max_d;

    if (G == 128) {
        // 128 threads = 4 warps per block: often improves reg/occ tradeoff for single-pass caching.
        int threads = 128;
        int warps_per_block = threads / 32;
        int total_groups = M * Ng;
        int blocks = (total_groups + warps_per_block - 1) / warps_per_block;

        // bf16x2 vector loads require 4-byte alignment (x base pointer).
        // (y is float-aligned already; we only guard x for safe __nv_bfloat162*)
        uintptr_t x_ptr = (uintptr_t)x.data_ptr<at::BFloat16>();
        int aligned_4b = (x_ptr % 4u) == 0u;

        dy_quantize_per_token_group_g128_singlepass_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
            (float*)y.data_ptr<float>(),
            (float*)scale.data_ptr<float>(),
            M, N,
            fp8_max,
            aligned_4b
        );
    } else {
        int threads = 256;
        dim3 block(threads);
        dim3 grid((unsigned int)(M * Ng));
        dy_quantize_per_token_group_generic_kernel<<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
            (float*)y.data_ptr<float>(),
            (float*)scale.data_ptr<float>(),
            M, N, G,
            fp8_max
        );
    }

    return {y, scale};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dy_quantize_per_token_group_fwd_cuda(torch::Tensor x, int64_t group_size, double fp8_max_d);
"""

custom_ops_lib = load_inline(
    name="custom_dy_quantize_ops_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["dy_quantize_per_token_group_fwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Dynamic Quantization (DyQuantize).
    CUDA fast path for mode='per_token_group' returning float32 scaled+clamped values
    (exactly matching the reference semantics), plus float32 scales.
    Other modes fall back to PyTorch reference implementations.
    """

    FP8_MAX = 448.0

    def __init__(self, mode="per_token_group", group_size=128):
        super().__init__()
        self.mode = str(mode)
        self.group_size = int(group_size)
        self.custom_ops_lib = custom_ops_lib

    def _per_tensor(self, x):
        x_f = x.float()
        x_amax = x_f.abs().amax().clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = x_f / scale
        return x_scaled.clamp(-self.FP8_MAX, self.FP8_MAX), scale.view(1)

    def _per_token(self, x):
        m, _n = x.shape
        x_f = x.float()
        x_amax = x_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = x_f / scale
        return x_scaled.clamp(-self.FP8_MAX, self.FP8_MAX), scale.view(m)

    def _per_token_group(self, x):
        m, n = x.shape
        assert n % self.group_size == 0
        x_f = x.float()
        x_view = x_f.view(m, -1, self.group_size)
        x_amax = x_view.abs().amax(dim=2, keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = (x_view / scale).clamp(-self.FP8_MAX, self.FP8_MAX)
        return x_scaled.view(m, n), scale.view(m, -1)

    def _per_block(self, x):
        m, n = x.shape
        block = self.group_size
        m_pad = math.ceil(m / block) * block
        n_pad = math.ceil(n / block) * block
        x_padded = torch.zeros(m_pad, n_pad, dtype=torch.float32, device=x.device)
        x_padded[:m, :n] = x.float()

        x_view = x_padded.view(m_pad // block, block, n_pad // block, block)
        x_amax = x_view.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
        scale = x_amax / self.FP8_MAX
        x_scaled = (x_view / scale).clamp(-self.FP8_MAX, self.FP8_MAX)

        result = x_scaled.view(m_pad, n_pad)[:m, :n].contiguous()
        scale_out = scale.view(m_pad // block, n_pad // block)
        return result, scale_out

    def forward(self, x: torch.Tensor):
        if self.mode == "per_token_group" and x.is_cuda and x.dtype == torch.bfloat16:
            x_c = x.contiguous()
            y, scale = self.custom_ops_lib.dy_quantize_per_token_group_fwd_cuda(
                x_c, int(self.group_size), float(self.FP8_MAX)
            )
            return y, scale

        if self.mode == "per_tensor":
            return self._per_tensor(x)
        elif self.mode == "per_token":
            return self._per_token(x)
        elif self.mode == "per_token_group":
            return self._per_token_group(x)
        elif self.mode == "per_block":
            return self._per_block(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")