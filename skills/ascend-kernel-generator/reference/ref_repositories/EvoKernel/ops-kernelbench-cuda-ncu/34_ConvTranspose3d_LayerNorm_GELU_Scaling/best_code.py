import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    float x2 = x * x;
    float x3 = x2 * x;
    float t = kAlpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// W==64: one warp owns one row, each lane loads 2 floats once and stores 2 floats once.
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
void ln_gelu_scale_w64_singlepass_kernel(
    const float* __restrict__ x,     // [rows, 64]
    const float* __restrict__ gamma, // [64]
    const float* __restrict__ beta,  // [64]
    float* __restrict__ y,           // [rows, 64]
    int rows,
    float eps,
    float scale
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    const int warp_grid_stride = (int)gridDim.x * WARPS_PER_BLOCK;

    // Preload gamma/beta as float2 for this lane's 2 outputs (reuse for all rows this warp processes)
    const int w0 = lane * 2;
    float2 g2 = *reinterpret_cast<const float2*>(gamma + w0);
    float2 b2 = *reinterpret_cast<const float2*>(beta + w0);

    while (row < rows) {
        const int64_t base = (int64_t)row * 64;

        // Load two elements per lane (coalesced)
        float v0 = x[base + lane];
        float v1 = x[base + lane + 32];

        // Stats
        float sum = v0 + v1;
        float sumsq = fmaf(v0, v0, v1 * v1);

        sum = warp_reduce_sum(sum);
        sumsq = warp_reduce_sum(sumsq);
        sum = __shfl_sync(0xffffffff, sum, 0);
        sumsq = __shfl_sync(0xffffffff, sumsq, 0);

        float mean = sum * (1.0f / 64.0f);
        float var = sumsq * (1.0f / 64.0f) - mean * mean;
        var = fmaxf(var, 0.0f);
        float inv_std = rsqrtf(var + eps);

        // Epilogue for the 2 contiguous outputs this lane owns: indices [2*lane, 2*lane+1]
        float2 out;
        // Map lane's two scalar values to contiguous positions:
        // We loaded (lane) and (lane+32). Need contiguous indices:
        // Use v_contig0 = x[2*lane], v_contig1 = x[2*lane+1]
        // So load them now as float2. This is the only extra read; it is still single-pass overall
        // relative to baseline (baseline reads every element twice; here only this float2 once, stats from v0/v1).
        // But we can avoid extra read by using a permutation; instead, do a single float2 load and compute stats from it.
        // Implement that: load float2 first, and for stats load also the second float2 32 elements ahead.
        // For correctness and bandwidth, we do: float2 a = x[2*lane], float2 b = x[2*lane+32]
        // and compute stats on (a.x,a.y,b.x,b.y) with two lanes? Not possible.
        // Therefore, we restructure: use float2 load for contiguous store only; stats already computed from v0/v1.
        // This costs +1 read per lane (2 floats), but removes the full-row second pass and keeps stores vectorized.

        float2 xv = *reinterpret_cast<const float2*>(x + base + w0);

        float n0 = (xv.x - mean) * inv_std;
        float n1 = (xv.y - mean) * inv_std;

        float a0 = fmaf(n0, g2.x, b2.x);
        float a1 = fmaf(n1, g2.y, b2.y);

        out.x = gelu_tanh(a0) * scale;
        out.y = gelu_tanh(a1) * scale;

        *reinterpret_cast<float2*>(y + base + w0) = out;

        row += warp_grid_stride;
    }
}

// Generic W: warp-per-row; reads x twice (stats + epilogue) like baseline, but leaner and WARPS_PER_BLOCK=4.
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
void ln_gelu_scale_generic_kernel(
    const float* __restrict__ x,     // [rows, W]
    const float* __restrict__ gamma, // [W]
    const float* __restrict__ beta,  // [W]
    float* __restrict__ y,           // [rows, W]
    int rows,
    int W,
    float eps,
    float scale
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    const int warp_grid_stride = (int)gridDim.x * WARPS_PER_BLOCK;

    while (row < rows) {
        const int64_t base = (int64_t)row * (int64_t)W;

        float sum = 0.0f;
        float sumsq = 0.0f;

        for (int w = lane; w < W; w += 32) {
            float v = x[base + w];
            sum += v;
            sumsq = fmaf(v, v, sumsq);
        }

        sum = warp_reduce_sum(sum);
        sumsq = warp_reduce_sum(sumsq);
        sum = __shfl_sync(0xffffffff, sum, 0);
        sumsq = __shfl_sync(0xffffffff, sumsq, 0);

        float invW = 1.0f / (float)W;
        float mean = sum * invW;
        float var = sumsq * invW - mean * mean;
        var = fmaxf(var, 0.0f);
        float inv_std = rsqrtf(var + eps);

        for (int w = lane; w < W; w += 32) {
            float v = x[base + w];
            float n = (v - mean) * inv_std;
            float a = fmaf(n, ldg_f32(gamma + w), ldg_f32(beta + w));
            y[base + w] = gelu_tanh(a) * scale;
        }

        row += warp_grid_stride;
    }
}

torch::Tensor ln_lastdim_gelu_scale_cuda_v4(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps,
    double scaling
) {
    CHECK_CUDA(x);
    CHECK_CUDA(gamma);
    CHECK_CUDA(beta);

    CHECK_F32(x);
    CHECK_F32(gamma);
    CHECK_F32(beta);

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");

    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [W]");
    TORCH_CHECK(beta.dim() == 1, "beta must be [W]");

    const int64_t W64 = x.size(4);
    TORCH_CHECK(W64 > 0 && W64 <= INT32_MAX, "invalid W");
    const int W = (int)W64;

    TORCH_CHECK(gamma.numel() == W, "gamma must have numel == W (last dim)");
    TORCH_CHECK(beta.numel() == W, "beta must have numel == W (last dim)");

    int64_t rows64 = x.size(0) * x.size(1) * x.size(2) * x.size(3);
    TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "rows too large");
    const int rows = (int)rows64;

    auto y = torch::empty_like(x);

    c10::cuda::CUDAGuard device_guard(x.device());

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    int blocks = sm * 10;
    int max_blocks_for_rows = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (blocks > max_blocks_for_rows) blocks = max_blocks_for_rows;
    if (blocks < 1) blocks = 1;

    if (W == 64) {
        ln_gelu_scale_w64_singlepass_kernel<WARPS_PER_BLOCK><<<blocks, THREADS, 0>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            rows,
            (float)eps,
            (float)scaling
        );
    } else {
        ln_gelu_scale_generic_kernel<WARPS_PER_BLOCK><<<blocks, THREADS, 0>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            rows,
            W,
            (float)eps,
            (float)scaling
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor ln_lastdim_gelu_scale_cuda_v4(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps,
    double scaling
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convt3d_lnlast_gelu_scale_opt7",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["ln_lastdim_gelu_scale_cuda_v4"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps nn.ConvTranspose3d on cuDNN, fuses:
      LayerNorm(over last dim) -> GELU -> scaling

    Correctness constraint vs nn.LayerNorm(out_channels):
      For an NCDHW tensor, LayerNorm(out_channels) is only equivalent to LN over last dim
      when x.size(-1) == out_channels at runtime.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.custom_ops = custom_ops_lib

        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(int(out_channels), eps=float(eps))
        self.scaling_factor = float(scaling_factor)
        self.eps = float(eps)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if x.size(-1) != self.out_channels:
            raise RuntimeError(
                f"ModelNew fast-path requires x.size(-1) == out_channels, got W={x.size(-1)} "
                f"and out_channels={self.out_channels}."
            )

        if not x.is_cuda:
            raise RuntimeError("ModelNew supports CUDA tensors only")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.layer_norm.weight
        b = self.layer_norm.bias
        if not w.is_contiguous():
            w = w.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()

        return self.custom_ops.ln_lastdim_gelu_scale_cuda_v4(
            x,
            w,
            b,
            float(self.eps),
            float(self.scaling_factor),
        )