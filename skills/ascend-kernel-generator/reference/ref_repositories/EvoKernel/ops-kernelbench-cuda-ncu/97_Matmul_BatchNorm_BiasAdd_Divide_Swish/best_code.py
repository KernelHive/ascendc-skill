import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

static inline __device__ float ld_g(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x)); // --use_fast_math
}
__device__ __forceinline__ float swishf(float x) {
    float s = sigmoidf_fast(x);
    return x * s;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static inline bool is_aligned_16_host(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// stats[c] = {mean, rstd}
// Multi-warp per channel to increase parallelism over N without CTA-wide reductions.
// WARPS_PER_C=2 works well for N=1024.
template<int WARPS_PER_C, int UNROLL>
__global__ __launch_bounds__(256, 2) void bn_train_stats_multiwarp_kernel(
    const float* __restrict__ x,  // [N,C]
    float2* __restrict__ stats,   // [C]
    int N, int C, float eps)
{
    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;   // 0..(warps_per_block-1)
    int lane    = tid & 31;

    int warps_per_block = (int)blockDim.x >> 5;
    int channels_per_block = warps_per_block / WARPS_PER_C;
    if (channels_per_block < 1) return;

    int local_group = warp_id / WARPS_PER_C;          // which channel inside the block
    int warp_in_grp = warp_id - local_group*WARPS_PER_C; // 0..WARPS_PER_C-1

    int c = (int)(blockIdx.x * channels_per_block + local_group);
    if (c >= C) return;

    // Each warp in group processes n starting at (warp_in_grp*32 + lane) with step WARPS_PER_C*32
    float sum = 0.f;
    float sumsq = 0.f;
    int step = WARPS_PER_C * 32;

    int n = warp_in_grp * 32 + lane;
    // Unroll by UNROLL: process UNROLL consecutive iterations separated by 'step'
#pragma unroll
    for (; n + (UNROLL-1)*step < N; n += UNROLL*step) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int nn = n + u*step;
            float v = x[(size_t)nn * C + c];
            sum += v;
            sumsq += v * v;
        }
    }
    for (; n < N; n += step) {
        float v = x[(size_t)n * C + c];
        sum += v;
        sumsq += v * v;
    }

    // Reduce within warp
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    // Now reduce across WARPS_PER_C warps in group using shared for lane0 only (minimal traffic).
    __shared__ float sh_sum[8 * 32];   // enough for up to 8 warps (256 threads)
    __shared__ float sh_sumsq[8 * 32];

    int warp_linear = warp_id; // 0..warps_per_block-1
    if (lane == 0) {
        sh_sum[warp_linear] = sum;
        sh_sumsq[warp_linear] = sumsq;
    }
    __syncthreads();

    if (warp_in_grp == 0) {
        float gsum = 0.f, gsumsq = 0.f;
        if (lane < WARPS_PER_C) {
            int src_warp = local_group*WARPS_PER_C + lane;
            gsum = sh_sum[src_warp];
            gsumsq = sh_sumsq[src_warp];
        }
        gsum = warp_reduce_sum(gsum);
        gsumsq = warp_reduce_sum(gsumsq);

        if (lane == 0) {
            float invN = 1.0f / (float)N;
            float mean = gsum * invN;
            float ex2 = gsumsq * invN;
            float var = ex2 - mean * mean;
            float rstd = rsqrtf(var + eps);
            stats[c] = make_float2(mean, rstd);
        }
    }
}

// Vectorized apply: process 4 channels at once (float4) for coalesced loads/stores.
// Launch: 2D grid over (C4, N). Each thread handles one float4 at one row.
template<bool SCALAR_BIAS>
__global__ __launch_bounds__(256, 3) void bn_bias_div_swish_apply_vec4_kernel(
    const float* __restrict__ x,            // [N,C]
    float* __restrict__ y,                  // [N,C]
    const float2* __restrict__ stats,       // [C]
    const float* __restrict__ gamma,        // [C]
    const float* __restrict__ beta,         // [C]
    const float* __restrict__ extra_bias,   // [1] or [C]
    float inv_div,
    int N, int C)
{
    int C4 = C >> 2; // requires C%4==0
    int c4 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n0 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (c4 >= C4) return;

    int c = c4 << 2;

    // Load per-channel params and stats (4-wide). Use scalar loads to avoid alignment requirements on params.
    float2 st0 = stats[c + 0];
    float2 st1 = stats[c + 1];
    float2 st2 = stats[c + 2];
    float2 st3 = stats[c + 3];

    float g0 = ld_g(gamma + c + 0);
    float g1 = ld_g(gamma + c + 1);
    float g2 = ld_g(gamma + c + 2);
    float g3 = ld_g(gamma + c + 3);

    float b0 = ld_g(beta + c + 0);
    float b1 = ld_g(beta + c + 1);
    float b2 = ld_g(beta + c + 2);
    float b3 = ld_g(beta + c + 3);

    float eb0, eb1, eb2, eb3;
    if constexpr (SCALAR_BIAS) {
        float s = ld_g(extra_bias);
        eb0 = s; eb1 = s; eb2 = s; eb3 = s;
    } else {
        eb0 = ld_g(extra_bias + c + 0);
        eb1 = ld_g(extra_bias + c + 1);
        eb2 = ld_g(extra_bias + c + 2);
        eb3 = ld_g(extra_bias + c + 3);
    }

    int n_stride = (int)(gridDim.y * blockDim.y);
    for (int n = n0; n < N; n += n_stride) {
        size_t base = (size_t)n * C + c;
        const float4* x4p = reinterpret_cast<const float4*>(x + base);
        float4 v = x4p[0];

        float z0 = (((v.x - st0.x) * st0.y) * g0 + b0 + eb0) * inv_div;
        float z1 = (((v.y - st1.x) * st1.y) * g1 + b1 + eb1) * inv_div;
        float z2 = (((v.z - st2.x) * st2.y) * g2 + b2 + eb2) * inv_div;
        float z3 = (((v.w - st3.x) * st3.y) * g3 + b3 + eb3) * inv_div;

        float4 out;
        out.x = swishf(z0);
        out.y = swishf(z1);
        out.z = swishf(z2);
        out.w = swishf(z3);

        float4* y4p = reinterpret_cast<float4*>(y + base);
        y4p[0] = out;
    }
}

// Scalar apply fallback (no vectorization).
template<bool SCALAR_BIAS>
__global__ __launch_bounds__(256, 3) void bn_bias_div_swish_apply_scalar_kernel(
    const float* __restrict__ x,            // [N,C]
    float* __restrict__ y,                  // [N,C]
    const float2* __restrict__ stats,       // [C]
    const float* __restrict__ gamma,        // [C]
    const float* __restrict__ beta,         // [C]
    const float* __restrict__ extra_bias,   // [1] or [C]
    float inv_div,
    int N, int C)
{
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n0 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (c >= C) return;

    float2 st = stats[c];
    float mean = st.x;
    float rstd = st.y;
    float ga = ld_g(gamma + c);
    float be = ld_g(beta + c);
    float eb = SCALAR_BIAS ? ld_g(extra_bias) : ld_g(extra_bias + c);

    int n_stride = (int)(gridDim.y * blockDim.y);
    for (int n = n0; n < N; n += n_stride) {
        size_t idx = (size_t)n * C + c;
        float v = x[idx];
        float z = (((v - mean) * rstd) * ga + be + eb) * inv_div;
        y[idx] = swishf(z);
    }
}

static inline void cuda_check_last_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));
}

torch::Tensor bn_bias_div_swish_forward_train_cuda_opt(
    torch::Tensor x,             // [N,C]
    torch::Tensor gamma,         // [C]
    torch::Tensor beta,          // [C]
    torch::Tensor extra_bias,    // [1] or [C]
    double eps,
    double divide_value)
{
    CHECK_CUDA(x);
    CHECK_CUDA(gamma);
    CHECK_CUDA(beta);
    CHECK_CUDA(extra_bias);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(extra_bias);
    CHECK_FLOAT(x);
    CHECK_FLOAT(gamma);
    CHECK_FLOAT(beta);
    CHECK_FLOAT(extra_bias);

    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, C]");
    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    TORCH_CHECK(N64 > 0 && C64 > 0, "Invalid x shape");
    TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX, "N/C too large");
    int N = (int)N64;
    int C = (int)C64;

    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta must have shape [C]");
    int eb_numel = (int)extra_bias.numel();
    TORCH_CHECK(eb_numel == 1 || eb_numel == C, "extra_bias must have numel 1 or C");

    TORCH_CHECK(divide_value != 0.0, "divide_value must be non-zero");
    float inv_div = (float)(1.0 / divide_value);

    auto stats = torch::empty({C, 2}, x.options());
    auto y = torch::empty_like(x);

    // ---- Stats kernel launch ----
    // Use 256 threads (8 warps). With WARPS_PER_C=2 => 4 channels per block.
    constexpr int THREADS_STATS = 256;
    constexpr int WARPS_PER_C = 2;
    constexpr int UNROLL = 4;
    int warps_per_block = THREADS_STATS / 32;
    int channels_per_block = warps_per_block / WARPS_PER_C; // 4
    int blocks_stats = (C + channels_per_block - 1) / channels_per_block;
    // Cap blocks to reduce overhead but keep enough for occupancy
    int maxBlocksStats = 8192;
    if (blocks_stats > maxBlocksStats) blocks_stats = maxBlocksStats;
    if (blocks_stats < 1) blocks_stats = 1;

    bn_train_stats_multiwarp_kernel<WARPS_PER_C, UNROLL><<<blocks_stats, THREADS_STATS>>>(
        (const float*)x.data_ptr<float>(),
        (float2*)stats.data_ptr<float>(),
        N, C, (float)eps
    );
    cuda_check_last_error();

    // ---- Apply kernel launch ----
    bool scalar_bias = (eb_numel == 1);

    // Less-brittle gating: only require x/y 16B alignment and C%4==0 for vec4.
    bool vec4_ok = ((C & 3) == 0) &&
                   is_aligned_16_host(x.data_ptr()) &&
                   is_aligned_16_host(y.data_ptr());

    if (vec4_ok) {
        const int TX = 128; // threads in x for C4
        const int TY = 2;   // rows per block in y-dim
        dim3 threads(TX, TY, 1);
        int C4 = C >> 2;
        int blocks_x = (C4 + TX - 1) / TX;
        int blocks_y = (N + TY - 1) / TY;
        // cap Y; grid-stride handles the rest
        int maxBlocksY = 256;
        if (blocks_y > maxBlocksY) blocks_y = maxBlocksY;
        if (blocks_y < 1) blocks_y = 1;

        dim3 blocks(blocks_x, blocks_y, 1);

        if (scalar_bias) {
            bn_bias_div_swish_apply_vec4_kernel<true><<<blocks, threads>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float2*)stats.data_ptr<float>(),
                (const float*)gamma.data_ptr<float>(),
                (const float*)beta.data_ptr<float>(),
                (const float*)extra_bias.data_ptr<float>(),
                inv_div, N, C
            );
        } else {
            bn_bias_div_swish_apply_vec4_kernel<false><<<blocks, threads>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float2*)stats.data_ptr<float>(),
                (const float*)gamma.data_ptr<float>(),
                (const float*)beta.data_ptr<float>(),
                (const float*)extra_bias.data_ptr<float>(),
                inv_div, N, C
            );
        }
        cuda_check_last_error();
    } else {
        const int TX = 128;
        const int TY = 4;
        dim3 threads(TX, TY, 1);
        int blocks_x = (C + TX - 1) / TX;
        int blocks_y = (N + TY - 1) / TY;
        int maxBlocksY = 256;
        if (blocks_y > maxBlocksY) blocks_y = maxBlocksY;
        if (blocks_y < 1) blocks_y = 1;
        dim3 blocks(blocks_x, blocks_y, 1);

        if (scalar_bias) {
            bn_bias_div_swish_apply_scalar_kernel<true><<<blocks, threads>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float2*)stats.data_ptr<float>(),
                (const float*)gamma.data_ptr<float>(),
                (const float*)beta.data_ptr<float>(),
                (const float*)extra_bias.data_ptr<float>(),
                inv_div, N, C
            );
        } else {
            bn_bias_div_swish_apply_scalar_kernel<false><<<blocks, threads>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                (const float2*)stats.data_ptr<float>(),
                (const float*)gamma.data_ptr<float>(),
                (const float*)beta.data_ptr<float>(),
                (const float*)extra_bias.data_ptr<float>(),
                inv_div, N, C
            );
        }
        cuda_check_last_error();
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor bn_bias_div_swish_forward_train_cuda_opt(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor extra_bias,
    double eps,
    double divide_value);
"""

custom_ops_lib = load_inline(
    name="custom_matmul_bn_bias_div_swish_ops_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["bn_bias_div_swish_forward_train_cuda_opt"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized replacement:
      - Keep matmul (Linear) in PyTorch/cuBLAS.
      - Fuse BN(training stats over N) + extra bias + scalar divide + Swish via custom CUDA.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.divide_value = float(divide_value)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        y = self.matmul(x)
        if y.dtype != torch.float32:
            y = y.float()
        if not y.is_contiguous():
            y = y.contiguous()

        gamma = self.bn.weight
        beta = self.bn.bias
        extra_bias = self.bias

        if y.is_cuda:
            dev = y.device
            if gamma.device != dev:
                gamma = gamma.to(dev)
            if beta.device != dev:
                beta = beta.to(dev)
            if extra_bias.device != dev:
                extra_bias = extra_bias.to(dev)

            return self.custom_ops_lib.bn_bias_div_swish_forward_train_cuda_opt(
                y,
                gamma.contiguous(),
                beta.contiguous(),
                extra_bias.contiguous(),
                float(self.bn.eps),
                float(self.divide_value),
            )

        # CPU fallback
        y = nn.functional.batch_norm(
            y, running_mean=None, running_var=None,
            weight=gamma, bias=beta,
            training=True, momentum=self.bn.momentum, eps=self.bn.eps
        )
        y = y + extra_bias
        y = y / self.divide_value
        y = y * torch.sigmoid(y)
        return y