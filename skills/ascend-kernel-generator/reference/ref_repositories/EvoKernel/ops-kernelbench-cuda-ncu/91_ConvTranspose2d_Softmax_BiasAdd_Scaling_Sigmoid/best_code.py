import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# CUDA/C++ extension: fused softmax(dim=1) + bias(add, per-C) + scaling + sigmoid
# Output:
#   y[n,c,h,w] = sigmoid( (softmax_c(x[n,:,h,w]) + bias[c]) * scale )
#
# Optimizations vs baseline:
#   1) Cache logits (x values) in shared memory to avoid rereading x 3 times.
#   2) Warp-specialized reductions (shfl) for max/sum -> fewer barriers.
#   3) Block size tuned by C (128/256) + __launch_bounds__ to help occupancy.
# ============================================================

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // fast math exp already enabled; keep classic sigmoid
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask=0xffffffffu) {
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v, unsigned mask=0xffffffffu) {
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

// Optional fast exp via exp2f(x * log2(e)). Usually close to __expf on modern GPUs with --use_fast_math.
// Kept as a helper to allow easy switching.
__device__ __forceinline__ float fast_exp(float x) {
    // return __expf(x);
    return exp2f(x * 1.4426950408889634f); // log2(e)
}

// One block per (n,h,w). Threads iterate channels.
// Cache logits in shared memory then do warp-specialized reductions.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_softmax_bias_scale_sigmoid_cached_logits_kernel(
    const float* __restrict__ x,    // [N,C,H,W] contiguous
    float* __restrict__ y,          // [N,C,H,W] contiguous
    const float* __restrict__ bias, // [C]
    float scale,
    int N, int C, int H, int W
) {
    int nhw = (int)blockIdx.x; // 0..N*H*W-1
    int HW = H * W;
    int n = nhw / HW;
    int hw = nhw - n * HW;
    if (n >= N) return;
    int h = hw / W;
    int w = hw - h * W;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    constexpr int WARP = 32;
    constexpr int NWARPS = (THREADS + WARP - 1) / WARP;
    unsigned mask = 0xffffffffu;

    // x index base for channel 0 at this (n,h,w)
    int64_t HW64 = (int64_t)H * (int64_t)W;
    int64_t x_base = ((int64_t)n * (int64_t)C * HW64) + (int64_t)h * (int64_t)W + (int64_t)w;

    extern __shared__ float smem[];
    float* sh_logits = smem;           // [C]
    float* sh_warp   = sh_logits + C;  // [2*NWARPS] (max, sum) partials

    // 1) Load logits once from global -> shared
    for (int c = tid; c < C; c += THREADS) {
        sh_logits[c] = LDG(x + x_base + (int64_t)c * HW64);
    }
    __syncthreads();

    // 2) Warp partial max over strided channels for this block
    float local_max = -INFINITY;
    for (int c = tid; c < C; c += THREADS) {
        local_max = fmaxf(local_max, sh_logits[c]);
    }
    float wmax = warp_reduce_max(local_max, mask);
    if (lane == 0) sh_warp[wid] = wmax;
    __syncthreads();

    // 3) Reduce warp maxima with warp0
    float maxv = -INFINITY;
    if (wid == 0) {
        float v = (lane < NWARPS) ? sh_warp[lane] : -INFINITY;
        float r = warp_reduce_max(v, mask);
        if (lane == 0) sh_warp[0] = r;
    }
    __syncthreads();
    maxv = sh_warp[0];

    // 4) Warp partial sum exp(logit - max)
    float local_sum = 0.0f;
    for (int c = tid; c < C; c += THREADS) {
        local_sum += fast_exp(sh_logits[c] - maxv);
    }
    float wsum = warp_reduce_sum(local_sum, mask);
    if (lane == 0) sh_warp[NWARPS + wid] = wsum;
    __syncthreads();

    // 5) Reduce warp sums with warp0
    float sumv = 0.0f;
    if (wid == 0) {
        float v = (lane < NWARPS) ? sh_warp[NWARPS + lane] : 0.0f;
        float r = warp_reduce_sum(v, mask);
        if (lane == 0) sh_warp[1] = r;
    }
    __syncthreads();
    sumv = sh_warp[1];
    float invsum = 1.0f / fmaxf(sumv, 1e-20f);

    // 6) Final write: softmax + bias + scale + sigmoid
    for (int c = tid; c < C; c += THREADS) {
        float p = fast_exp(sh_logits[c] - maxv) * invsum;
        float b = LDG(bias + c);
        float outv = sigmoidf_fast((p + b) * scale);
        y[x_base + (int64_t)c * HW64] = outv;
    }
}

torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(torch::Tensor x,
                                                   torch::Tensor bias_c,
                                                   double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias_c.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(bias_c.dtype() == torch::kFloat32, "only float32 bias supported");
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

    int64_t blocks64 = (int64_t)N * (int64_t)H * (int64_t)W;
    TORCH_CHECK(blocks64 > 0, "Invalid launch size");
    int blocks = (blocks64 > (int64_t)INT_MAX) ? INT_MAX : (int)blocks64;

    // Tune threads by channel count.
    // For C<=128, 128 threads (4 warps) is generally enough and reduces wasted work.
    // Otherwise 256 threads (8 warps) to increase ILP/latency hiding.
    if (C <= 128) {
        constexpr int THREADS = 128;
        size_t shmem = (size_t)C * sizeof(float) + (size_t)(2 * ((THREADS + 31) / 32)) * sizeof(float);
        fused_softmax_bias_scale_sigmoid_cached_logits_kernel<THREADS><<<blocks, THREADS, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias_c.data_ptr<float>(),
            (float)scale,
            N, C, H, W
        );
    } else {
        constexpr int THREADS = 256;
        size_t shmem = (size_t)C * sizeof(float) + (size_t)(2 * ((THREADS + 31) / 32)) * sizeof(float);
        fused_softmax_bias_scale_sigmoid_cached_logits_kernel<THREADS><<<blocks, THREADS, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias_c.data_ptr<float>(),
            (float)scale,
            N, C, H, W
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(torch::Tensor x,
                                                   torch::Tensor bias_c,
                                                   double scale);
"""

custom_ops_lib = load_inline(
    name="custom_conv_transpose2d_softmax_bias_add_scaling_sigmoid_ops_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_softmax_bias_scale_sigmoid_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keep ConvTranspose2d; fuse:
        softmax(dim=1) -> +bias (broadcast per-channel) -> *scaling_factor -> sigmoid
    into one CUDA op for float32 contiguous NCHW.

    Bias semantics: original expects broadcastable (C,1,1); we flatten to [C].
    Includes CPU/eager fallback.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        bias_c = self.bias
        if bias_c.dtype != torch.float32:
            bias_c = bias_c.float()
        bias_c = bias_c.contiguous().view(-1)  # [C]

        if not x.is_cuda:
            y = torch.softmax(x, dim=1)
            y = y + bias_c.view(1, -1, 1, 1)
            y = y * self.scaling_factor
            y = torch.sigmoid(y)
            return y

        if (not bias_c.is_cuda) or (bias_c.device != x.device):
            bias_c = bias_c.to(device=x.device)

        return self.custom_ops_lib.fused_softmax_bias_scale_sigmoid_cuda(
            x, bias_c, float(self.scaling_factor)
        )