import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
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

// ---------------- Warp helpers ----------------
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

__device__ __forceinline__ float fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f); // log2(e)
}

// Lighter depth-mean loop to reduce register pressure.
// (The previous ILP4 variant tended to keep more temporaries live.)
__device__ __forceinline__ float depth_mean_light_ldg(
    const float* __restrict__ xptr,
    int64_t base,
    int D,
    int64_t strideD
) {
    float sum = 0.0f;
    #pragma unroll 1
    for (int d = 0; d < D; ++d) {
        sum += LDG(xptr + base + (int64_t)d * strideD);
    }
    return sum * (1.0f / (float)D);
}

// ============================================================
// Hot path: C == 64
// Each warp processes TWO pixels (two hw positions) for the same n.
// Each lane handles 2 channels (c and c+32) for each pixel.
// This increases MLP (two independent depth reductions) and improves latency hiding.
// ============================================================
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void fused_mean_add_softmax_tanh_scale_warp64_2pix_f32(
    const float* __restrict__ x,       // [N,64,D,H,W] contiguous
    const float* __restrict__ bias_c,  // [64]
    float* __restrict__ out,           // [N,64,H,W] contiguous
    int N, int D, int H, int W,
    float scaling
) {
    constexpr int WARP = 32;
    int tid = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid >> 5;
    int warps_per_block = THREADS / WARP;

    int HW = H * W;
    int64_t strideD = (int64_t)H * (int64_t)W;

    int64_t total_pixels = (int64_t)N * (int64_t)HW;
    int64_t total_pairs = (total_pixels + 1) >> 1; // 2 pixels per warp-iteration

    int64_t warp_global = (int64_t)blockIdx.x * (int64_t)warps_per_block + (int64_t)warp_id;
    int64_t warp_stride = (int64_t)gridDim.x * (int64_t)warps_per_block;

    int c0 = lane;
    int c1 = lane + 32;
    float b0 = LDG(bias_c + c0);
    float b1 = LDG(bias_c + c1);

    for (int64_t pair = warp_global; pair < total_pairs; pair += warp_stride) {
        int64_t pix0 = pair << 1;
        int64_t pix1 = pix0 + 1;

        // ---- pixel 0 decode
        int n0 = (int)(pix0 / HW);
        int hw0 = (int)(pix0 - (int64_t)n0 * (int64_t)HW);
        int h0 = hw0 / W;
        int w0 = hw0 - h0 * W;

        int64_t base_n0 = (int64_t)n0 * (int64_t)64 * (int64_t)D * strideD;
        int64_t base_hw0 = (int64_t)h0 * (int64_t)W + (int64_t)w0;

        int64_t base0_c0 = base_n0 + ((int64_t)c0 * (int64_t)D) * strideD + base_hw0;
        int64_t base0_c1 = base_n0 + ((int64_t)c1 * (int64_t)D) * strideD + base_hw0;

        float logit0_p0 = depth_mean_light_ldg(x, base0_c0, D, strideD) + b0;
        float logit1_p0 = depth_mean_light_ldg(x, base0_c1, D, strideD) + b1;

        // ---- pixel 1 decode (guarded)
        float logit0_p1 = -INFINITY;
        float logit1_p1 = -INFINITY;
        int n1 = 0, h1 = 0, w1 = 0;
        if (pix1 < total_pixels) {
            n1 = (int)(pix1 / HW);
            int hw1 = (int)(pix1 - (int64_t)n1 * (int64_t)HW);
            h1 = hw1 / W;
            w1 = hw1 - h1 * W;

            int64_t base_n1 = (int64_t)n1 * (int64_t)64 * (int64_t)D * strideD;
            int64_t base_hw1 = (int64_t)h1 * (int64_t)W + (int64_t)w1;

            int64_t base1_c0 = base_n1 + ((int64_t)c0 * (int64_t)D) * strideD + base_hw1;
            int64_t base1_c1 = base_n1 + ((int64_t)c1 * (int64_t)D) * strideD + base_hw1;

            logit0_p1 = depth_mean_light_ldg(x, base1_c0, D, strideD) + b0;
            logit1_p1 = depth_mean_light_ldg(x, base1_c1, D, strideD) + b1;
        }

        // ---- softmax pixel 0
        float local_max0 = fmaxf(logit0_p0, logit1_p0);
        float max32_0 = warp_reduce_max(local_max0);
        float maxv0 = __shfl_sync(0xffffffffu, max32_0, 0);

        float e0_0 = fast_exp(logit0_p0 - maxv0);
        float e1_0 = fast_exp(logit1_p0 - maxv0);
        float local_sum0 = e0_0 + e1_0;
        float sum32_0 = warp_reduce_sum(local_sum0);
        float sumv0 = __shfl_sync(0xffffffffu, sum32_0, 0);
        float invsum0 = 1.0f / fmaxf(sumv0, 1e-20f);

        float p0_0 = e0_0 * invsum0;
        float p1_0 = e1_0 * invsum0;

        float y0_0 = tanhf(p0_0) * scaling;
        float y1_0 = tanhf(p1_0) * scaling;

        // ---- softmax pixel 1 (if valid)
        float y0_1 = 0.0f, y1_1 = 0.0f;
        if (pix1 < total_pixels) {
            float local_max1 = fmaxf(logit0_p1, logit1_p1);
            float max32_1 = warp_reduce_max(local_max1);
            float maxv1 = __shfl_sync(0xffffffffu, max32_1, 0);

            float e0_1 = fast_exp(logit0_p1 - maxv1);
            float e1_1 = fast_exp(logit1_p1 - maxv1);
            float local_sum1 = e0_1 + e1_1;
            float sum32_1 = warp_reduce_sum(local_sum1);
            float sumv1 = __shfl_sync(0xffffffffu, sum32_1, 0);
            float invsum1 = 1.0f / fmaxf(sumv1, 1e-20f);

            float p0_1 = e0_1 * invsum1;
            float p1_1 = e1_1 * invsum1;

            y0_1 = tanhf(p0_1) * scaling;
            y1_1 = tanhf(p1_1) * scaling;
        }

        // Stores: out is [N,64,H,W] contiguous (NCHW)
        int64_t HW64 = (int64_t)H * (int64_t)W;

        int64_t out_base0 = ((int64_t)n0 * (int64_t)64 * HW64) + (int64_t)h0 * (int64_t)W + (int64_t)w0;
        out[out_base0 + (int64_t)c0 * HW64] = y0_0;
        out[out_base0 + (int64_t)c1 * HW64] = y1_0;

        if (pix1 < total_pixels) {
            int64_t out_base1 = ((int64_t)n1 * (int64_t)64 * HW64) + (int64_t)h1 * (int64_t)W + (int64_t)w1;
            out[out_base1 + (int64_t)c0 * HW64] = y0_1;
            out[out_base1 + (int64_t)c1 * HW64] = y1_1;
        }
    }
}

// ============================================================
// General fallback (cached logits in shared memory, block per pixel).
// Uses depth_mean_light_ldg to reduce register pressure.
// ============================================================
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void fused_mean_add_softmax_tanh_scale_cached_logits_f32(
    const float* __restrict__ x,       // [N,C,D,H,W]
    const float* __restrict__ bias_c,  // [C]
    float* __restrict__ out,           // [N,C,H,W] contiguous
    int N, int C, int D, int H, int W,
    float scaling
) {
    int nhw = (int)blockIdx.x;
    int HW = H * W;
    int n = nhw / HW;
    int hw = nhw - n * HW;
    if (n >= N) return;
    int h = hw / W;
    int w = hw - h * W;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int nwarps = (THREADS + 31) >> 5;
    unsigned mask = 0xffffffffu;

    extern __shared__ float shmem[];
    float* sh_logits = shmem;           // [C]
    float* sh_warp = sh_logits + C;     // [2*nwarps] floats

    int64_t strideD = (int64_t)H * (int64_t)W;
    int64_t base_n = (int64_t)n * (int64_t)C * (int64_t)D * strideD;

    for (int c = tid; c < C; c += THREADS) {
        int64_t base = base_n + ((int64_t)c * (int64_t)D) * strideD + (int64_t)h * (int64_t)W + (int64_t)w;
        float mean = depth_mean_light_ldg(x, base, D, strideD);
        float b = LDG(bias_c + c);
        sh_logits[c] = mean + b;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int c = tid; c < C; c += THREADS) local_max = fmaxf(local_max, sh_logits[c]);
    float wmax = warp_reduce_max(local_max, mask);
    if (lane == 0) sh_warp[wid] = wmax;
    __syncthreads();

    float maxv = -INFINITY;
    if (wid == 0) {
        float v = (lane < nwarps) ? sh_warp[lane] : -INFINITY;
        float r = warp_reduce_max(v, mask);
        if (lane == 0) sh_warp[0] = r;
    }
    __syncthreads();
    maxv = sh_warp[0];

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += THREADS) local_sum += fast_exp(sh_logits[c] - maxv);
    float wsum = warp_reduce_sum(local_sum, mask);
    if (lane == 0) sh_warp[nwarps + wid] = wsum;
    __syncthreads();

    float sumv = 0.0f;
    if (wid == 0) {
        float v = (lane < nwarps) ? sh_warp[nwarps + lane] : 0.0f;
        float r = warp_reduce_sum(v, mask);
        if (lane == 0) sh_warp[1] = r;
    }
    __syncthreads();
    sumv = sh_warp[1];
    float invsum = 1.0f / fmaxf(sumv, 1e-20f);

    for (int c = tid; c < C; c += THREADS) {
        float p = fast_exp(sh_logits[c] - maxv) * invsum;
        float y = tanhf(p) * scaling;
        int64_t out_idx = (((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)H + (int64_t)h) * (int64_t)W + (int64_t)w;
        out[out_idx] = y;
    }
}

torch::Tensor fused_mean_add_softmax_tanh_scale_cuda(
    torch::Tensor x,      // [N,C,D,H,W] float32 contiguous cuda
    torch::Tensor bias,   // [1,C,1,1,1] float32 contiguous cuda
    double scaling
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCDHW)");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(bias.dim() == 5, "bias must be 5D [1,C,1,1,1]");

    int64_t N64 = x.size(0), C64 = x.size(1), D64 = x.size(2), H64 = x.size(3), W64 = x.size(4);
    TORCH_CHECK(bias.size(0) == 1, "bias must have size(0)=1");
    TORCH_CHECK(bias.size(1) == C64, "bias C mismatch");
    TORCH_CHECK(bias.size(2) == 1 && bias.size(3) == 1 && bias.size(4) == 1, "bias must be [1,C,1,1,1]");

    int N = (int)N64, C = (int)C64, D = (int)D64, H = (int)H64, W = (int)W64;
    auto bias_c = bias.view({C}); // [C]

    auto out = torch::empty({N, C, 1, H, W}, x.options());
    auto out_flat = out.view({N, C, H, W});

    if (C == 64) {
        constexpr int THREADS = 128; // 4 warps/block, better reg-limited occupancy than 256 on many GPUs
        int warps_per_block = THREADS / 32;
        int64_t total_pixels = (int64_t)N * (int64_t)H * (int64_t)W;
        int64_t total_pairs = (total_pixels + 1) >> 1;

        int blocks = (int)((total_pairs + warps_per_block - 1) / warps_per_block);
        if (blocks > 65535) blocks = 65535;

        fused_mean_add_softmax_tanh_scale_warp64_2pix_f32<THREADS><<<blocks, THREADS, 0>>>(
            x.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            out_flat.data_ptr<float>(),
            N, D, H, W,
            (float)scaling
        );
        return out;
    }

    int64_t blocks64 = (int64_t)N * (int64_t)H * (int64_t)W;
    TORCH_CHECK(blocks64 > 0, "Invalid launch size");
    int blocks = (blocks64 > (int64_t)INT_MAX) ? INT_MAX : (int)blocks64;

    if (C <= 128) {
        constexpr int THREADS = 128;
        int nwarps = (THREADS + 31) >> 5;
        size_t shmem = (size_t)C * sizeof(float) + (size_t)(2 * nwarps) * sizeof(float);
        fused_mean_add_softmax_tanh_scale_cached_logits_f32<THREADS><<<blocks, THREADS, shmem>>>(
            x.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            out_flat.data_ptr<float>(),
            N, C, D, H, W,
            (float)scaling
        );
    } else {
        constexpr int THREADS = 256;
        int nwarps = (THREADS + 31) >> 5;
        size_t shmem = (size_t)C * sizeof(float) + (size_t)(2 * nwarps) * sizeof(float);
        fused_mean_add_softmax_tanh_scale_cached_logits_f32<THREADS><<<blocks, THREADS, shmem>>>(
            x.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            out_flat.data_ptr<float>(),
            N, C, D, H, W,
            (float)scaling
        );
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_mean_add_softmax_tanh_scale_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double scaling
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_mean_add_softmax_tanh_scaling_v5",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_mean_add_softmax_tanh_scale_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps ConvTranspose3d as-is, then fuses:
      mean(depth, keepdim=True) + add(self.bias) + softmax(dim=1) + tanh + scaling
    into a custom CUDA kernel for CUDA float32 contiguous tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            stride=int(stride),
            padding=int(padding),
        )
        self.bias = nn.Parameter(torch.randn(1, int(out_channels), 1, 1, 1))
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            y = x.mean(dim=2, keepdim=True)
            y = y + self.bias
            y = torch.softmax(y, dim=1)
            y = torch.tanh(y)
            y = y * self.scaling_factor
            return y

        if not x.is_contiguous():
            x = x.contiguous()

        bias = self.bias
        if (not bias.is_cuda) or (bias.device != x.device):
            bias = bias.to(device=x.device)
        if bias.dtype != torch.float32:
            bias = bias.float()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        if bias.dim() != 5 or bias.size(0) != 1 or bias.size(2) != 1 or bias.size(3) != 1 or bias.size(4) != 1:
            y = x.mean(dim=2, keepdim=True)
            y = y + self.bias
            y = torch.softmax(y, dim=1)
            y = torch.tanh(y)
            y = y * self.scaling_factor
            return y
        if bias.size(1) != x.size(1):
            y = x.mean(dim=2, keepdim=True)
            y = y + self.bias
            y = torch.softmax(y, dim=1)
            y = torch.tanh(y)
            y = y * self.scaling_factor
            return y

        return self.custom_ops_lib.fused_mean_add_softmax_tanh_scale_cuda(x, bias, self.scaling_factor)