import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized SE attention CUDA (incremental improvement vs baseline):
# - Replace large fused 256-thread kernel with:
#   (1) avgpool + FC1 + ReLU kernel: one block per batch, 256 threads, FP32 accum,
#       FP16 weights (w1_half) loaded as half2 to reduce instruction count/bandwidth.
#   (2) FC2 + sigmoid + apply kernel: one warp per (b,c), 32 threads, no __syncthreads,
#       FP16 weights (w2_half) half2 vector loads, vectorized float4 stores for x*out.
# - This targets register pressure / occupancy-limited bottleneck and barrier stalls.
# - Fast path specialized for C=512, Cr=32, HW=49 (7x7), float32 contiguous.
# - Fallback: uses PyTorch ops on GPU (safe, correct) for non-fast shapes.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float relu_fast(float x) { return x > 0.0f ? x : 0.0f; }

__device__ __forceinline__ float sigmoid_fast(float x) {
#if __CUDA_ARCH__ >= 200
    return 1.0f / (1.0f + __expf(-x));
#else
    return 1.0f / (1.0f + expf(-x));
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v,  8);
    v += __shfl_down_sync(mask, v,  4);
    v += __shfl_down_sync(mask, v,  2);
    v += __shfl_down_sync(mask, v,  1);
    return v;
}

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ half ldg_h(const half* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// (1) avgpool + FC1 + ReLU
// Specialization: C=512, Cr=32, HW=49
// One block per batch element, 256 threads.
// Writes hidden[b,32] in FP32.
__global__ __launch_bounds__(256, 2)
void se_avgpool_fc1_relu_512_32_49(
    const float* __restrict__ x,     // (B,512,49)
    const half*  __restrict__ w1h,   // (32,512) FP16
    float* __restrict__ hidden,      // (B,32) FP32
    int B
) {
    constexpr int C = 512;
    constexpr int Cr = 32;
    constexpr int HW = 49;

    int b = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    // Compute y[c] into shared (FP32). Keep shared small: 512 floats = 2KB.
    __shared__ float y_s[C];

    for (int c = tid; c < C; c += 256) {
        const float* xptr = x + ((int64_t)b * C + c) * (int64_t)HW;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < HW; ++i) sum += ldg_f(xptr + i);
        y_s[c] = sum * (1.0f / 49.0f);
    }
    __syncthreads();

    // FC1: each warp computes 4 cr outputs (8 warps * 4 = 32).
    int lane = tid & 31;
    int warp = tid >> 5; // 0..7
    int cr_base = warp * 4;
    if (cr_base < Cr) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int cr = cr_base + j;
            const half* wrow = w1h + (int64_t)cr * C;
            float acc = 0.0f;

            // Use half2 vectorization: 512 half = 256 half2
            // Each lane handles stride 32 half2 -> covers 256 half2 across warp.
            for (int i2 = lane; i2 < (C / 2); i2 += 32) {
                // load half2
                const half2* w2 = reinterpret_cast<const half2*>(wrow);
                half2 wv2 = __ldg(w2 + i2);
                // y in float, pack two y values:
                float y0 = y_s[2 * i2 + 0];
                float y1 = y_s[2 * i2 + 1];
                // convert weights
                float2 wf = __half22float2(wv2);
                acc += wf.x * y0 + wf.y * y1;
            }
            acc = warp_sum(acc);
            if (lane == 0) hidden[(int64_t)b * Cr + cr] = relu_fast(acc);
        }
    }
}

// (2) FC2 + sigmoid + apply
// One warp per (b,c). blockDim=32, grid=(B,C).
// Specialization: C=512, Cr=32, HW=49.
// w2 in FP16, hidden in FP32, x/out FP32.
__global__ __launch_bounds__(32, 8)
void se_fc2_sigmoid_apply_512_32_49_warp(
    const float* __restrict__ x,     // (B,512,49)
    const float* __restrict__ hidden,// (B,32)
    const half*  __restrict__ w2h,   // (512,32) FP16
    float* __restrict__ out,         // (B,512,49)
    int B
) {
    constexpr int C = 512;
    constexpr int Cr = 32;
    constexpr int HW = 49;

    int b = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    int lane = (int)threadIdx.x & 31;

    const float* hrow = hidden + (int64_t)b * Cr;
    const half*  wrow = w2h + (int64_t)c * Cr;

    // Dot(hidden, w2[c,:]) using half2 vectorization (Cr=32 -> 16 half2)
    float acc = 0.0f;
    const half2* w2p = reinterpret_cast<const half2*>(wrow);

    // Each lane handles at most one half2 (since 16 < 32):
    if (lane < 16) {
        half2 wv2 = __ldg(w2p + lane);
        float2 wf = __half22float2(wv2);
        float h0 = hrow[2 * lane + 0];
        float h1 = hrow[2 * lane + 1];
        acc = wf.x * h0 + wf.y * h1;
    }

    acc = warp_sum(acc);
    float gate = __shfl_sync(0xffffffffu, acc, 0);
    gate = sigmoid_fast(gate);

    const int64_t base = ((int64_t)b * C + c) * (int64_t)HW;
    const float* xptr = x + base;
    float* optr = out + base;

    // Apply gate; use float4 when aligned: 12 float4 + tail 1
    bool aligned16 = (((uintptr_t)xptr & 0xF) == 0) && (((uintptr_t)optr & 0xF) == 0);
    if (aligned16) {
        const float4* x4 = reinterpret_cast<const float4*>(xptr);
        float4* o4 = reinterpret_cast<float4*>(optr);
        // HW=49 => 12 float4 (=48 floats) + 1 tail
        for (int i4 = lane; i4 < 12; i4 += 32) {
            float4 v = x4[i4];
            v.x *= gate; v.y *= gate; v.z *= gate; v.w *= gate;
            o4[i4] = v;
        }
        if (lane == 0) optr[48] = xptr[48] * gate;
    } else {
        if (lane < 49) optr[lane] = xptr[lane] * gate;
        if (lane + 32 < 49) optr[lane + 32] = xptr[lane + 32] * gate;
    }
}

torch::Tensor se_attention_fast_cuda(
    torch::Tensor x,   // (B,512,7,7) float32 contiguous
    torch::Tensor w1h, // (32,512) float16 contiguous
    torch::Tensor w2h  // (512,32) float16 contiguous
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w1h.is_cuda() && w2h.is_cuda(), "w1h/w2h must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w1h.scalar_type() == torch::kFloat16, "w1h must be float16");
    TORCH_CHECK(w2h.scalar_type() == torch::kFloat16, "w2h must be float16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w1h.is_contiguous() && w2h.is_contiguous(), "w1h/w2h must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be (B,C,H,W)");
    TORCH_CHECK(w1h.dim() == 2 && w2h.dim() == 2, "w1h/w2h must be 2D");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    TORCH_CHECK(C == 512 && H * W == 49, "fast path requires C=512, HW=49");
    TORCH_CHECK(w1h.size(0) == 32 && w1h.size(1) == 512, "w1h must be (32,512)");
    TORCH_CHECK(w2h.size(0) == 512 && w2h.size(1) == 32, "w2h must be (512,32)");

    auto out = torch::empty_like(x);
    auto hidden = torch::empty({B, 32}, x.options());

    // Kernel (1)
    {
        dim3 block(256, 1, 1);
        dim3 grid(B, 1, 1);
        se_avgpool_fc1_relu_512_32_49<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const half*)w1h.data_ptr<at::Half>(),
            (float*)hidden.data_ptr<float>(),
            B
        );
    }

    // Kernel (2)
    {
        dim3 block(32, 1, 1);
        dim3 grid(B, 512, 1);
        se_fc2_sigmoid_apply_512_32_49_warp<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)hidden.data_ptr<float>(),
            (const half*)w2h.data_ptr<at::Half>(),
            (float*)out.data_ptr<float>(),
            B
        );
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor se_attention_fast_cuda(torch::Tensor x, torch::Tensor w1h, torch::Tensor w2h);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_se_attention_opt_warp_halfw",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["se_attention_fast_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """SE Attention optimized CUDA fast-path for (C=512, reduction=16, H=W=7)."""
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        cr = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, cr, bias=False)
        self.fc2 = nn.Linear(cr, channel, bias=False)
        self.custom_ops_lib = custom_ops_lib

        # Cached half weights for fast path (updated on-demand)
        self.register_buffer("_w1h_cache", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_w2h_cache", torch.empty(0, dtype=torch.float16), persistent=False)
        self._cache_valid = False

    def _update_half_weight_cache(self):
        # Recreate half caches (contiguous) if needed.
        w1h = self.fc1.weight.detach().to(dtype=torch.float16).contiguous()
        w2h = self.fc2.weight.detach().to(dtype=torch.float16).contiguous()
        self._w1h_cache = w1h
        self._w2h_cache = w2h
        self._cache_valid = True

    def forward(self, x):
        # CPU fallback
        if not x.is_cuda:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc2(torch.relu(self.fc1(y))).sigmoid().view(b, c, 1, 1)
            return x * y.expand_as(x)

        b, c, h, w = x.size()

        # Fast path: only for the target configuration, float32 contiguous
        if c == 512 and (h * w) == 49:
            if x.dtype != torch.float32:
                x_fp32 = x.float()
            else:
                x_fp32 = x
            x_fp32 = x_fp32.contiguous()

            # Update cached FP16 weights if missing / wrong device / training updates likely.
            # In training, weights change every step, so refresh every forward (safe).
            # In eval, cache remains valid unless user modifies weights.
            if self.training or (not self._cache_valid) or (self._w1h_cache.device != x.device):
                self._update_half_weight_cache()
                self._w1h_cache = self._w1h_cache.to(device=x.device, non_blocking=True)
                self._w2h_cache = self._w2h_cache.to(device=x.device, non_blocking=True)

            return self.custom_ops_lib.se_attention_fast_cuda(x_fp32, self._w1h_cache, self._w2h_cache)

        # General GPU fallback (correctness over speed)
        # Keep everything in float32 for numerical parity with baseline custom kernel behavior.
        if x.dtype != torch.float32:
            x = x.float()
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc2(torch.relu(self.fc1(y))).sigmoid().view(b, c, 1, 1)
        return x * y