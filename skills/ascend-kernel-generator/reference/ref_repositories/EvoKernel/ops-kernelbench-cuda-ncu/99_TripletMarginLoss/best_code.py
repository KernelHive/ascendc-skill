import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------
# TripletMarginLoss optimized CUDA (p=2, swap=False, reduction='mean')
# loss = mean_i max(||a-p||_2 - ||a-n||_2 + margin, 0)
#
# Key optimizations vs current baseline:
# - CTA-per-sample (256 threads) to increase MLP and hide DRAM latency
# - Vectorization ladder: float4 (16B aligned) else float2 (8B aligned) else scalar
# - Prefetch + dual accumulators to increase ILP with bounded registers
# - Two-stage reduction: stage1 -> partial sums, stage2 -> single-block mean (no atomics)
# -------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float clamp_min0(float x) { return x > 0.0f ? x : 0.0f; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    static_assert(THREADS % 32 == 0, "THREADS must be multiple of warp size");
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int WARPS = THREADS / 32;

    v = warp_reduce_sum(v);

    __shared__ float warp_sums[WARPS];
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        float x = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) out = x;
    }
    return out; // valid for all threads after sync? only warp0 lane0 sets; callers should use thread0
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void triplet_margin_loss_stage1_cta_f32(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ partial_sums, // [gridDim.x]
    int B,
    int D,
    float margin,
    float eps,
    int vec_mode // 4=float4, 2=float2, 1=scalar
) {
    int sample = (int)blockIdx.x;
    if (sample >= B) return;

    const float* a = anchor   + (int64_t)sample * D;
    const float* p = positive + (int64_t)sample * D;
    const float* n = negative + (int64_t)sample * D;

    float ap0 = 0.0f, ap1 = 0.0f;
    float an0 = 0.0f, an1 = 0.0f;

    if (vec_mode == 4) {
        int D4 = D >> 2;
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* p4 = reinterpret_cast<const float4*>(p);
        const float4* n4 = reinterpret_cast<const float4*>(n);

        // Each thread processes indices t = tid, tid+THREADS, ...
        // Prefetch next iteration to increase ILP slightly.
        for (int t = (int)threadIdx.x; t < D4; t += THREADS) {
            float4 av = a4[t];
            float4 pv = p4[t];
            float4 nv = n4[t];

            float dx0 = av.x - pv.x; ap0 = fmaf(dx0, dx0, ap0);
            float dy0 = av.x - nv.x; an0 = fmaf(dy0, dy0, an0);

            float dx1 = av.y - pv.y; ap1 = fmaf(dx1, dx1, ap1);
            float dy1 = av.y - nv.y; an1 = fmaf(dy1, dy1, an1);

            float dx2 = av.z - pv.z; ap0 = fmaf(dx2, dx2, ap0);
            float dy2 = av.z - nv.z; an0 = fmaf(dy2, dy2, an0);

            float dx3 = av.w - pv.w; ap1 = fmaf(dx3, dx3, ap1);
            float dy3 = av.w - nv.w; an1 = fmaf(dy3, dy3, an1);
        }
    } else if (vec_mode == 2) {
        int D2 = D >> 1;
        const float2* a2 = reinterpret_cast<const float2*>(a);
        const float2* p2 = reinterpret_cast<const float2*>(p);
        const float2* n2 = reinterpret_cast<const float2*>(n);

        for (int t = (int)threadIdx.x; t < D2; t += THREADS) {
            float2 av = a2[t];
            float2 pv = p2[t];
            float2 nv = n2[t];

            float dx0 = av.x - pv.x; ap0 = fmaf(dx0, dx0, ap0);
            float dy0 = av.x - nv.x; an0 = fmaf(dy0, dy0, an0);

            float dx1 = av.y - pv.y; ap1 = fmaf(dx1, dx1, ap1);
            float dy1 = av.y - nv.y; an1 = fmaf(dy1, dy1, an1);
        }
    } else {
        for (int t = (int)threadIdx.x; t < D; t += THREADS) {
            float av = a[t];
            float pv = p[t];
            float nv = n[t];
            float da_p = av - pv;
            float da_n = av - nv;
            ap0 = fmaf(da_p, da_p, ap0);
            an0 = fmaf(da_n, da_n, an0);
        }
    }

    float dap2 = ap0 + ap1;
    float dan2 = an0 + an1;

    // Reduce across block
    float dap2_blk = block_reduce_sum<THREADS>(dap2);
    __syncthreads();
    float dan2_blk = block_reduce_sum<THREADS>(dan2);
    __syncthreads();

    if (threadIdx.x == 0) {
        float dap = sqrtf(dap2_blk + eps);
        float dan = sqrtf(dan2_blk + eps);
        float l = clamp_min0(dap - dan + margin);
        partial_sums[sample] = l;
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void reduce_losses_to_mean_single_block(
    const float* __restrict__ losses,
    float* __restrict__ out_mean,
    int B
) {
    // Single-block reduction (assumes gridDim.x == 1)
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < B; i += THREADS) sum += losses[i];

    sum = block_reduce_sum<THREADS>(sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        *out_mean = (B > 0) ? (sum / (float)B) : 0.0f;
    }
}

torch::Tensor triplet_margin_loss_mean_cuda(torch::Tensor anchor,
                                            torch::Tensor positive,
                                            torch::Tensor negative,
                                            double margin,
                                            double eps) {
    TORCH_CHECK(anchor.is_cuda() && positive.is_cuda() && negative.is_cuda(),
                "triplet_margin_loss_mean_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(anchor.scalar_type() == torch::kFloat32 &&
                positive.scalar_type() == torch::kFloat32 &&
                negative.scalar_type() == torch::kFloat32,
                "triplet_margin_loss_mean_cuda: only float32 is supported");
    TORCH_CHECK(anchor.is_contiguous() && positive.is_contiguous() && negative.is_contiguous(),
                "triplet_margin_loss_mean_cuda: inputs must be contiguous");
    TORCH_CHECK(anchor.dim() == 2 && positive.dim() == 2 && negative.dim() == 2,
                "triplet_margin_loss_mean_cuda: expected 2D tensors [B, D]");
    TORCH_CHECK(anchor.sizes() == positive.sizes() && anchor.sizes() == negative.sizes(),
                "triplet_margin_loss_mean_cuda: input shapes must match");

    int64_t B64 = anchor.size(0);
    int64_t D64 = anchor.size(1);
    TORCH_CHECK(B64 <= INT32_MAX && D64 <= INT32_MAX, "B and D must fit in int32");
    const int B = (int)B64;
    const int D = (int)D64;

    auto out = torch::empty({}, anchor.options()); // scalar
    if (B == 0) {
        out.zero_();
        return out;
    }

    // Decide vectorization mode based on alignment and D divisibility
    uintptr_t pa = (uintptr_t)anchor.data_ptr<float>();
    uintptr_t pp = (uintptr_t)positive.data_ptr<float>();
    uintptr_t pn = (uintptr_t)negative.data_ptr<float>();

    int vec_mode = 1;
    if ((D % 4) == 0 && (pa % 16) == 0 && (pp % 16) == 0 && (pn % 16) == 0) vec_mode = 4;
    else if ((D % 2) == 0 && (pa % 8) == 0 && (pp % 8) == 0 && (pn % 8) == 0) vec_mode = 2;

    // Stage1: one block per sample, but cap by a reasonable max to avoid absurd grid sizes
    // For this workload B is large (32768), this is fine.
    constexpr int THREADS1 = 256;

    // In case B is huge on other uses, we could cap blocks and grid-stride samples, but
    // current task shape benefits from 1 block per sample.
    int blocks1 = B;

    auto losses = torch::empty({B}, anchor.options());

    triplet_margin_loss_stage1_cta_f32<THREADS1><<<blocks1, THREADS1>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        losses.data_ptr<float>(),
        B, D,
        (float)margin,
        (float)eps,
        vec_mode
    );

    // Stage2: single-block reduction to mean (no atomics)
    constexpr int THREADS2 = 256;
    reduce_losses_to_mean_single_block<THREADS2><<<1, THREADS2>>>(
        losses.data_ptr<float>(),
        out.data_ptr<float>(),
        B
    );

    return out;
}
"""

cpp_source = r"""
torch::Tensor triplet_margin_loss_mean_cuda(torch::Tensor anchor,
                                            torch::Tensor positive,
                                            torch::Tensor negative,
                                            double margin,
                                            double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_triplet_margin_loss_opt6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["triplet_margin_loss_mean_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    TripletMarginLoss replacement using an optimized custom CUDA kernel for:
      - p=2, swap=False, reduction='mean'
    Falls back to PyTorch for other cases.
    """
    def __init__(self, margin=1.0, eps=1e-6, reduction="mean", p=2.0, swap=False):
        super().__init__()
        self.margin = float(margin)
        self.eps = float(eps)
        self.reduction = reduction
        self.p = float(p)
        self.swap = bool(swap)
        self.custom_ops_lib = custom_ops_lib
        self._fallback = torch.nn.TripletMarginLoss(
            margin=self.margin, p=self.p, eps=self.eps, swap=self.swap, reduction=self.reduction
        )

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if (
            anchor.is_cuda
            and positive.is_cuda
            and negative.is_cuda
            and anchor.dtype == torch.float32
            and positive.dtype == torch.float32
            and negative.dtype == torch.float32
            and anchor.dim() == 2
            and positive.dim() == 2
            and negative.dim() == 2
            and self.reduction == "mean"
            and self.p == 2.0
            and (not self.swap)
        ):
            if not anchor.is_contiguous():
                anchor = anchor.contiguous()
            if not positive.is_contiguous():
                positive = positive.contiguous()
            if not negative.is_contiguous():
                negative = negative.contiguous()
            return self.custom_ops_lib.triplet_margin_loss_mean_cuda(anchor, positive, negative, self.margin, self.eps)

        return self._fallback(anchor, positive, negative)