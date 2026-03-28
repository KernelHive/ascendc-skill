import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FP32
#define CHECK_FP32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float smem[32]; // up to 1024 threads => 32 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (wid == 0) {
        out = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.f;
        out = warp_reduce_sum(out);
    }
    return out;
}

static __forceinline__ __host__ __device__ bool is_aligned_16_hostdev(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// Stage A: per-channel reduction via partial CTAs + atomic add into sum/sumsq.
// grid: (C, partials)
__global__ __launch_bounds__(256, 2) void nhw_sum_sumsq_atomic_kernel_vec4(
    const float* __restrict__ X,   // [N,C,H,W]
    float* __restrict__ sum,       // [C]
    float* __restrict__ sumsq,     // [C]
    int N, int C, int H, int W,
    int partials
) {
    int c = (int)blockIdx.x;
    int pid = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    int HW = H * W;
    int NHW = N * HW;

    int s0 = (int)((long long)pid * NHW / partials);
    int s1 = (int)((long long)(pid + 1) * NHW / partials);

    const float* __restrict__ Xc = X + (int64_t)c * (int64_t)HW; // base for n=0,c

    float lsum = 0.f;
    float lss  = 0.f;

    // Vectorize over the flattened NHW dimension.
    // For NCHW, addresses are: Xc[s + n*(C-1)*HW] effectively strideN = C*HW.
    // But using flattened s => we must map to idx = (n*C + c)*HW + hw.
    // To avoid div/mod on every element, we still use s->(n,hw) mapping, but vectorize hw within a fixed n.
    // Better approach: iterate by n and within each n do contiguous HW (vectorized). However partial slicing is on s,
    // so we keep s loop and only vectorize when s spans full contiguous HW segments. To keep it simple and safe,
    // we use scalar for general case and a fast path when the slice covers full NHW and is aligned; partials typically small.
    //
    // Instead, we do pointer-increment scalar with reduced math: compute n, hw using fast div/mod once per iteration group.
    // The big win will come from Stage B vectorization; Stage A gets modest improvement.
    for (int s = s0 + tid; s < s1; s += (int)blockDim.x) {
        int n  = s / HW;
        int hw = s - n * HW;
        const float* ptr = Xc + (int64_t)n * (int64_t)C * (int64_t)HW + (int64_t)hw;
        float v = *ptr;
        lsum += v;
        lss = fmaf(v, v, lss);
    }

    lsum = block_reduce_sum(lsum);
    lss  = block_reduce_sum(lss);

    if (tid == 0) {
        atomicAdd(sum + c, lsum);
        atomicAdd(sumsq + c, lss);
    }
}

// Stage B: compute mean/var and apply BN+affine+scaling.
// grid: (C, tiles_over_NHW)
__global__ __launch_bounds__(256, 2) void apply_bn_scale_train_2d_kernel_vec4(
    const float* __restrict__ X,     // [N,C,H,W]
    const float* __restrict__ sum,   // [C]
    const float* __restrict__ sumsq, // [C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ Y,           // [N,C,H,W]
    int N, int C, int H, int W,
    float eps,
    float scaling_factor
) {
    int c = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    int HW = H * W;
    int NHW = N * HW;
    float invNHW = 1.0f / (float)NHW;

    float s  = __ldg(sum + c);
    float ss = __ldg(sumsq + c);

    float mean = s * invNHW;
    float ex2  = ss * invNHW;
    float var  = ex2 - mean * mean;
    if (var < 0.f) var = 0.f;

    float g  = __ldg(gamma + c);
    float b0 = __ldg(beta + c);

    float invstd = rsqrtf(var + eps);
    float a = g * invstd * scaling_factor;
    float b = (b0 - mean * g * invstd) * scaling_factor;

    int tile = (int)blockIdx.y;
    int base = tile * (int)blockDim.x + tid;
    int stride = (int)blockDim.x * (int)gridDim.y;

    // Vectorization condition for the whole tensor pointer (strong enough for contiguous allocations).
    bool vec_ok = ((NHW & 3) == 0) && is_aligned_16_hostdev(X) && is_aligned_16_hostdev(Y);

    if (vec_ok) {
        // Work in units of float4 over the flattened NHW dimension.
        int NHW4 = NHW >> 2;
        int base4 = base;      // base is in "elements"; for float4, treat it as index into float4 by dividing by 4
        int stride4 = stride;  // same
        // To avoid expensive /4 per iteration, shift indices:
        base4 >>= 2;
        stride4 >>= 2;

        const float4* __restrict__ X4 = reinterpret_cast<const float4*>(X);
        float4* __restrict__ Y4 = reinterpret_cast<float4*>(Y);

        // Each float4 corresponds to 4 consecutive elements in memory in NCHW linear layout.
        // However, across channels, memory is interleaved every HW. Our grid is per-channel,
        // so we must only touch elements belonging to channel c.
        // Therefore, we cannot simply flatten X as float4 and stride over NHW;
        // we still need per-(n,hw) mapping. Vectorizing here safely is only possible over hw
        // within fixed (n,c). So we use an inner loop over n and vectorize hw.
        //
        // Fallback to efficient scalar pointer math below (still good).
    }

    // Efficient scalar traversal with reduced integer ops: loop over n and hw blocks.
    // We cover NHW via sidx progression (as before) but compute base pointers per n.
    for (int sidx = base; sidx < NHW; sidx += stride) {
        int n  = sidx / HW;
        int hw = sidx - n * HW;
        int64_t idx = ((int64_t)n * C + c) * (int64_t)HW + hw;
        float x = X[idx];
        Y[idx] = fmaf(x, a, b);
    }
}

// Eval apply: uses running stats, with per-(n,c) plane vectorization over HW.
__global__ __launch_bounds__(256, 2) void apply_bn_scale_eval_2d_kernel_vec_hw4(
    const float* __restrict__ X,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ Y,
    int N, int C, int H, int W,
    float eps,
    float scaling_factor
) {
    int c = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    int HW = H * W;

    float mean = __ldg(running_mean + c);
    float var  = __ldg(running_var + c);
    if (var < 0.f) var = 0.f;

    float g  = __ldg(gamma + c);
    float b0 = __ldg(beta + c);

    float invstd = rsqrtf(var + eps);
    float a = g * invstd * scaling_factor;
    float b = (b0 - mean * g * invstd) * scaling_factor;

    // 2D tiling across n and hw:
    // blockIdx.y selects a chunk of HW within each n.
    int tiles = (int)gridDim.y;
    int tile = (int)blockIdx.y;

    // Split HW among tiles; keep contiguous in hw for coalescing.
    int hw0 = (int)((long long)tile * HW / tiles);
    int hw1 = (int)((long long)(tile + 1) * HW / tiles);
    int span = hw1 - hw0;

    int64_t HW64 = (int64_t)HW;
    int64_t strideN = (int64_t)C * HW64;

    for (int n = 0; n < N; ++n) {
        const float* x_nc = X + (int64_t)n * strideN + (int64_t)c * HW64 + (int64_t)hw0;
        float* y_nc       = Y + (int64_t)n * strideN + (int64_t)c * HW64 + (int64_t)hw0;

        bool vec_ok = ((span & 3) == 0) && is_aligned_16_hostdev(x_nc) && is_aligned_16_hostdev(y_nc);

        if (vec_ok) {
            const float4* x4 = reinterpret_cast<const float4*>(x_nc);
            float4* y4 = reinterpret_cast<float4*>(y_nc);
            int span4 = span >> 2;
            for (int i = tid; i < span4; i += (int)blockDim.x) {
                float4 v = x4[i];
                v.x = fmaf(v.x, a, b);
                v.y = fmaf(v.y, a, b);
                v.z = fmaf(v.z, a, b);
                v.w = fmaf(v.w, a, b);
                y4[i] = v;
            }
        } else {
            for (int i = tid; i < span; i += (int)blockDim.x) {
                float x = x_nc[i];
                y_nc[i] = fmaf(x, a, b);
            }
        }
    }
}

// Optional running stats update to match BatchNorm2d training semantics.
__global__ void update_running_stats_kernel(
    const float* __restrict__ sum,   // [C]
    const float* __restrict__ sumsq, // [C]
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int64_t* __restrict__ num_batches_tracked,
    int C, int NHW,
    float momentum
) {
    int c = (int)blockIdx.x;
    if (c >= C) return;

    float invNHW = 1.0f / (float)NHW;
    float s  = sum[c];
    float ss = sumsq[c];

    float mean = s * invNHW;
    float ex2  = ss * invNHW;
    float var  = ex2 - mean * mean;
    if (var < 0.f) var = 0.f;

    float unbiased = var;
    if (NHW > 1) unbiased = var * ((float)NHW / (float)(NHW - 1));

    float rm = running_mean[c];
    float rv = running_var[c];
    rm = rm + momentum * (mean - rm);
    rv = rv + momentum * (unbiased - rv);
    running_mean[c] = rm;
    running_var[c]  = rv;

    if (c == 0 && num_batches_tracked != nullptr) {
        atomicAdd((unsigned long long*)num_batches_tracked, 1ULL);
    }
}

torch::Tensor bn2d_scale_train_cuda(
    torch::Tensor x,       // [N,C,H,W]
    torch::Tensor gamma,   // [C]
    torch::Tensor beta,    // [C]
    torch::Tensor running_mean, // [C]
    torch::Tensor running_var,  // [C]
    torch::Tensor num_batches_tracked, // scalar int64
    double eps,
    double scaling_factor,
    double momentum,
    bool update_running
) {
    CHECK_CUDA(x); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_FP32(x); CHECK_FP32(gamma); CHECK_FP32(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta);

    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK((int)gamma.numel() == C, "gamma size mismatch");
    TORCH_CHECK((int)beta.numel() == C, "beta size mismatch");

    if (update_running) {
        CHECK_CUDA(running_mean); CHECK_CUDA(running_var); CHECK_CUDA(num_batches_tracked);
        CHECK_FP32(running_mean); CHECK_FP32(running_var);
        CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var);
        TORCH_CHECK(running_mean.numel() == C && running_var.numel() == C, "running stats size mismatch");
        TORCH_CHECK(num_batches_tracked.numel() == 1 && num_batches_tracked.dtype() == torch::kInt64, "num_batches_tracked must be int64 scalar");
        CHECK_CONTIGUOUS(num_batches_tracked);
    }

    at::cuda::CUDAGuard device_guard(x.device());

    auto y = torch::empty_like(x);
    auto sum   = torch::zeros({C}, x.options());
    auto sumsq = torch::zeros({C}, x.options());

    int NHW = N * H * W;

    // Retuned partials: slightly higher for better latency hiding at large NHW.
    int partials = 8;
    if (NHW >= (1 << 18)) partials = 16;
    if (NHW >= (1 << 20)) partials = 32;
    if (NHW >= (1 << 22)) partials = 48;
    if (NHW < 4096) partials = 4;
    if (NHW < 1024) partials = 2;
    if (partials > 64) partials = 64;
    if (partials < 1) partials = 1;

    dim3 gridA((unsigned)C, (unsigned)partials, 1);
    nhw_sum_sumsq_atomic_kernel_vec4<<<gridA, 256>>>(
        (const float*)x.data_ptr<float>(),
        (float*)sum.data_ptr<float>(),
        (float*)sumsq.data_ptr<float>(),
        N, C, H, W, partials
    );

    if (update_running) {
        update_running_stats_kernel<<<(unsigned)C, 1>>>(
            (const float*)sum.data_ptr<float>(),
            (const float*)sumsq.data_ptr<float>(),
            (float*)running_mean.data_ptr<float>(),
            (float*)running_var.data_ptr<float>(),
            (int64_t*)num_batches_tracked.data_ptr<int64_t>(),
            C, NHW, (float)momentum
        );
    }

    // Retuned tiles: scale up more aggressively with NHW to increase CTAs.
    int tiles = 8;
    if (NHW >= (1 << 15)) tiles = 16;
    if (NHW >= (1 << 17)) tiles = 32;
    if (NHW >= (1 << 19)) tiles = 48;
    if (NHW >= (1 << 21)) tiles = 64;
    if (tiles > 64) tiles = 64;

    dim3 gridB((unsigned)C, (unsigned)tiles, 1);
    apply_bn_scale_train_2d_kernel_vec4<<<gridB, 256>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)sum.data_ptr<float>(),
        (const float*)sumsq.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, C, H, W,
        (float)eps,
        (float)scaling_factor
    );

    return y;
}

torch::Tensor bn2d_scale_eval_cuda(
    torch::Tensor x,             // [N,C,H,W]
    torch::Tensor running_mean,  // [C]
    torch::Tensor running_var,   // [C]
    torch::Tensor gamma,         // [C]
    torch::Tensor beta,          // [C]
    double eps,
    double scaling_factor
) {
    CHECK_CUDA(x); CHECK_CUDA(running_mean); CHECK_CUDA(running_var); CHECK_CUDA(gamma); CHECK_CUDA(beta);
    CHECK_FP32(x); CHECK_FP32(running_mean); CHECK_FP32(running_var); CHECK_FP32(gamma); CHECK_FP32(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var); CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta);

    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    TORCH_CHECK((int)running_mean.numel() == C, "running_mean size mismatch");
    TORCH_CHECK((int)running_var.numel() == C, "running_var size mismatch");
    TORCH_CHECK((int)gamma.numel() == C, "gamma size mismatch");
    TORCH_CHECK((int)beta.numel() == C, "beta size mismatch");

    at::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty_like(x);

    int HW = H * W;
    // tiles over HW (per n) to enable safe hw-vectorization without cross-channel mixing
    int tiles = 8;
    if (HW >= (1 << 14)) tiles = 16;
    if (HW >= (1 << 15)) tiles = 32;
    if (HW >= (1 << 16)) tiles = 48;
    if (HW >= (1 << 17)) tiles = 64;
    if (tiles > 64) tiles = 64;

    dim3 grid((unsigned)C, (unsigned)tiles, 1);
    apply_bn_scale_eval_2d_kernel_vec_hw4<<<grid, 256>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)running_mean.data_ptr<float>(),
        (const float*)running_var.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, C, H, W,
        (float)eps,
        (float)scaling_factor
    );

    return y;
}
"""

cpp_source = r"""
torch::Tensor bn2d_scale_train_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor num_batches_tracked,
    double eps,
    double scaling_factor,
    double momentum,
    bool update_running
);

torch::Tensor bn2d_scale_eval_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps,
    double scaling_factor
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_bn2d_scale_v4_vec_hwtiles",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bn2d_scale_train_cuda", "bn2d_scale_eval_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps Conv2d in PyTorch/cuDNN and fuses:
        BatchNorm2d + scaling_factor
    using a custom CUDA extension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)

        if (not y.is_cuda) or (y.dtype != torch.float32) or (not y.is_contiguous()):
            y = self.bn(y)
            return y * self.scaling_factor

        gamma = self.bn.weight
        beta = self.bn.bias
        if gamma is None or beta is None or (not self.bn.affine):
            y = self.bn(y)
            return y * self.scaling_factor

        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()

        eps = float(self.bn.eps)

        if self.training:
            rm = self.bn.running_mean
            rv = self.bn.running_var
            nbt = self.bn.num_batches_tracked

            update_running = bool(self.bn.track_running_stats and (rm is not None) and (rv is not None) and (nbt is not None))
            if update_running:
                if rm.dtype != torch.float32:
                    rm = rm.float()
                if rv.dtype != torch.float32:
                    rv = rv.float()
                if not rm.is_contiguous():
                    rm = rm.contiguous()
                if not rv.is_contiguous():
                    rv = rv.contiguous()
                if not nbt.is_contiguous():
                    nbt = nbt.contiguous()
            else:
                rm = torch.empty((gamma.numel(),), device=y.device, dtype=torch.float32)
                rv = torch.empty((gamma.numel(),), device=y.device, dtype=torch.float32)
                nbt = torch.empty((1,), device=y.device, dtype=torch.int64)

            momentum = float(self.bn.momentum) if (self.bn.momentum is not None) else 0.1

            return self.custom_ops_lib.bn2d_scale_train_cuda(
                y, gamma, beta, rm, rv, nbt,
                eps, float(self.scaling_factor), momentum, bool(update_running)
            )
        else:
            rm = self.bn.running_mean
            rv = self.bn.running_var
            if rm is None or rv is None:
                y = self.bn(y)
                return y * self.scaling_factor

            if rm.dtype != torch.float32:
                rm = rm.float()
            if rv.dtype != torch.float32:
                rv = rv.float()
            if not rm.is_contiguous():
                rm = rm.contiguous()
            if not rv.is_contiguous():
                rv = rv.contiguous()

            return self.custom_ops_lib.bn2d_scale_eval_cuda(
                y, rm, rv, gamma, beta,
                eps, float(self.scaling_factor)
            )