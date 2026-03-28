import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Two-phase split-K LayerNorm forward (no affine):
# Phase1: many CTAs per row compute partial sum/sumsq, atomicAdd to row buffers; one CTA finalizes mean/rstd
# Phase2: many CTAs per row normalize and write output using mean/rstd
#
# Designed for regime: very small M (e.g., batch=16) and huge K (e.g., 64*256*256).
# Avoids spin-wait polling and increases parallelism to hide DRAM latency.

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int THREADS>
static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float smem[32]; // up to 1024 threads -> 32 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nwarps = (THREADS + 31) >> 5;
    float out = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum(out);
    __shared__ float total;
    if (threadIdx.x == 0) total = out;
    __syncthreads();
    return total;
}

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Map blocks: blockIdx.x in [0, M*splits)
// row = blockIdx.x / splits
// split = blockIdx.x % splits
// each split processes a contiguous segment [split*chunk, min((split+1)*chunk, K))
template<int THREADS>
__global__ __launch_bounds__(THREADS, 6)
void layernorm_stats_splitk(const float* __restrict__ x,
                            float* __restrict__ row_sum,
                            float* __restrict__ row_sumsq,
                            int*   __restrict__ row_ctr,
                            float* __restrict__ mean,
                            float* __restrict__ rstd,
                            int64_t M, int64_t K,
                            int splits,
                            float eps) {
    int64_t row = (int64_t)(blockIdx.x / (uint32_t)splits);
    int split = (int)(blockIdx.x - (int64_t)row * splits);
    if (row >= M) return;

    const float* __restrict__ xr = x + row * K;

    // chunking
    int64_t chunk = (K + splits - 1) / splits;
    int64_t start = (int64_t)split * chunk;
    int64_t end = start + chunk;
    if (end > K) end = K;

    float sum = 0.0f;
    float sumsq = 0.0f;

    // Prefer float4 if aligned and region mostly divisible; handle tail scalars.
    uintptr_t ax = (uintptr_t)(xr + start);
    bool can_vec4 = ((ax & 0xF) == 0);

    if (can_vec4) {
        int64_t start4 = (start + 3) & ~3LL;
        // head (to align)
        for (int64_t j = start + (int64_t)threadIdx.x; j < start4 && j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            sum += v;
            sumsq += v * v;
        }
        // vec body
        int64_t end4 = end & ~3LL;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xr);
        int64_t j4 = (start4 >> 2) + (int64_t)threadIdx.x;
        int64_t e4 = (end4 >> 2);
        for (; j4 < e4; j4 += (int64_t)THREADS) {
            float4 a = x4[j4];
            sum   += (a.x + a.y) + (a.z + a.w);
            sumsq += (a.x*a.x + a.y*a.y) + (a.z*a.z + a.w*a.w);
        }
        // tail
        for (int64_t j = end4 + (int64_t)threadIdx.x; j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            sum += v;
            sumsq += v * v;
        }
    } else {
        for (int64_t j = start + (int64_t)threadIdx.x; j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            sum += v;
            sumsq += v * v;
        }
    }

    float sum_all = block_reduce_sum<THREADS>(sum);
    float sumsq_all = block_reduce_sum<THREADS>(sumsq);

    if (threadIdx.x == 0) {
        atomicAdd(row_sum + row, sum_all);
        atomicAdd(row_sumsq + row, sumsq_all);
        int c = atomicAdd(row_ctr + row, 1) + 1;
        // The CTA that observes completion finalizes stats.
        if (c == splits) {
            float s = row_sum[row];
            float ss = row_sumsq[row];
            float invK = 1.0f / (float)K;
            float m = s * invK;
            float var = ss * invK - m * m;
            var = (var > 0.0f) ? var : 0.0f;
            mean[row] = m;
            rstd[row] = rsqrtf(var + eps);
        }
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 6)
void layernorm_norm_splitk(const float* __restrict__ x,
                           float* __restrict__ y,
                           const float* __restrict__ mean,
                           const float* __restrict__ rstd,
                           int64_t M, int64_t K,
                           int splits) {
    int64_t row = (int64_t)(blockIdx.x / (uint32_t)splits);
    int split = (int)(blockIdx.x - (int64_t)row * splits);
    if (row >= M) return;

    const float* __restrict__ xr = x + row * K;
    float* __restrict__ yr = y + row * K;

    int64_t chunk = (K + splits - 1) / splits;
    int64_t start = (int64_t)split * chunk;
    int64_t end = start + chunk;
    if (end > K) end = K;

    float m = mean[row];
    float rs = rstd[row];

    uintptr_t ax = (uintptr_t)(xr + start);
    uintptr_t ay = (uintptr_t)(yr + start);
    bool can_vec4 = ((ax & 0xF) == 0) && ((ay & 0xF) == 0);

    if (can_vec4) {
        int64_t start4 = (start + 3) & ~3LL;
        for (int64_t j = start + (int64_t)threadIdx.x; j < start4 && j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            yr[j] = (v - m) * rs;
        }
        int64_t end4 = end & ~3LL;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xr);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(yr);

        int64_t j4 = (start4 >> 2) + (int64_t)threadIdx.x;
        int64_t e4 = (end4 >> 2);
        for (; j4 < e4; j4 += (int64_t)THREADS) {
            float4 a = x4[j4];
            a.x = (a.x - m) * rs;
            a.y = (a.y - m) * rs;
            a.z = (a.z - m) * rs;
            a.w = (a.w - m) * rs;
            y4[j4] = a;
        }
        for (int64_t j = end4 + (int64_t)threadIdx.x; j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            yr[j] = (v - m) * rs;
        }
    } else {
        for (int64_t j = start + (int64_t)threadIdx.x; j < end; j += (int64_t)THREADS) {
            float v = ldg_f32(xr + j);
            yr[j] = (v - m) * rs;
        }
    }
}

// Host entry: allocates temps and launches the two phases.
// Notes:
// - We rely on kernel launch ordering on the default stream for phase1->phase2.
// - Temp buffers are per-call; for repeated calls you could cache, but keep it simple here.
torch::Tensor layernorm_fwd_cuda(torch::Tensor x, int64_t K, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(x.size(1) == K, "x.size(1) must equal K");

    auto y = torch::empty_like(x);
    int64_t M = x.size(0);

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(x.device());

    // Per-row accumulators for phase1
    auto row_sum   = torch::zeros({M}, opts_f);
    auto row_sumsq = torch::zeros({M}, opts_f);
    auto row_ctr   = torch::zeros({M}, opts_i);
    auto mean      = torch::empty({M}, opts_f);
    auto rstd      = torch::empty({M}, opts_f);

    // Choose splits to ensure enough CTAs when M is tiny, but cap to limit atomics.
    int dev = x.get_device();
    int sm_count = 80;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) sm_count = prop.multiProcessorCount;

    constexpr int THREADS = 128;

    // Target blocks ~ sm_count * 12 for good latency hiding
    int target_blocks = sm_count * 12;
    int splits = (int)((target_blocks + (int)M - 1) / (int)M);
    if (splits < 1) splits = 1;
    // Avoid excessive atomic contention; also don't exceed K/THREADS granularity too much
    int max_splits = 64;
    if (splits > max_splits) splits = max_splits;

    // If K is smaller, fewer splits needed
    // (rough heuristic: want at least ~4096 elements per CTA to amortize overhead)
    int64_t min_elems_per_cta = 4096;
    int64_t splits_by_k = (int64_t)((K + min_elems_per_cta - 1) / min_elems_per_cta);
    if (splits_by_k < 1) splits_by_k = 1;
    if (splits > (int)splits_by_k) splits = (int)splits_by_k;
    if (splits < 1) splits = 1;

    dim3 block(THREADS);
    dim3 grid((unsigned)(M * (int64_t)splits));

    layernorm_stats_splitk<THREADS><<<grid, block>>>(
        (const float*)x.data_ptr<float>(),
        (float*)row_sum.data_ptr<float>(),
        (float*)row_sumsq.data_ptr<float>(),
        (int*)row_ctr.data_ptr<int>(),
        (float*)mean.data_ptr<float>(),
        (float*)rstd.data_ptr<float>(),
        (int64_t)M, (int64_t)K,
        (int)splits,
        (float)eps
    );

    layernorm_norm_splitk<THREADS><<<grid, block>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (const float*)mean.data_ptr<float>(),
        (const float*)rstd.data_ptr<float>(),
        (int64_t)M, (int64_t)K,
        (int)splits
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor layernorm_fwd_cuda(torch::Tensor x, int64_t K, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_layernorm_ops_splitk_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["layernorm_fwd_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


class ModelNew(nn.Module):
    """
    Replacement for nn.LayerNorm over normalized_shape=(features, dim1, dim2)
    using a fused custom CUDA implementation (no affine).
    """
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.ln = nn.LayerNorm(
            normalized_shape=self.normalized_shape,
            elementwise_affine=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32):
            return self.ln(x)

        nd = len(self.normalized_shape)
        if nd == 0:
            return x

        if tuple(x.shape[-nd:]) != self.normalized_shape:
            return self.ln(x)

        x_contig = x.contiguous()
        K = 1
        for d in self.normalized_shape:
            K *= int(d)
        M = x_contig.numel() // K
        x2d = x_contig.view(M, K)

        y2d = custom_ops_lib.layernorm_fwd_cuda(x2d, K, self.eps)
        return y2d.view_as(x_contig)