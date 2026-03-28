import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>
#include <limits>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

template<int THREADS>
static __forceinline__ __device__ float block_reduce_max(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float sm[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float w = warp_reduce_max(v);
    if (lane == 0) sm[warp] = w;
    __syncthreads();
    float out = -INFINITY;
    if (warp == 0) {
        float x = (tid < WARPS) ? sm[tid] : -INFINITY;
        x = warp_reduce_max(x);
        if (lane == 0) sm[0] = x;
    }
    __syncthreads();
    out = sm[0];
    return out;
}

template<int THREADS>
static __forceinline__ __device__ float block_reduce_sum(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float sm[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float w = warp_reduce_sum(v);
    if (lane == 0) sm[warp] = w;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        float x = (tid < WARPS) ? sm[tid] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) sm[0] = x;
    }
    __syncthreads();
    out = sm[0];
    return out;
}

// Each CTA handles one (row, segment). Segment length is SEG_ELEMS.
// We compute:
//  - seg_max
//  - seg_sum = sum(exp(x - seg_max)) over the segment
// These can be combined across segments later:
// row_max = max(seg_max)
// row_sum = sum( seg_sum * exp(seg_max - row_max) )
template<int THREADS, int SEG_ELEMS>
__global__ __launch_bounds__(THREADS, 4)
void log_softmax_stage1_seg_stats_f32(
    const float* __restrict__ x,
    float* __restrict__ seg_max_out,   // [B, S]
    float* __restrict__ seg_sum_out,   // [B, S]
    int B, int D, int S)
{
    int row = (int)blockIdx.y;
    int seg = (int)blockIdx.x;
    if (row >= B || seg >= S) return;

    int tid = (int)threadIdx.x;
    int start = seg * SEG_ELEMS;
    int end = start + SEG_ELEMS;
    if (end > D) end = D;

    const float* xptr = x + (int64_t)row * (int64_t)D + start;

    // Pass A: segment max
    float local_max = -INFINITY;

    int len = end - start;

    // vectorized if aligned and length allows
    uintptr_t addr = (uintptr_t)(xptr);
    bool aligned16 = ((addr & 0xF) == 0);
    int vecN = len >> 2;
    int tail = vecN << 2;

    if (aligned16) {
        const float4* x4 = reinterpret_cast<const float4*>(xptr);
        for (int i = tid; i < vecN; i += THREADS) {
            float4 v = x4[i];
            local_max = fmaxf(local_max, v.x);
            local_max = fmaxf(local_max, v.y);
            local_max = fmaxf(local_max, v.z);
            local_max = fmaxf(local_max, v.w);
        }
        for (int j = tail + tid; j < len; j += THREADS) {
            local_max = fmaxf(local_max, ldg_f32(xptr + j));
        }
    } else {
        for (int j = tid; j < len; j += THREADS) {
            local_max = fmaxf(local_max, ldg_f32(xptr + j));
        }
    }

    float seg_max = block_reduce_max<THREADS>(local_max);

    // Pass B: segment sumexp around seg_max
    float local_sum = 0.0f;
    if (aligned16) {
        const float4* x4 = reinterpret_cast<const float4*>(xptr);
        for (int i = tid; i < vecN; i += THREADS) {
            float4 v = x4[i];
            local_sum += __expf(v.x - seg_max);
            local_sum += __expf(v.y - seg_max);
            local_sum += __expf(v.z - seg_max);
            local_sum += __expf(v.w - seg_max);
        }
        for (int j = tail + tid; j < len; j += THREADS) {
            local_sum += __expf(ldg_f32(xptr + j) - seg_max);
        }
    } else {
        for (int j = tid; j < len; j += THREADS) {
            local_sum += __expf(ldg_f32(xptr + j) - seg_max);
        }
    }

    float seg_sum = block_reduce_sum<THREADS>(local_sum);

    if (tid == 0) {
        seg_max_out[row * S + seg] = seg_max;
        seg_sum_out[row * S + seg] = seg_sum;
    }
}

// Reduce per-row over S segments to produce row_max and row_sum.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 8)
void log_softmax_stage2_row_reduce_f32(
    const float* __restrict__ seg_max,
    const float* __restrict__ seg_sum,
    float* __restrict__ row_max_out,   // [B]
    float* __restrict__ row_sum_out,   // [B]
    int B, int S)
{
    int row = (int)blockIdx.x;
    if (row >= B) return;
    int tid = (int)threadIdx.x;

    const float* mx = seg_max + row * S;
    const float* sm = seg_sum + row * S;

    // row max
    float local_max = -INFINITY;
    for (int i = tid; i < S; i += THREADS) {
        local_max = fmaxf(local_max, ldg_f32(mx + i));
    }
    float row_max = block_reduce_max<THREADS>(local_max);

    // row sum adjusted
    float local_sum = 0.0f;
    for (int i = tid; i < S; i += THREADS) {
        float m = ldg_f32(mx + i);
        float s = ldg_f32(sm + i);
        local_sum += s * __expf(m - row_max);
    }
    float row_sum = block_reduce_sum<THREADS>(local_sum);

    if (tid == 0) {
        row_max_out[row] = row_max;
        row_sum_out[row] = row_sum;
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void log_softmax_stage3_finalize_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    int B, int D)
{
    int row = (int)blockIdx.x;
    if (row >= B) return;
    int tid = (int)threadIdx.x;

    float mx = ldg_f32(row_max + row);
    float sm = ldg_f32(row_sum + row);
    sm = fmaxf(sm, 1e-20f);
    float log_denom = logf(sm);

    const float* xptr = x + (int64_t)row * (int64_t)D;
    float* yptr = y + (int64_t)row * (int64_t)D;

    uintptr_t xa = (uintptr_t)xptr;
    uintptr_t ya = (uintptr_t)yptr;
    bool aligned16 = ((xa & 0xF) == 0) && ((ya & 0xF) == 0);

    int vecD = D >> 2;
    int tail = vecD << 2;

    if (aligned16) {
        const float4* x4 = reinterpret_cast<const float4*>(xptr);
        float4* y4 = reinterpret_cast<float4*>(yptr);
        for (int i = tid; i < vecD; i += THREADS) {
            float4 v = x4[i];
            float4 o;
            o.x = v.x - mx - log_denom;
            o.y = v.y - mx - log_denom;
            o.z = v.z - mx - log_denom;
            o.w = v.w - mx - log_denom;
            y4[i] = o;
        }
        for (int j = tail + tid; j < D; j += THREADS) {
            yptr[j] = ldg_f32(xptr + j) - mx - log_denom;
        }
    } else {
        for (int j = tid; j < D; j += THREADS) {
            yptr[j] = ldg_f32(xptr + j) - mx - log_denom;
        }
    }
}

std::vector<torch::Tensor> log_softmax_dim1_f32_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only supports float32");
    TORCH_CHECK(x.dim() == 2, "expects 2D tensor");
    const int B = (int)x.size(0);
    const int D = (int)x.size(1);

    auto y = torch::empty_like(x);

    // Segment sizing tuned for very wide D to create enough CTAs per row for latency hiding,
    // while keeping overhead small.
    constexpr int SEG_ELEMS = 16384; // 64KB per segment
    int S = (D + SEG_ELEMS - 1) / SEG_ELEMS;

    // Cap segments per row to avoid excessive overhead on smaller D.
    // For this workload D=393216 -> S=24 (fine).
    TORCH_CHECK(S > 0, "invalid S");

    auto opts = x.options();
    auto seg_max = torch::empty({B, S}, opts);
    auto seg_sum = torch::empty({B, S}, opts);
    auto row_max = torch::empty({B}, opts);
    auto row_sum = torch::empty({B}, opts);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    // Stage1: 2D grid: (segments, rows)
    {
        constexpr int THREADS = 128;
        dim3 blocks((unsigned)S, (unsigned)B, 1);
        dim3 threads(THREADS, 1, 1);
        log_softmax_stage1_seg_stats_f32<THREADS, SEG_ELEMS><<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)seg_max.data_ptr<float>(),
            (float*)seg_sum.data_ptr<float>(),
            B, D, S
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage2: per row reduce over segments
    {
        constexpr int THREADS = 128;
        dim3 blocks((unsigned)B, 1, 1);
        dim3 threads(THREADS, 1, 1);
        log_softmax_stage2_row_reduce_f32<THREADS><<<blocks, threads, 0, stream>>>(
            (const float*)seg_max.data_ptr<float>(),
            (const float*)seg_sum.data_ptr<float>(),
            (float*)row_max.data_ptr<float>(),
            (float*)row_sum.data_ptr<float>(),
            B, S
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage3: finalize
    {
        constexpr int THREADS = 256;
        dim3 blocks((unsigned)B, 1, 1);
        dim3 threads(THREADS, 1, 1);
        log_softmax_stage3_finalize_f32<THREADS><<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)row_max.data_ptr<float>(),
            (const float*)row_sum.data_ptr<float>(),
            B, D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {y, seg_max, seg_sum, row_max, row_sum};
}
"""

cpp_source = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> log_softmax_dim1_f32_cuda(torch::Tensor x);
"""

_ext_name = "custom_ops_lib_log_softmax_segmented_v2"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["log_softmax_dim1_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    LogSoftmax model using a segmented multi-CTA CUDA kernel (dim=1, float32, 2D contiguous).
    Falls back to torch.log_softmax for unsupported cases.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.dim == 1
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 2
            and x.is_contiguous()
        ):
            # extension returns multiple tensors; first is output
            return self.custom_ops_lib.log_softmax_dim1_f32_cuda(x)[0]
        return torch.log_softmax(x, dim=self.dim)