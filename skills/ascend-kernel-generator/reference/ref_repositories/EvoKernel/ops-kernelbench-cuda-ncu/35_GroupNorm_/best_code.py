import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static inline __device__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static inline __device__ float block_sum(float v) {
    __shared__ float sh[32]; // up to 1024 threads => 32 warps
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float w = warp_sum(v);
    if (lane == 0) sh[warp] = w;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        float x = (lane < num_warps) ? sh[lane] : 0.0f;
        out = warp_sum(x);
    }
    return out; // valid only in warp0; broadcast by caller if needed
}

// Stage 1: compute partial sums/sumsq per (n,g,split)
__global__ void groupnorm_partials_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum_out,     // [total*Splits]
    float* __restrict__ sumsq_out,   // [total*Splits]
    int N, int C, int HxW, int G, int Splits)
{
    int total = N * G;
    int idx = blockIdx.x;
    int split = blockIdx.y;
    if (idx >= total || split >= Splits) return;

    int tid = threadIdx.x;

    int Cg = C / G;
    int D = Cg * HxW;

    int n = idx / G;
    int g = idx - n * G;

    const float* x_ptr = x + ((n * C + g * Cg) * HxW);

    // Split D into contiguous chunks
    int chunk = (D + Splits - 1) / Splits;
    int start = split * chunk;
    int end = start + chunk;
    if (end > D) end = D;

    float sum = 0.0f;
    float sumsq = 0.0f;

    // Vectorized when aligned; handle tail
    uintptr_t addr = reinterpret_cast<uintptr_t>(x_ptr + start);
    bool vec_ok = ((addr & 0xF) == 0);
    int len = end - start;

    if (vec_ok) {
        int len4 = len >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x_ptr + start);
        for (int i = tid; i < len4; i += blockDim.x) {
            float4 v = x4[i];
            sum   += (v.x + v.y + v.z + v.w);
            sumsq += (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w);
        }
        for (int i = (len4 << 2) + tid; i < len; i += blockDim.x) {
            float v = x_ptr[start + i];
            sum += v;
            sumsq += v * v;
        }
    } else {
        for (int i = start + tid; i < end; i += blockDim.x) {
            float v = x_ptr[i];
            sum += v;
            sumsq += v * v;
        }
    }

    float bsum = block_sum(sum);
    float bsumsq = block_sum(sumsq);

    if (threadIdx.x < 32) {
        // broadcast from warp0 lane0
        float out_sum = __shfl_sync(0xffffffff, bsum, 0);
        float out_sumsq = __shfl_sync(0xffffffff, bsumsq, 0);
        if ((threadIdx.x & 31) == 0) {
            int off = idx * Splits + split;
            sum_out[off] = out_sum;
            sumsq_out[off] = out_sumsq;
        }
    }
}

// Stage 2: reduce partials to mean/rstd then normalize+affine.
// Uses shared memory to cache per-channel affine params for this (n,g): a[c]=gamma*c_rstd, b[c]=beta - mean*a_base.
__global__ void groupnorm_finalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ sum_in,    // [total*Splits]
    const float* __restrict__ sumsq_in,  // [total*Splits]
    int N, int C, int HxW, int G, int Splits, float eps)
{
    int total = N * G;
    int idx = blockIdx.x;
    int split = blockIdx.y;
    if (idx >= total || split >= Splits) return;

    int tid = threadIdx.x;
    int Cg = C / G;
    int D = Cg * HxW;

    int n = idx / G;
    int g = idx - n * G;

    const float* x_ptr = x + ((n * C + g * Cg) * HxW);
    float* y_ptr = y + ((n * C + g * Cg) * HxW);

    // Reduce partial sums across Splits (small, e.g., 8/16) using all threads
    float psum = 0.0f;
    float psumsq = 0.0f;
    for (int s = tid; s < Splits; s += blockDim.x) {
        int off = idx * Splits + s;
#if __CUDA_ARCH__ >= 350
        float a = __ldg(sum_in + off);
        float b = __ldg(sumsq_in + off);
#else
        float a = sum_in[off];
        float b = sumsq_in[off];
#endif
        psum += a;
        psumsq += b;
    }

    float bsum = block_sum(psum);
    float bsumsq = block_sum(psumsq);

    __shared__ float s_mean;
    __shared__ float s_rstd;

    if (tid == 0) {
        float sum_all = bsum;     // lane0 warp0 holds final
        float sumsq_all = bsumsq;
        float mean = sum_all / (float)D;
        float var = sumsq_all / (float)D - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // Precompute per-channel affine in shared memory (only once per (idx, split) block).
    // This is redundant across splits, but still dramatically reduces gamma/beta reads vs per-element.
    extern __shared__ float shmem[];
    float* sh_a = shmem;         // [Cg]
    float* sh_b = shmem + Cg;    // [Cg]

    // Load gamma/beta and compute a,b for channels in this group
    for (int c = tid; c < Cg; c += blockDim.x) {
        int ch = g * Cg + c;
#if __CUDA_ARCH__ >= 350
        float ga = __ldg(gamma + ch);
        float be = __ldg(beta + ch);
#else
        float ga = gamma[ch];
        float be = beta[ch];
#endif
        float a = ga * rstd;
        float b = be - mean * a;
        sh_a[c] = a;
        sh_b[c] = b;
    }
    __syncthreads();

    // Split D into contiguous chunks
    int chunk = (D + Splits - 1) / Splits;
    int start = split * chunk;
    int end = start + chunk;
    if (end > D) end = D;
    int len = end - start;

    // Normalize + affine: y = x*sh_a[c] + sh_b[c]
    // Avoid i/HxW division by using nested loops over channels then spatial.
    // Work assignment: linear over elements in [start,end) but compute c via multiplication/subtraction (still needs / for mapping),
    // so we instead process by channels: for each channel, its range is [c*HxW,(c+1)*HxW).
    // Determine which channel range overlaps our [start,end) chunk.
    int c_start = start / HxW;
    int c_end = (end + HxW - 1) / HxW;
    if (c_end > Cg) c_end = Cg;

    // Iterate channels and vectorize within each channel spatial segment.
    for (int c = c_start; c < c_end; ++c) {
        int seg0 = c * HxW;
        int seg1 = seg0 + HxW;
        int s0 = (start > seg0) ? start : seg0;
        int s1 = (end < seg1) ? end : seg1;
        int nelt = s1 - s0;
        if (nelt <= 0) continue;

        float a = sh_a[c];
        float b = sh_b[c];

        // Vectorize if possible for this segment
        uintptr_t xaddr = reinterpret_cast<uintptr_t>(x_ptr + s0);
        uintptr_t yaddr = reinterpret_cast<uintptr_t>(y_ptr + s0);
        bool vec_ok = ((xaddr & 0xF) == 0) && ((yaddr & 0xF) == 0);

        if (vec_ok) {
            int n4 = nelt >> 2;
            const float4* x4 = reinterpret_cast<const float4*>(x_ptr + s0);
            float4* y4 = reinterpret_cast<float4*>(y_ptr + s0);
            for (int i = tid; i < n4; i += blockDim.x) {
                float4 v = x4[i];
                float4 o;
                o.x = v.x * a + b;
                o.y = v.y * a + b;
                o.z = v.z * a + b;
                o.w = v.w * a + b;
                y4[i] = o;
            }
            for (int i = (n4 << 2) + tid; i < nelt; i += blockDim.x) {
                float v = x_ptr[s0 + i];
                y_ptr[s0 + i] = v * a + b;
            }
        } else {
            for (int i = tid; i < nelt; i += blockDim.x) {
                float v = x_ptr[s0 + i];
                y_ptr[s0 + i] = v * a + b;
            }
        }
        __syncthreads(); // keep shmem stable across channels (conservative)
    }
}

torch::Tensor groupnorm_forward_cuda(torch::Tensor x,
                                     torch::Tensor gamma,
                                     torch::Tensor beta,
                                     int64_t num_groups,
                                     double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32, "only float32 gamma/beta supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW contiguous)");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims (N, C, ...)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    int64_t HxW = 1;
    for (int i = 2; i < x.dim(); ++i) HxW *= x.size(i);

    auto y = torch::empty_like(x);

    int total = (int)(N * num_groups);
    int Cg = (int)(C / num_groups);
    int D = (int)(Cg * HxW);

    // Choose splits to increase parallelism; keep small to reduce overhead.
    int Splits = 8;
    if (D >= 1<<18) Splits = 16;
    if (D <= 1<<14) Splits = 4;
    if (Splits < 1) Splits = 1;

    // Intermediate partial sums
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sum_buf = torch::empty({total * Splits}, opts);
    auto sumsq_buf = torch::empty({total * Splits}, opts);

    int threads = 256;

    dim3 grid1((unsigned)total, (unsigned)Splits, 1);
    groupnorm_partials_kernel<<<grid1, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)sum_buf.data_ptr<float>(),
        (float*)sumsq_buf.data_ptr<float>(),
        (int)N, (int)C, (int)HxW, (int)num_groups, (int)Splits
    );

    // Shared memory for a,b arrays: 2*Cg floats
    size_t shmem = (size_t)(2 * Cg) * sizeof(float);
    dim3 grid2((unsigned)total, (unsigned)Splits, 1);
    groupnorm_finalize_kernel<<<grid2, threads, shmem>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (const float*)sum_buf.data_ptr<float>(),
        (const float*)sumsq_buf.data_ptr<float>(),
        (int)N, (int)C, (int)HxW, (int)num_groups, (int)Splits, (float)eps
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor groupnorm_forward_cuda(torch::Tensor x,
                                     torch::Tensor gamma,
                                     torch::Tensor beta,
                                     int64_t num_groups,
                                     double eps);
"""

custom_ops_lib = load_inline(
    name="custom_groupnorm_ops_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["groupnorm_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized GroupNorm forward for float32 contiguous NCHW inputs (affine=True).
    """
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = int(num_groups)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        b = self.bias
        if x.is_cuda and (not w.is_cuda):
            w = w.to(device=x.device)
            b = b.to(device=x.device)

        return self.custom_ops_lib.groupnorm_forward_cuda(
            x, w.contiguous(), b.contiguous(), self.num_groups, float(self.eps)
        )