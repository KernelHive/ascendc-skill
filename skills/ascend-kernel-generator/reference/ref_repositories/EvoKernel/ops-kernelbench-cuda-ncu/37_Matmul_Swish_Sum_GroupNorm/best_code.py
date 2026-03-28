import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float swish(float x) { return x * sigmoidf_fast(x); }

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Hot path: Cg==64, 4 warps/CTA, each warp processes one (n,g).
// Lanes 0..15 load/store float4 (16 vectors = 64 floats). Lanes 16..31 idle (warp-uniform).
__global__ __launch_bounds__(128, 6)
void swish_bias_groupnorm_cg64_warp4_kernel(
    const float* __restrict__ x,      // [N, C]
    float* __restrict__ y,            // [N, C]
    const float* __restrict__ bias,   // [C]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    int total_groups, int C, int G, float eps)
{
    const int Cg = 64;
    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;        // 0..3
    int lane = tid & 31;        // 0..31

    int group_idx = (int)blockIdx.x * 4 + warp;
    if (group_idx >= total_groups) return;

    int n = group_idx / G;
    int g = group_idx - n * G;
    int base = g * Cg;

    const float* xg = x + (size_t)n * (size_t)C + (size_t)base;
    float* yg = y + (size_t)n * (size_t)C + (size_t)base;

    // Vector alignment check (warp-uniform)
    bool vec_ok =
        ((((uintptr_t)xg) & 0xF) == 0) &&
        ((((uintptr_t)yg) & 0xF) == 0) &&
        ((((uintptr_t)(bias  + base)) & 0xF) == 0) &&
        ((((uintptr_t)(gamma + base)) & 0xF) == 0) &&
        ((((uintptr_t)(beta  + base)) & 0xF) == 0);

    float sum = 0.0f;
    float sumsq = 0.0f;

    // Keep swish+bias values for writeback (only used by active lanes)
    float sb0=0.f, sb1=0.f, sb2=0.f, sb3=0.f;

    if (vec_ok) {
        if (lane < 16) {
            const float4* x4 = reinterpret_cast<const float4*>(xg);
            const float4* b4 = reinterpret_cast<const float4*>(bias + base);

            float4 xv = x4[lane];
            float4 bv = b4[lane];

            sb0 = swish(xv.x) + bv.x;
            sb1 = swish(xv.y) + bv.y;
            sb2 = swish(xv.z) + bv.z;
            sb3 = swish(xv.w) + bv.w;

            sum = (sb0 + sb1) + (sb2 + sb3);
            sumsq = (sb0*sb0 + sb1*sb1) + (sb2*sb2 + sb3*sb3);
        }
        // lanes 16..31 contribute zeros
    } else {
        // Scalar path: lanes 0..31 cover first 32 elems; all lanes cover one elem, and also one elem+32.
        // This matches the v3 mapping (2 elems/lane), no extra sync.
        int i0 = lane;
        int i1 = lane + 32;
        float v0 = xg[i0];
        float v1 = xg[i1];
        float b0 = bias[base + i0];
        float b1 = bias[base + i1];
        float a0 = swish(v0) + b0;
        float a1 = swish(v1) + b1;
        // reuse sb0/sb1 for writeback
        sb0 = a0;
        sb1 = a1;
        sum = a0 + a1;
        sumsq = a0*a0 + a1*a1;
    }

    float total_sum = warp_sum(sum);
    float total_sqs = warp_sum(sumsq);
    total_sum = __shfl_sync(0xffffffff, total_sum, 0);
    total_sqs = __shfl_sync(0xffffffff, total_sqs, 0);

    float inv = 1.0f / 64.0f;
    float mean = total_sum * inv;
    float var = total_sqs * inv - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float rstd = rsqrtf(var + eps);

    if (vec_ok) {
        if (lane < 16) {
            const float4* g4  = reinterpret_cast<const float4*>(gamma + base);
            const float4* be4 = reinterpret_cast<const float4*>(beta  + base);
            float4 gv  = g4[lane];
            float4 bev = be4[lane];

            float4 out;
            out.x = ((sb0 - mean) * rstd) * gv.x + bev.x;
            out.y = ((sb1 - mean) * rstd) * gv.y + bev.y;
            out.z = ((sb2 - mean) * rstd) * gv.z + bev.z;
            out.w = ((sb3 - mean) * rstd) * gv.w + bev.w;

            float4* y4 = reinterpret_cast<float4*>(yg);
            y4[lane] = out;
        }
    } else {
        int i0 = lane;
        int i1 = lane + 32;
        float ga0 = gamma[base + i0];
        float ga1 = gamma[base + i1];
        float be0 = beta[base + i0];
        float be1 = beta[base + i1];
        yg[i0] = ((sb0 - mean) * rstd) * ga0 + be0;
        yg[i1] = ((sb1 - mean) * rstd) * ga1 + be1;
    }
}

// Generic fallback (similar to v3, small tuning: 192 threads can improve residency on some GPUs)
__global__ __launch_bounds__(192, 3)
void swish_bias_groupnorm_generic_kernel(
    const float* __restrict__ x,      // [N, C]
    float* __restrict__ y,            // [N, C]
    const float* __restrict__ bias,   // [C]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    int N, int C, int G, float eps)
{
    int idx = (int)blockIdx.x; // [0, N*G)
    int n = idx / G;
    int g = idx - n * G;
    int Cg = C / G;

    const float* xg = x + (size_t)n * (size_t)C + (size_t)g * (size_t)Cg;
    float* yg = y + (size_t)n * (size_t)C + (size_t)g * (size_t)Cg;

    const float* b_ptr  = bias  + (size_t)g * (size_t)Cg;
    const float* ga_ptr = gamma + (size_t)g * (size_t)Cg;
    const float* be_ptr = beta  + (size_t)g * (size_t)Cg;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (int)blockDim.x >> 5;

    float thread_sum = 0.0f;
    float thread_sumsq = 0.0f;

    for (int i = tid; i < Cg; i += (int)blockDim.x) {
        float sb = swish(xg[i]) + b_ptr[i];
        thread_sum += sb;
        thread_sumsq += sb * sb;
    }

    float wsum = warp_sum(thread_sum);
    float wsq  = warp_sum(thread_sumsq);

    extern __shared__ float shmem[];
    float* sh_sum = shmem;
    float* sh_sqs = shmem + num_warps;

    if (lane == 0) {
        sh_sum[warp] = wsum;
        sh_sqs[warp] = wsq;
    }
    __syncthreads();

    __shared__ float s_mean;
    __shared__ float s_rstd;

    if (warp == 0) {
        float v1 = (lane < num_warps) ? sh_sum[lane] : 0.0f;
        float v2 = (lane < num_warps) ? sh_sqs[lane] : 0.0f;
        float total_sum = warp_sum(v1);
        float total_sqs = warp_sum(v2);
        if (lane == 0) {
            float inv = 1.0f / (float)Cg;
            float mean = total_sum * inv;
            float var = total_sqs * inv - mean * mean;
            if (var < 0.0f) var = 0.0f;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    for (int i = tid; i < Cg; i += (int)blockDim.x) {
        float sb = swish(xg[i]) + b_ptr[i];
        float nv = (sb - mean) * rstd;
        yg[i] = nv * ga_ptr[i] + be_ptr[i];
    }
}

torch::Tensor swish_bias_groupnorm_forward_cuda(torch::Tensor x,
                                               torch::Tensor bias,
                                               torch::Tensor gamma,
                                               torch::Tensor beta,
                                               int64_t num_groups,
                                               double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "bias/gamma/beta must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported for x");
    TORCH_CHECK(bias.dtype() == torch::kFloat32 && gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32,
                "only float32 supported for bias/gamma/beta");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "bias/gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, C]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t G = num_groups;

    TORCH_CHECK(bias.numel() == C, "bias must have shape [C]");
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta must have shape [C]");
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    auto y = torch::empty_like(x);

    int total_groups = (int)(N * G);
    int Cg = (int)(C / G);

    if (Cg == 64) {
        // 4 warps per block, each warp does one group => blocks = ceil(total_groups/4)
        int threads = 128;
        int blocks = (total_groups + 3) / 4;
        swish_bias_groupnorm_cg64_warp4_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            total_groups, (int)C, (int)G, (float)eps
        );
    } else {
        int threads = 192;
        int num_warps = threads / 32;
        size_t shmem = (size_t)(2 * num_warps) * sizeof(float);
        swish_bias_groupnorm_generic_kernel<<<total_groups, threads, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (int)N, (int)C, (int)G, (float)eps
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor swish_bias_groupnorm_forward_cuda(torch::Tensor x,
                                               torch::Tensor bias,
                                               torch::Tensor gamma,
                                               torch::Tensor beta,
                                               int64_t num_groups,
                                               double eps);
"""

custom_ops_lib = load_inline(
    name="custom_matmul_swish_sum_groupnorm_ops_fused_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["swish_bias_groupnorm_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized replacement:
      - Linear (GEMM) stays in PyTorch/cuBLAS
      - Fuse Swish + extra bias + GroupNorm(+affine) into one CUDA kernel

    v5 improvements over v3:
      - For Cg==64: use 128-thread CTA (4 warps) with warp-specialization:
        each warp handles one (n,g), no __syncthreads(), warp-shuffle reductions only.
      - Vectorized float4 IO for the hot path when aligned (lanes 0..15 active).
      - Generic fallback retained (slightly tuned CTA size).
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.group_norm = nn.GroupNorm(num_groups, out_features, affine=True)
        self.num_groups = int(num_groups)
        self.eps = float(self.group_norm.eps)
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

        bias = self.bias
        gamma = self.group_norm.weight
        beta = self.group_norm.bias

        if y.is_cuda:
            dev = y.device
            if bias.device != dev:
                bias = bias.to(dev)
            if gamma.device != dev:
                gamma = gamma.to(dev)
            if beta.device != dev:
                beta = beta.to(dev)

            return self.custom_ops_lib.swish_bias_groupnorm_forward_cuda(
                y,
                bias.contiguous(),
                gamma.contiguous(),
                beta.contiguous(),
                self.num_groups,
                float(self.eps),
            )

        z = torch.sigmoid(y) * y
        z = z + self.bias
        return nn.functional.group_norm(z, self.num_groups, self.group_norm.weight, self.group_norm.bias, self.eps)