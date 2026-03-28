import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused avgpool1d + gelu + scale + rowwise max (CTA-per-row, fixed build) ----
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// ---- GELU tanh approximation (fast) ----
__device__ __forceinline__ float gelu_tanh_approx(float x) {
    // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = kAlpha * (x + kBeta * x3);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float smem[8]; // up to 8 warps for 256 threads
    int lane = (int)(threadIdx.x & 31);
    int wid  = (int)(threadIdx.x >> 5);

    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    float out = -INFINITY;
    if (wid == 0) {
        out = (lane < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
        out = warp_reduce_max(out);
    }
    return out;
}

// CTA-per-row kernel, specialized for K==16.
// Vectorized float4 loads when 16B-aligned; scalar fallback otherwise.
// Unrolls the grid-stride loop to increase ILP.
__global__ __launch_bounds__(256, 2)
void avgpool16_gelu_scale_rowmax_cta_kernel(
    const float* __restrict__ x, // [B, C]
    float* __restrict__ out,     // [B]
    int B, int C, float scale
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    const float* row = x + (int64_t)b * (int64_t)C;
    int L = C >> 4; // C/16

    float tmax = -INFINITY;

    // Alignment check for safe float4 loads
    uintptr_t addr = (uintptr_t)(row);
    bool aligned16 = ((addr & 0xF) == 0);

    int tid = (int)threadIdx.x;
    int stride = (int)blockDim.x;

    if (aligned16) {
        // process i, i+stride, i+2*stride, i+3*stride in one loop body when possible
        for (int i = tid; i < L; i += 4 * stride) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int ii = i + u * stride;
                if (ii < L) {
                    int start = ii << 4; // ii*16
                    const float4* p4 = reinterpret_cast<const float4*>(row + start);
                    float4 v0 = __ldg(p4 + 0);
                    float4 v1 = __ldg(p4 + 1);
                    float4 v2 = __ldg(p4 + 2);
                    float4 v3 = __ldg(p4 + 3);

                    float sum = 0.0f;
                    sum += v0.x + v0.y + v0.z + v0.w;
                    sum += v1.x + v1.y + v1.z + v1.w;
                    sum += v2.x + v2.y + v2.z + v2.w;
                    sum += v3.x + v3.y + v3.z + v3.w;

                    float avg = sum * (1.0f / 16.0f);
                    float v = gelu_tanh_approx(avg) * scale;
                    tmax = fmaxf(tmax, v);
                }
            }
        }
    } else {
        for (int i = tid; i < L; i += 4 * stride) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int ii = i + u * stride;
                if (ii < L) {
                    int start = ii << 4;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        sum += __ldg(row + start + j);
                    }
                    float avg = sum * (1.0f / 16.0f);
                    float v = gelu_tanh_approx(avg) * scale;
                    tmax = fmaxf(tmax, v);
                }
            }
        }
    }

    float bmax = block_reduce_max(tmax);
    if (threadIdx.x == 0) out[b] = bmax;
}

// Generic CTA-per-row kernel for any K (C divisible by K).
__global__ __launch_bounds__(256, 2)
void avgpoolK_gelu_scale_rowmax_cta_kernel(
    const float* __restrict__ x, // [B, C]
    float* __restrict__ out,     // [B]
    int B, int C, int K, float scale
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    const float* row = x + (int64_t)b * (int64_t)C;
    int L = C / K;

    float tmax = -INFINITY;

    int tid = (int)threadIdx.x;
    int stride = (int)blockDim.x;

    for (int i = tid; i < L; i += stride) {
        int start = i * K;
        float sum = 0.0f;

        // mild unroll; K is typically small
        int j = 0;
        for (; j + 3 < K; j += 4) {
            float a0 = __ldg(row + start + j + 0);
            float a1 = __ldg(row + start + j + 1);
            float a2 = __ldg(row + start + j + 2);
            float a3 = __ldg(row + start + j + 3);
            sum += a0 + a1 + a2 + a3;
        }
        for (; j < K; ++j) sum += __ldg(row + start + j);

        float avg = sum / (float)K;
        float v = gelu_tanh_approx(avg) * scale;
        tmax = fmaxf(tmax, v);
    }

    float bmax = block_reduce_max(tmax);
    if (threadIdx.x == 0) out[b] = bmax;
}

torch::Tensor avgpool_gelu_scale_rowmax_forward_cuda(
    torch::Tensor x,
    int64_t pool_kernel_size,
    double scale_factor
) {
    TORCH_CHECK(x.is_cuda(), "avgpool_gelu_scale_rowmax_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "avgpool_gelu_scale_rowmax_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "avgpool_gelu_scale_rowmax_forward_cuda: x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "avgpool_gelu_scale_rowmax_forward_cuda: x must be 2D (B, C)");

    const int64_t B64 = x.size(0);
    const int64_t C64 = x.size(1);
    TORCH_CHECK(B64 > 0 && C64 > 0, "avgpool_gelu_scale_rowmax_forward_cuda: invalid shape");
    TORCH_CHECK(pool_kernel_size > 0, "avgpool_gelu_scale_rowmax_forward_cuda: pool_kernel_size must be > 0");
    TORCH_CHECK(C64 >= pool_kernel_size, "avgpool_gelu_scale_rowmax_forward_cuda: C must be >= pool_kernel_size");
    TORCH_CHECK((C64 - pool_kernel_size) % pool_kernel_size == 0,
                "avgpool_gelu_scale_rowmax_forward_cuda: expected out_features divisible by pool_kernel_size for stride=kernel_size pooling");

    const int B = (int)B64;
    const int C = (int)C64;
    const int K = (int)pool_kernel_size;
    const float scale = (float)scale_factor;

    auto out = torch::empty({B64}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));

    dim3 grid(B);
    dim3 block(256);

    if (K == 16 && (C % 16) == 0) {
        avgpool16_gelu_scale_rowmax_cta_kernel<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, scale
        );
    } else {
        avgpoolK_gelu_scale_rowmax_cta_kernel<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, K, scale
        );
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor avgpool_gelu_scale_rowmax_forward_cuda(torch::Tensor x, int64_t pool_kernel_size, double scale_factor);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_avg_pool_gelu_scale_max_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["avgpool_gelu_scale_rowmax_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs GEMM (nn.Linear) followed by a fused custom CUDA kernel:
    AvgPool1d (over feature dim, stride=kernel_size) + GELU + scale + rowwise max.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)  # cuBLAS GEMM
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.avgpool_gelu_scale_rowmax_forward_cuda(
            x, self.pool_kernel_size, self.scale_factor
        )


# Keep the same input helpers for compatibility with the original harness.
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]