import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: pooled-max (stride=K) + rowwise sum (+ scale) with split-CTAs + no atomics ----
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <math.h>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {                                  \
  cudaError_t err = cudaGetLastError();                                      \
  if (err != cudaSuccess) {                                                  \
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));       \
  }                                                                          \
} while(0)
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template<int THREADS>
__device__ __forceinline__ float block_sum(float v) {
    constexpr int WARP = 32;
    __shared__ float smem[THREADS / WARP];
    int tid = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int wid  = tid >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (wid == 0) {
        // lane indexes warp-sums
        out = (lane < (THREADS / WARP)) ? smem[lane] : 0.f;
        out = warp_sum(out);
    }
    return out;
}

// partials: [B, splits] in row-major (contiguous)
template<int THREADS>
__global__ __launch_bounds__(THREADS, 8)
void maxpool_sum_partials_kernel(
    const float* __restrict__ x,     // [B, C]
    float* __restrict__ partials,    // [B, splits]
    int B, int C, int K, int L,
    int splits
) {
    int b = (int)blockIdx.x;
    int split = (int)blockIdx.y;
    if (b >= B) return;

    const float* row = x + (int64_t)b * (int64_t)C;

    float acc = 0.f;

    // Prefer vector paths when aligned.
    const uintptr_t addr = (uintptr_t)row;
    const bool vec4_ok = ((addr & 15) == 0) && ((C & 3) == 0);
    const bool vec2_ok = ((addr & 7) == 0);

    if (K == 2 && vec4_ok) {
        // Partition in float4 units to avoid overlap:
        // float4 j corresponds to elements [4j..4j+3] -> pooled indices (2j, 2j+1)
        const int n4 = C >> 2; // number of float4 vectors
        int j0 = (int)((int64_t)n4 * split / splits);
        int j1 = (int)((int64_t)n4 * (split + 1) / splits);

        const float4* __restrict__ row4 = (const float4*)row;

        // thread-stride over contiguous float4s (coalesced)
        // mild unroll for ILP without exploding regs
        for (int j = j0 + (int)threadIdx.x; j < j1; j += THREADS * 2) {
            float4 v0 = __ldg(row4 + j);
            acc += fmaxf(v0.x, v0.y) + fmaxf(v0.z, v0.w);

            int j2 = j + THREADS;
            if (j2 < j1) {
                float4 v1 = __ldg(row4 + j2);
                acc += fmaxf(v1.x, v1.y) + fmaxf(v1.z, v1.w);
            }
        }
    } else if (K == 4 && vec4_ok) {
        // Each float4 corresponds to one pooled output.
        const int n4 = C >> 2; // == L
        int i0 = (int)((int64_t)n4 * split / splits);
        int i1 = (int)((int64_t)n4 * (split + 1) / splits);

        const float4* __restrict__ row4 = (const float4*)row;
        for (int i = i0 + (int)threadIdx.x; i < i1; i += THREADS * 2) {
            float4 v0 = __ldg(row4 + i);
            acc += fmaxf(fmaxf(v0.x, v0.y), fmaxf(v0.z, v0.w));

            int i2 = i + THREADS;
            if (i2 < i1) {
                float4 v1 = __ldg(row4 + i2);
                acc += fmaxf(fmaxf(v1.x, v1.y), fmaxf(v1.z, v1.w));
            }
        }
    } else {
        // Generic partitioning in pooled-index space [l0, l1)
        int l0 = (int)((int64_t)L * split / splits);
        int l1 = (int)((int64_t)L * (split + 1) / splits);

        if (K == 2 && vec2_ok) {
            const float2* __restrict__ row2 = (const float2*)row; // length L
            for (int i = l0 + (int)threadIdx.x; i < l1; i += THREADS * 2) {
                float2 v0 = __ldg(row2 + i);
                acc += fmaxf(v0.x, v0.y);
                int i2 = i + THREADS;
                if (i2 < l1) {
                    float2 v1 = __ldg(row2 + i2);
                    acc += fmaxf(v1.x, v1.y);
                }
            }
        } else if (K == 2) {
            for (int i = l0 + (int)threadIdx.x; i < l1; i += THREADS * 2) {
                int base0 = i << 1;
                float a0 = ldg_f32(row + base0 + 0);
                float a1 = ldg_f32(row + base0 + 1);
                acc += fmaxf(a0, a1);

                int i2 = i + THREADS;
                if (i2 < l1) {
                    int base1 = i2 << 1;
                    float b0 = ldg_f32(row + base1 + 0);
                    float b1 = ldg_f32(row + base1 + 1);
                    acc += fmaxf(b0, b1);
                }
            }
        } else if (K == 4) {
            for (int i = l0 + (int)threadIdx.x; i < l1; i += THREADS) {
                int base = i << 2;
                float a0 = ldg_f32(row + base + 0);
                float a1 = ldg_f32(row + base + 1);
                float a2 = ldg_f32(row + base + 2);
                float a3 = ldg_f32(row + base + 3);
                acc += fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
            }
        } else {
            for (int i = l0 + (int)threadIdx.x; i < l1; i += THREADS) {
                int base = i * K;
                float mv = -INFINITY;
#pragma unroll 1
                for (int j = 0; j < K; ++j) mv = fmaxf(mv, ldg_f32(row + base + j));
                acc += mv;
            }
        }
    }

    float total = block_sum<THREADS>(acc);
    if ((int)threadIdx.x == 0) {
        partials[(int64_t)b * (int64_t)splits + split] = total;
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 16)
void reduce_partials_scale_kernel(
    const float* __restrict__ partials, // [B, splits]
    float* __restrict__ out,            // [B]
    int B, int splits,
    float scale
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    const float* p = partials + (int64_t)b * (int64_t)splits;

    float acc = 0.f;
    for (int i = (int)threadIdx.x; i < splits; i += THREADS) {
        acc += ldg_f32(p + i);
    }
    float total = block_sum<THREADS>(acc);
    if ((int)threadIdx.x == 0) out[b] = total * scale;
}

torch::Tensor maxpool_sum_scale_forward_cuda(
    torch::Tensor x,
    int64_t kernel_size,
    double scale_factor
) {
    TORCH_CHECK(x.is_cuda(), "maxpool_sum_scale_forward_cuda: x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "maxpool_sum_scale_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "maxpool_sum_scale_forward_cuda: x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "maxpool_sum_scale_forward_cuda: x must be 2D (B, C)");

    const int64_t B64 = x.size(0);
    const int64_t C64 = x.size(1);
    TORCH_CHECK(B64 > 0 && C64 > 0, "maxpool_sum_scale_forward_cuda: invalid shape");
    TORCH_CHECK(kernel_size > 0, "maxpool_sum_scale_forward_cuda: kernel_size must be > 0");
    TORCH_CHECK((C64 % kernel_size) == 0,
                "maxpool_sum_scale_forward_cuda: expected C divisible by kernel_size for stride=kernel_size pooling");

    const int B = (int)B64;
    const int C = (int)C64;
    const int K = (int)kernel_size;
    const int L = C / K;
    const float scale = (float)scale_factor;

    // choose splits (CTAs per row)
    int splits = 1;
    if (C >= 65536) splits = 8;
    else if (C >= 32768) splits = 6;
    else if (C >= 16384) splits = 4;
    else if (C >= 8192)  splits = 2;
    else splits = 1;

    // Keep splits <= 8 to limit overhead and temp size
    if (splits > 8) splits = 8;

    auto out = torch::empty({B64}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (splits == 1) {
        // No temp: reuse partials kernel writing directly into out (as "partials")
        dim3 grid((unsigned)B, 1u, 1u);
        maxpool_sum_partials_kernel<128><<<grid, 128, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, K, L, 1
        );
        // scale in-place via tiny kernel: reduce_partials on splits=1 is overkill; do a simple scale
        // but keep code size small: call reduce kernel with splits=1 and partials==out into out2 then copy? no.
        // Instead do a one-block-per-row scale in-place.
        // Launch a light kernel that reads out[b] and scales; reuse reduce kernel by treating out as partials and writing to out.
        reduce_partials_scale_kernel<32><<<B, 32, 0, stream>>>(
            (const float*)out.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, 1, scale
        );
    } else {
        auto partials = torch::empty({B64, (int64_t)splits}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
        dim3 grid((unsigned)B, (unsigned)splits, 1u);
        maxpool_sum_partials_kernel<128><<<grid, 128, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)partials.data_ptr<float>(),
            B, C, K, L, splits
        );
        reduce_partials_scale_kernel<128><<<B, 128, 0, stream>>>(
            (const float*)partials.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, splits, scale
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_src = r"""
torch::Tensor maxpool_sum_scale_forward_cuda(torch::Tensor x, int64_t kernel_size, double scale_factor);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_max_pool_sum_scale_v6_split_noatomics",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["maxpool_sum_scale_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    GEMM (nn.Linear) followed by a fused CUDA extension:
    MaxPool1d over features with stride=kernel_size + sum over pooled features + scale.
    Output: (batch_size,)
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.maxpool_sum_scale_forward_cuda(
            x, self.kernel_size, self.scale_factor
        )


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]