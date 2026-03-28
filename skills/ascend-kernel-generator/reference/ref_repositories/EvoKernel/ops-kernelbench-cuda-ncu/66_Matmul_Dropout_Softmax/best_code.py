import torch
import torch.nn as nn
import torch.nn.functional as F
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

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

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

// Lightweight counter-based hash RNG (not Philox): good enough for dropout, cheap.
// Deterministic by (seed,row,idx4). Produces 4x uint32 then convert to float (0,1).
static __forceinline__ __device__ uint32_t mix32(uint32_t x) {
    // finalizer-like mix (Murmur-inspired)
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __forceinline__ __device__ float u32_to_uniform01(uint32_t x) {
    // (0,1), use 24 MSBs
    return ((x >> 8) + 1) * 5.960464477539063e-08f; // 2^-24
}

static __forceinline__ __device__ void rng4_uniform(uint64_t seed, uint32_t row, uint32_t idx4,
                                                    float &r0, float &r1, float &r2, float &r3) {
    uint32_t s0 = (uint32_t)(seed) ^ 0x9e3779b9u;
    uint32_t s1 = (uint32_t)(seed >> 32) ^ 0x85ebca6bu;
    uint32_t key = mix32(s0 ^ mix32(s1) ^ mix32(row));
    uint32_t b = idx4 * 4u;
    uint32_t x0 = mix32(key ^ (b + 0u));
    uint32_t x1 = mix32(key ^ (b + 1u));
    uint32_t x2 = mix32(key ^ (b + 2u));
    uint32_t x3 = mix32(key ^ (b + 3u));
    r0 = u32_to_uniform01(x0);
    r1 = u32_to_uniform01(x1);
    r2 = u32_to_uniform01(x2);
    r3 = u32_to_uniform01(x3);
}

template<int THREADS, bool DO_DROPOUT>
__global__ __launch_bounds__(THREADS, 3)
void softmax_dim1_f32_kernel_vec4(
    const float* __restrict__ x,  // [B, N]
    float* __restrict__ y,        // [B, N]
    int B, int N,
    float dropout_p,
    uint64_t seed
) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float sh_warp[WARPS];
    __shared__ float sh_row; // max then sum

    int row0 = (int)blockIdx.x;
    int stride = (int)gridDim.x;

    const float inv_keep = DO_DROPOUT ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    for (int row = row0; row < B; row += stride) {
        const int tid = (int)threadIdx.x;
        const int lane = tid & 31;
        const int warp = tid >> 5;

        const int base = row * N;
        const float* xrow = x + base;
        float* yrow = y + base;

        // Require vectorization safety (N multiple of 4 for our target; still handle tail safely).
        const int n4 = N >> 2;
        const int tail = N & 3;

        // -----------------------
        // Pass 1: row_max
        // -----------------------
        float local_max = -INFINITY;

        const bool aligned16 = ((((uintptr_t)xrow) & 0xF) == 0) && ((((uintptr_t)yrow) & 0xF) == 0);

        if (aligned16) {
            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xrow);
            #pragma unroll 2
            for (int i4 = tid; i4 < n4; i4 += THREADS) {
                float4 v = __ldg(x4 + i4);
                if constexpr (DO_DROPOUT) {
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, (uint32_t)i4, r0, r1, r2, r3);
                    v.x = (r0 > dropout_p) ? (v.x * inv_keep) : 0.0f;
                    v.y = (r1 > dropout_p) ? (v.y * inv_keep) : 0.0f;
                    v.z = (r2 > dropout_p) ? (v.z * inv_keep) : 0.0f;
                    v.w = (r3 > dropout_p) ? (v.w * inv_keep) : 0.0f;
                }
                local_max = fmaxf(local_max, v.x);
                local_max = fmaxf(local_max, v.y);
                local_max = fmaxf(local_max, v.z);
                local_max = fmaxf(local_max, v.w);
            }
            if (tail) {
                for (int j = (n4 << 2) + tid; j < N; j += THREADS) {
                    float v = __ldg(xrow + j);
                    if constexpr (DO_DROPOUT) {
                        // Derive from idx4 and lane within 4 deterministically
                        uint32_t idx4 = (uint32_t)(j >> 2);
                        float r0, r1, r2, r3;
                        rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                        float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                        v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                    }
                    local_max = fmaxf(local_max, v);
                }
            }
        } else {
            for (int j = tid; j < N; j += THREADS) {
                float v = __ldg(xrow + j);
                if constexpr (DO_DROPOUT) {
                    uint32_t idx4 = (uint32_t)(j >> 2);
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                    float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                    v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                }
                local_max = fmaxf(local_max, v);
            }
        }

        float wmax = warp_reduce_max(local_max);
        if (lane == 0) sh_warp[warp] = wmax;
        __syncthreads();

        if (warp == 0) {
            float v = (tid < WARPS) ? sh_warp[tid] : -INFINITY;
            v = warp_reduce_max(v);
            if (lane == 0) sh_row = v;
        }
        __syncthreads();
        const float row_max = sh_row;

        // -----------------------
        // Pass 2: row_sum
        // -----------------------
        float local_sum = 0.0f;

        if (aligned16) {
            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xrow);
            #pragma unroll 2
            for (int i4 = tid; i4 < n4; i4 += THREADS) {
                float4 v = __ldg(x4 + i4);
                if constexpr (DO_DROPOUT) {
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, (uint32_t)i4, r0, r1, r2, r3);
                    v.x = (r0 > dropout_p) ? (v.x * inv_keep) : 0.0f;
                    v.y = (r1 > dropout_p) ? (v.y * inv_keep) : 0.0f;
                    v.z = (r2 > dropout_p) ? (v.z * inv_keep) : 0.0f;
                    v.w = (r3 > dropout_p) ? (v.w * inv_keep) : 0.0f;
                }
                local_sum += __expf(v.x - row_max);
                local_sum += __expf(v.y - row_max);
                local_sum += __expf(v.z - row_max);
                local_sum += __expf(v.w - row_max);
            }
            if (tail) {
                for (int j = (n4 << 2) + tid; j < N; j += THREADS) {
                    float v = __ldg(xrow + j);
                    if constexpr (DO_DROPOUT) {
                        uint32_t idx4 = (uint32_t)(j >> 2);
                        float r0, r1, r2, r3;
                        rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                        float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                        v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                    }
                    local_sum += __expf(v - row_max);
                }
            }
        } else {
            for (int j = tid; j < N; j += THREADS) {
                float v = __ldg(xrow + j);
                if constexpr (DO_DROPOUT) {
                    uint32_t idx4 = (uint32_t)(j >> 2);
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                    float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                    v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                }
                local_sum += __expf(v - row_max);
            }
        }

        float wsum = warp_reduce_sum(local_sum);
        if (lane == 0) sh_warp[warp] = wsum;
        __syncthreads();

        if (warp == 0) {
            float v = (tid < WARPS) ? sh_warp[tid] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) sh_row = v;
        }
        __syncthreads();
        const float inv_sum = 1.0f / fmaxf(sh_row, 1e-20f);

        // -----------------------
        // Pass 3: write
        // -----------------------
        if (aligned16) {
            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xrow);
            float4* __restrict__ y4 = reinterpret_cast<float4*>(yrow);

            #pragma unroll 2
            for (int i4 = tid; i4 < n4; i4 += THREADS) {
                float4 v = __ldg(x4 + i4);
                if constexpr (DO_DROPOUT) {
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, (uint32_t)i4, r0, r1, r2, r3);
                    v.x = (r0 > dropout_p) ? (v.x * inv_keep) : 0.0f;
                    v.y = (r1 > dropout_p) ? (v.y * inv_keep) : 0.0f;
                    v.z = (r2 > dropout_p) ? (v.z * inv_keep) : 0.0f;
                    v.w = (r3 > dropout_p) ? (v.w * inv_keep) : 0.0f;
                }
                float4 o;
                o.x = __expf(v.x - row_max) * inv_sum;
                o.y = __expf(v.y - row_max) * inv_sum;
                o.z = __expf(v.z - row_max) * inv_sum;
                o.w = __expf(v.w - row_max) * inv_sum;
                y4[i4] = o;
            }
            if (tail) {
                for (int j = (n4 << 2) + tid; j < N; j += THREADS) {
                    float v = __ldg(xrow + j);
                    if constexpr (DO_DROPOUT) {
                        uint32_t idx4 = (uint32_t)(j >> 2);
                        float r0, r1, r2, r3;
                        rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                        float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                        v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                    }
                    yrow[j] = __expf(v - row_max) * inv_sum;
                }
            }
        } else {
            for (int j = tid; j < N; j += THREADS) {
                float v = __ldg(xrow + j);
                if constexpr (DO_DROPOUT) {
                    uint32_t idx4 = (uint32_t)(j >> 2);
                    float r0, r1, r2, r3;
                    rng4_uniform(seed, (uint32_t)row, idx4, r0, r1, r2, r3);
                    float r = ((j & 3) == 0) ? r0 : ((j & 3) == 1) ? r1 : ((j & 3) == 2) ? r2 : r3;
                    v = (r > dropout_p) ? (v * inv_keep) : 0.0f;
                }
                yrow[j] = __expf(v - row_max) * inv_sum;
            }
        }

        __syncthreads(); // keep shared reuse safe before next row iteration
    }
}

torch::Tensor dropout_softmax_dim1_f32_cuda_opt(
    torch::Tensor x,
    double dropout_p,
    uint64_t seed,
    bool training
) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "supports float32 only");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, N]");
    TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must be in [0, 1)");

    const int B = (int)x.size(0);
    const int N = (int)x.size(1);

    auto y = torch::empty_like(x);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    // For large N (16384), 128 threads tends to reduce regs and increase residency vs 256.
    constexpr int THREADS = 128;

    // Overprovision blocks to improve SM utilization when B is smallish.
    int dev = x.get_device();
    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    int blocks = B;
    int target = sm_count * 4;
    if (blocks < target) blocks = target;

    dim3 grid(blocks);
    dim3 block(THREADS);

    if (training && dropout_p > 0.0) {
        softmax_dim1_f32_kernel_vec4<THREADS, true><<<grid, block, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N, (float)dropout_p, seed
        );
    } else {
        softmax_dim1_f32_kernel_vec4<THREADS, false><<<grid, block, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N, 0.0f, 0
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor dropout_softmax_dim1_f32_cuda_opt(torch::Tensor x, double dropout_p, uint64_t seed, bool training);
"""

_ext_name = "custom_ops_lib_matmul_dropout_softmax_v7"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["dropout_softmax_dim1_f32_cuda_opt"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        "--maxrregcount=64",
    ],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    Linear (cuBLAS) -> fused Dropout+Softmax (custom CUDA) for float32 2D contiguous CUDA inputs.
    Falls back to PyTorch ops otherwise.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = float(dropout_p)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.matmul.weight
        b = self.matmul.bias

        if (
            x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous()
            and w.is_cuda and w.dtype == torch.float32 and w.dim() == 2 and w.is_contiguous()
            and (b is None or (b.is_cuda and b.dtype == torch.float32 and b.dim() == 1 and b.is_contiguous()))
        ):
            z = F.linear(x, w, b)
            if self.training and self.dropout_p > 0.0:
                seed = int(torch.empty((), device=x.device, dtype=torch.int64).random_().item())
            else:
                seed = 0
            return self.custom_ops_lib.dropout_softmax_dim1_f32_cuda_opt(
                z, self.dropout_p, seed, bool(self.training)
            )

        y = self.matmul(x)
        y = F.dropout(y, p=self.dropout_p, training=self.training)
        return torch.softmax(y, dim=1)