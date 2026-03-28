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
static __forceinline__ __device__ float block_reduce_max_shfl(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_buf[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float w = warp_reduce_max(v);
    if (lane == 0) warp_buf[warp] = w;
    __syncthreads();
    float out = -INFINITY;
    if (warp == 0) {
        float t = (tid < WARPS) ? warp_buf[tid] : -INFINITY;
        t = warp_reduce_max(t);
        if (lane == 0) warp_buf[0] = t;
    }
    __syncthreads();
    out = warp_buf[0];
    return out;
}

template<int THREADS>
static __forceinline__ __device__ float block_reduce_sum_shfl(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_buf[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float w = warp_reduce_sum(v);
    if (lane == 0) warp_buf[warp] = w;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        float t = (tid < WARPS) ? warp_buf[tid] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) warp_buf[0] = t;
    }
    __syncthreads();
    out = warp_buf[0];
    return out;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_pipeline.h>
#endif

template<int THREADS, int TILE_FLOATS, bool USE_CP_ASYNC>
__global__ __launch_bounds__(THREADS, 3)
void softmax_fused_row_f32_kernel(
    const float* __restrict__ x, // [B, D]
    float* __restrict__ y,       // [B, D]
    int B, int D
) {
    int row = (int)blockIdx.x;
    if (row >= B) return;

    int tid = (int)threadIdx.x;
    const float* xrow = x + (int64_t)row * D;
    float* yrow = y + (int64_t)row * D;

    // Two shared buffers for double-buffered tiles (even if not using cp.async, keeps code simple).
    __align__(16) extern __shared__ float shmem[];
    float* sh0 = shmem;
    float* sh1 = sh0 + TILE_FLOATS;

    auto load_tile_scalar = [&](float* dst, int off, int n) {
        // vectorized float4 when aligned and n multiple-friendly; otherwise scalar.
        bool aligned16 = ((((uintptr_t)(xrow + off)) & 0xF) == 0) && ((((uintptr_t)dst) & 0xF) == 0);
        int vec4 = n >> 2;
        int tail = n & 3;
        if (aligned16 && vec4 > 0) {
            const float4* __restrict__ src4 = reinterpret_cast<const float4*>(xrow + off);
            float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst);
#pragma unroll 2
            for (int i = tid; i < vec4; i += THREADS) {
                // ldg through float4 pointer: fine; compiler will emit vector loads
                float4 v = src4[i];
                dst4[i] = v;
            }
        } else {
            for (int i = tid; i < n; i += THREADS) dst[i] = ldg_f32(xrow + off + i);
        }
        if (tail) {
            int start = (vec4 << 2);
            for (int i = start + tid; i < n; i += THREADS) dst[i] = ldg_f32(xrow + off + i);
        }
    };

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    auto load_tile_cp_async = [&](float* dst, int off, int n) {
        // cp.async supports 16B; we handle main body in float4 and tail scalar after wait.
        int vec4 = n >> 2;
        const float4* __restrict__ src4 = reinterpret_cast<const float4*>(xrow + off);
        float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst);
#pragma unroll 2
        for (int i = tid; i < vec4; i += THREADS) {
            __pipeline_memcpy_async(dst4 + i, src4 + i, sizeof(float4));
        }
        __pipeline_commit();
    };
#endif

    int num_tiles = (D + TILE_FLOATS - 1) / TILE_FLOATS;

    // -------------------------
    // Pass 1: compute row_max and row_sum (stable online merge) while streaming tiles.
    // Optional cp.async prefetch for next tile to overlap memory with compute.
    // -------------------------
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Prefetch tile 0
    int off0 = 0;
    int n0 = D;
    if (n0 > TILE_FLOATS) n0 = TILE_FLOATS;

    if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        // Require 16B alignment for cp.async correctness/perf; if misaligned, behavior is still defined but can be slower.
        load_tile_cp_async(sh0, off0, n0);
        __pipeline_wait_prior(0);
        __syncthreads();
        // tail scalar (rare when n0 % 4 != 0)
        if (n0 & 3) {
            int start = (n0 & ~3);
            for (int i = start + tid; i < n0; i += THREADS) sh0[i] = ldg_f32(xrow + off0 + i);
        }
#else
        load_tile_scalar(sh0, off0, n0);
        __syncthreads();
#endif
    } else {
        load_tile_scalar(sh0, off0, n0);
        __syncthreads();
    }

    for (int t = 0; t < num_tiles; ++t) {
        int off = t * TILE_FLOATS;
        int n = D - off;
        if (n > TILE_FLOATS) n = TILE_FLOATS;

        float* cur = (t & 1) ? sh1 : sh0;
        float* nxt = (t & 1) ? sh0 : sh1;

        // Prefetch next tile early (overlap with reductions/exp of current tile).
        if (t + 1 < num_tiles) {
            int offn = (t + 1) * TILE_FLOATS;
            int nn = D - offn;
            if (nn > TILE_FLOATS) nn = TILE_FLOATS;
            if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                load_tile_cp_async(nxt, offn, nn);
#endif
            } else {
                // no async path: do nothing now; we'll load after finishing current tile
            }
        }

        // Compute tile max with ILP: process float4 when aligned.
        float tmax = -INFINITY;
        bool sh_aligned16 = ((((uintptr_t)cur) & 0xF) == 0);
        int vec4 = n >> 2;
        int tail = n & 3;

        if (sh_aligned16 && vec4 > 0) {
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(cur);
#pragma unroll 2
            for (int i = tid; i < vec4; i += THREADS) {
                float4 v = p4[i];
                tmax = fmaxf(tmax, v.x);
                tmax = fmaxf(tmax, v.y);
                tmax = fmaxf(tmax, v.z);
                tmax = fmaxf(tmax, v.w);
            }
        } else {
            for (int i = tid; i < n; i += THREADS) tmax = fmaxf(tmax, cur[i]);
        }
        if (tail) {
            int start = (vec4 << 2);
            for (int i = start + tid; i < n; i += THREADS) tmax = fmaxf(tmax, cur[i]);
        }

        tmax = block_reduce_max_shfl<THREADS>(tmax);

        // Compute tile sumexp wrt tile max.
        float tsum = 0.0f;
        if (sh_aligned16 && vec4 > 0) {
            const float4* __restrict__ p4 = reinterpret_cast<const float4*>(cur);
#pragma unroll 2
            for (int i = tid; i < vec4; i += THREADS) {
                float4 v = p4[i];
                tsum += __expf(v.x - tmax);
                tsum += __expf(v.y - tmax);
                tsum += __expf(v.z - tmax);
                tsum += __expf(v.w - tmax);
            }
        } else {
            for (int i = tid; i < n; i += THREADS) tsum += __expf(cur[i] - tmax);
        }
        if (tail) {
            int start = (vec4 << 2);
            for (int i = start + tid; i < n; i += THREADS) tsum += __expf(cur[i] - tmax);
        }

        tsum = block_reduce_sum_shfl<THREADS>(tsum);

        // Online stable merge (row_max, row_sum) with (tmax, tsum).
        float new_max = fmaxf(row_max, tmax);
        float merged_sum;
        if (row_max == -INFINITY) {
            merged_sum = tsum;
            new_max = tmax;
        } else {
            merged_sum = row_sum * __expf(row_max - new_max) + tsum * __expf(tmax - new_max);
        }
        row_max = new_max;
        row_sum = merged_sum;

        // Finish prefetch if cp.async; or load next tile now if not using async.
        if (t + 1 < num_tiles) {
            int offn = (t + 1) * TILE_FLOATS;
            int nn = D - offn;
            if (nn > TILE_FLOATS) nn = TILE_FLOATS;

            if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                __pipeline_wait_prior(0);
                __syncthreads();
                if (nn & 3) {
                    int start = (nn & ~3);
                    for (int i = start + tid; i < nn; i += THREADS) nxt[i] = ldg_f32(xrow + offn + i);
                }
                __syncthreads();
#endif
            } else {
                // Synchronous load of next tile into nxt.
                load_tile_scalar(nxt, offn, nn);
                __syncthreads();
            }
        }
        __syncthreads();
    }

    float inv = 1.0f / fmaxf(row_sum, 1e-20f);

    // -------------------------
    // Pass 2: write outputs, streaming x again (fundamental), optional cp.async for overlap.
    // -------------------------
    // Prefetch tile 0 again
    if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        load_tile_cp_async(sh0, 0, n0);
        __pipeline_wait_prior(0);
        __syncthreads();
        if (n0 & 3) {
            int start = (n0 & ~3);
            for (int i = start + tid; i < n0; i += THREADS) sh0[i] = ldg_f32(xrow + i);
        }
#else
        load_tile_scalar(sh0, 0, n0);
        __syncthreads();
#endif
    } else {
        load_tile_scalar(sh0, 0, n0);
        __syncthreads();
    }

    for (int t = 0; t < num_tiles; ++t) {
        int off = t * TILE_FLOATS;
        int n = D - off;
        if (n > TILE_FLOATS) n = TILE_FLOATS;

        float* cur = (t & 1) ? sh1 : sh0;
        float* nxt = (t & 1) ? sh0 : sh1;

        // Prefetch next tile
        if (t + 1 < num_tiles) {
            int offn = (t + 1) * TILE_FLOATS;
            int nn = D - offn;
            if (nn > TILE_FLOATS) nn = TILE_FLOATS;
            if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                load_tile_cp_async(nxt, offn, nn);
#endif
            }
        }

        // Compute & store output (vectorized when aligned).
        bool sh_aligned16 = ((((uintptr_t)cur) & 0xF) == 0);
        bool y_aligned16  = ((((uintptr_t)(yrow + off)) & 0xF) == 0);
        int vec4 = n >> 2;
        int tail = n & 3;

        if (sh_aligned16 && y_aligned16 && vec4 > 0) {
            const float4* __restrict__ in4 = reinterpret_cast<const float4*>(cur);
            float4* __restrict__ out4 = reinterpret_cast<float4*>(yrow + off);
#pragma unroll 2
            for (int i = tid; i < vec4; i += THREADS) {
                float4 v = in4[i];
                float4 o;
                o.x = __expf(v.x - row_max) * inv;
                o.y = __expf(v.y - row_max) * inv;
                o.z = __expf(v.z - row_max) * inv;
                o.w = __expf(v.w - row_max) * inv;
                out4[i] = o;
            }
        } else {
            for (int i = tid; i < n; i += THREADS) {
                yrow[off + i] = __expf(cur[i] - row_max) * inv;
            }
        }
        if (tail) {
            int start = (vec4 << 2);
            for (int i = start + tid; i < n; i += THREADS) {
                yrow[off + i] = __expf(cur[i] - row_max) * inv;
            }
        }

        // finalize next prefetch or do sync load for non-async
        if (t + 1 < num_tiles) {
            int offn = (t + 1) * TILE_FLOATS;
            int nn = D - offn;
            if (nn > TILE_FLOATS) nn = TILE_FLOATS;

            if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                __pipeline_wait_prior(0);
                __syncthreads();
                if (nn & 3) {
                    int start = (nn & ~3);
                    for (int i = start + tid; i < nn; i += THREADS) nxt[i] = ldg_f32(xrow + offn + i);
                }
                __syncthreads();
#endif
            } else {
                load_tile_scalar(nxt, offn, nn);
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

torch::Tensor softmax_dim1_f32_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == at::kFloat, "softmax_dim1_f32_cuda only supports float32");
    TORCH_CHECK(x.dim() == 2, "softmax_dim1_f32_cuda expects a 2D tensor");

    const int B = (int)x.size(0);
    const int D = (int)x.size(1);

    auto y = torch::empty_like(x);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    int dev = x.get_device();
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    bool sm80plus = (prop.major > 8) || (prop.major == 8 && prop.minor >= 0);

    // Use 128 threads for huge D to reduce registers and improve residency; otherwise 256.
    // TILE_FLOATS chosen to keep shared memory low: 2048 floats = 8KB per buffer, double-buffer = 16KB.
    constexpr int TILE = 2048;

    if (D >= (1 << 18)) {
        constexpr int THREADS = 128;
        dim3 blocks(B);
        dim3 threads(THREADS);
        size_t shmem = (size_t)(2 * TILE) * sizeof(float); // sh0 + sh1
        if (sm80plus) {
            softmax_fused_row_f32_kernel<THREADS, TILE, true><<<blocks, threads, shmem, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                B, D
            );
        } else {
            softmax_fused_row_f32_kernel<THREADS, TILE, false><<<blocks, threads, shmem, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                B, D
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        constexpr int THREADS = 256;
        dim3 blocks(B);
        dim3 threads(THREADS);
        size_t shmem = (size_t)(2 * TILE) * sizeof(float);
        // For smaller D, async gains are smaller; still enable on sm80+.
        if (sm80plus) {
            softmax_fused_row_f32_kernel<THREADS, TILE, true><<<blocks, threads, shmem, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                B, D
            );
        } else {
            softmax_fused_row_f32_kernel<THREADS, TILE, false><<<blocks, threads, shmem, stream>>>(
                (const float*)x.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                B, D
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor softmax_dim1_f32_cuda(torch::Tensor x);
"""

_ext_name = "custom_ops_lib_softmax_fused_cpasync_v1"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_dim1_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        # keep some pressure control; not too low to avoid spills
        "--maxrregcount=120",
    ],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Softmax model using an optimized custom CUDA kernel (dim=1, float32, 2D contiguous).
    Falls back to torch.softmax for unsupported cases.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous():
            return self.custom_ops_lib.softmax_dim1_f32_cuda(x)
        return torch.softmax(x, dim=1)