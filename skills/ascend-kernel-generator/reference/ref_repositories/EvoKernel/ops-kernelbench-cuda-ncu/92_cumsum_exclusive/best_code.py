import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static __device__ __forceinline__ float warp_inclusive_scan(float v, unsigned mask=0xffffffffu) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, v, offset);
        if ((int)(threadIdx.x & 31) >= offset) v += n;
    }
    return v;
}
static __device__ __forceinline__ float warp_exclusive_scan(float v, unsigned mask=0xffffffffu) {
    float incl = warp_inclusive_scan(v, mask);
    return incl - v;
}

static __device__ __forceinline__ float4 ld4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
static __device__ __forceinline__ void st4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

/*
  Hot-path kernel: [B, 32768], exclusive cumsum along dim=1.
  - One block per row.
  - 256 threads, each handles float4 => 1024 elems/tile, 32 tiles total.
  - Double-buffer prefetch of next tile.
  - Carry update broadcast via warp shuffle (no shared tile_sum + extra barriers).
*/
__global__ __launch_bounds__(256, 2)
void cumsum_exclusive_dim1_N32768_vec4_prefetch_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B
) {
    constexpr int THREADS = 256;
    constexpr int VEC = 4;
    constexpr int N = 32768;
    constexpr int TILE = THREADS * VEC;  // 1024
    constexpr int NTILES = N / TILE;     // 32

    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int NWARPS = THREADS / 32;

    __shared__ float warp_sums[NWARPS];
    __shared__ float warp_prefix[NWARPS];

    const float* row = x + (size_t)b * (size_t)N;
    float* orow = y + (size_t)b * (size_t)N;

    // Strict vectorized fast path only when 16B aligned for both pointers.
    // (Tensor allocations are typically aligned, so this should usually hit.)
    if ((((uintptr_t)row | (uintptr_t)orow) & 0xF) != 0) {
        // Fallback: scalar per-thread for correctness (still uses same scan skeleton).
        float carry = 0.0f;
        #pragma unroll 1
        for (int t = 0; t < NTILES; ++t) {
            int base = t * TILE;
            int j0 = base + tid * VEC;

            float v0 = row[j0 + 0];
            float v1 = row[j0 + 1];
            float v2 = row[j0 + 2];
            float v3 = row[j0 + 3];

            float p0 = 0.f;
            float p1 = v0;
            float p2 = v0 + v1;
            float p3 = v0 + v1 + v2;
            float thread_sum = v0 + v1 + v2 + v3;

            float w_excl = warp_exclusive_scan(thread_sum);
            float w_incl = w_excl + thread_sum;
            float w_total = __shfl_sync(0xffffffffu, w_incl, 31);

            if (lane == 31) warp_sums[warp] = w_total;
            __syncthreads();

            if (warp == 0) {
                float wv = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
                float wincl2 = warp_inclusive_scan(wv);
                if (lane < NWARPS) warp_prefix[lane] = wincl2 - wv;
            }
            __syncthreads();

            float block_excl = warp_prefix[warp] + w_excl;
            float base_out = carry + block_excl;

            orow[j0 + 0] = base_out + p0;
            orow[j0 + 1] = base_out + p1;
            orow[j0 + 2] = base_out + p2;
            orow[j0 + 3] = base_out + p3;

            // tile sum: last warp prefix + last warp total
            float tile_sum = warp_prefix[NWARPS - 1] + warp_sums[NWARPS - 1];

            // Broadcast tile_sum from lane 0 of warp 0 to all threads via warp shuffle,
            // then cross-warp via shared (single barrier already exists next iter).
            // Here we keep it simple with one __syncthreads already paid above.
            if (warp == 0) {
                // lane 0 holds tile_sum; broadcast within warp0
                float v = (lane == 0) ? tile_sum : 0.0f;
                v = __shfl_sync(0xffffffffu, v, 0);
                if (lane == 0) warp_sums[0] = v;
            }
            __syncthreads();
            carry += warp_sums[0];
            __syncthreads();
        }
        return;
    }

    // Aligned vectorized + prefetch path
    float carry = 0.0f;

    // Prefetch tile 0
    int j0_pref = tid * VEC;
    float4 v_next = ld4(row + j0_pref);

    #pragma unroll 1
    for (int t = 0; t < NTILES; ++t) {
        // Current tile values from prefetch
        float4 v = v_next;

        // Prefetch next tile early (except last)
        if (t + 1 < NTILES) {
            int base_next = (t + 1) * TILE;
            int j_next = base_next + tid * VEC;
            v_next = ld4(row + j_next);
        }

        // local exclusive prefixes within float4
        float p0 = 0.f;
        float p1 = v.x;
        float p2 = v.x + v.y;
        float p3 = v.x + v.y + v.z;
        float thread_sum = v.x + v.y + v.z + v.w;

        // block exclusive scan over thread_sum
        float w_excl = warp_exclusive_scan(thread_sum);
        float w_incl = w_excl + thread_sum;
        float w_total = __shfl_sync(0xffffffffu, w_incl, 31);

        if (lane == 31) warp_sums[warp] = w_total;
        __syncthreads();

        if (warp == 0) {
            float wv = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
            float wincl2 = warp_inclusive_scan(wv);
            if (lane < NWARPS) warp_prefix[lane] = wincl2 - wv;
        }
        __syncthreads();

        float block_excl = warp_prefix[warp] + w_excl;
        float base_out = carry + block_excl;

        float4 outv;
        outv.x = base_out + p0;
        outv.y = base_out + p1;
        outv.z = base_out + p2;
        outv.w = base_out + p3;

        int base = t * TILE;
        int j0 = base + tid * VEC;
        st4(orow + j0, outv);

        // Compute tile sum (same for all threads) and update carry with minimal overhead.
        float tile_sum = warp_prefix[NWARPS - 1] + warp_sums[NWARPS - 1];

        // Broadcast tile_sum to all threads:
        // 1) lane0 of warp0 loads tile_sum into a register and broadcasts within warp0
        // 2) lane0 writes to shared, single sync to make visible to all warps
        if (warp == 0) {
            float vts = (lane == 0) ? tile_sum : 0.0f;
            vts = __shfl_sync(0xffffffffu, vts, 0);
            if (lane == 0) warp_sums[0] = vts;
        }
        __syncthreads();
        carry += warp_sums[0];
        __syncthreads();
    }
}

// General kernel from baseline: persistent-CTA per row, VEC in {1,2,4}.
template<int VEC, int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void cumsum_exclusive_dim1_row_persistent_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int N
) {
    constexpr int WARP = 32;
    int tid  = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp = tid >> 5;
    int nwarps = THREADS / WARP;

    __shared__ float warp_sums[32];
    __shared__ float warp_prefix[32];
    __shared__ float tile_sum_sh;

    int b = (int)blockIdx.x;
    if (b >= B) return;

    const float* row = x + (size_t)b * (size_t)N;
    float* orow = y + (size_t)b * (size_t)N;

    const int tile_elems = THREADS * VEC;
    int nt = (N + tile_elems - 1) / tile_elems;

    float carry = 0.0f;

    for (int t = 0; t < nt; ++t) {
        int base = t * tile_elems;
        int j0 = base + tid * VEC;

        float v0=0.f, v1=0.f, v2=0.f, v3=0.f;

        if constexpr (VEC == 4) {
            bool inb = (j0 + 3) < N;
            bool aligned16 = (((uintptr_t)(row + j0) & 0xF) == 0);
            if (inb && aligned16) {
                float4 vv = ld4(row + j0);
                v0 = vv.x; v1 = vv.y; v2 = vv.z; v3 = vv.w;
            } else {
                if (j0 + 0 < N) v0 = row[j0 + 0];
                if (j0 + 1 < N) v1 = row[j0 + 1];
                if (j0 + 2 < N) v2 = row[j0 + 2];
                if (j0 + 3 < N) v3 = row[j0 + 3];
            }
        } else if constexpr (VEC == 2) {
            bool inb = (j0 + 1) < N;
            bool aligned8 = (((uintptr_t)(row + j0) & 0x7) == 0);
            if (inb && aligned8) {
                float2 vv = *reinterpret_cast<const float2*>(row + j0);
                v0 = vv.x; v1 = vv.y;
            } else {
                if (j0 + 0 < N) v0 = row[j0 + 0];
                if (j0 + 1 < N) v1 = row[j0 + 1];
            }
        } else {
#if __CUDA_ARCH__ >= 350
            if (j0 < N) v0 = __ldg(row + j0);
#else
            if (j0 < N) v0 = row[j0];
#endif
        }

        float thread_sum;
        float p0, p1, p2, p3;
        if constexpr (VEC == 4) {
            p0 = 0.f;
            p1 = v0;
            p2 = v0 + v1;
            p3 = v0 + v1 + v2;
            thread_sum = v0 + v1 + v2 + v3;
        } else if constexpr (VEC == 2) {
            p0 = 0.f;
            p1 = v0;
            thread_sum = v0 + v1;
        } else {
            p0 = 0.f;
            thread_sum = v0;
        }

        float warp_excl = warp_exclusive_scan(thread_sum);
        float warp_incl = warp_excl + thread_sum;
        float warp_total = __shfl_sync(0xffffffffu, warp_incl, 31);

        if (lane == 31) warp_sums[warp] = warp_total;
        __syncthreads();

        if (warp == 0) {
            float wv = (lane < nwarps) ? warp_sums[lane] : 0.0f;
            float wincl = warp_inclusive_scan(wv);
            if (lane < nwarps) warp_prefix[lane] = wincl - wv;
        }
        __syncthreads();

        float block_excl = warp_prefix[warp] + warp_excl;
        float base_out = carry + block_excl;

        if constexpr (VEC == 4) {
            bool inb = (j0 + 3) < N;
            bool aligned16 = (((uintptr_t)(orow + j0) & 0xF) == 0);
            if (inb && aligned16) {
                float4 oo;
                oo.x = base_out + p0;
                oo.y = base_out + p1;
                oo.z = base_out + p2;
                oo.w = base_out + p3;
                st4(orow + j0, oo);
            } else {
                if (j0 + 0 < N) orow[j0 + 0] = base_out + p0;
                if (j0 + 1 < N) orow[j0 + 1] = base_out + p1;
                if (j0 + 2 < N) orow[j0 + 2] = base_out + p2;
                if (j0 + 3 < N) orow[j0 + 3] = base_out + p3;
            }
        } else if constexpr (VEC == 2) {
            bool inb = (j0 + 1) < N;
            bool aligned8 = (((uintptr_t)(orow + j0) & 0x7) == 0);
            if (inb && aligned8) {
                float2 oo;
                oo.x = base_out + p0;
                oo.y = base_out + p1;
                *reinterpret_cast<float2*>(orow + j0) = oo;
            } else {
                if (j0 + 0 < N) orow[j0 + 0] = base_out + p0;
                if (j0 + 1 < N) orow[j0 + 1] = base_out + p1;
            }
        } else {
            if (j0 < N) orow[j0] = base_out + p0;
        }

        if (tid == 0) {
            float tsum = warp_prefix[nwarps - 1] + warp_sums[nwarps - 1];
            tile_sum_sh = tsum;
        }
        __syncthreads();
        carry += tile_sum_sh;
        __syncthreads();
    }
}

torch::Tensor cumsum_exclusive_dim1_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [B, N]");

    const int64_t B64 = x.size(0);
    const int64_t N64 = x.size(1);
    auto y = torch::empty_like(x);
    if (B64 == 0 || N64 == 0) return y;

    TORCH_CHECK(B64 <= INT_MAX && N64 <= INT_MAX, "tensor too large");
    int B = (int)B64;
    int N = (int)N64;

    dim3 grid(B);

    // Hot specialization for benchmark shape
    if (N == 32768) {
        dim3 block(256);
        cumsum_exclusive_dim1_N32768_vec4_prefetch_f32<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B
        );
        return y;
    }

    // General fallback
    constexpr int THREADS = 128;
    dim3 block(THREADS);

    if (N >= 512) {
        cumsum_exclusive_dim1_row_persistent_f32<4, THREADS><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N
        );
    } else if (N >= 128) {
        cumsum_exclusive_dim1_row_persistent_f32<2, THREADS><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N
        );
    } else {
        cumsum_exclusive_dim1_row_persistent_f32<1, THREADS><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N
        );
    }
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor cumsum_exclusive_dim1_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_cumsum_exclusive_ext_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["cumsum_exclusive_dim1_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replacement model using a custom CUDA kernel for exclusive cumulative sum.

    Fast path:
      - dim == 1
      - x: contiguous CUDA float32 tensor of shape [B, N]
    Fallback: original PyTorch expression.
    """
    def __init__(self, dim: int):
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
            return self.custom_ops_lib.cumsum_exclusive_dim1_cuda(x)

        cumsum = torch.cumsum(
            x.narrow(dim=self.dim, start=0, length=x.size(self.dim) - 1),
            dim=self.dim,
        )
        return torch.cat(
            (torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum),
            dim=self.dim,
        )