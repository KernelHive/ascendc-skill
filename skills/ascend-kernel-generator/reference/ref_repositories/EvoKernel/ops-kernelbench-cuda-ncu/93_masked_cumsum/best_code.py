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
#ifndef CHECK_BOOL
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be bool")
#endif

static __device__ __forceinline__ float warp_exclusive_sum(float v, unsigned mask=0xFFFFFFFFu) {
    // Exclusive scan: returns sum of lanes < lane_id.
    float x = v;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(mask, x, offset);
        if ((int)(threadIdx.x & 31) >= offset) x += y;
    }
    return x - v;
}

template<int BLOCK_THREADS, int VEC>
__global__ __launch_bounds__(BLOCK_THREADS, 3)
void masked_cumsum_dim1_warp_f32_u8_vec(
    const float* __restrict__ x,
    const uint8_t* __restrict__ m,
    float* __restrict__ y,
    int B, int N
) {
    constexpr int WARPS = BLOCK_THREADS / 32;
    static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of warp");

    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const int row_off = b * N;

    // Very small shared state: warp totals and warp-prefix offsets
    __shared__ float warp_totals[WARPS];
    __shared__ float warp_offsets[WARPS];

    float carry = 0.0f;

    const int tile_elems = BLOCK_THREADS * VEC; // elements per tile

    for (int base = 0; base < N; base += tile_elems) {
        int i0 = base + tid * VEC;

        // Load VEC masked values
        float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
        if constexpr (VEC == 4) {
            if (i0 + 3 < N) {
                // vector load
                uchar4 mm = *reinterpret_cast<const uchar4*>(m + row_off + i0);
                float4 xx = *reinterpret_cast<const float4*>(x + row_off + i0);
                v0 = mm.x ? xx.x : 0.f;
                v1 = mm.y ? xx.y : 0.f;
                v2 = mm.z ? xx.z : 0.f;
                v3 = mm.w ? xx.w : 0.f;
            } else {
                // tail
                if (i0 + 0 < N) { float xv = x[row_off + i0 + 0]; uint8_t mk = m[row_off + i0 + 0]; v0 = mk ? xv : 0.f; }
                if (i0 + 1 < N) { float xv = x[row_off + i0 + 1]; uint8_t mk = m[row_off + i0 + 1]; v1 = mk ? xv : 0.f; }
                if (i0 + 2 < N) { float xv = x[row_off + i0 + 2]; uint8_t mk = m[row_off + i0 + 2]; v2 = mk ? xv : 0.f; }
                if (i0 + 3 < N) { float xv = x[row_off + i0 + 3]; uint8_t mk = m[row_off + i0 + 3]; v3 = mk ? xv : 0.f; }
            }

            // in-thread inclusive prefix
            float p0 = v0;
            float p1 = p0 + v1;
            float p2 = p1 + v2;
            float p3 = p2 + v3;
            float thread_sum = p3;

            // warp exclusive of thread sums
            float warp_excl = warp_exclusive_sum(thread_sum);

            // warp total (last lane writes)
            float warp_total = warp_excl + thread_sum;
            if (lane == 31) warp_totals[warp] = warp_total;

            __syncthreads();

            // warp0 computes prefix over warp totals (WARPS <= 8 for 256 threads)
            if (warp == 0) {
                float wt = (lane < WARPS) ? warp_totals[lane] : 0.f;
                float wex = warp_exclusive_sum(wt);
                if (lane < WARPS) warp_offsets[lane] = wex;
            }

            __syncthreads();

            float add = carry + warp_offsets[warp] + warp_excl;

            // store outputs
            if (i0 + 3 < N) {
                float4 out;
                out.x = add + p0;
                out.y = add + p1;
                out.z = add + p2;
                out.w = add + p3;
                *reinterpret_cast<float4*>(y + row_off + i0) = out;
            } else {
                if (i0 + 0 < N) y[row_off + i0 + 0] = add + p0;
                if (i0 + 1 < N) y[row_off + i0 + 1] = add + p1;
                if (i0 + 2 < N) y[row_off + i0 + 2] = add + p2;
                if (i0 + 3 < N) y[row_off + i0 + 3] = add + p3;
            }

            // update carry by tile total (sum of warp totals)
            // warp0 lane WARPS-1 has prefix excluding itself; easiest: sum warp_totals in warp0 lanes
            if (warp == 0) {
                float wt = (lane < WARPS) ? warp_totals[lane] : 0.f;
                // reduce within warp0
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    wt += __shfl_down_sync(0xFFFFFFFFu, wt, offset);
                }
                if (lane == 0) warp_totals[0] = wt;
            }
            __syncthreads();
            carry += warp_totals[0];
            __syncthreads();
        } else {
            // VEC==1 scalar path (still warp-scan)
            float v = 0.0f;
            if (i0 < N) {
                float xv = x[row_off + i0];
                uint8_t mk = m[row_off + i0];
                v = mk ? xv : 0.0f;
            }

            float warp_excl = warp_exclusive_sum(v);
            float warp_total = warp_excl + v;
            if (lane == 31) warp_totals[warp] = warp_total;
            __syncthreads();

            if (warp == 0) {
                float wt = (lane < WARPS) ? warp_totals[lane] : 0.f;
                float wex = warp_exclusive_sum(wt);
                if (lane < WARPS) warp_offsets[lane] = wex;
            }
            __syncthreads();

            float out = carry + warp_offsets[warp] + warp_excl + v;
            if (i0 < N) y[row_off + i0] = out;

            if (warp == 0) {
                float wt = (lane < WARPS) ? warp_totals[lane] : 0.f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    wt += __shfl_down_sync(0xFFFFFFFFu, wt, offset);
                }
                if (lane == 0) warp_totals[0] = wt;
            }
            __syncthreads();
            carry += warp_totals[0];
            __syncthreads();
        }
    }
}

torch::Tensor masked_cumsum_dim1_cuda(torch::Tensor x, torch::Tensor mask) {
    CHECK_CUDA(x);
    CHECK_CUDA(mask);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(mask);
    CHECK_FLOAT(x);
    CHECK_BOOL(mask);

    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [B, N]");
    TORCH_CHECK(mask.dim() == 2, "mask must be a 2D tensor [B, N]");
    TORCH_CHECK(x.sizes() == mask.sizes(), "mask must have the same shape as x");

    int64_t B64 = x.size(0);
    int64_t N64 = x.size(1);
    TORCH_CHECK(B64 >= 0 && N64 >= 0, "Invalid tensor sizes");

    int B = (int)B64;
    int N = (int)N64;

    auto y = torch::empty_like(x);
    if (B == 0 || N == 0) return y;

    dim3 grid(B);
    constexpr int BLOCK = 256;

    const uintptr_t x_addr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t y_addr = (uintptr_t)y.data_ptr<float>();
    const uintptr_t m_addr = (uintptr_t)mask.data_ptr<bool>();
    bool vec_ok = ((x_addr | y_addr) & 0xF) == 0 && (m_addr & 0x3) == 0;

    const uint8_t* mbytes = reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());

    if (vec_ok && N >= 128) {
        masked_cumsum_dim1_warp_f32_u8_vec<BLOCK, 4><<<grid, BLOCK, 0>>>(
            x.data_ptr<float>(), mbytes, y.data_ptr<float>(), B, N
        );
    } else {
        masked_cumsum_dim1_warp_f32_u8_vec<BLOCK, 1><<<grid, BLOCK, 0>>>(
            x.data_ptr<float>(), mbytes, y.data_ptr<float>(), B, N
        );
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor masked_cumsum_dim1_cuda(torch::Tensor x, torch::Tensor mask);
"""

custom_ops_lib = load_inline(
    name="custom_masked_cumsum_dim1_ext_warp_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["masked_cumsum_dim1_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized masked cumsum along dim=1 for 2D contiguous CUDA tensors:
      x: float32 [B, N]
      mask: bool [B, N]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.cumsum(x * mask, dim=self.dim)

        if (
            (not x.is_cuda) or (not mask.is_cuda) or
            x.dtype != torch.float32 or mask.dtype != torch.bool or
            x.dim() != 2 or mask.dim() != 2 or
            (not x.is_contiguous()) or (not mask.is_contiguous()) or
            (x.shape != mask.shape)
        ):
            return torch.cumsum(x * mask, dim=self.dim)

        return self.custom_ops_lib.masked_cumsum_dim1_cuda(x, mask)