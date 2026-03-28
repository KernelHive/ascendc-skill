import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA: optimized reverse-cumsum for 1D (and optional 2D dim=1) float32 contiguous CUDA tensors.
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

static __device__ __forceinline__ float4 load_rev4(const float* __restrict__ x, int N, int r_base) {
    // Load values for r = r_base..r_base+3 (increasing r) => j = N-1-r (decreasing j).
    // When all in-bounds and aligned, can load x[j3..j0] as float4 and reverse lanes.
    float4 out;
    out.x = out.y = out.z = out.w = 0.0f;
    int r3 = r_base + 3;
    if (r3 < N) {
        int j0 = (N - 1) - (r_base + 0);
        int j3 = (N - 1) - (r_base + 3); // j3 = j0 - 3
        // x[j3..j3+3] corresponds to [r3,r2,r1,r0] in that order.
        float4 vv = *reinterpret_cast<const float4*>(x + j3);
        // Map back to r0..r3 order:
        out.x = vv.w; // r0
        out.y = vv.z; // r1
        out.z = vv.y; // r2
        out.w = vv.x; // r3
    } else {
        // tail-safe scalar loads
        if (r_base + 0 < N) out.x = x[(N - 1) - (r_base + 0)];
        if (r_base + 1 < N) out.y = x[(N - 1) - (r_base + 1)];
        if (r_base + 2 < N) out.z = x[(N - 1) - (r_base + 2)];
        if (r_base + 3 < N) out.w = x[(N - 1) - (r_base + 3)];
    }
    return out;
}

static __device__ __forceinline__ void store_rev4(float* __restrict__ out, int N, int r_base, const float4& s) {
    // Store scan results for r=r_base..r_base+3 in original order at j=N-1-r.
    int r3 = r_base + 3;
    if (r3 < N) {
        int j0 = (N - 1) - (r_base + 0);
        int j3 = (N - 1) - (r_base + 3); // j3 = j0 - 3
        // We want to store to out[j3..j0] == [r3,r2,r1,r0] = [s.w,s.z,s.y,s.x]
        float4 vv;
        vv.x = s.w;
        vv.y = s.z;
        vv.z = s.y;
        vv.w = s.x;
        *reinterpret_cast<float4*>(out + j3) = vv;
    } else {
        if (r_base + 0 < N) out[(N - 1) - (r_base + 0)] = s.x;
        if (r_base + 1 < N) out[(N - 1) - (r_base + 1)] = s.y;
        if (r_base + 2 < N) out[(N - 1) - (r_base + 2)] = s.z;
        if (r_base + 3 < N) out[(N - 1) - (r_base + 3)] = s.w;
    }
}

template<int NTILES_FIXED> // -1 means dynamic loop
__global__ __launch_bounds__(256, 2)
void cumsum_reverse_1d_vec4_warpscan_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N
) {
    // One block processes the full vector in sequential tiles; tiles are large (1024 elts) so overhead is small.
    // Tile size in elements: blockDim.x * 4
    constexpr int BLOCK = 256;
    constexpr int TILE = BLOCK * 4; // 1024
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = BLOCK >> 5;

    __shared__ float warp_sums[8];   // 8 warps
    __shared__ float warp_prefix[8];

    // Alignment check for vector loads/stores (16B)
    bool vec_ok = ((((uintptr_t)x | (uintptr_t)out) & 0xF) == 0);

    float carry = 0.0f;

    int nt = (NTILES_FIXED >= 0) ? NTILES_FIXED : ( (N + TILE - 1) / TILE );

    #pragma unroll 1
    for (int t = 0; t < nt; ++t) {
        int base_r = t * TILE;
        int r_base = base_r + tid * 4;

        float4 v;
        if (vec_ok) {
            v = load_rev4(x, N, r_base);
        } else {
            // scalar load but still 4-per-thread for ILP
            v.x = (r_base + 0 < N) ? x[(N - 1) - (r_base + 0)] : 0.0f;
            v.y = (r_base + 1 < N) ? x[(N - 1) - (r_base + 1)] : 0.0f;
            v.z = (r_base + 2 < N) ? x[(N - 1) - (r_base + 2)] : 0.0f;
            v.w = (r_base + 3 < N) ? x[(N - 1) - (r_base + 3)] : 0.0f;
        }

        // Convert 4 values into:
        // local inclusive prefix within the thread: p0, p1, p2, p3 and thread_total
        float p0 = v.x;
        float p1 = v.x + v.y;
        float p2 = v.x + v.y + v.z;
        float p3 = v.x + v.y + v.z + v.w;
        float thread_total = p3;

        // Warp inclusive scan over thread_total gives prefix at thread granularity (each thread=4 elts)
        float wscan = warp_inclusive_scan(thread_total);

        if (lane == 31) warp_sums[warp] = wscan;
        __syncthreads();

        if (warp == 0) {
            float wv = (lane < nwarps) ? warp_sums[lane] : 0.0f;
            float wscan2 = warp_inclusive_scan(wv);
            if (lane < nwarps) warp_prefix[lane] = wscan2 - wv; // exclusive
        }
        __syncthreads();

        float thread_excl = (wscan - thread_total) + warp_prefix[warp] + carry;

        float4 s;
        s.x = p0 + thread_excl;
        s.y = p1 + thread_excl;
        s.z = p2 + thread_excl;
        s.w = p3 + thread_excl;

        // Store (vectorized when possible)
        if (vec_ok) {
            store_rev4(out, N, r_base, s);
        } else {
            if (r_base + 0 < N) out[(N - 1) - (r_base + 0)] = s.x;
            if (r_base + 1 < N) out[(N - 1) - (r_base + 1)] = s.y;
            if (r_base + 2 < N) out[(N - 1) - (r_base + 2)] = s.z;
            if (r_base + 3 < N) out[(N - 1) - (r_base + 3)] = s.w;
        }

        __syncthreads();
        if (tid == 0) {
            float tile_sum = warp_prefix[nwarps - 1] + warp_sums[nwarps - 1];
            warp_sums[0] = tile_sum;
        }
        __syncthreads();
        carry += warp_sums[0];
        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 2)
void cumsum_reverse_2d_dim1_vec4_warpscan_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int N
) {
    // One block per row (grid.x = B). Same tile scan logic as 1D.
    constexpr int BLOCK = 256;
    constexpr int TILE = BLOCK * 4;

    int b = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = BLOCK >> 5;

    __shared__ float warp_sums[8];
    __shared__ float warp_prefix[8];

    const float* row = x + (size_t)b * (size_t)N;
    float* orow = out + (size_t)b * (size_t)N;

    bool vec_ok = ((((uintptr_t)row | (uintptr_t)orow) & 0xF) == 0);

    float carry = 0.0f;
    int nt = (N + TILE - 1) / TILE;

    for (int t = 0; t < nt; ++t) {
        int base_r = t * TILE;
        int r_base = base_r + tid * 4;

        float4 v;
        if (vec_ok) v = load_rev4(row, N, r_base);
        else {
            v.x = (r_base + 0 < N) ? row[(N - 1) - (r_base + 0)] : 0.0f;
            v.y = (r_base + 1 < N) ? row[(N - 1) - (r_base + 1)] : 0.0f;
            v.z = (r_base + 2 < N) ? row[(N - 1) - (r_base + 2)] : 0.0f;
            v.w = (r_base + 3 < N) ? row[(N - 1) - (r_base + 3)] : 0.0f;
        }

        float p0 = v.x;
        float p1 = v.x + v.y;
        float p2 = v.x + v.y + v.z;
        float p3 = v.x + v.y + v.z + v.w;
        float thread_total = p3;

        float wscan = warp_inclusive_scan(thread_total);

        if (lane == 31) warp_sums[warp] = wscan;
        __syncthreads();

        if (warp == 0) {
            float wv = (lane < nwarps) ? warp_sums[lane] : 0.0f;
            float wscan2 = warp_inclusive_scan(wv);
            if (lane < nwarps) warp_prefix[lane] = wscan2 - wv;
        }
        __syncthreads();

        float thread_excl = (wscan - thread_total) + warp_prefix[warp] + carry;

        float4 s;
        s.x = p0 + thread_excl;
        s.y = p1 + thread_excl;
        s.z = p2 + thread_excl;
        s.w = p3 + thread_excl;

        if (vec_ok) store_rev4(orow, N, r_base, s);
        else {
            if (r_base + 0 < N) orow[(N - 1) - (r_base + 0)] = s.x;
            if (r_base + 1 < N) orow[(N - 1) - (r_base + 1)] = s.y;
            if (r_base + 2 < N) orow[(N - 1) - (r_base + 2)] = s.z;
            if (r_base + 3 < N) orow[(N - 1) - (r_base + 3)] = s.w;
        }

        __syncthreads();
        if (tid == 0) {
            float tile_sum = warp_prefix[nwarps - 1] + warp_sums[nwarps - 1];
            warp_sums[0] = tile_sum;
        }
        __syncthreads();
        carry += warp_sums[0];
        __syncthreads();
    }
}

torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int64_t dim) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);

    TORCH_CHECK(dim >= -x.dim() && dim < x.dim(), "dim out of range");
    int64_t d = dim < 0 ? dim + x.dim() : dim;

    if (x.dim() == 1) {
        TORCH_CHECK(d == 0, "For 1D input, dim must be 0");
        int64_t N64 = x.size(0);
        TORCH_CHECK(N64 > 0 && N64 <= (int64_t)2147483647, "Invalid N");
        int N = (int)N64;
        auto out = torch::empty({N}, x.options());

        dim3 block(256);
        dim3 grid(1);

        // Specialize common hot size N=32768 => nt = 32768/1024 = 32 tiles.
        if (N == 32768) {
            cumsum_reverse_1d_vec4_warpscan_f32<32><<<grid, block>>>(
                (const float*)x.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                N
            );
        } else {
            cumsum_reverse_1d_vec4_warpscan_f32<-1><<<grid, block>>>(
                (const float*)x.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                N
            );
        }
        return out;
    }

    if (x.dim() == 2) {
        TORCH_CHECK(d == 1, "For 2D input, dim must be 1");
        int64_t B64 = x.size(0);
        int64_t N64 = x.size(1);
        TORCH_CHECK(B64 > 0 && N64 > 0, "Invalid sizes");
        TORCH_CHECK(B64 <= (int64_t)2147483647 && N64 <= (int64_t)2147483647, "Too large");
        int B = (int)B64;
        int N = (int)N64;

        auto out = torch::empty({B, N}, x.options());

        dim3 block(256);
        dim3 grid(B);
        cumsum_reverse_2d_dim1_vec4_warpscan_f32<<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, N
        );
        return out;
    }

    TORCH_CHECK(false, "Only 1D (dim=0) and 2D (dim=1) are supported for the CUDA fast path");
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int64_t dim);
"""

custom_ops_lib = load_inline(
    name="custom_cumsum_reverse_ext_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["cumsum_reverse_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Custom reverse-cumsum.

    Fast path:
      - CUDA, float32, contiguous
      - 1D with dim in {0,-1} OR 2D with dim in {1,-1}

    Fallback: torch.cumsum(x.flip(dim), dim).flip(dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            if x.dim() == 1 and (self.dim == 0 or self.dim == -1):
                dim = self.dim if self.dim >= 0 else self.dim + x.dim()
                return self.custom_ops_lib.cumsum_reverse_cuda(x, dim)
            if x.dim() == 2 and (self.dim == 1 or self.dim == -1):
                dim = self.dim if self.dim >= 0 else self.dim + x.dim()
                return self.custom_ops_lib.cumsum_reverse_cuda(x, dim)

        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)