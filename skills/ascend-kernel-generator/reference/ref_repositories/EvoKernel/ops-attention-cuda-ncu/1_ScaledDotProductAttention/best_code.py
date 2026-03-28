import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------
# Custom CUDA extension
# ------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
static __device__ __forceinline__ float warp_bcast0(float v) {
    return __shfl_sync(0xffffffff, v, 0);
}

struct __align__(16) float4a { float x, y, z, w; };

static __device__ __forceinline__ float4a ld_float4(const float* p) {
    return *reinterpret_cast<const float4a*>(p);
}
static __device__ __forceinline__ void st_float4(float* p, const float4a& v) {
    *reinterpret_cast<float4a*>(p) = v;
}

// -------------------------
// Fast path: D=64, S=128
// One warp computes one row (bh,i) and produces 64 outputs (2 per lane).
// -------------------------
__global__ __launch_bounds__(32, 8)
void sdpa_fwd_warp_d64_s128(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int BH,
    float inv_sqrt_d
) {
    int row = (int)blockIdx.x;           // 0 .. (BH*128 - 1)
    int lane = (int)(threadIdx.x & 31);  // 0..31
    if (row >= BH * 128) return;

    constexpr int D = 64;
    constexpr int S = 128;

    int i  = row & 127;
    int bh = row >> 7;

    const float* __restrict__ Qrow = Q + ((bh * S + i) * D);
    const float* __restrict__ Kbh  = K + (bh * S * D);
    const float* __restrict__ Vbh  = V + (bh * S * D);
    float* __restrict__ Orow = Out + ((bh * S + i) * D);

    int d0 = lane;
    int d1 = lane + 32;

    // Keep Q in registers
    float q0 = ldg_f32(Qrow + d0);
    float q1 = ldg_f32(Qrow + d1);

    float o0 = 0.f;
    float o1 = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    // Unroll to reduce loop overhead; S is fixed
#pragma unroll 8
    for (int kpos = 0; kpos < S; kpos++) {
        const float* __restrict__ Krow = Kbh + kpos * D;
        const float* __restrict__ Vrow = Vbh + kpos * D;

        float part = fmaf(q0, ldg_f32(Krow + d0), q1 * ldg_f32(Krow + d1));
        float dot = warp_reduce_sum(part);
        dot = warp_bcast0(dot);

        float s = dot * inv_sqrt_d;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p = __expf(s - m_new);

        l = fmaf(l, alpha, p);
        o0 *= alpha;
        o1 *= alpha;

        float v0 = ldg_f32(Vrow + d0);
        float v1 = ldg_f32(Vrow + d1);
        o0 = fmaf(p, v0, o0);
        o1 = fmaf(p, v1, o1);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    Orow[d0] = o0 * inv_l;
    Orow[d1] = o1 * inv_l;
}

// -------------------------
// Generic path: D=64, any S
// Multi-warp blocks: each warp computes one row; K/V are staged per-tile in shared memory
// and reused across multiple rows in the block (multiple warps), improving bandwidth.
// -------------------------
template<int WARPS_PER_BLOCK, int TILE>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
void sdpa_fwd_d64_tiled_multiwarp(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int BH, int S,
    float inv_sqrt_d
) {
    constexpr int D = 64;
    constexpr int VEC = 4;
    constexpr int D4 = D / VEC; // 16 float4s

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (warp >= WARPS_PER_BLOCK) return;

    // Shared K/V tiles: [TILE, D4] float4 each
    extern __shared__ float4a smem[];
    float4a* Ks = smem;                     // TILE*D4
    float4a* Vs = smem + (TILE * D4);       // TILE*D4

    // If out of bounds row, we still participate in loads/sync to keep block coherent,
    // but skip math/stores.
    bool row_valid = (row < BH * S);
    int i = 0, bh = 0;
    const float* __restrict__ Qrow = nullptr;
    const float* __restrict__ Kbh  = nullptr;
    const float* __restrict__ Vbh  = nullptr;
    float* __restrict__ Orow = nullptr;

    if (row_valid) {
        i = row % S;
        bh = row / S;
        Qrow = Q + ((bh * S + i) * D);
        Kbh  = K + (bh * S * D);
        Vbh  = V + (bh * S * D);
        Orow = Out + ((bh * S + i) * D);
    }

    // Each lane handles 2 dims (d0,d1) for output
    int d0 = lane;
    int d1 = lane + 32;

    float q0 = row_valid ? ldg_f32(Qrow + d0) : 0.f;
    float q1 = row_valid ? ldg_f32(Qrow + d1) : 0.f;

    float o0 = 0.f, o1 = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    // tile loop over keys/values
    for (int tj = 0; tj < S; tj += TILE) {
        int tile_len = min(TILE, S - tj);

        // Cooperative load K/V tile as float4
        int total_vec = tile_len * D4;
        for (int t = tid; t < total_vec; t += WARPS_PER_BLOCK * 32) {
            int j = t / D4;
            int d4 = t - j * D4;
            const float* kptr = K + ((bh * S + (tj + j)) * D) + d4 * VEC;
            const float* vptr = V + ((bh * S + (tj + j)) * D) + d4 * VEC;
            // If row_invalid, bh is undefined; but row_invalid warps still run.
            // Guard loads by having only row_valid warps load? That would reduce load parallelism.
            // Instead, base loads on bh from block's first valid warp.
        }
        // To keep correct for blocks where some warps invalid, define a "block bh base" from first warp.
        // We compute it once and use it for loads; invalid warps still help load.
        __shared__ int bh_block;
        if (tid == 0) {
            int first_row = (int)blockIdx.x * WARPS_PER_BLOCK;
            if (first_row < BH * S) bh_block = first_row / S;
            else bh_block = 0;
        }
        __syncthreads();
        int bh_load = bh_block;

        for (int t = tid; t < total_vec; t += WARPS_PER_BLOCK * 32) {
            int j = t / D4;
            int d4 = t - j * D4;
            const float* kptr = K + ((bh_load * S + (tj + j)) * D) + d4 * VEC;
            const float* vptr = V + ((bh_load * S + (tj + j)) * D) + d4 * VEC;
            Ks[j * D4 + d4] = ld_float4(kptr);
            Vs[j * D4 + d4] = ld_float4(vptr);
        }
        __syncthreads();

        if (row_valid) {
            // Process tile
#pragma unroll 1
            for (int j = 0; j < tile_len; j++) {
                // dot(Q, K[tj+j]) using warp reduction; each lane contributes 2 muls
                // Load K scalars from shared (float4) with minimal indexing
                // Convert d0/d1 into float4 index + lane-within-float4
                int d4_0 = d0 >> 2; int off0 = d0 & 3;
                int d4_1 = d1 >> 2; int off1 = d1 & 3;

                float4a k40 = Ks[j * D4 + d4_0];
                float4a k41 = Ks[j * D4 + d4_1];

                float k0 = (off0 == 0) ? k40.x : (off0 == 1) ? k40.y : (off0 == 2) ? k40.z : k40.w;
                float k1 = (off1 == 0) ? k41.x : (off1 == 1) ? k41.y : (off1 == 2) ? k41.z : k41.w;

                float part = fmaf(q0, k0, q1 * k1);
                float dot = warp_reduce_sum(part);
                dot = warp_bcast0(dot);

                float s = dot * inv_sqrt_d;

                float m_new = fmaxf(m, s);
                float alpha = __expf(m - m_new);
                float p = __expf(s - m_new);

                l = fmaf(l, alpha, p);
                o0 *= alpha;
                o1 *= alpha;

                float4a v40 = Vs[j * D4 + d4_0];
                float4a v41 = Vs[j * D4 + d4_1];

                float v0 = (off0 == 0) ? v40.x : (off0 == 1) ? v40.y : (off0 == 2) ? v40.z : v40.w;
                float v1 = (off1 == 0) ? v41.x : (off1 == 1) ? v41.y : (off1 == 2) ? v41.z : v41.w;

                o0 = fmaf(p, v0, o0);
                o1 = fmaf(p, v1, o1);

                m = m_new;
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        float inv_l = 1.0f / (l + 1e-9f);
        Orow[d0] = o0 * inv_l;
        Orow[d1] = o1 * inv_l;
    }
}

torch::Tensor sdpa_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B, H, S, D]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(D == 64, "sdpa_forward_cuda: optimized extension supports only D=64");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D, "V shape mismatch");

    auto Out = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    int BH = B * H;

    if (S == 128) {
        int blocks = BH * 128;
        sdpa_fwd_warp_d64_s128<<<blocks, 32>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            BH,
            inv_sqrt_d
        );
        return Out;
    }

    // Generic tiled kernel
    // Use 4 warps per block to increase occupancy and allow K/V tile reuse across 4 rows.
    constexpr int WARPS = 4;
    constexpr int TILE = 32; // matches prior fallback tile size
    int total_rows = BH * S;
    int blocks = (total_rows + WARPS - 1) / WARPS;

    size_t shmem = 2 * (TILE * (64/4)) * sizeof(float4a); // Ks + Vs
    sdpa_fwd_d64_tiled_multiwarp<WARPS, TILE><<<blocks, WARPS * 32, shmem>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        BH, S,
        inv_sqrt_d
    );
    return Out;
}
"""

cpp_src = r"""
torch::Tensor sdpa_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_sdpa_ops_v8_multiwarp_tiled",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sdpa_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
)

# ------------------------
# Model using custom op
# ------------------------

class ModelNew(nn.Module):
    """
    Fused Scaled Dot-Product Attention (inference-oriented).
    Custom CUDA path supports: CUDA, float32, contiguous, D==64.
    Dropout is not applied in the custom path (expected dropout=0.0 for inference).
    """
    def __init__(self, d_k, dropout=0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout_p = float(dropout)

    def forward(self, Q, K, V):
        if (
            Q.is_cuda and K.is_cuda and V.is_cuda and
            Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32 and
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and
            Q.dim() == 4 and K.dim() == 4 and V.dim() == 4 and
            Q.size(-1) == self.d_k and K.size(-1) == self.d_k and V.size(-1) == self.d_k and
            self.dropout_p == 0.0 and
            self.d_k == 64 and
            Q.size(2) == K.size(2) and Q.size(2) == V.size(2)
        ):
            return custom_ops_lib.sdpa_forward_cuda(Q, K, V)

        import torch.nn.functional as F
        batch_size, n_heads, seq_len, d_k = Q.size()
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_p != 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        output = torch.matmul(attn_weights, V)
        return output