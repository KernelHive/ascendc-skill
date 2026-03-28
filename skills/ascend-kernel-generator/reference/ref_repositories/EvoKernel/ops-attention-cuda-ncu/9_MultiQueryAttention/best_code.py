import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Fused MQA attention core (FP32):
#   O = softmax(QK^T / sqrt(dk)) V
#
# Fast path (dominant): D == 64
#   - 1 warp computes two query rows (q-tiling=2): (i0, i1) for a given (b,h)
#   - Streams K/V from global with __ldg (read-only cache)
#   - No shared memory, no __syncthreads
#   - Each lane computes two output channels: lane and lane+32
#
# Fallback: shared-memory tiled kernel for general D (previous baseline).
# ---------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// --------------------
// Fast path: D == 64, q-tiling = 2
// --------------------
template<int WARPS_PER_BLOCK>
__global__ void mqa_fused_attn_mqaKV_f32_d64_q2_kernel(
    const float* __restrict__ Q,   // [B,H,S,64]
    const float* __restrict__ K,   // [B,1,S,64]
    const float* __restrict__ V,   // [B,1,S,64]
    float* __restrict__ O,         // [B,H,S,64]
    int B, int H, int S,
    float inv_sqrt_d
) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // global warp id
    const int warp_g = (int)blockIdx.x * WARPS_PER_BLOCK + warp;

    // each (b,h) has ceil(S/2) warp-tiles (two queries per tile)
    const int tiles_per_bh = (S + 1) >> 1;
    const int bh = warp_g / tiles_per_bh;
    if (bh >= B * H) return;

    const int t  = warp_g - bh * tiles_per_bh;
    const int i0 = (t << 1);
    const int i1 = i0 + 1;

    const int b = bh / H;
    const int h = bh - b * H;

    // base pointers
    const float* Q0 = Q + (((b * H + h) * S + i0) * 64);
    const float* Q1 = (i1 < S) ? (Q + (((b * H + h) * S + i1) * 64)) : nullptr;

    const float* Kb = K + ((b * S) * 64);   // [b,0,:,:]
    const float* Vb = V + ((b * S) * 64);

    // per-query online softmax state
    float m0 = -FLT_MAX, l0 = 0.0f;
    float m1 = -FLT_MAX, l1 = 0.0f;

    // output accumulators: each lane computes two dims (lane and lane+32)
    float acc0_lo = 0.0f, acc0_hi = 0.0f;
    float acc1_lo = 0.0f, acc1_hi = 0.0f;

    // load Q fragments into registers (2 floats per query per lane)
    // (guard i0 < S is always true by construction)
    float q0_lo = Q0[lane];
    float q0_hi = Q0[lane + 32];

    float q1_lo = 0.0f, q1_hi = 0.0f;
    if (Q1) {
        q1_lo = Q1[lane];
        q1_hi = Q1[lane + 32];
    }

    // Mild unrolling over j to increase ILP without exploding registers.
    // We keep it to 2-way unroll; each step loads K/V once and updates both queries.
    int j = 0;
    for (; j + 1 < S; j += 2) {
        // ---- j ----
        float k0_lo = ld_ro(Kb + j * 64 + lane);
        float k0_hi = ld_ro(Kb + j * 64 + lane + 32);
        float v0_lo = ld_ro(Vb + j * 64 + lane);
        float v0_hi = ld_ro(Vb + j * 64 + lane + 32);

        float d0_0 = q0_lo * k0_lo + q0_hi * k0_hi;
        float d1_0 = q1_lo * k0_lo + q1_hi * k0_hi;

        float sum0 = warp_reduce_sum(d0_0);
        float sum1 = warp_reduce_sum(d1_0);
        sum0 = __shfl_sync(0xffffffff, sum0, 0);
        sum1 = __shfl_sync(0xffffffff, sum1, 0);

        float s0 = sum0 * inv_sqrt_d;
        float s1 = sum1 * inv_sqrt_d;

        // update q0
        {
            float m_new = fmaxf(m0, s0);
            float alpha = __expf(m0 - m_new);
            float beta  = __expf(s0 - m_new);
            acc0_lo = acc0_lo * alpha + beta * v0_lo;
            acc0_hi = acc0_hi * alpha + beta * v0_hi;
            l0      = l0      * alpha + beta;
            m0 = m_new;
        }
        // update q1 if valid
        if (Q1) {
            float m_new = fmaxf(m1, s1);
            float alpha = __expf(m1 - m_new);
            float beta  = __expf(s1 - m_new);
            acc1_lo = acc1_lo * alpha + beta * v0_lo;
            acc1_hi = acc1_hi * alpha + beta * v0_hi;
            l1      = l1      * alpha + beta;
            m1 = m_new;
        }

        // ---- j+1 ----
        int jn = j + 1;
        float k1_lo = ld_ro(Kb + jn * 64 + lane);
        float k1_hi = ld_ro(Kb + jn * 64 + lane + 32);
        float v1_lo = ld_ro(Vb + jn * 64 + lane);
        float v1_hi = ld_ro(Vb + jn * 64 + lane + 32);

        float d0_1 = q0_lo * k1_lo + q0_hi * k1_hi;
        float d1_1 = q1_lo * k1_lo + q1_hi * k1_hi;

        float sum0b = warp_reduce_sum(d0_1);
        float sum1b = warp_reduce_sum(d1_1);
        sum0b = __shfl_sync(0xffffffff, sum0b, 0);
        sum1b = __shfl_sync(0xffffffff, sum1b, 0);

        float s0b = sum0b * inv_sqrt_d;
        float s1b = sum1b * inv_sqrt_d;

        // update q0
        {
            float m_new = fmaxf(m0, s0b);
            float alpha = __expf(m0 - m_new);
            float beta  = __expf(s0b - m_new);
            acc0_lo = acc0_lo * alpha + beta * v1_lo;
            acc0_hi = acc0_hi * alpha + beta * v1_hi;
            l0      = l0      * alpha + beta;
            m0 = m_new;
        }
        // update q1 if valid
        if (Q1) {
            float m_new = fmaxf(m1, s1b);
            float alpha = __expf(m1 - m_new);
            float beta  = __expf(s1b - m_new);
            acc1_lo = acc1_lo * alpha + beta * v1_lo;
            acc1_hi = acc1_hi * alpha + beta * v1_hi;
            l1      = l1      * alpha + beta;
            m1 = m_new;
        }
    }

    // tail
    if (j < S) {
        float k_lo = ld_ro(Kb + j * 64 + lane);
        float k_hi = ld_ro(Kb + j * 64 + lane + 32);
        float v_lo = ld_ro(Vb + j * 64 + lane);
        float v_hi = ld_ro(Vb + j * 64 + lane + 32);

        float d0t = q0_lo * k_lo + q0_hi * k_hi;
        float d1t = q1_lo * k_lo + q1_hi * k_hi;

        float sum0 = warp_reduce_sum(d0t);
        float sum1 = warp_reduce_sum(d1t);
        sum0 = __shfl_sync(0xffffffff, sum0, 0);
        sum1 = __shfl_sync(0xffffffff, sum1, 0);

        float s0 = sum0 * inv_sqrt_d;
        float s1 = sum1 * inv_sqrt_d;

        {
            float m_new = fmaxf(m0, s0);
            float alpha = __expf(m0 - m_new);
            float beta  = __expf(s0 - m_new);
            acc0_lo = acc0_lo * alpha + beta * v_lo;
            acc0_hi = acc0_hi * alpha + beta * v_hi;
            l0      = l0      * alpha + beta;
            m0 = m_new;
        }
        if (Q1) {
            float m_new = fmaxf(m1, s1);
            float alpha = __expf(m1 - m_new);
            float beta  = __expf(s1 - m_new);
            acc1_lo = acc1_lo * alpha + beta * v_lo;
            acc1_hi = acc1_hi * alpha + beta * v_hi;
            l1      = l1      * alpha + beta;
            m1 = m_new;
        }
    }

    // store
    {
        float invl0 = 1.0f / l0;
        float* O0 = O + (((b * H + h) * S + i0) * 64);
        O0[lane]      = acc0_lo * invl0;
        O0[lane + 32] = acc0_hi * invl0;
    }
    if (Q1) {
        float invl1 = 1.0f / l1;
        float* O1 = O + (((b * H + h) * S + i1) * 64);
        O1[lane]      = acc1_lo * invl1;
        O1[lane + 32] = acc1_hi * invl1;
    }
}

// -----------------------------------------
// Fallback: shared-memory tiled kernel (any D)
// -----------------------------------------
template<int WARPS_PER_BLOCK, int JTILE>
__global__ void mqa_fused_attn_mqaKV_f32_kernel(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ K,   // [B,1,S,D]
    const float* __restrict__ V,   // [B,1,S,D]
    float* __restrict__ O,         // [B,H,S,D]
    int B, int H, int S, int D,
    float inv_sqrt_d
) {
    extern __shared__ float smem[];
    float* Ksm = smem;
    float* Vsm = smem + (JTILE * D);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int d_tile = (int)blockIdx.y;
    const int d0 = d_tile * 32 + lane;

    const int rows_per_block = WARPS_PER_BLOCK;
    int row0 = (int)blockIdx.x * rows_per_block;
    int row = row0 + warp;
    int total_rows = B * H * S;
    if (row >= total_rows) return;

    int bh = row / S;
    int i  = row - bh * S;
    int b  = bh / H;
    int h  = bh - b * H;

    const float* Q_bhs = Q + (((b * H + h) * S + i) * D);
    const float* K_b   = K + (b * S * D); // [b,0,:,:]
    const float* V_b   = V + (b * S * D);
    float* O_bhs = O + (((b * H + h) * S + i) * D);

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    const bool vec4_ok = ((D & 3) == 0);

    for (int jb = 0; jb < S; jb += JTILE) {
        int jmax = min(S - jb, JTILE);

        int tile_elems = jmax * D;
        for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
            int tj = idx / D;
            int td = idx - tj * D;
            Ksm[tj * D + td] = K_b[(jb + tj) * D + td];
            Vsm[tj * D + td] = V_b[(jb + tj) * D + td];
        }
        __syncthreads();

        for (int tj = 0; tj < jmax; ++tj) {
            float dot = 0.0f;

            if (vec4_ok) {
                const float4* Q4 = reinterpret_cast<const float4*>(Q_bhs);
                const float4* K4 = reinterpret_cast<const float4*>(Ksm + tj * D);
                int D4 = D >> 2;
                for (int k4 = lane; k4 < D4; k4 += 32) {
                    float4 qv = Q4[k4];
                    float4 kv = K4[k4];
                    dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    dot += Q_bhs[d] * Ksm[tj * D + d];
                }
            }

            float dot_sum = warp_reduce_sum(dot);
            dot_sum = __shfl_sync(0xffffffff, dot_sum, 0);

            float score = dot_sum * inv_sqrt_d;

            float m_new = fmaxf(m, score);
            float exp_m = __expf(m - m_new);
            float exp_s = __expf(score - m_new);

            float vj = (d0 < D) ? Vsm[tj * D + d0] : 0.0f;
            acc = acc * exp_m + exp_s * vj;
            l   = l   * exp_m + exp_s;
            m = m_new;
        }

        __syncthreads();
    }

    if (d0 < D) O_bhs[d0] = acc / l;
}

torch::Tensor mqa_fused_attn_mqaKV_f32_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,1,S,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,1,S,D]");

    TORCH_CHECK(K.size(1) == 1 && V.size(1) == 1, "K and V must have head dim = 1 for MQA");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0), "B mismatch");
    TORCH_CHECK(Q.size(2) == K.size(2) && Q.size(2) == V.size(2), "S mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3), "D mismatch");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    auto O = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // Fast path for D=64
    if (D == 64) {
        // Use more warps/block to hide latency; still modest threads and no smem.
        constexpr int WARPS = 8; // 256 threads
        dim3 block(32 * WARPS, 1, 1);

        int tiles_per_bh = (S + 1) >> 1;
        int total_warps = (B * H) * tiles_per_bh;
        dim3 grid(div_up_int(total_warps, WARPS), 1, 1);

        mqa_fused_attn_mqaKV_f32_d64_q2_kernel<WARPS><<<grid, block, 0>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, S, inv_sqrt_d
        );
        return O;
    }

    // Generic fallback
    constexpr int WARPS = 4;
    constexpr int JTILE = 32;
    dim3 block(32 * WARPS, 1, 1);
    int total_rows = B * H * S;
    dim3 grid(div_up_int(total_rows, WARPS), div_up_int(D, 32), 1);
    size_t smem_bytes = (size_t)(2 * JTILE * D) * sizeof(float);

    mqa_fused_attn_mqaKV_f32_kernel<WARPS, JTILE><<<grid, block, smem_bytes>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)O.data_ptr<float>(),
        B, H, S, D, inv_sqrt_d
    );
    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mqa_fused_attn_mqaKV_f32_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mqa_opt5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mqa_fused_attn_mqaKV_f32_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Multi-Query Attention (MQA) using an optimized fused CUDA attention core.

    - K/V remain single-head (no expansion).
    - Training-time dropout falls back to PyTorch for exact semantics.
    - Inference/no-dropout uses fused CUDA:
        * fast path for D=64
        * generic fallback otherwise
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()  # [B,H,S,D]
        K = self.W_k(x).view(B, S, 1, self.d_k).transpose(1, 2).contiguous()              # [B,1,S,D]
        V = self.W_v(x).view(B, S, 1, self.d_k).transpose(1, 2).contiguous()              # [B,1,S,D]

        if self.dropout.p != 0.0 and self.training:
            Kexp = K.expand(-1, self.n_heads, -1, -1)
            Vexp = V.expand(-1, self.n_heads, -1, -1)
            scores = torch.matmul(Q, Kexp.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            O = torch.matmul(attn, Vexp)
        else:
            O = self.custom_ops_lib.mqa_fused_attn_mqaKV_f32_cuda(Q, K, V)

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O