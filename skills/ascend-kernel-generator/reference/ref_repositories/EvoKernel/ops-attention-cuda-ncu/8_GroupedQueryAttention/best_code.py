import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

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

__host__ __device__ __forceinline__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ------------------------------
// Fast path: D == 64, Q_TILE=2
// One warp computes full D=64 output for up to two query rows.
// Two phases for d: lane covers d and d+32.
// K/V tiled in shared memory.
// ------------------------------
template<int WARPS_PER_BLOCK, int JTILE>
__global__ void gqa_fused_attn_kv_compact_qtile_d64_f32_kernel(
    const float* __restrict__ Q,   // [B,H,S,64]
    const float* __restrict__ K,   // [B,Hkv,S,64]
    const float* __restrict__ V,   // [B,Hkv,S,64]
    float* __restrict__ O,         // [B,H,S,64]
    int B, int H, int Hkv, int S,
    int n_groups,
    float inv_sqrt_d
) {
    extern __shared__ float smem[];
    float* Ksm = smem;                    // [JTILE, 64]
    float* Vsm = smem + (JTILE * 64);     // [JTILE, 64]

    const int tid  = threadIdx.x;
    const int warp = tid >> 5; // 0..WARPS_PER_BLOCK-1
    const int lane = tid & 31;

    // Linearize groups of 2 queries per (b,h)
    constexpr int Q_TILE = 2;
    const int rows_per_bh_groups = div_up_int(S, Q_TILE);
    const int total_bh = B * H;
    const int total_groups = total_bh * rows_per_bh_groups;

    const int group0 = (int)blockIdx.x * WARPS_PER_BLOCK;
    const int group = group0 + warp;
    if (group >= total_groups) return;

    const int bh = group / rows_per_bh_groups;
    const int g_in_bh = group - bh * rows_per_bh_groups;
    const int i0 = g_in_bh * Q_TILE;
    const int i1 = i0 + 1;

    const int b = bh / H;
    const int h = bh - b * H;
    const int hkv = h / n_groups;

    const float* __restrict__ K_bhkv = K + (((b * Hkv + hkv) * S) * 64);
    const float* __restrict__ V_bhkv = V + (((b * Hkv + hkv) * S) * 64);

    const float* __restrict__ Q0_ptr = Q + (((b * H + h) * S + i0) * 64);
    const float* __restrict__ Q1_ptr = (i1 < S) ? (Q + (((b * H + h) * S + i1) * 64)) : nullptr;

    float* __restrict__ O0_ptr = O + (((b * H + h) * S + i0) * 64);
    float* __restrict__ O1_ptr = (i1 < S) ? (O + (((b * H + h) * S + i1) * 64)) : nullptr;

    // Online softmax scalars for both queries
    float m0 = -FLT_MAX, l0 = 0.0f;
    float m1 = -FLT_MAX, l1 = 0.0f;

    // Two-phase accumulators for D=64: d = lane and d+32
    float acc0a = 0.0f, acc0b = 0.0f;
    float acc1a = 0.0f, acc1b = 0.0f;

    // Use float4 loads for cooperative K/V tile load
    // Ksm/Vsm are 64-wide so always aligned to 16 bytes when base is aligned.
    for (int jb = 0; jb < S; jb += JTILE) {
        const int jmax = min(S - jb, JTILE);

        // Cooperative load: total float4 per row = 64/4 = 16, tile float4 = jmax*16
        const int tile_elems4 = jmax * 16;
        const float4* __restrict__ K4 = reinterpret_cast<const float4*>(K_bhkv + jb * 64);
        const float4* __restrict__ V4 = reinterpret_cast<const float4*>(V_bhkv + jb * 64);
        float4* __restrict__ Ksm4 = reinterpret_cast<float4*>(Ksm);
        float4* __restrict__ Vsm4 = reinterpret_cast<float4*>(Vsm);

        for (int idx4 = tid; idx4 < tile_elems4; idx4 += blockDim.x) {
            Ksm4[idx4] = K4[idx4];
            Vsm4[idx4] = V4[idx4];
        }
        __syncthreads();

        #pragma unroll
        for (int tj = 0; tj < JTILE; ++tj) {
            if (tj >= jmax) break;

            // Dot products. Each lane computes partial sum over D=64 (two scalars: lane and lane+32)
            float dot0 = 0.0f;
            float dot1 = 0.0f;

            const float k_a = Ksm[tj * 64 + lane];
            const float k_b = Ksm[tj * 64 + (lane + 32)];

            dot0 = __ldg(&Q0_ptr[lane]) * k_a + __ldg(&Q0_ptr[lane + 32]) * k_b;
            if (Q1_ptr) {
                dot1 = __ldg(&Q1_ptr[lane]) * k_a + __ldg(&Q1_ptr[lane + 32]) * k_b;
            }

            float sum0 = warp_reduce_sum(dot0);
            sum0 = __shfl_sync(0xffffffff, sum0, 0);
            const float score0 = sum0 * inv_sqrt_d;

            float score1 = -INFINITY;
            if (Q1_ptr) {
                float sum1 = warp_reduce_sum(dot1);
                sum1 = __shfl_sync(0xffffffff, sum1, 0);
                score1 = sum1 * inv_sqrt_d;
            }

            // Load v for both phases
            const float v_a = Vsm[tj * 64 + lane];
            const float v_b = Vsm[tj * 64 + (lane + 32)];

            // Update q0
            {
                const float m_new = fmaxf(m0, score0);
                const float exp_m = __expf(m0 - m_new);
                const float exp_s = __expf(score0 - m_new);
                acc0a = acc0a * exp_m + exp_s * v_a;
                acc0b = acc0b * exp_m + exp_s * v_b;
                l0    = l0    * exp_m + exp_s;
                m0 = m_new;
            }
            // Update q1
            if (Q1_ptr) {
                const float m_new = fmaxf(m1, score1);
                const float exp_m = __expf(m1 - m_new);
                const float exp_s = __expf(score1 - m_new);
                acc1a = acc1a * exp_m + exp_s * v_a;
                acc1b = acc1b * exp_m + exp_s * v_b;
                l1    = l1    * exp_m + exp_s;
                m1 = m_new;
            }
        }

        __syncthreads();
    }

    const float inv_l0 = 1.0f / (l0 + 1e-9f);
    O0_ptr[lane]      = acc0a * inv_l0;
    O0_ptr[lane + 32] = acc0b * inv_l0;

    if (O1_ptr) {
        const float inv_l1 = 1.0f / (l1 + 1e-9f);
        O1_ptr[lane]      = acc1a * inv_l1;
        O1_ptr[lane + 32] = acc1b * inv_l1;
    }
}

// ------------------------------
// General fallback: prior kernel structure (D tiled by 32)
// Kept for correctness on arbitrary D.
// ------------------------------
template<int WARPS_PER_BLOCK, int JTILE>
__global__ void gqa_fused_attn_kv_compact_f32_kernel(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ K,   // [B,Hkv,S,D]
    const float* __restrict__ V,   // [B,Hkv,S,D]
    float* __restrict__ O,         // [B,H,S,D]
    int B, int H, int Hkv, int S, int D,
    int n_groups,
    float inv_sqrt_d
) {
    extern __shared__ float smem[];
    float* Ksm = smem;
    float* Vsm = smem + (JTILE * D);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int d_tile = (int)blockIdx.y;
    const int d0 = d_tile * 32 + lane;

    const int total_rows = B * H * S;
    const int row0 = (int)blockIdx.x * WARPS_PER_BLOCK;
    const int row = row0 + warp;
    if (row >= total_rows) return;

    const int bh = row / S;
    const int i  = row - bh * S;
    const int b  = bh / H;
    const int h  = bh - b * H;
    const int hkv = h / n_groups;

    const float* Q_ptr = Q + (((b * H + h) * S + i) * D);
    const float* K_bhkv = K + (((b * Hkv + hkv) * S) * D);
    const float* V_bhkv = V + (((b * Hkv + hkv) * S) * D);
    float* O_ptr = O + (((b * H + h) * S + i) * D);

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    const bool vec4_ok = ((D & 3) == 0);

    for (int jb = 0; jb < S; jb += JTILE) {
        int jmax = min(S - jb, JTILE);
        int tile_elems = jmax * D;

        if (vec4_ok) {
            int D4 = D >> 2;
            int tile_elems4 = jmax * D4;
            const float4* K4 = reinterpret_cast<const float4*>(K_bhkv + jb * D);
            const float4* V4 = reinterpret_cast<const float4*>(V_bhkv + jb * D);
            float4* Ksm4 = reinterpret_cast<float4*>(Ksm);
            float4* Vsm4 = reinterpret_cast<float4*>(Vsm);

            for (int idx4 = tid; idx4 < tile_elems4; idx4 += blockDim.x) {
                Ksm4[idx4] = K4[idx4];
                Vsm4[idx4] = V4[idx4];
            }
        } else {
            for (int idx = tid; idx < tile_elems; idx += blockDim.x) {
                int tj = idx / D;
                int td = idx - tj * D;
                Ksm[tj * D + td] = K_bhkv[(jb + tj) * D + td];
                Vsm[tj * D + td] = V_bhkv[(jb + tj) * D + td];
            }
        }
        __syncthreads();

        for (int tj = 0; tj < jmax; ++tj) {
            float dot = 0.0f;

            if (vec4_ok) {
                const float4* Q4  = reinterpret_cast<const float4*>(Q_ptr);
                const float4* Kt4 = reinterpret_cast<const float4*>(Ksm + tj * D);
                int D4 = D >> 2;
                for (int k4 = lane; k4 < D4; k4 += 32) {
                    float4 qv = __ldg(&Q4[k4]);
                    float4 kv = Kt4[k4];
                    dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    dot += __ldg(&Q_ptr[d]) * Ksm[tj * D + d];
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

    if (d0 < D) {
        O_ptr[d0] = acc / (l + 1e-9f);
    }
}

torch::Tensor gqa_attention_fwd_cuda_compactKV(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,Hkv,S,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,Hkv,S,D]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);
    int Hkv = (int)K.size(1);

    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "B mismatch");
    TORCH_CHECK(K.size(2) == S && V.size(2) == S, "S mismatch");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "D mismatch");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");
    int n_groups = H / Hkv;

    auto O = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // Tunables
    constexpr int WARPS = 4;
    constexpr int JTILE_FAST = 16;
    constexpr int JTILE_FALLBACK = 32;

    if (D == 64) {
        dim3 block(32 * WARPS, 1, 1);

        // q-tiling: 2 queries per warp => groups = (B*H)*ceil(S/2)
        const int rows_per_bh_groups = div_up_int(S, 2);
        const int total_groups = (B * H) * rows_per_bh_groups;
        dim3 grid(div_up_int(total_groups, WARPS), 1, 1);

        size_t smem_bytes = (size_t)(2 * JTILE_FAST * 64) * sizeof(float);

        gqa_fused_attn_kv_compact_qtile_d64_f32_kernel<WARPS, JTILE_FAST><<<grid, block, smem_bytes>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, Hkv, S,
            n_groups,
            inv_sqrt_d
        );
        return O;
    }

    // Fallback to general kernel (D tiled by 32)
    {
        dim3 block(32 * WARPS, 1, 1);
        int total_rows = B * H * S;
        dim3 grid(div_up_int(total_rows, WARPS), div_up_int(D, 32), 1);
        size_t smem_bytes = (size_t)(2 * JTILE_FALLBACK * D) * sizeof(float);

        gqa_fused_attn_kv_compact_f32_kernel<WARPS, JTILE_FALLBACK><<<grid, block, smem_bytes>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, Hkv, S, D,
            n_groups,
            inv_sqrt_d
        );
        return O;
    }
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gqa_attention_fwd_cuda_compactKV(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_gqa_ops_opt_compactkv_d64qtile",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gqa_attention_fwd_cuda_compactKV"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Grouped Query Attention (GQA) with optimized fused CUDA attention core.
    - Uses compact K/V layout [B, Hkv, S, D] and maps query head->kv head in kernel.
    - Fast path specialized for D=64 with q-tiling (2 queries/warp) and no D-tiling grid dimension.
    - Fallback path supports general D.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = float(dropout)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()      # [B,H,S,D]
        K = self.W_k(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2).contiguous()   # [B,Hkv,S,D]
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2).contiguous()   # [B,Hkv,S,D]

        if self.dropout_p != 0.0 and self.training:
            Kexp = K.repeat_interleave(self.n_groups, dim=1).contiguous()
            Vexp = V.repeat_interleave(self.n_groups, dim=1).contiguous()
            scores = torch.matmul(Q, Kexp.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=True)
            O = torch.matmul(attn, Vexp)
        else:
            O = self.custom_ops_lib.gqa_attention_fwd_cuda_compactKV(Q, K, V)

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O