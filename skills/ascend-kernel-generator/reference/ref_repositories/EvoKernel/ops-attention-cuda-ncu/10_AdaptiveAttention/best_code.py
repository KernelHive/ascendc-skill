import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA: KV-head aware fused attention core for GQA/MQA/MHA without K/V expansion.
# v3 improvements over baseline:
#  - D==64 fast path: 2 queries per warp (i, i+1) + 4 warps/block
#    * amortizes K/V loads across 2 queries
#    * no shared memory, no __syncthreads()
#  - General fallback: double-buffered shared-memory tiling (ping-pong) + smaller JTILE
#    * reduces barrier/pipeline gaps by overlapping "load next tile" with "compute current tile"
#    * reduces shared memory footprint to improve occupancy
# Forward-only; dropout handled by PyTorch fallback when active in training.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

__host__ __device__ __forceinline__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float2 ldg_f32x2(const float* p) {
    float2 out;
#if __CUDA_ARCH__ >= 350
    out.x = __ldg(p + 0);
    out.y = __ldg(p + 1);
#else
    out.x = p[0];
    out.y = p[1];
#endif
    return out;
}

template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK, 2)
void gqa_kvhead_warp64_q2_fwd_f32_kernel(
    const float* __restrict__ Q, // [B,H,S,64]
    const float* __restrict__ K, // [B,Hkv,S,64]
    const float* __restrict__ V, // [B,Hkv,S,64]
    float* __restrict__ O,       // [B,H,S,64]
    int B, int H, int Hkv, int S,
    float scale
) {
    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;      // 0..WARPS_PER_BLOCK-1
    int lane = tid & 31;      // 0..31
    int kvh  = (int)blockIdx.z;

    int groups = H / Hkv;
    int S2 = div_up_int(S, 2);

    int total_pair_rows = B * groups * S2;
    int pair_row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (pair_row >= total_pair_rows) return;

    int tmp = pair_row;
    int ipair = tmp % S2; tmp /= S2;
    int g = tmp % groups;
    int b = tmp / groups;

    int i0 = ipair * 2;
    int i1 = i0 + 1;
    bool has1 = (i1 < S);
    int h = kvh * groups + g;

    const float* Qrow0 = Q + (((b * H + h) * S + i0) * 64);
    const float* Qrow1 = has1 ? (Q + (((b * H + h) * S + i1) * 64)) : nullptr;

    const float* Kbase = K + (((b * Hkv + kvh) * S) * 64);
    const float* Vbase = V + (((b * Hkv + kvh) * S) * 64);

    float* Orow0 = O + (((b * H + h) * S + i0) * 64);
    float* Orow1 = has1 ? (O + (((b * H + h) * S + i1) * 64)) : nullptr;

    int d0 = lane * 2;

    float2 q0 = ldg_f32x2(Qrow0 + d0);
    float2 q1;
    if (has1) q1 = ldg_f32x2(Qrow1 + d0);
    else { q1.x = 0.0f; q1.y = 0.0f; }

    float m0 = -FLT_MAX, l0 = 0.0f, acc00 = 0.0f, acc01 = 0.0f;
    float m1 = -FLT_MAX, l1 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Stream over keys
    #pragma unroll 1
    for (int j = 0; j < S; ++j) {
        const float* Kj = Kbase + j * 64;
        const float* Vj = Vbase + j * 64;

        float2 k2 = ldg_f32x2(Kj + d0);
        float2 v2 = ldg_f32x2(Vj + d0);

        float dot0 = q0.x * k2.x + q0.y * k2.y;
        float dot1 = has1 ? (q1.x * k2.x + q1.y * k2.y) : 0.0f;

        float sum0 = warp_reduce_sum(dot0);
        float sum1 = warp_reduce_sum(dot1);
        sum0 = __shfl_sync(0xffffffff, sum0, 0);
        sum1 = __shfl_sync(0xffffffff, sum1, 0);

        float s0 = sum0 * scale;
        float s1 = sum1 * scale;

        // online softmax update query0
        float m0_new = fmaxf(m0, s0);
        float a0 = __expf(m0 - m0_new);
        float b0 = __expf(s0 - m0_new);
        acc00 = acc00 * a0 + b0 * v2.x;
        acc01 = acc01 * a0 + b0 * v2.y;
        l0    = l0    * a0 + b0;
        m0 = m0_new;

        // online softmax update query1
        if (has1) {
            float m1_new = fmaxf(m1, s1);
            float a1 = __expf(m1 - m1_new);
            float b1 = __expf(s1 - m1_new);
            acc10 = acc10 * a1 + b1 * v2.x;
            acc11 = acc11 * a1 + b1 * v2.y;
            l1    = l1    * a1 + b1;
            m1 = m1_new;
        }
    }

    float invl0 = 1.0f / (l0 + 1e-9f);
    Orow0[d0 + 0] = acc00 * invl0;
    Orow0[d0 + 1] = acc01 * invl0;

    if (has1) {
        float invl1 = 1.0f / (l1 + 1e-9f);
        Orow1[d0 + 0] = acc10 * invl1;
        Orow1[d0 + 1] = acc11 * invl1;
    }
}

// General fallback: double-buffered shared-memory tiling + online softmax
template<int WARPS_PER_BLOCK, int JTILE>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK, 2)
void gqa_kvhead_flash_db_fwd_f32_kernel(
    const float* __restrict__ Q, // [B,H,S,D]
    const float* __restrict__ K, // [B,Hkv,S,D]
    const float* __restrict__ V, // [B,Hkv,S,D]
    float* __restrict__ O,       // [B,H,S,D]
    int B, int H, int Hkv, int S, int D,
    float scale
) {
    extern __shared__ float smem[];
    // Layout: K0 [JTILE,D], V0 [JTILE,D], K1 [JTILE,D], V1 [JTILE,D]
    float* Ksm0 = smem;
    float* Vsm0 = Ksm0 + (JTILE * D);
    float* Ksm1 = Vsm0 + (JTILE * D);
    float* Vsm1 = Ksm1 + (JTILE * D);

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;   // 0..WARPS_PER_BLOCK-1
    int lane = tid & 31;

    int d_tile = (int)blockIdx.y;
    int d0 = d_tile * 32 + lane;
    int kvh = (int)blockIdx.z;

    int groups = H / Hkv;
    int total_rows = B * groups * S;

    int row0 = (int)blockIdx.x * WARPS_PER_BLOCK;
    int row = row0 + warp;
    if (row >= total_rows) return;

    int tmp = row;
    int i = tmp % S; tmp /= S;
    int g = tmp % groups;
    int b = tmp / groups;
    int h = kvh * groups + g;

    const float* Qrow  = Q + (((b * H + h) * S + i) * D);
    const float* Kbase = K + (((b * Hkv + kvh) * S) * D);
    const float* Vbase = V + (((b * Hkv + kvh) * S) * D);
    float* Orow        = O + (((b * H + h) * S + i) * D);

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    const bool vec4_ok = ((D & 3) == 0);

    auto load_tile = [&](int jb, float* Ksm, float* Vsm) {
        int jmax = min(JTILE, S - jb);
        int tile_elems = jmax * D;
        for (int idx = tid; idx < tile_elems; idx += (int)blockDim.x) {
            int tj = idx / D;
            int td = idx - tj * D;
            Ksm[tj * D + td] = Kbase[(jb + tj) * D + td];
            Vsm[tj * D + td] = Vbase[(jb + tj) * D + td];
        }
        // For unused rows in tile, nothing to do; jmax governs compute.
    };

    // Preload first tile into buffer0
    int jb0 = 0;
    load_tile(jb0, Ksm0, Vsm0);
    __syncthreads();

    int cur = 0;
    for (int jb = 0; jb < S; jb += JTILE) {
        int jmax = min(JTILE, S - jb);

        // Prefetch next tile into the other buffer
        int jb_next = jb + JTILE;
        if (jb_next < S) {
            if (cur == 0) load_tile(jb_next, Ksm1, Vsm1);
            else          load_tile(jb_next, Ksm0, Vsm0);
        }

        // Compute using current buffer
        float* Ksm = (cur == 0) ? Ksm0 : Ksm1;
        float* Vsm = (cur == 0) ? Vsm0 : Vsm1;

        // No need to sync before compute: current buffer is ready.
        // But we MUST ensure loads to next buffer don't clobber current buffer.
        // Since we used the alternate buffer, it's safe; still, threads are concurrently loading next while others compute.
        // To avoid race on shared memory bank conflicts and ensure correctness, we synchronize after issuing prefetch loads.
        __syncthreads();

        #pragma unroll
        for (int tj = 0; tj < JTILE; ++tj) {
            if (tj >= jmax) break;
            float dot = 0.0f;

            if (vec4_ok) {
                const float4* Q4 = reinterpret_cast<const float4*>(Qrow);
                const float4* K4 = reinterpret_cast<const float4*>(Ksm + tj * D);
                int D4 = D >> 2;
                for (int k4 = lane; k4 < D4; k4 += 32) {
                    float4 qv = Q4[k4];
                    float4 kv = K4[k4];
                    dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                for (int d = lane; d < D; d += 32) {
                    dot += Qrow[d] * Ksm[tj * D + d];
                }
            }

            float sum = warp_reduce_sum(dot);
            sum = __shfl_sync(0xffffffff, sum, 0);
            float s = sum * scale;

            float m_new = fmaxf(m, s);
            float alpha = __expf(m - m_new);
            float beta  = __expf(s - m_new);

            float vj = (d0 < D) ? Vsm[tj * D + d0] : 0.0f;
            acc = acc * alpha + beta * vj;
            l   = l   * alpha + beta;
            m = m_new;
        }

        // Ensure next buffer load completes before next iteration uses it
        __syncthreads();
        cur ^= 1;
    }

    if (d0 < D) {
        Orow[d0] = acc / (l + 1e-9f);
    }
}

torch::Tensor attn_fwd_gqa_kvhead_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
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

    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "B mismatch");
    int Hkv = (int)K.size(1);
    TORCH_CHECK(V.size(1) == Hkv, "Hkv mismatch");
    TORCH_CHECK(K.size(2) == S && V.size(2) == S, "S mismatch");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "D mismatch");
    TORCH_CHECK(H % Hkv == 0, "H must be divisible by Hkv");

    auto O = torch::empty_like(Q);

    // Fast path: D == 64
    if (D == 64) {
        constexpr int WARPS = 4;
        int groups = H / Hkv;
        int total_pair_rows_per_kvh = B * groups * div_up_int(S, 2);
        dim3 block(32 * WARPS, 1, 1);
        dim3 grid(div_up_int(total_pair_rows_per_kvh, WARPS), 1, Hkv);
        gqa_kvhead_warp64_q2_fwd_f32_kernel<WARPS><<<grid, block, 0>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, Hkv, S,
            (float)scale
        );
        return O;
    }

    // General fallback
    constexpr int WARPS = 4;   // 128 threads
    constexpr int JTILE = 16;  // smaller tile for higher occupancy

    dim3 block(32 * WARPS, 1, 1);

    int groups = H / Hkv;
    int total_rows_per_kvh = B * groups * S;
    dim3 grid(div_up_int(total_rows_per_kvh, WARPS), div_up_int(D, 32), Hkv);

    size_t smem_bytes = (size_t)(4 * JTILE * D) * sizeof(float);

    gqa_kvhead_flash_db_fwd_f32_kernel<WARPS, JTILE><<<grid, block, smem_bytes>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)O.data_ptr<float>(),
        B, H, Hkv, S, D,
        (float)scale
    );

    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor attn_fwd_gqa_kvhead_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_adaptive_attention_ops_kvhead_v3_db",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["attn_fwd_gqa_kvhead_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class GroupedQueryAttentionNew(nn.Module):
    """
    GQA using a KV-head aware fused CUDA attention core.
    Avoids K/V repeat_interleave materialization entirely (in inference/no-dropout path).
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.n_groups = self.n_heads // self.n_kv_heads
        self.d_k = self.d_model // self.n_heads
        self.max_seq_len = int(max_seq_len)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = float(dropout)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2).contiguous()

        if self.dropout_p != 0.0 and self.training:
            # Preserve exact dropout semantics
            Kexp = K.repeat_interleave(self.n_groups, dim=1)
            Vexp = V.repeat_interleave(self.n_groups, dim=1)
            scores = torch.matmul(Q, Kexp.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=True)
            O = torch.matmul(attn, Vexp)
        else:
            scale = 1.0 / math.sqrt(self.d_k)
            O = self.custom_ops_lib.attn_fwd_gqa_kvhead_cuda(Q, K, V, scale)

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O


class ModelNew(nn.Module):
    """
    Adaptive Attention:
    - Router + dynamic batching in PyTorch.
    - Attention core uses KV-head aware fused CUDA op (no K/V expansion) when dropout inactive.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads_options: list = None, dropout: float = 0.0):
        super().__init__()
        if n_kv_heads_options is None:
            n_kv_heads_options = [8, 4, 1]

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads_options = list(map(int, n_kv_heads_options))
        self.dropout_p = float(dropout)

        self.attention_layers = nn.ModuleList([
            GroupedQueryAttentionNew(d_model, n_heads, n_kv_heads, dropout)
            for n_kv_heads in self.n_kv_heads_options
        ])

        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(self.n_kv_heads_options)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_input = x.mean(dim=1)
        routing_logits = self.router(routing_input)
        routing_probs = F.softmax(routing_logits, dim=-1)
        selected_idx = routing_probs.argmax(dim=-1)

        outputs = torch.zeros_like(x)
        for idx in range(len(self.n_kv_heads_options)):
            mask = (selected_idx == idx)
            if mask.any():
                batch_input = x[mask]
                outputs[mask] = self.attention_layers[idx](batch_input)
        return outputs