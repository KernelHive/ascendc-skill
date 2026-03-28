import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------
# Custom CUDA: online-softmax attention
# - Baseline: 1 warp per row (generic and D=64)
# - Fast path: D=64, even-q rows only, 2 queries per warp, float4 K/V loads + shuffle distribution
# -----------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

static __forceinline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __forceinline__ __device__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// -----------------------------
// Generic: each lane owns 4 consecutive dims (d0..d0+3).
// One warp per attention row.
// -----------------------------
__global__ void attn_online_warp_generic(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ K,   // [B,H,S,D]
    const float* __restrict__ V,   // [B,H,S,D]
    float* __restrict__ O,         // [B,H,S,D]
    int total_rows, int S, int D, float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int q = row % S;
    int bh = row / S; // b*H + h

    int base = bh * S * D;
    const float* q_ptr = Q + base + q * D;
    float* o_ptr = O + base + q * D;

    int d0 = lane * 4;

    float q0=0.f,q1=0.f,q2=0.f,q3=0.f;
    if (d0 + 0 < D) q0 = ldg_f32(q_ptr + d0 + 0);
    if (d0 + 1 < D) q1 = ldg_f32(q_ptr + d0 + 1);
    if (d0 + 2 < D) q2 = ldg_f32(q_ptr + d0 + 2);
    if (d0 + 3 < D) q3 = ldg_f32(q_ptr + d0 + 3);

    float o0=0.f,o1=0.f,o2=0.f,o3=0.f;
    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll 1
    for (int kidx = 0; kidx < S; kidx++) {
        const float* k_ptr = K + base + kidx * D;
        float part = 0.f;
        if (d0 + 0 < D) part += q0 * ldg_f32(k_ptr + d0 + 0);
        if (d0 + 1 < D) part += q1 * ldg_f32(k_ptr + d0 + 1);
        if (d0 + 2 < D) part += q2 * ldg_f32(k_ptr + d0 + 2);
        if (d0 + 3 < D) part += q3 * ldg_f32(k_ptr + d0 + 3);

        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;
        o0 *= alpha; o1 *= alpha; o2 *= alpha; o3 *= alpha;

        const float* v_ptr = V + base + kidx * D;
        if (d0 + 0 < D) o0 += p * ldg_f32(v_ptr + d0 + 0);
        if (d0 + 1 < D) o1 += p * ldg_f32(v_ptr + d0 + 1);
        if (d0 + 2 < D) o2 += p * ldg_f32(v_ptr + d0 + 2);
        if (d0 + 3 < D) o3 += p * ldg_f32(v_ptr + d0 + 3);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    if (d0 + 0 < D) o_ptr[d0 + 0] = o0 * inv_l;
    if (d0 + 1 < D) o_ptr[d0 + 1] = o1 * inv_l;
    if (d0 + 2 < D) o_ptr[d0 + 2] = o2 * inv_l;
    if (d0 + 3 < D) o_ptr[d0 + 3] = o3 * inv_l;
}

// -----------------------------
// D=64 baseline: each lane owns dims lane and lane+32.
// One warp per row.
// -----------------------------
__global__ void attn_online_warp_d64(
    const float* __restrict__ Q,   // [B,H,S,64]
    const float* __restrict__ K,   // [B,H,S,64]
    const float* __restrict__ V,   // [B,H,S,64]
    float* __restrict__ O,         // [B,H,S,64]
    int total_rows, int S, float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int q = row % S;
    int bh = row / S;

    int base = bh * S * 64;

    const float* q_ptr = Q + base + q * 64;
    float* o_ptr = O + base + q * 64;

    float qA = ldg_f32(q_ptr + lane);
    float qB = ldg_f32(q_ptr + lane + 32);

    float oA = 0.f, oB = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll 1
    for (int kidx = 0; kidx < S; kidx++) {
        const float* k_ptr = K + base + kidx * 64;
        float part = qA * ldg_f32(k_ptr + lane) + qB * ldg_f32(k_ptr + lane + 32);

        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;
        oA *= alpha; oB *= alpha;

        const float* v_ptr = V + base + kidx * 64;
        oA += p * ldg_f32(v_ptr + lane);
        oB += p * ldg_f32(v_ptr + lane + 32);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane] = oA * inv_l;
    o_ptr[lane + 32] = oB * inv_l;
}

// -----------------------------
// D=64 fast path: even-q only, 2 queries per warp (q and q+1).
// One warp per block.
// K/V loaded as float4 by lanes 0..15, distributed via shuffles of scalar components.
// No inner-loop branching; tail handled by launching this kernel only for q_even in [0, S-2].
// -----------------------------
__global__ __launch_bounds__(32, 8) void attn_online_qtile2_d64_evenq_vec4(
    const float* __restrict__ Q,   // [B,H,S,64]
    const float* __restrict__ K,   // [B,H,S,64]
    const float* __restrict__ V,   // [B,H,S,64]
    float* __restrict__ O,         // [B,H,S,64]
    int total_pairs, int S, float scale
) {
    int lane = threadIdx.x & 31;
    int pair = (int)blockIdx.x;
    if (pair >= total_pairs) return;

    // Pair index enumerates (bh, q_even) in row-major:
    // pairs_per_bh = S/2 floor (only even q with q+1 valid)
    int pairs_per_bh = S >> 1; // S assumed >=2; this kernel only launched when S>=2
    int bh = pair / pairs_per_bh;
    int q_even = (pair - bh * pairs_per_bh) * 2;

    // Pointers
    int base = bh * S * 64;
    const float* q_ptr0 = Q + base + q_even * 64;
    const float* q_ptr1 = q_ptr0 + 64;
    float* o_ptr0 = O + base + q_even * 64;
    float* o_ptr1 = o_ptr0 + 64;

    // Q in registers
    float q0A = ldg_f32(q_ptr0 + lane);
    float q0B = ldg_f32(q_ptr0 + lane + 32);
    float q1A = ldg_f32(q_ptr1 + lane);
    float q1B = ldg_f32(q_ptr1 + lane + 32);

    float o0A=0.f, o0B=0.f;
    float o1A=0.f, o1B=0.f;
    float m0=-INFINITY, l0=0.f;
    float m1=-INFINITY, l1=0.f;

    const float* k_base = K + base;
    const float* v_base = V + base;
    const bool vec_ok = is_aligned_16(k_base) && is_aligned_16(v_base);

    #pragma unroll 1
    for (int kidx = 0; kidx < S; kidx++) {
        const float* k_ptr = k_base + kidx * 64;
        const float* v_ptr = v_base + kidx * 64;

        float kA, kB, vA, vB;

        if (vec_ok) {
            float kx=0.f,ky=0.f,kz=0.f,kw=0.f;
            float vx=0.f,vy=0.f,vz=0.f,vw=0.f;

            if (lane < 16) {
                float4 k4 = *reinterpret_cast<const float4*>(k_ptr + lane * 4);
                float4 v4 = *reinterpret_cast<const float4*>(v_ptr + lane * 4);
                kx = k4.x; ky = k4.y; kz = k4.z; kw = k4.w;
                vx = v4.x; vy = v4.y; vz = v4.z; vw = v4.w;
            }

            // For our indices lane (0..31) and lane+32 (32..63):
            int idxA = lane;
            int idxB = lane + 32;
            int loaderA = idxA >> 2; // 0..7
            int compA   = idxA & 3;
            int loaderB = idxB >> 2; // 8..15
            int compB   = idxB & 3;

            float kA0 = __shfl_sync(0xffffffff, kx, loaderA);
            float kA1 = __shfl_sync(0xffffffff, ky, loaderA);
            float kA2 = __shfl_sync(0xffffffff, kz, loaderA);
            float kA3 = __shfl_sync(0xffffffff, kw, loaderA);
            float vA0 = __shfl_sync(0xffffffff, vx, loaderA);
            float vA1 = __shfl_sync(0xffffffff, vy, loaderA);
            float vA2 = __shfl_sync(0xffffffff, vz, loaderA);
            float vA3 = __shfl_sync(0xffffffff, vw, loaderA);

            kA = (compA==0)?kA0:(compA==1)?kA1:(compA==2)?kA2:kA3;
            vA = (compA==0)?vA0:(compA==1)?vA1:(compA==2)?vA2:vA3;

            float kB0 = __shfl_sync(0xffffffff, kx, loaderB);
            float kB1 = __shfl_sync(0xffffffff, ky, loaderB);
            float kB2 = __shfl_sync(0xffffffff, kz, loaderB);
            float kB3 = __shfl_sync(0xffffffff, kw, loaderB);
            float vB0 = __shfl_sync(0xffffffff, vx, loaderB);
            float vB1 = __shfl_sync(0xffffffff, vy, loaderB);
            float vB2 = __shfl_sync(0xffffffff, vz, loaderB);
            float vB3 = __shfl_sync(0xffffffff, vw, loaderB);

            kB = (compB==0)?kB0:(compB==1)?kB1:(compB==2)?kB2:kB3;
            vB = (compB==0)?vB0:(compB==1)?vB1:(compB==2)?vB2:vB3;
        } else {
            kA = ldg_f32(k_ptr + lane);
            kB = ldg_f32(k_ptr + lane + 32);
            vA = ldg_f32(v_ptr + lane);
            vB = ldg_f32(v_ptr + lane + 32);
        }

        // Query 0 update
        float part0 = q0A * kA + q0B * kB;
        float dot0 = warp_reduce_sum(part0);
        dot0 = __shfl_sync(0xffffffff, dot0, 0) * scale;

        float m0_new = fmaxf(m0, dot0);
        float a0 = (m0 == -INFINITY) ? 0.0f : __expf(m0 - m0_new);
        float p0 = __expf(dot0 - m0_new);
        l0 = l0 * a0 + p0;
        o0A = o0A * a0 + p0 * vA;
        o0B = o0B * a0 + p0 * vB;
        m0 = m0_new;

        // Query 1 update
        float part1 = q1A * kA + q1B * kB;
        float dot1 = warp_reduce_sum(part1);
        dot1 = __shfl_sync(0xffffffff, dot1, 0) * scale;

        float m1_new = fmaxf(m1, dot1);
        float a1 = (m1 == -INFINITY) ? 0.0f : __expf(m1 - m1_new);
        float p1 = __expf(dot1 - m1_new);
        l1 = l1 * a1 + p1;
        o1A = o1A * a1 + p1 * vA;
        o1B = o1B * a1 + p1 * vB;
        m1 = m1_new;
    }

    float inv_l0 = 1.0f / (l0 + 1e-9f);
    o_ptr0[lane] = o0A * inv_l0;
    o_ptr0[lane + 32] = o0B * inv_l0;

    float inv_l1 = 1.0f / (l1 + 1e-9f);
    o_ptr1[lane] = o1A * inv_l1;
    o_ptr1[lane + 32] = o1B * inv_l1;
}

torch::Tensor optimized_flash_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4, "K,V must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q,K,V must have same shape");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    int total_rows = B * H * S;

    const float* q = (const float*)Q.data_ptr<float>();
    const float* k = (const float*)K.data_ptr<float>();
    const float* v = (const float*)V.data_ptr<float>();
    float* o = (float*)O.data_ptr<float>();

    // Prefer D=64 fast path if possible
    if (D == 64 && S >= 2) {
        // Launch q-tiling kernel for even q in [0, S-2]
        int pairs_per_bh = S >> 1;            // floor(S/2)
        int total_pairs = (B * H) * pairs_per_bh;
        if (total_pairs > 0) {
            attn_online_qtile2_d64_evenq_vec4<<<total_pairs, 32>>>(q, k, v, o, total_pairs, S, scale);
        }
        // If S is odd, compute the last query (q=S-1) via baseline d64 kernel.
        if (S & 1) {
            int last_q = S - 1;
            int last_rows = (B * H); // one row per (b,h)
            // We will call baseline kernel on a packed view of just those rows by offsetting pointers per bh.
            // Simplest: launch a custom grid where row index maps to (bh, q=last_q).
            // Reuse attn_online_warp_d64 by treating total_rows=last_rows and S=1 with base strides? Not possible.
            // Instead, launch a small bespoke mapping using the generic D=64 kernel by indexing into full tensors:
            // We'll launch one warp per (bh) and compute q=last_q directly.
            // Implemented here by invoking attn_online_warp_d64 with a custom "row" mapping would require another kernel.
            // So we fallback to the generic kernel for just those rows by calling a tiny kernel below.
        }
    } else if (D == 64) {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;
        int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        attn_online_warp_d64<<<blocks, THREADS>>>(q, k, v, o, total_rows, S, scale);
    } else {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;
        int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        attn_online_warp_generic<<<blocks, THREADS>>>(q, k, v, o, total_rows, S, D, scale);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "optimized_flash_attention_cuda kernel launch failed: ", cudaGetErrorString(err));

    // Handle S odd tail for D=64 by launching a dedicated tiny kernel (defined via ATen CUDA? must be here).
    // We'll implement it as a second launch via a lambda-like static kernel in this TU:
    if (D == 64 && (S & 1) && S >= 2) {
        // Define and launch a tiny kernel: one warp per (bh), compute q = S-1.
        // We can't define a kernel inside a function in CUDA C++, so we predeclare it below.
    }

    return O;
}

// --- Tail kernel declaration/definition (must be at global scope) ---
__global__ void attn_online_tail_lastq_d64(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH, int S, float scale
) {
    int lane = threadIdx.x & 31;
    int bh = (int)blockIdx.x;
    if (bh >= BH) return;
    int q = S - 1;
    int base = bh * S * 64;

    const float* q_ptr = Q + base + q * 64;
    float* o_ptr = O + base + q * 64;

    float qA = ldg_f32(q_ptr + lane);
    float qB = ldg_f32(q_ptr + lane + 32);

    float oA = 0.f, oB = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll 1
    for (int kidx = 0; kidx < S; kidx++) {
        const float* k_ptr = K + base + kidx * 64;
        float part = qA * ldg_f32(k_ptr + lane) + qB * ldg_f32(k_ptr + lane + 32);

        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;
        oA *= alpha; oB *= alpha;

        const float* v_ptr = V + base + kidx * 64;
        oA += p * ldg_f32(v_ptr + lane);
        oB += p * ldg_f32(v_ptr + lane + 32);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane] = oA * inv_l;
    o_ptr[lane + 32] = oB * inv_l;
}

// Separate entrypoint to include tail handling cleanly (called from Python wrapper when needed).
torch::Tensor optimized_flash_attention_cuda_with_tail(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = optimized_flash_attention_cuda(Q, K, V);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);
    if (D == 64 && (S & 1) && S >= 2) {
        int BH = B * H;
        float scale = 1.0f / sqrtf(64.0f);
        const float* q = (const float*)Q.data_ptr<float>();
        const float* k = (const float*)K.data_ptr<float>();
        const float* v = (const float*)V.data_ptr<float>();
        float* o = (float*)O.data_ptr<float>();
        attn_online_tail_lastq_d64<<<BH, 32>>>(q, k, v, o, BH, S, scale);
        auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "tail kernel launch failed: ", cudaGetErrorString(err));
    }
    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor optimized_flash_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor optimized_flash_attention_cuda_with_tail(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_optimized_flash_attention_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["optimized_flash_attention_cuda", "optimized_flash_attention_cuda_with_tail"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# -----------------------------
# Model
# -----------------------------
class OptimizedFlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True, use_alibi: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        self.use_alibi = use_alibi

        if use_rope:
            self._init_rope()
        if use_alibi:
            self._init_alibi()

    def _init_rope(self):
        dim = self.d_k
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("rope_inv_freq", inv_freq)

    def _init_alibi(self):
        slopes = torch.tensor([2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)])
        self.register_buffer("alibi_slopes", slopes)

    def apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        sincos = torch.einsum("bi,j->bij", position_ids.float(), self.rope_inv_freq)
        sin = sincos.sin().repeat_interleave(2, dim=-1)
        cos = sincos.cos().repeat_interleave(2, dim=-1)
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos.unsqueeze(1) + x_rot * sin.unsqueeze(1)

    def _compute_alibi_bias(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_pos = -(q_pos - k_pos).abs()
        return self.alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()
        position_ids = torch.arange(S, device=x.device).unsqueeze(0)

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        if self.use_rope:
            Q = self.apply_rope(Q, position_ids).contiguous()
            K = self.apply_rope(K, position_ids).contiguous()

        if self.use_alibi:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            position_bias = self._compute_alibi_bias(S, S, x.device)
            scores = scores + position_bias.unsqueeze(0)
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, V)
        else:
            if Q.dtype != torch.float32:
                out = custom_ops_lib.optimized_flash_attention_cuda_with_tail(Q.float(), K.float(), V.float()).to(Q.dtype)
            else:
                out = custom_ops_lib.optimized_flash_attention_cuda_with_tail(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, d_model, n_heads, use_rope, use_alibi):
        super().__init__()
        self.attn = OptimizedFlashAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_rope=use_rope,
            use_alibi=use_alibi,
        )

    def forward(self, x):
        return self.attn(x)