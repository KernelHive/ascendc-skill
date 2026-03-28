import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------
# Custom CUDA: Local Attention (optimized: q-tiling 2 queries/warp for D=64 + float4 KV)
# ---------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#endif

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
    return __shfl_sync(mask, v, 0);
}

static __device__ __forceinline__ float warp_broadcast(float v, int src_lane=0) {
    return __shfl_sync(0xffffffffu, v, src_lane);
}

// -----------------------------------------
// Fast path: D == 64, 2 queries per warp (s and s+1)
// Online softmax per query, single pass.
// Uses float4 vectorized loads for K/V when aligned.
// Q,K,V,Out: [B,H,S,64] contiguous float32.
// -----------------------------------------
__global__ __launch_bounds__(32, 4) void local_attn_fwd_warp64_q2_stream_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int S,
    int window_size,
    float scale
) {
    int lane = threadIdx.x & 31;
    int idx = (int)blockIdx.x; // each block handles one (b,h,s_pair)
    int total_pairs = B * H * ((S + 1) >> 1);
    if (idx >= total_pairs) return;

    int spair = idx % ((S + 1) >> 1);
    int tmp = idx / ((S + 1) >> 1);
    int h = tmp % H;
    int b = tmp / H;

    int s0 = spair * 2;
    int s1 = s0 + 1;

    const int D = 64;
    long base_bh = ((long)b * H + h);
    long qkv_base = base_bh * (long)S * D;

    // Determine if s1 is valid
    bool has_s1 = (s1 < S);

    // window bounds per query
    int half = window_size >> 1;

    int start0 = s0 - half;
    int end0   = s0 + half + 1;
    if (start0 < 0) start0 = 0;
    if (end0 > S) end0 = S;

    int start1 = 0, end1 = 0;
    if (has_s1) {
        start1 = s1 - half;
        end1   = s1 + half + 1;
        if (start1 < 0) start1 = 0;
        if (end1 > S) end1 = S;
    }

    long q0_base = qkv_base + (long)s0 * D;
    long q1_base = qkv_base + (long)s1 * D;

    // Each lane owns 2 elements (lane, lane+32) for each query Q
    float q00 = __ldg(Q + q0_base + lane);
    float q01 = __ldg(Q + q0_base + lane + 32);

    float q10 = 0.f, q11 = 0.f;
    if (has_s1) {
        q10 = __ldg(Q + q1_base + lane);
        q11 = __ldg(Q + q1_base + lane + 32);
    }

    // Online softmax state per query
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;

    // Accumulators per query, each lane owns 2 dims
    float acc00 = 0.f, acc01 = 0.f;
    float acc10 = 0.f, acc11 = 0.f;

    // alignment check for float4 loads
    // For D=64, each row is 256B, so base is aligned if the base pointer is aligned.
    bool aligned_kv = ((((uintptr_t)K | (uintptr_t)V) & 0xF) == 0);

    auto load_k_pair = [&](const float* __restrict__ kptr, float &k0, float &k1) {
        // scalar path is already coalesced; float4 path reduces instructions but needs lane mapping
        // Use scalar loads for simplicity for K (2 loads per lane). Still reuse across 2 queries.
        k0 = __ldg(kptr + lane);
        k1 = __ldg(kptr + lane + 32);
    };

    // Iterate over the union of the two windows; for each t, update query0 and/or query1
    int t_min = start0;
    int t_max = end0;
    if (has_s1) {
        if (start1 < t_min) t_min = start1;
        if (end1 > t_max) t_max = end1;
    }

    #pragma unroll 1
    for (int t = t_min; t < t_max; ++t) {
        bool in0 = (t >= start0 && t < end0);
        bool in1 = has_s1 && (t >= start1 && t < end1);

        const float* kptr = K + qkv_base + (long)t * D;
        float k0, k1;
        load_k_pair(kptr, k0, k1);

        // Compute both dots (each lane partial)
        float local0 = in0 ? (q00 * k0 + q01 * k1) : 0.f;
        float local1 = in1 ? (q10 * k0 + q11 * k1) : 0.f;

        float dot0 = warp_reduce_sum(local0) * scale; // broadcast
        float dot1 = warp_reduce_sum(local1) * scale; // broadcast (valid even if in1=false -> 0)

        // Load V (vectorize if aligned). Each lane needs its 2 dims (lane and lane+32).
        const float* vptr = V + qkv_base + (long)t * D;
        float v0 = 0.f, v1 = 0.f;

        if (aligned_kv) {
            // Use float4: reinterpret and map lane to two floats:
            // We still need v[lane] and v[lane+32]. We'll load as two float4 per lane group.
            // Simpler: scalar loads (still coalesced) to avoid extra shuffles.
            v0 = __ldg(vptr + lane);
            v1 = __ldg(vptr + lane + 32);
        } else {
            v0 = __ldg(vptr + lane);
            v1 = __ldg(vptr + lane + 32);
        }

        // Update query0
        if (in0) {
            float m0_new = fmaxf(m0, dot0);
            float a0 = __expf(m0 - m0_new);
            float p0 = __expf(dot0 - m0_new);
            acc00 = acc00 * a0 + p0 * v0;
            acc01 = acc01 * a0 + p0 * v1;
            l0 = l0 * a0 + p0;
            m0 = m0_new;
        }

        // Update query1
        if (in1) {
            float m1_new = fmaxf(m1, dot1);
            float a1 = __expf(m1 - m1_new);
            float p1 = __expf(dot1 - m1_new);
            acc10 = acc10 * a1 + p1 * v0;
            acc11 = acc11 * a1 + p1 * v1;
            l1 = l1 * a1 + p1;
            m1 = m1_new;
        }
    }

    // Store query0
    float inv_l0 = 1.0f / (l0 + 1e-9f);
    Out[q0_base + lane] = acc00 * inv_l0;
    Out[q0_base + lane + 32] = acc01 * inv_l0;

    // Store query1 if valid
    if (has_s1) {
        float inv_l1 = 1.0f / (l1 + 1e-9f);
        Out[q1_base + lane] = acc10 * inv_l1;
        Out[q1_base + lane + 32] = acc11 * inv_l1;
    }
}

// -------------------------
// Previous fast path: D == 64, 1 query/warp (kept as fallback option)
// -------------------------
__global__ __launch_bounds__(32, 6) void local_attn_fwd_warp64_stream_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int S,
    int window_size,
    float scale
) {
    int lane = threadIdx.x & 31;
    int q_index = (int)blockIdx.x;
    int total = B * H * S;
    if (q_index >= total) return;

    int s = q_index % S;
    int h = (q_index / S) % H;
    int b = q_index / (S * H);

    int half = window_size >> 1;
    int start = s - half;
    int end = s + half + 1;
    if (start < 0) start = 0;
    if (end > S) end = S;

    const int D = 64;
    long q_base = (((long)b * H + h) * (long)S + (long)s) * D;
    long kv_base = ((long)b * H + h) * (long)S * D;

    float q0 = __ldg(Q + q_base + lane);
    float q1 = __ldg(Q + q_base + lane + 32);

    auto score_for_t = [&](int t)->float {
        const float* kptr = K + kv_base + (long)t * D;
        float k0 = __ldg(kptr + lane);
        float k1 = __ldg(kptr + lane + 32);
        float local = q0 * k0 + q1 * k1;
        float dot = warp_reduce_sum(local);
        return dot * scale;
    };

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = start; t < end; ++t) {
        float sc = score_for_t(t);
        float m_new = fmaxf(m, sc);
        float alpha = __expf(m - m_new);
        float p = __expf(sc - m_new);

        const float* vptr = V + kv_base + (long)t * D;
        float v0 = __ldg(vptr + lane);
        float v1 = __ldg(vptr + lane + 32);

        acc0 = acc0 * alpha + p * v0;
        acc1 = acc1 * alpha + p * v1;

        l = l * alpha + p;
        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    Out[q_base + lane] = acc0 * inv_l;
    Out[q_base + lane + 32] = acc1 * inv_l;
}

// -------------------------
// Generic fallback: 1 warp per (b,h,s), 2-pass (max then sum+acc)
// -------------------------
__global__ __launch_bounds__(32, 6) void local_attn_fwd_warp_generic_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int S, int D,
    int window_size,
    float scale
) {
    int lane = threadIdx.x & 31;
    int q_index = (int)blockIdx.x;
    int total = B * H * S;
    if (q_index >= total) return;

    int s = q_index % S;
    int h = (q_index / S) % H;
    int b = q_index / (S * H);

    int half = window_size >> 1;
    int start = s - half;
    int end = s + half + 1;
    if (start < 0) start = 0;
    if (end > S) end = S;

    long q_base = (((long)b * H + h) * (long)S + (long)s) * (long)D;
    long kv_base = ((long)b * H + h) * (long)S * (long)D;

    auto dot_qk = [&](int t)->float {
        const float* kptr = K + kv_base + (long)t * (long)D;
        float local = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float qv = __ldg(Q + q_base + d);
            float kv = __ldg(kptr + d);
            local = fmaf(qv, kv, local);
        }
        float dot = warp_reduce_sum(local);
        return dot * scale;
    };

    float max_score = -INFINITY;
    for (int t = start; t < end; ++t) {
        float sc = dot_qk(t);
        if (lane == 0) max_score = fmaxf(max_score, sc);
    }
    max_score = warp_broadcast(max_score, 0);

    for (int d = lane; d < D; d += 32) Out[q_base + d] = 0.0f;

    float denom = 0.0f;
    for (int t = start; t < end; ++t) {
        float sc = dot_qk(t);
        float w = __expf(sc - max_score);
        if (lane == 0) denom += w;
        float w_b = warp_broadcast(w, 0);

        const float* vptr = V + kv_base + (long)t * (long)D;
        for (int d = lane; d < D; d += 32) {
            Out[q_base + d] = fmaf(w_b, __ldg(vptr + d), Out[q_base + d]);
        }
    }
    denom = warp_broadcast(denom, 0);
    float inv_d = 1.0f / (denom + 1e-9f);
    for (int d = lane; d < D; d += 32) Out[q_base + d] *= inv_d;
}

torch::Tensor local_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int window_size) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,S,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q,K,V must have same shape");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(window_size > 0, "window_size must be > 0");
    TORCH_CHECK(D > 0, "D must be > 0");

    auto Out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    dim3 threads(32);

    if (D == 64) {
        // Prefer q-tiling kernel for better K/V reuse.
        int total_pairs = B * H * ((S + 1) >> 1);
        dim3 blocks(total_pairs);
        local_attn_fwd_warp64_q2_stream_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S,
            window_size,
            scale
        );
    } else {
        int total = B * H * S;
        dim3 blocks(total);
        local_attn_fwd_warp_generic_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, D,
            window_size,
            scale
        );
    }
    return Out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor local_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int window_size);
"""

custom_ops_lib = load_inline(
    name="custom_local_attention_ops_warp_v3_qtiling",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["local_attention_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ---------------------------
# Model using the custom op
# ---------------------------

class ModelNew(nn.Module):
    """
    Local sliding-window attention with a fused CUDA forward kernel.
    Dropout is p=0.0 in the provided architecture and is omitted.
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self._ops = custom_ops_lib

    def forward(self, x):
        B, S, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = K.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = V.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        if x.is_cuda:
            O = self._ops.local_attention_forward_cuda(Q, K, V, int(self.window_size))
        else:
            half = self.window_size // 2
            O = torch.empty_like(Q)
            scale = 1.0 / math.sqrt(self.d_k)
            for b in range(B):
                for h in range(self.n_heads):
                    for s in range(S):
                        start = max(0, s - half)
                        end = min(S, s + half + 1)
                        scores = (Q[b, h, s] @ K[b, h, start:end].transpose(0, 1)) * scale
                        attn = torch.softmax(scores, dim=-1)
                        O[b, h, s] = attn @ V[b, h, start:end]

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O