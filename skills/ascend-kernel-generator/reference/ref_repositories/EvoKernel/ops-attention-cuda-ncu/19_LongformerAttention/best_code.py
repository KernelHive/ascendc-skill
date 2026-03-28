import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------
# Custom CUDA: Longformer Attention
# ---------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#endif
#ifndef CHECK_INT
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")
#endif

__constant__ int c_gidx[8];
__constant__ int c_gcount;

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
    return __shfl_sync(mask, v, 0);
}

static __device__ __forceinline__ bool is_global_pos_fast(int s) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= c_gcount) break;
        if (c_gidx[i] == s) return true;
    }
    return false;
}

// Online softmax update driven by a single authoritative score (lane 0).
// Updates m,l and provides scaling for accumulator renorm + new weight.
static __device__ __forceinline__ void online_softmax_update_lane0(
    float score,
    float &m, float &l,
    float &scale_old, float &w_new
) {
    float m_new = fmaxf(m, score);
    scale_old = __expf(m - m_new);
    w_new = __expf(score - m_new);
    l = l * scale_old + w_new;
    m = m_new;
}

// -------------------------
// Fast path: D == 64 only
// One warp per (b,h,s).
// Online softmax single pass.
// Each lane owns 2 output channels => all lanes active.
// -------------------------
__global__ __launch_bounds__(32, 4) void longformer_attn_fwd_warp64_online_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int S,
    int window_size,
    float scale
) {
    int lane = threadIdx.x & 31;
    int q_index = (int)blockIdx.x; // 1 warp/block, 1 block/query
    int total_q = B * H * S;
    if (q_index >= total_q) return;

    int s = q_index % S;
    int h = (q_index / S) % H;
    int b = q_index / (S * H);

    int half = window_size >> 1;
    int start = s - half;
    int end = s + half + 1;
    if (start < 0) start = 0;
    if (end > S) end = S;

    bool q_is_global = is_global_pos_fast(s);

    const int D = 64;
    long q_base = (((long)b * H + h) * (long)S + s) * D;
    long kv_base = ((long)b * H + h) * (long)S * D;

    // Each lane loads 2 Q elements: lane and lane+32
    float q0 = __ldg(Q + q_base + lane);
    float q1 = __ldg(Q + q_base + lane + 32);

    auto dot_qk_lane0 = [&](int t) -> float {
        const float* k_ptr = K + kv_base + (long)t * D;
        float k0 = __ldg(k_ptr + lane);
        float k1 = __ldg(k_ptr + lane + 32);
        float local = q0 * k0 + q1 * k1;
        float sum = warp_reduce_sum(local);
        // sum is broadcasted from lane0 by warp_reduce_sum
        return sum * scale;
    };

    // Each lane owns 2 output dims: lane and lane+32
    float acc0 = 0.f;
    float acc1 = 0.f;

    float m = -INFINITY;
    float l = 0.f;

    auto process_t = [&](int t) {
        // score: lane0 authoritative
        float sc = dot_qk_lane0(t);

        float scale_old = 0.f;
        float w_new = 0.f;
        if (lane == 0) {
            online_softmax_update_lane0(sc, m, l, scale_old, w_new);
        }
        // Broadcast deterministic scalars
        scale_old = __shfl_sync(0xffffffffu, scale_old, 0);
        w_new     = __shfl_sync(0xffffffffu, w_new, 0);

        // Renorm previous accumulator if needed
        acc0 *= scale_old;
        acc1 *= scale_old;

        // Load V (2 floats per lane, fully coalesced across warp for each half)
        const float* v_ptr = V + kv_base + (long)t * D;
        float v0 = __ldg(v_ptr + lane);
        float v1 = __ldg(v_ptr + lane + 32);

        acc0 = fmaf(w_new, v0, acc0);
        acc1 = fmaf(w_new, v1, acc1);
    };

    if (q_is_global) {
        for (int t = 0; t < S; ++t) process_t(t);
    } else {
        for (int t = start; t < end; ++t) process_t(t);
        #pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi >= c_gcount) break;
            int t = c_gidx[gi];
            if ((unsigned)t >= (unsigned)S) continue;
            if (t >= start && t < end) continue;
            process_t(t);
        }
    }

    float inv_l = 0.f;
    if (lane == 0) inv_l = 1.0f / (l + 1e-9f);
    inv_l = __shfl_sync(0xffffffffu, inv_l, 0);

    // Store
    float* o_ptr = Out + q_base;
    o_ptr[lane] = acc0 * inv_l;
    o_ptr[lane + 32] = acc1 * inv_l;
}

// -------------------------
// Generic fallback kernel
// 2-pass softmax (max then exp-sum)
// -------------------------
__global__ __launch_bounds__(32, 4) void longformer_attn_fwd_warp_generic_kernel(
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
    int total_q = B * H * S;
    if (q_index >= total_q) return;

    int s = q_index % S;
    int h = (q_index / S) % H;
    int b = q_index / (S * H);

    int half = window_size >> 1;
    int start = s - half;
    int end = s + half + 1;
    if (start < 0) start = 0;
    if (end > S) end = S;

    bool q_is_global = is_global_pos_fast(s);

    long q_base = (((long)b * H + h) * (long)S + s) * (long)D;
    long kv_base = ((long)b * H + h) * (long)S * (long)D;

    auto dot_qk = [&](int t) -> float {
        const float* k_ptr = K + kv_base + (long)t * (long)D;
        float local = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float qv = __ldg(Q + q_base + d);
            float kv = __ldg(k_ptr + d);
            local += qv * kv;
        }
        float sum = warp_reduce_sum(local);
        return sum * scale;
    };

    float max_score = -INFINITY;
    if (q_is_global) {
        for (int t = 0; t < S; ++t) {
            float sc = dot_qk(t);
            if (lane == 0) max_score = fmaxf(max_score, sc);
        }
    } else {
        for (int t = start; t < end; ++t) {
            float sc = dot_qk(t);
            if (lane == 0) max_score = fmaxf(max_score, sc);
        }
        #pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi >= c_gcount) break;
            int t = c_gidx[gi];
            if ((unsigned)t >= (unsigned)S) continue;
            if (t >= start && t < end) continue;
            float sc = dot_qk(t);
            if (lane == 0) max_score = fmaxf(max_score, sc);
        }
    }
    max_score = __shfl_sync(0xffffffffu, max_score, 0);

    for (int d = lane; d < D; d += 32) Out[q_base + d] = 0.0f;

    float denom = 0.0f;

    auto add_token = [&](int t) {
        float sc = dot_qk(t);
        float w = __expf(sc - max_score);
        if (lane == 0) denom += w;
        float w_b = __shfl_sync(0xffffffffu, w, 0);

        const float* v_ptr = V + kv_base + (long)t * (long)D;
        for (int d = lane; d < D; d += 32) {
            Out[q_base + d] += w_b * __ldg(v_ptr + d);
        }
    };

    if (q_is_global) {
        for (int t = 0; t < S; ++t) add_token(t);
    } else {
        for (int t = start; t < end; ++t) add_token(t);
        #pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi >= c_gcount) break;
            int t = c_gidx[gi];
            if ((unsigned)t >= (unsigned)S) continue;
            if (t >= start && t < end) continue;
            add_token(t);
        }
    }

    denom = __shfl_sync(0xffffffffu, denom, 0);
    float inv_d = 1.0f / (denom + 1e-9f);
    for (int d = lane; d < D; d += 32) Out[q_base + d] *= inv_d;
}

torch::Tensor longformer_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int64_t window_size,
    torch::Tensor global_indices
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V); CHECK_CUDA(global_indices);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V); CHECK_CONTIGUOUS(global_indices);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V);
    CHECK_INT(global_indices);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q,K,V must have same shape");
    TORCH_CHECK(global_indices.dim() == 1, "global_indices must be 1D [G]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(window_size > 0, "window_size must be > 0");
    TORCH_CHECK(D > 0 && S > 0, "S and D must be > 0");

    int gcount = (int)global_indices.numel();
    TORCH_CHECK(gcount <= 8, "supports up to 8 global indices");
    if (gcount > 0) {
        cudaMemcpyToSymbol(c_gidx, global_indices.data_ptr<int>(), gcount * sizeof(int), 0, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpyToSymbol(c_gcount, &gcount, sizeof(int), 0, cudaMemcpyHostToDevice);

    auto Out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    int total_q = B * H * S;
    dim3 threads(32);
    dim3 blocks(total_q);

    if (D == 64) {
        longformer_attn_fwd_warp64_online_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S,
            (int)window_size,
            scale
        );
    } else {
        longformer_attn_fwd_warp_generic_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, D,
            (int)window_size,
            scale
        );
    }

    return Out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor longformer_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int64_t window_size,
    torch::Tensor global_indices
);
"""

custom_ops_lib = load_inline(
    name="custom_longformer_attention_ops_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["longformer_attention_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    verbose=False,
)

# ---------------------------
# Model using the custom op
# ---------------------------

class ModelNew(nn.Module):
    """
    Longformer Attention with fused CUDA forward:
      scores = Q @ K^T / sqrt(d_k)
      apply longformer hybrid mask (local window + global tokens)
      softmax
      output = attn @ V
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.global_attention_indices = [0, 511]
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Kept for state compatibility (not used in fused path)
        self.W_q_global = nn.Linear(d_model, d_model)
        self.W_k_global = nn.Linear(d_model, d_model)
        self.W_v_global = nn.Linear(d_model, d_model)

        self._ops = custom_ops_lib
        self._gidx_cache = {}

    def _get_gidx(self, device, S: int):
        key = (device.index if device.type == "cuda" else -1, S)
        t = self._gidx_cache.get(key, None)
        if t is None or (not t.is_cuda) or t.device != device:
            idx = [int(i) for i in self.global_attention_indices if 0 <= int(i) < S]
            t = torch.tensor(idx, device=device, dtype=torch.int32).contiguous()
            self._gidx_cache[key] = t
        return t

    def forward(self, x):
        B, S, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = K.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = V.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        if x.is_cuda:
            gidx = self._get_gidx(x.device, S)
            O = self._ops.longformer_attention_forward_cuda(Q, K, V, int(self.window_size), gidx)
        else:
            half = self.window_size // 2
            scale = 1.0 / math.sqrt(self.d_k)
            gset = set(int(i) for i in self.global_attention_indices)
            O = torch.empty_like(Q)

            for b in range(B):
                for h in range(self.n_heads):
                    for s in range(S):
                        if s in gset:
                            key_pos = list(range(S))
                        else:
                            start = max(0, s - half)
                            end = min(S, s + half + 1)
                            key_pos = list(range(start, end))
                            for gi in self.global_attention_indices:
                                gi = int(gi)
                                if 0 <= gi < S and not (start <= gi < end):
                                    key_pos.append(gi)

                        Ksel = K[b, h, key_pos]
                        Vsel = V[b, h, key_pos]
                        scores = (Q[b, h, s] @ Ksel.transpose(0, 1)) * scale
                        attn = torch.softmax(scores, dim=-1)
                        O[b, h, s] = attn @ Vsel

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O