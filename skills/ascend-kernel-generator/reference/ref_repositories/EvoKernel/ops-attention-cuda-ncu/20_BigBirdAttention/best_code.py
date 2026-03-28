import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA: BigBird dense-mask attention forward (fused)
#   Computes: Out[b,h,s,:] = softmax_j( (Q·K)/sqrt(D) with masked_fill(mask==0,-1e9) ) @ V
#   Inputs:
#     Q,K,V: [B,H,S,D] float32 contiguous CUDA
#     maskSS: [S,S] float32 contiguous CUDA with 0/1 (shared across B/H)
#   Output:
#     Out: [B,H,S,D] float32
#
# Design:
#   - One warp computes one (b,h,s) query.
#   - Online softmax (streaming): maintains running max m and normalizer l.
#   - Accumulates numerator vector in registers across D with lane-strided loop.
#   - Avoids latent truncation hazards for D>256 (no fixed-size arrays).
#   - Uses maskSS exactly (no dedup / sparsification that could change semantics).
# ------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#endif

static __device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
    return __shfl_sync(mask, v, 0);
}

static __device__ __forceinline__ float warp_bcast(float v, int src=0) {
    return __shfl_sync(0xffffffffu, v, src);
}

__global__ __launch_bounds__(32, 6) void bigbird_attn_fwd_warp_online(
    const float* __restrict__ Q,      // [B,H,S,D]
    const float* __restrict__ K,      // [B,H,S,D]
    const float* __restrict__ V,      // [B,H,S,D]
    const float* __restrict__ maskSS, // [S,S] 0/1
    float* __restrict__ Out,          // [B,H,S,D]
    int B, int H, int S, int D,
    float scale
) {
    int lane = threadIdx.x & 31;
    int q_index = (int)blockIdx.x;
    int total = B * H * S;
    if (q_index >= total) return;

    int s = q_index % S;
    int h = (q_index / S) % H;
    int b = q_index / (S * H);

    int64_t q_base  = (((int64_t)b * H + h) * (int64_t)S + (int64_t)s) * (int64_t)D;
    int64_t bh_base = ((int64_t)b * H + h) * (int64_t)S * (int64_t)D;
    const float* qptr = Q + q_base;
    const float* mask_row = maskSS + (int64_t)s * (int64_t)S;

    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;

    // We'll accumulate output numerator in Out temporarily, then scale by 1/l at end.
    // Initialize Out[q_base + d] = 0
    for (int d = lane; d < D; d += 32) {
        Out[q_base + d] = 0.0f;
    }

    // Stream over all keys j
    for (int j = 0; j < S; ++j) {
        float score = -1.0e9f;  // masked_fill value
        float mj = mask_row[j];

        if (mj != 0.0f) {
            const float* kptr = K + bh_base + (int64_t)j * (int64_t)D;
            float local = 0.0f;
            for (int d = lane; d < D; d += 32) {
                local = fmaf(__ldg(qptr + d), __ldg(kptr + d), local);
            }
            float dot = warp_sum(local) * scale;
            if (lane == 0) score = dot;
        }
        score = warp_bcast(score, 0);

        // Online softmax update
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float beta  = __expf(score - m_new);
        l = l * alpha + beta;

        // Out = Out*alpha + beta*V_j
        const float* vptr = V + bh_base + (int64_t)j * (int64_t)D;
        for (int d = lane; d < D; d += 32) {
            float o = Out[q_base + d];
            float v = __ldg(vptr + d);
            Out[q_base + d] = o * alpha + beta * v;
        }

        m = m_new;
    }

    float inv = 1.0f / (l + 1e-9f);
    for (int d = lane; d < D; d += 32) {
        Out[q_base + d] *= inv;
    }
}

torch::Tensor big_bird_attention_forward_cuda(
    torch::Tensor Q,      // [B,H,S,D]
    torch::Tensor K,      // [B,H,S,D]
    torch::Tensor V,      // [B,H,S,D]
    torch::Tensor maskSS  // [S,S] float32 0/1
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V); CHECK_CUDA(maskSS);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V); CHECK_CONTIGUOUS(maskSS);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V); CHECK_FLOAT(maskSS);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(), "K and V must match Q");
    TORCH_CHECK(maskSS.dim() == 2, "maskSS must be [S,S]");
    TORCH_CHECK(maskSS.size(0) == Q.size(2) && maskSS.size(1) == Q.size(2), "maskSS must match S");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    auto Out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    int total = B * H * S;
    dim3 threads(32);
    dim3 blocks(total);

    bigbird_attn_fwd_warp_online<<<blocks, threads>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (const float*)maskSS.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        B, H, S, D,
        scale
    );

    return Out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor big_bird_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor maskSS
);
"""

custom_ops_lib = load_inline(
    name="custom_bigbird_attention_ops_online_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["big_bird_attention_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    BigBird Attention module with a fused CUDA forward kernel for:
      scores = Q @ K^T / sqrt(d_k)
      scores.masked_fill(mask == 0, -1e9)
      softmax(scores)
      output = attn @ V

    Dropout is p=0.0 in the provided architecture and is omitted.
    """

    def __init__(self, d_model, n_heads, window_size, num_random_blocks):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.num_random_blocks = num_random_blocks
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self._ops = custom_ops_lib

    def create_bigbird_mask(self, seq_len, window_size, num_random_blocks, device):
        mask = torch.zeros(seq_len, seq_len, device=device)

        # 1) Local window attention
        half = window_size // 2
        for i in range(seq_len):
            start = max(0, i - half)
            end = min(seq_len, i + half + 1)
            mask[i, start:end] = 1

        # 2) Global attention (first and last tokens)
        mask[0, :] = 1
        mask[-1, :] = 1
        mask[:, 0] = 1
        mask[:, -1] = 1

        # 3) Random attention (match reference: per-row randperm on same device; no dedup needed for binary mask)
        for i in range(1, seq_len - 1):
            random_indices = torch.randperm(seq_len, device=device)[:num_random_blocks]
            mask[i, random_indices] = 1

        return mask

    def forward(self, x):
        B, S, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = K.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = V.view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        if x.is_cuda:
            maskSS = self.create_bigbird_mask(S, self.window_size, self.num_random_blocks, x.device)
            maskSS = maskSS.to(dtype=torch.float32).contiguous()
            O = self._ops.big_bird_attention_forward_cuda(Q, K, V, maskSS)
        else:
            mask = self.create_bigbird_mask(S, self.window_size, self.num_random_blocks, x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            O = torch.matmul(attn_weights, V)

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O