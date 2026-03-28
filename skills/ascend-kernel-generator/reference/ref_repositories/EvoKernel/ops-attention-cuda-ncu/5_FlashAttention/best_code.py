import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA extension: FlashAttention (single-pass online softmax, warp-per-row) ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_FP32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DTYPE_FP32(x)

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

// Generic kernel: one warp computes one attention row (bh, i). Each lane owns 4 dims.
__global__ void fa_online_warp_generic(
    const float* __restrict__ Q,   // [B,H,Sq,D]
    const float* __restrict__ K,   // [B,H,Skv,D]
    const float* __restrict__ V,   // [B,H,Skv,D]
    float* __restrict__ O,         // [B,H,Sq,D]
    int total_rows, int Sq, int Skv, int D,
    float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int i = row % Sq;
    int bh = row / Sq; // bh = b*H + h

    // Base pointers for bh slice
    int q_base = bh * Sq * D;
    int kv_base = bh * Skv * D;

    const float* q_ptr = Q + q_base + i * D;
    float* o_ptr = O + q_base + i * D;

    int d0 = lane * 4;

    float q0=0.f,q1=0.f,q2=0.f,q3=0.f;
    if (d0 + 0 < D) q0 = ldg_f32(q_ptr + d0 + 0);
    if (d0 + 1 < D) q1 = ldg_f32(q_ptr + d0 + 1);
    if (d0 + 2 < D) q2 = ldg_f32(q_ptr + d0 + 2);
    if (d0 + 3 < D) q3 = ldg_f32(q_ptr + d0 + 3);

    float o0=0.f,o1=0.f,o2=0.f,o3=0.f;

    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int j = 0; j < Skv; j++) {
        const float* k_ptr = K + kv_base + j * D;
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

        const float* v_ptr = V + kv_base + j * D;
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

// D=64 specialized: each lane owns dims lane and lane+32.
__global__ void fa_online_warp_d64(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int total_rows, int Sq, int Skv,
    float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int i = row % Sq;
    int bh = row / Sq;

    int q_base = bh * Sq * 64;
    int kv_base = bh * Skv * 64;

    const float* q_ptr = Q + q_base + i * 64;
    float* o_ptr = O + q_base + i * 64;

    float qA = ldg_f32(q_ptr + lane);
    float qB = ldg_f32(q_ptr + lane + 32);

    float oA = 0.f, oB = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    #pragma unroll 1
    for (int j = 0; j < Skv; j++) {
        const float* k_ptr = K + kv_base + j * 64;
        float part = qA * ldg_f32(k_ptr + lane) + qB * ldg_f32(k_ptr + lane + 32);

        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;
        oA *= alpha;
        oB *= alpha;

        const float* v_ptr = V + kv_base + j * 64;
        oA += p * ldg_f32(v_ptr + lane);
        oB += p * ldg_f32(v_ptr + lane + 32);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane] = oA * inv_l;
    o_ptr[lane + 32] = oB * inv_l;
}

torch::Tensor flash_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,Sq,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,Skv,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,Skv,D]");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0), "Batch mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1), "Head mismatch");
    TORCH_CHECK(K.size(2) == V.size(2), "K/V seq mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3), "D mismatch");

    const int B = (int)Q.size(0);
    const int H = (int)Q.size(1);
    const int Sq = (int)Q.size(2);
    const int Skv = (int)K.size(2);
    const int D = (int)Q.size(3);

    auto O = torch::empty_like(Q);

    int total_rows = B * H * Sq;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    const float* q = (const float*)Q.data_ptr<float>();
    const float* k = (const float*)K.data_ptr<float>();
    const float* v = (const float*)V.data_ptr<float>();
    float* o = (float*)O.data_ptr<float>();

    if (D == 64) {
        fa_online_warp_d64<<<blocks, THREADS>>>(q, k, v, o, total_rows, Sq, Skv, (float)scale);
    } else {
        fa_online_warp_generic<<<blocks, THREADS>>>(q, k, v, o, total_rows, Sq, Skv, D, (float)scale);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_attention_forward_cuda kernel launch failed: ", cudaGetErrorString(err));
    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor flash_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_flash_attention_opt2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["flash_attention_forward_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
    verbose=False,
)

# ---- Model using the custom op ----

class FlashAttentionNew(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size_q: int = 64, block_size_kv: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _flash_attention_forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.flash_attention_forward_cuda(Q, K, V, float(self.scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        O = self._flash_attention_forward(Q, K, V)  # [B,H,S,D]

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_o(O)


class ModelNew(nn.Module):
    def __init__(self, d_model, n_heads, block_size_q, block_size_kv):
        super().__init__()
        self.attn = FlashAttentionNew(
            d_model=d_model,
            n_heads=n_heads,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
        )

    def forward(self, x):
        return self.attn(x)


def get_inputs():
    batch_size = 32
    seq_len = 512
    d_model = 512
    return [torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    d_model = 512
    n_heads = 8
    block_size_q = 64
    block_size_kv = 64
    return [d_model, n_heads, block_size_q, block_size_kv]