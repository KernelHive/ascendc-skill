import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA extension (D=64 optimized tiled kernel + fallback generic) ----

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

static __forceinline__ __device__ float4 ldg_f32x4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

// ---------------------- Fallback generic warp-per-row ----------------------
__global__ void fa2_online_warp_generic(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ K,   // [B,H,S,D]
    const float* __restrict__ V,   // [B,H,S,D]
    float* __restrict__ O,         // [B,H,S,D]
    int total_rows, int S, int D,
    float scale
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int i = row % S;
    int bh = row / S; // bh = b*H + h
    int base_bh = bh * S * D;

    const float* q_ptr = Q + base_bh + i * D;
    float* o_ptr = O + base_bh + i * D;

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
    for (int j = 0; j < S; j++) {
        const float* k_ptr = K + base_bh + j * D;
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

        o0 = o0 * alpha;
        o1 = o1 * alpha;
        o2 = o2 * alpha;
        o3 = o3 * alpha;

        const float* v_ptr = V + base_bh + j * D;
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

// ---------------------- Optimized D=64 kernel: larger Q tile, smaller K tile (fits 48KB SMEM) ----------------------
// CTA handles one (bh) and one q-tile. One warp per Q row in the tile.
// K/V are staged to shared and reused across all warps in CTA.
// SMEM bytes = 2*K_TILE*64*4. With K_TILE=48 => 24KB.
template<int Q_TILE, int K_TILE, int WARPS>
__global__ __launch_bounds__(WARPS*32, 2) void fa2_tiled_d64_q8k48(
    const float* __restrict__ Q,   // [BH,S,64]
    const float* __restrict__ K,   // [BH,S,64]
    const float* __restrict__ V,   // [BH,S,64]
    float* __restrict__ O,         // [BH,S,64]
    int BH, int S,
    float scale
) {
    int bh = (int)blockIdx.x;
    if (bh >= BH) return;

    int q_tile = (int)blockIdx.y;
    int q_start = q_tile * Q_TILE;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (warp >= WARPS) return;

    int qi = q_start + warp;
    bool valid_q = (qi < S);

    extern __shared__ float smem[];
    float* smemK = smem;
    float* smemV = smem + (K_TILE * 64);

    const float* q_ptr = Q + (bh * S + qi) * 64;
    float qA = 0.f, qB = 0.f;
    if (valid_q) {
        qA = ldg_f32(q_ptr + lane);
        qB = ldg_f32(q_ptr + lane + 32);
    }

    float oA = 0.f, oB = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    constexpr int vecs_per_row = 64 / 4; // 16 float4
    constexpr int total_vecs = K_TILE * vecs_per_row;

    for (int kt = 0; kt < S; kt += K_TILE) {
        int tile_len = min(K_TILE, S - kt);

        // cooperative load K/V tile -> shared (vectorized float4)
        for (int idx = tid; idx < total_vecs; idx += blockDim.x) {
            int r = idx / vecs_per_row;                 // 0..K_TILE-1
            int c4 = (idx - r * vecs_per_row) * 4;      // 0..60
            int j = kt + r;

            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (j < S) {
                const float* k_ptr = K + (bh * S + j) * 64 + c4;
                const float* v_ptr = V + (bh * S + j) * 64 + c4;
                kv = ldg_f32x4(k_ptr);
                vv = ldg_f32x4(v_ptr);
            }
            *reinterpret_cast<float4*>(&smemK[r * 64 + c4]) = kv;
            *reinterpret_cast<float4*>(&smemV[r * 64 + c4]) = vv;
        }
        __syncthreads();

        if (valid_q) {
            int r = 0;
            // unroll by 4 for better ILP (kept small to avoid register blowup)
            for (; r + 3 < tile_len; r += 4) {
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float* krow = smemK + (r + u) * 64;
                    float part = qA * krow[lane] + qB * krow[lane + 32];
                    float dot = warp_reduce_sum(part);
                    dot = __shfl_sync(0xffffffff, dot, 0) * scale;

                    float m_new = fmaxf(m, dot);
                    float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
                    float p = __expf(dot - m_new);

                    l = l * alpha + p;
                    oA *= alpha;
                    oB *= alpha;

                    const float* vrow = smemV + (r + u) * 64;
                    oA += p * vrow[lane];
                    oB += p * vrow[lane + 32];

                    m = m_new;
                }
            }
            for (; r < tile_len; r++) {
                const float* krow = smemK + r * 64;
                float part = qA * krow[lane] + qB * krow[lane + 32];
                float dot = warp_reduce_sum(part);
                dot = __shfl_sync(0xffffffff, dot, 0) * scale;

                float m_new = fmaxf(m, dot);
                float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
                float p = __expf(dot - m_new);

                l = l * alpha + p;
                oA *= alpha;
                oB *= alpha;

                const float* vrow = smemV + r * 64;
                oA += p * vrow[lane];
                oB += p * vrow[lane + 32];

                m = m_new;
            }
        }

        __syncthreads();
    }

    if (valid_q) {
        float inv_l = 1.0f / (l + 1e-9f);
        float* o_ptr = O + (bh * S + qi) * 64;
        o_ptr[lane] = oA * inv_l;
        o_ptr[lane + 32] = oB * inv_l;
    }
}

torch::Tensor flash_attention_v2_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,S,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");

    const int B = (int)Q.size(0);
    const int H = (int)Q.size(1);
    const int S = (int)Q.size(2);
    const int D = (int)Q.size(3);

    auto O = torch::empty_like(Q);

    const float* q = (const float*)Q.data_ptr<float>();
    const float* k = (const float*)K.data_ptr<float>();
    const float* v = (const float*)V.data_ptr<float>();
    float* o = (float*)O.data_ptr<float>();

    if (D == 64) {
        constexpr int Q_TILE = 8;     // increase reuse of staged K/V
        constexpr int K_TILE = 48;    // keep SMEM <= 24KB (K+V)
        constexpr int WARPS  = 8;
        constexpr int THREADS = WARPS * 32;

        dim3 grid((unsigned)(B * H), (unsigned)((S + Q_TILE - 1) / Q_TILE), 1);

        size_t smem_bytes = (size_t)(2 * K_TILE * 64) * sizeof(float);
        fa2_tiled_d64_q8k48<Q_TILE, K_TILE, WARPS><<<grid, THREADS, smem_bytes>>>(
            q, k, v, o, B * H, S, (float)scale
        );
    } else {
        int total_rows = B * H * S;
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;
        int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        fa2_online_warp_generic<<<blocks, THREADS>>>(q, k, v, o, total_rows, S, D, (float)scale);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_attention_v2_forward_cuda kernel launch failed: ", cudaGetErrorString(err));
    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor flash_attention_v2_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_flash_attention_v2_opt4_q8k48",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["flash_attention_v2_forward_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
    verbose=False,
)

# ---- Model definition using the custom op ----

class FlashAttentionV2New(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        scale = 1.0 / math.sqrt(self.d_k)
        O = custom_ops_lib.flash_attention_v2_forward_cuda(Q, K, V, scale)

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O


class ModelNew(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = FlashAttentionV2New(d_model=d_model, n_heads=n_heads)

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
    return [d_model, n_heads]