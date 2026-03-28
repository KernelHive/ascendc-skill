import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
static __device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// ------------------------------------
// Baseline general-D kernel (correct).
// 3 passes over T: max, sumexp, output.
// ------------------------------------
__global__ void cross_attn_forward_baseline(
    const float* __restrict__ Q, // [B,H,S,D]
    const float* __restrict__ K, // [B,H,T,D]
    const float* __restrict__ V, // [B,H,T,D]
    float* __restrict__ O,       // [B,H,S,D]
    int B, int H, int S, int T, int D,
    float inv_sqrt_d
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    const float* Qrow = Q + ((b * H + h) * (int64_t)S + s) * (int64_t)D;
    const float* Kbh  = K + (b * H + h) * (int64_t)T * (int64_t)D;
    const float* Vbh  = V + (b * H + h) * (int64_t)T * (int64_t)D;
    float* Orow       = O + ((b * H + h) * (int64_t)S + s) * (int64_t)D;

    float thread_max = -INFINITY;
    for (int tpos = tid; tpos < T; tpos += blockDim.x) {
        const float* Krow = Kbh + (int64_t)tpos * (int64_t)D;
        float dot = 0.f;
        #pragma unroll 4
        for (int d = 0; d < D; d++) dot = fmaf(Qrow[d], Krow[d], dot);
        dot *= inv_sqrt_d;
        thread_max = fmaxf(thread_max, dot);
    }

    __shared__ float smem_max[32];
    float wmax = warp_reduce_max(thread_max);
    if (lane == 0) smem_max[warp] = wmax;
    __syncthreads();

    if (warp == 0) {
        int nwarp = (blockDim.x + 31) >> 5;
        float val = (lane < nwarp) ? smem_max[lane] : -INFINITY;
        float r = warp_reduce_max(val);
        if (lane == 0) smem_max[0] = r;
    }
    __syncthreads();
    float block_max = smem_max[0];

    float thread_sum = 0.f;
    for (int tpos = tid; tpos < T; tpos += blockDim.x) {
        const float* Krow = Kbh + (int64_t)tpos * (int64_t)D;
        float dot = 0.f;
        #pragma unroll 4
        for (int d = 0; d < D; d++) dot = fmaf(Qrow[d], Krow[d], dot);
        float score = dot * inv_sqrt_d;
        thread_sum += __expf(score - block_max);
    }

    __shared__ float smem_sum[32];
    float wsum = warp_reduce_sum(thread_sum);
    if (lane == 0) smem_sum[warp] = wsum;
    __syncthreads();

    if (warp == 0) {
        int nwarp = (blockDim.x + 31) >> 5;
        float val = (lane < nwarp) ? smem_sum[lane] : 0.f;
        float r = warp_reduce_sum(val);
        if (lane == 0) smem_sum[0] = r;
    }
    __syncthreads();
    float block_sum = smem_sum[0];
    float inv_denom = 1.f / (block_sum + 1e-9f);

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int tpos = 0; tpos < T; tpos++) {
            const float* Krow = Kbh + (int64_t)tpos * (int64_t)D;
            const float* Vrow = Vbh + (int64_t)tpos * (int64_t)D;
            float dot = 0.f;
            #pragma unroll 4
            for (int kk = 0; kk < D; kk++) dot = fmaf(Qrow[kk], Krow[kk], dot);
            float score = dot * inv_sqrt_d;
            float w = __expf(score - block_max) * inv_denom;
            acc = fmaf(w, Vrow[d], acc);
        }
        Orow[d] = acc;
    }
}

// ------------------------------------
// Fast path: D=64, 3 queries per warp.
// Online softmax (lane0 owns m/l), safe unroll-by-2 over T.
// No per-iteration branching.
// ------------------------------------
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4) void cross_attn_forward_warp3q_d64_u2(
    const float* __restrict__ Q, // [B,H,S,64]
    const float* __restrict__ K, // [B,H,T,64]
    const float* __restrict__ V, // [B,H,T,64]
    float* __restrict__ O,       // [B,H,S,64]
    int B, int H, int S, int T,
    float inv_sqrt_d
) {
    int b = blockIdx.x;
    int h = blockIdx.y;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    // 3 queries per warp
    int warp_q_base = ((int)blockIdx.z * WARPS_PER_BLOCK + warp_id) * 3;
    int s0 = warp_q_base + 0;
    int s1 = warp_q_base + 1;
    int s2 = warp_q_base + 2;

    if (s0 >= S) return;

    const float* Kbh  = K + (b * H + h) * (int64_t)T * 64;
    const float* Vbh  = V + (b * H + h) * (int64_t)T * 64;

    const float* Qrow0 = Q + ((b * H + h) * (int64_t)S + s0) * 64;
    const float* Qrow1 = (s1 < S) ? (Q + ((b * H + h) * (int64_t)S + s1) * 64) : nullptr;
    const float* Qrow2 = (s2 < S) ? (Q + ((b * H + h) * (int64_t)S + s2) * 64) : nullptr;

    float* Orow0 = O + ((b * H + h) * (int64_t)S + s0) * 64;
    float* Orow1 = (s1 < S) ? (O + ((b * H + h) * (int64_t)S + s1) * 64) : nullptr;
    float* Orow2 = (s2 < S) ? (O + ((b * H + h) * (int64_t)S + s2) * 64) : nullptr;

    bool has1 = (s1 < S);
    bool has2 = (s2 < S);

    // Each lane owns two dims: lane and lane+32
    float q00 = ldg_f32(Qrow0 + lane);
    float q01 = ldg_f32(Qrow0 + lane + 32);

    float q10=0.f,q11=0.f,q20=0.f,q21=0.f;
    if (has1) { q10 = ldg_f32(Qrow1 + lane); q11 = ldg_f32(Qrow1 + lane + 32); }
    if (has2) { q20 = ldg_f32(Qrow2 + lane); q21 = ldg_f32(Qrow2 + lane + 32); }

    // Lane0 softmax scalars
    float m0=-INFINITY,l0=0.f,m1=-INFINITY,l1=0.f,m2=-INFINITY,l2=0.f;

    // Numerator accumulators per lane (2 dims per query)
    float o00=0.f,o01=0.f,o10=0.f,o11=0.f,o20=0.f,o21=0.f;

    auto process_one = [&](float k0, float k1, float v0, float v1) {
        // score0
        float dot0 = warp_reduce_sum(fmaf(q00, k0, q01 * k1));
        dot0 = __shfl_sync(0xffffffff, dot0, 0);
        float s0f = dot0 * inv_sqrt_d;

        float s1f=0.f,s2f=0.f;
        if (has1) { float d = warp_reduce_sum(fmaf(q10, k0, q11 * k1)); d = __shfl_sync(0xffffffff, d, 0); s1f = d * inv_sqrt_d; }
        if (has2) { float d = warp_reduce_sum(fmaf(q20, k0, q21 * k1)); d = __shfl_sync(0xffffffff, d, 0); s2f = d * inv_sqrt_d; }

        // row0 update
        float scale0=1.f,p0=0.f;
        if (lane == 0) {
            float m_new = fmaxf(m0, s0f);
            scale0 = __expf(m0 - m_new);
            p0     = __expf(s0f - m_new);
            l0 = l0 * scale0 + p0;
            m0 = m_new;
        }
        scale0 = __shfl_sync(0xffffffff, scale0, 0);
        p0     = __shfl_sync(0xffffffff, p0, 0);
        o00 = o00 * scale0 + p0 * v0;
        o01 = o01 * scale0 + p0 * v1;

        if (has1) {
            float scale=1.f,p=0.f;
            if (lane == 0) {
                float m_new = fmaxf(m1, s1f);
                scale = __expf(m1 - m_new);
                p     = __expf(s1f - m_new);
                l1 = l1 * scale + p;
                m1 = m_new;
            }
            scale = __shfl_sync(0xffffffff, scale, 0);
            p     = __shfl_sync(0xffffffff, p, 0);
            o10 = o10 * scale + p * v0;
            o11 = o11 * scale + p * v1;
        }

        if (has2) {
            float scale=1.f,p=0.f;
            if (lane == 0) {
                float m_new = fmaxf(m2, s2f);
                scale = __expf(m2 - m_new);
                p     = __expf(s2f - m_new);
                l2 = l2 * scale + p;
                m2 = m_new;
            }
            scale = __shfl_sync(0xffffffff, scale, 0);
            p     = __shfl_sync(0xffffffff, p, 0);
            o20 = o20 * scale + p * v0;
            o21 = o21 * scale + p * v1;
        }
    };

    int tpos = 0;
    int T2 = T & ~1;
    for (; tpos < T2; tpos += 2) {
        const float* Krow0 = Kbh + (int64_t)tpos * 64;
        const float* Vrow0 = Vbh + (int64_t)tpos * 64;
        const float* Krow1 = Kbh + (int64_t)(tpos + 1) * 64;
        const float* Vrow1 = Vbh + (int64_t)(tpos + 1) * 64;

        float k0a = ldg_f32(Krow0 + lane);
        float k1a = ldg_f32(Krow0 + lane + 32);
        float v0a = ldg_f32(Vrow0 + lane);
        float v1a = ldg_f32(Vrow0 + lane + 32);

        float k0b = ldg_f32(Krow1 + lane);
        float k1b = ldg_f32(Krow1 + lane + 32);
        float v0b = ldg_f32(Vrow1 + lane);
        float v1b = ldg_f32(Vrow1 + lane + 32);

        process_one(k0a, k1a, v0a, v1a);
        process_one(k0b, k1b, v0b, v1b);
    }

    if (tpos < T) {
        const float* Krow = Kbh + (int64_t)tpos * 64;
        const float* Vrow = Vbh + (int64_t)tpos * 64;
        float k0 = ldg_f32(Krow + lane);
        float k1 = ldg_f32(Krow + lane + 32);
        float v0 = ldg_f32(Vrow + lane);
        float v1 = ldg_f32(Vrow + lane + 32);
        process_one(k0, k1, v0, v1);
    }

    float inv_l0 = 1.f / (__shfl_sync(0xffffffff, l0, 0) + 1e-9f);
    Orow0[lane]      = o00 * inv_l0;
    Orow0[lane + 32] = o01 * inv_l0;

    if (has1) {
        float inv_l = 1.f / (__shfl_sync(0xffffffff, l1, 0) + 1e-9f);
        Orow1[lane]      = o10 * inv_l;
        Orow1[lane + 32] = o11 * inv_l;
    }
    if (has2) {
        float inv_l = 1.f / (__shfl_sync(0xffffffff, l2, 0) + 1e-9f);
        Orow2[lane]      = o20 * inv_l;
        Orow2[lane + 32] = o21 * inv_l;
    }
}

torch::Tensor cross_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,T,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,T,D]");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(1) == K.size(1) && Q.size(3) == K.size(3), "K shape mismatch");
    TORCH_CHECK(V.size(0) == K.size(0) && V.size(1) == K.size(1) && V.size(2) == K.size(2) && V.size(3) == K.size(3), "V shape mismatch");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);
    int T = (int)K.size(2);

    auto O = torch::empty({B, H, S, D}, Q.options());
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    if (D == 64) {
        constexpr int WARPS_PER_BLOCK = 2;
        int threads = WARPS_PER_BLOCK * 32;
        int warps_total = (S + 2) / 3;
        int grid_z = (warps_total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        dim3 grid(B, H, grid_z);
        cross_attn_forward_warp3q_d64_u2<WARPS_PER_BLOCK><<<grid, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, S, T, inv_sqrt_d
        );
    } else {
        dim3 grid(B, H, S);
        int threads = 256;
        if (T < 256) threads = 128;
        if (T < 128) threads = 64;
        cross_attn_forward_baseline<<<grid, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, S, T, D, inv_sqrt_d
        );
    }
    return O;
}
"""

cpp_source = r"""
torch::Tensor cross_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_cross_attention_ops_opt9_warp3q_u2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cross_attention_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Cross-Attention layer using a custom CUDA fused attention core (forward only).
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.custom_ops = custom_ops_lib

    def forward(self, query, context):
        B, S, _ = query.shape
        T = context.size(1)

        Q = self.W_q(query).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(context).view(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(context).view(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        attn_out = self.custom_ops.cross_attention_forward_cuda(Q, K, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(attn_out)
        out = self.dropout(out)
        return out