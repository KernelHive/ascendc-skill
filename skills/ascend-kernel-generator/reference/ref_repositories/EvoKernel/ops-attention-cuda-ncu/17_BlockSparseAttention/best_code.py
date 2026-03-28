import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------
# Custom CUDA extension: block-sparse attention (forward only)
# v6 improvements over baseline:
# - Specialized fast kernel for BS=32, D=64:
#   * 1 CTA per (B,H,NB) tile, 256 threads (8 warps)
#   * cooperative load of K and V into shared memory once per tile
#   * shared memory padding to reduce bank conflicts
#   * each warp computes 4 queries (32 queries/CTA)
#   * online softmax (no score materialization)
#   * lightweight register prefetch of next K/V row to reduce pipeline gaps
# - Keeps previous streaming q2 kernel as fallback for BS=32,D=64 if desired
# - Keeps general kernel fallback for D<=256
# ---------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline __device__ float warp_allreduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return __shfl_sync(0xffffffff, v, 0);
}

static inline __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// ----------------------------------------
// Fast path (dominant): BS=32, D=64
// 1 CTA per (b,h,nb), 256 threads (8 warps).
// Each warp computes 4 queries => 32 queries total.
// K/V staged into shared memory once.
// Adds D-padding in smem to mitigate bank conflicts.
// Adds lightweight prefetch of next K/V row into regs.
// ----------------------------------------
__global__ __launch_bounds__(256, 2)
void block_sparse_attn_fwd_bs32_d64_smem_q4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int NB,
    float inv_sqrt_d
) {
    const int BS = 32;
    const int D  = 64;
    // padding to reduce bank conflicts for lane and lane+32 patterns
    const int Dp = 68; // must be >= 64

    const int group = (int)blockIdx.x;
    int tmp = group;
    const int nb = tmp % NB; tmp /= NB;
    const int h  = tmp % H;  tmp /= H;
    const int b  = tmp;

    const int base = ((((b * H + h) * NB + nb) * BS) * D);

    extern __shared__ float smem[];
    float* Ksm = smem;                // [BS][Dp]
    float* Vsm = Ksm + BS * Dp;       // [BS][Dp]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5; // 0..7

    // cooperative load K and V into shared memory
    // Total elements: 2 * BS * D = 4096 floats.
    // We'll copy only D (64) and leave padding undefined.
    for (int idx = tid; idx < BS * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx - r * D;
        Ksm[r * Dp + c] = ldg_f32(K + base + r * D + c);
        Vsm[r * Dp + c] = ldg_f32(V + base + r * D + c);
    }
    __syncthreads();

    // Each warp handles 4 queries => total 32 queries
    const int q0 = warp * 4 + 0;
    const int q1 = warp * 4 + 1;
    const int q2 = warp * 4 + 2;
    const int q3 = warp * 4 + 3;

    // Load Q rows (each lane owns 2 dims: lane and lane+32)
    const float* q0_ptr = Q + base + q0 * D;
    const float* q1_ptr = Q + base + q1 * D;
    const float* q2_ptr = Q + base + q2 * D;
    const float* q3_ptr = Q + base + q3 * D;

    float q0a = ldg_f32(q0_ptr + lane), q0b = ldg_f32(q0_ptr + lane + 32);
    float q1a = ldg_f32(q1_ptr + lane), q1b = ldg_f32(q1_ptr + lane + 32);
    float q2a = ldg_f32(q2_ptr + lane), q2b = ldg_f32(q2_ptr + lane + 32);
    float q3a = ldg_f32(q3_ptr + lane), q3b = ldg_f32(q3_ptr + lane + 32);

    float m0 = -INFINITY, l0 = 0.0f, acc0a = 0.0f, acc0b = 0.0f;
    float m1 = -INFINITY, l1 = 0.0f, acc1a = 0.0f, acc1b = 0.0f;
    float m2 = -INFINITY, l2 = 0.0f, acc2a = 0.0f, acc2b = 0.0f;
    float m3 = -INFINITY, l3 = 0.0f, acc3a = 0.0f, acc3b = 0.0f;

    // Prefetch first row of K/V from smem into registers
    float ka = Ksm[0 * Dp + lane];
    float kb = Ksm[0 * Dp + lane + 32];
    float va = Vsm[0 * Dp + lane];
    float vb = Vsm[0 * Dp + lane + 32];

    #pragma unroll
    for (int kpos = 0; kpos < BS; ++kpos) {
        // prefetch next row (software pipelining-lite)
        float nka = 0.f, nkb = 0.f, nva = 0.f, nvb = 0.f;
        if (kpos + 1 < BS) {
            nka = Ksm[(kpos + 1) * Dp + lane];
            nkb = Ksm[(kpos + 1) * Dp + lane + 32];
            nva = Vsm[(kpos + 1) * Dp + lane];
            nvb = Vsm[(kpos + 1) * Dp + lane + 32];
        }

        float s0 = warp_allreduce_sum(q0a * ka + q0b * kb) * inv_sqrt_d;
        float s1 = warp_allreduce_sum(q1a * ka + q1b * kb) * inv_sqrt_d;
        float s2 = warp_allreduce_sum(q2a * ka + q2b * kb) * inv_sqrt_d;
        float s3 = warp_allreduce_sum(q3a * ka + q3b * kb) * inv_sqrt_d;

        float m0n = fmaxf(m0, s0); float a0 = __expf(m0 - m0n); float b0 = __expf(s0 - m0n); float l0n = l0 * a0 + b0;
        float m1n = fmaxf(m1, s1); float a1 = __expf(m1 - m1n); float b1 = __expf(s1 - m1n); float l1n = l1 * a1 + b1;
        float m2n = fmaxf(m2, s2); float a2 = __expf(m2 - m2n); float b2 = __expf(s2 - m2n); float l2n = l2 * a2 + b2;
        float m3n = fmaxf(m3, s3); float a3 = __expf(m3 - m3n); float b3 = __expf(s3 - m3n); float l3n = l3 * a3 + b3;

        acc0a = acc0a * a0 + b0 * va; acc0b = acc0b * a0 + b0 * vb;
        acc1a = acc1a * a1 + b1 * va; acc1b = acc1b * a1 + b1 * vb;
        acc2a = acc2a * a2 + b2 * va; acc2b = acc2b * a2 + b2 * vb;
        acc3a = acc3a * a3 + b3 * va; acc3b = acc3b * a3 + b3 * vb;

        m0 = m0n; l0 = l0n;
        m1 = m1n; l1 = l1n;
        m2 = m2n; l2 = l2n;
        m3 = m3n; l3 = l3n;

        // advance prefetched values
        ka = nka; kb = nkb; va = nva; vb = nvb;
    }

    float inv0 = 1.0f / (l0 + 1e-9f);
    float inv1 = 1.0f / (l1 + 1e-9f);
    float inv2 = 1.0f / (l2 + 1e-9f);
    float inv3 = 1.0f / (l3 + 1e-9f);

    float* o0_ptr = Out + base + q0 * D;
    float* o1_ptr = Out + base + q1 * D;
    float* o2_ptr = Out + base + q2 * D;
    float* o3_ptr = Out + base + q3 * D;

    o0_ptr[lane] = acc0a * inv0; o0_ptr[lane + 32] = acc0b * inv0;
    o1_ptr[lane] = acc1a * inv1; o1_ptr[lane + 32] = acc1b * inv1;
    o2_ptr[lane] = acc2a * inv2; o2_ptr[lane + 32] = acc2b * inv2;
    o3_ptr[lane] = acc3a * inv3; o3_ptr[lane + 32] = acc3b * inv3;
}

// ----------------------------------------
// Prior fast path: BS=32, D=64, 2 queries per warp (streaming global).
// Retained as fallback/alternative.
// ----------------------------------------
__global__ void block_sparse_attn_fwd_bs32_d64_q2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int NB,
    float inv_sqrt_d
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    int warp_linear = (int)blockIdx.x * warps_per_block + warp_in_block;

    const int groups = B * H * NB;
    const int warps_per_group = 16;
    const int total_warps = groups * warps_per_group;
    if (warp_linear >= total_warps) return;

    int tmp = warp_linear;
    const int qpair = tmp % warps_per_group; tmp /= warps_per_group;
    const int nb = tmp % NB; tmp /= NB;
    const int h  = tmp % H;  tmp /= H;
    const int b  = tmp;

    const int BS = 32;
    const int D  = 64;

    const int q0 = qpair;
    const int q1 = qpair + 16;

    const int base = ((((b * H + h) * NB + nb) * BS) * D);

    const float* q0_ptr = Q + base + q0 * D;
    const float* q1_ptr = Q + base + q1 * D;

    float q0a = ldg_f32(q0_ptr + lane);
    float q0b = ldg_f32(q0_ptr + lane + 32);
    float q1a = ldg_f32(q1_ptr + lane);
    float q1b = ldg_f32(q1_ptr + lane + 32);

    float m0 = -INFINITY, l0 = 0.0f;
    float m1 = -INFINITY, l1 = 0.0f;

    float acc0a = 0.0f, acc0b = 0.0f;
    float acc1a = 0.0f, acc1b = 0.0f;

    #pragma unroll
    for (int kpos = 0; kpos < 32; ++kpos) {
        const float* k_ptr = K + base + kpos * D;
        const float* v_ptr = V + base + kpos * D;

        float ka = ldg_f32(k_ptr + lane);
        float kb = ldg_f32(k_ptr + lane + 32);

        float dot0 = warp_allreduce_sum(q0a * ka + q0b * kb);
        float dot1 = warp_allreduce_sum(q1a * ka + q1b * kb);

        float s0 = dot0 * inv_sqrt_d;
        float s1 = dot1 * inv_sqrt_d;

        float m0_new = fmaxf(m0, s0);
        float a0 = __expf(m0 - m0_new);
        float b0 = __expf(s0 - m0_new);
        float l0_new = l0 * a0 + b0;

        float m1_new = fmaxf(m1, s1);
        float a1 = __expf(m1 - m1_new);
        float b1 = __expf(s1 - m1_new);
        float l1_new = l1 * a1 + b1;

        float va = ldg_f32(v_ptr + lane);
        float vb = ldg_f32(v_ptr + lane + 32);

        acc0a = acc0a * a0 + b0 * va;
        acc0b = acc0b * a0 + b0 * vb;

        acc1a = acc1a * a1 + b1 * va;
        acc1b = acc1b * a1 + b1 * vb;

        m0 = m0_new; l0 = l0_new;
        m1 = m1_new; l1 = l1_new;
    }

    float inv_l0 = 1.0f / (l0 + 1e-9f);
    float inv_l1 = 1.0f / (l1 + 1e-9f);

    float* o0_ptr = Out + base + q0 * D;
    float* o1_ptr = Out + base + q1 * D;

    o0_ptr[lane] = acc0a * inv_l0;
    o0_ptr[lane + 32] = acc0b * inv_l0;

    o1_ptr[lane] = acc1a * inv_l1;
    o1_ptr[lane + 32] = acc1b * inv_l1;
}

// ----------------------------------------
// General fallback: 1 query per warp, online softmax.
// Supports D <= 256.
// ----------------------------------------
__global__ void block_sparse_attn_fwd_general_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int NB, int BS, int D,
    float inv_sqrt_d
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    int q_linear = (int)blockIdx.x * warps_per_block + warp_in_block;
    const int total_q = B * H * NB * BS;
    if (q_linear >= total_q) return;

    int tmp = q_linear;
    const int qpos = tmp % BS; tmp /= BS;
    const int nb   = tmp % NB; tmp /= NB;
    const int h    = tmp % H;  tmp /= H;
    const int b    = tmp;

    const int base = ((((b * H + h) * NB + nb) * BS) * D);
    const float* q_ptr = Q + base + qpos * D;

    float m = -INFINITY;
    float l = 0.0f;

    float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
    const int chunks = (D + 31) >> 5;

    for (int kpos = 0; kpos < BS; ++kpos) {
        const float* k_ptr = K + base + kpos * D;
        const float* v_ptr = V + base + kpos * D;

        float partial = 0.0f;
        for (int d = lane; d < D; d += 32) {
            partial += ldg_f32(q_ptr + d) * ldg_f32(k_ptr + d);
        }
        float s = warp_allreduce_sum(partial) * inv_sqrt_d;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float beta  = __expf(s - m_new);
        float l_new = l * alpha + beta;

        if (chunks > 0) { int d = lane + 0;   if (d < D) acc0 = acc0 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 1) { int d = lane + 32;  if (d < D) acc1 = acc1 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 2) { int d = lane + 64;  if (d < D) acc2 = acc2 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 3) { int d = lane + 96;  if (d < D) acc3 = acc3 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 4) { int d = lane + 128; if (d < D) acc4 = acc4 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 5) { int d = lane + 160; if (d < D) acc5 = acc5 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 6) { int d = lane + 192; if (d < D) acc6 = acc6 * alpha + beta * ldg_f32(v_ptr + d); }
        if (chunks > 7) { int d = lane + 224; if (d < D) acc7 = acc7 * alpha + beta * ldg_f32(v_ptr + d); }

        m = m_new;
        l = l_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    float* out_ptr = Out + base + qpos * D;

    if (chunks > 0) { int d = lane + 0;   if (d < D) out_ptr[d] = acc0 * inv_l; }
    if (chunks > 1) { int d = lane + 32;  if (d < D) out_ptr[d] = acc1 * inv_l; }
    if (chunks > 2) { int d = lane + 64;  if (d < D) out_ptr[d] = acc2 * inv_l; }
    if (chunks > 3) { int d = lane + 96;  if (d < D) out_ptr[d] = acc3 * inv_l; }
    if (chunks > 4) { int d = lane + 128; if (d < D) out_ptr[d] = acc4 * inv_l; }
    if (chunks > 5) { int d = lane + 160; if (d < D) out_ptr[d] = acc5 * inv_l; }
    if (chunks > 6) { int d = lane + 192; if (d < D) out_ptr[d] = acc6 * inv_l; }
    if (chunks > 7) { int d = lane + 224; if (d < D) out_ptr[d] = acc7 * inv_l; }
}

torch::Tensor block_sparse_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 5, "Q must be [B,H,NB,BS,D]");
    TORCH_CHECK(K.dim() == 5, "K must be [B,H,NB,BS,D]");
    TORCH_CHECK(V.dim() == 5, "V must be [B,H,NB,BS,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q,K,V must have the same shape");

    const int B  = (int)Q.size(0);
    const int H  = (int)Q.size(1);
    const int NB = (int)Q.size(2);
    const int BS = (int)Q.size(3);
    const int D  = (int)Q.size(4);

    auto Out = torch::empty_like(Q);
    const float inv_sqrt_d = 1.0f / sqrtf((float)D);

    if (BS == 32 && D == 64) {
        // Prefer shared-memory tiled kernel
        const int threads = 256; // 8 warps
        const int blocks = B * H * NB;
        // shared memory: 2 * BS * Dp floats, Dp=68
        const int Dp = 68;
        const size_t shmem = (size_t)(2 * BS * Dp) * sizeof(float);

        block_sparse_attn_fwd_bs32_d64_smem_q4_kernel<<<blocks, threads, shmem>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, NB,
            inv_sqrt_d
        );
        return Out;
    }

    TORCH_CHECK(D <= 256, "General kernel supports D <= 256 (got D=", D, ")");
    const int total_q = B * H * NB * BS;

    const int threads = 128; // 4 warps
    const int warps_per_block = threads / 32;
    const int blocks = (total_q + warps_per_block - 1) / warps_per_block;

    block_sparse_attn_fwd_general_kernel<<<blocks, threads, 0>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        B, H, NB, BS, D,
        inv_sqrt_d
    );
    return Out;
}
"""

cpp_src = r"""
torch::Tensor block_sparse_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_block_sparse_attention_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["block_sparse_attention_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Block Sparse Attention mechanism with optimized fused intra-block attention kernel.
    """
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        assert seq_len % self.block_size == 0, \
            f"Sequence length {seq_len} must be divisible by block size {self.block_size}"
        n_blocks = seq_len // self.block_size

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        Qb = Q.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k).contiguous()
        Kb = K.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k).contiguous()
        Vb = V.view(batch_size, self.n_heads, n_blocks, self.block_size, self.d_k).contiguous()

        Ob = self.custom_ops_lib.block_sparse_attention_cuda(Qb, Kb, Vb)

        out = Ob.view(batch_size, self.n_heads, seq_len, self.d_k).transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out


def get_inputs():
    batch_size = 32
    seq_len = 512
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    d_model = 512
    n_heads = 8
    block_size = 32
    return [d_model, n_heads, block_size]