import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static inline __device__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// -----------------------------
// Projection kernel (unchanged baseline v2, vectorized float4 when possible)
// -----------------------------
__global__ void linformer_project_kernel_v2(
    const float* __restrict__ K,   // [B,H,S,D]
    const float* __restrict__ V,   // [B,H,S,D]
    const float* __restrict__ E,   // [S,KK]
    const float* __restrict__ Fm,  // [S,KK]
    float* __restrict__ Kp,        // [B,H,KK,D]
    float* __restrict__ Vp,        // [B,H,KK,D]
    int B, int H, int S, int KK, int D,
    int vec4
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (vec4) {
        int D4 = D >> 2;
        long total4 = (long)B * H * KK * D4;
        if (idx >= total4) return;

        int d4 = (int)(idx % D4);
        long tmp = idx / D4;
        int kk = (int)(tmp % KK);
        tmp /= KK;
        int h = (int)(tmp % H);
        int b = (int)(tmp / H);

        float4 accK = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 accV = make_float4(0.f, 0.f, 0.f, 0.f);

        const float* Kbase = K + (((b * H + h) * S) * D);
        const float* Vbase = V + (((b * H + h) * S) * D);

        const float4* K4 = reinterpret_cast<const float4*>(Kbase);
        const float4* V4 = reinterpret_cast<const float4*>(Vbase);

        #pragma unroll 1
        for (int s = 0; s < S; ++s) {
            float e = __ldg(E + s * KK + kk);
            float f = __ldg(Fm + s * KK + kk);

            float4 kv = K4[s * D4 + d4];
            float4 vv = V4[s * D4 + d4];

            accK.x = fmaf(e, kv.x, accK.x);
            accK.y = fmaf(e, kv.y, accK.y);
            accK.z = fmaf(e, kv.z, accK.z);
            accK.w = fmaf(e, kv.w, accK.w);

            accV.x = fmaf(f, vv.x, accV.x);
            accV.y = fmaf(f, vv.y, accV.y);
            accV.z = fmaf(f, vv.z, accV.z);
            accV.w = fmaf(f, vv.w, accV.w);
        }

        float* Kp_out = Kp + (((b * H + h) * KK + kk) * D);
        float* Vp_out = Vp + (((b * H + h) * KK + kk) * D);
        reinterpret_cast<float4*>(Kp_out)[d4] = accK;
        reinterpret_cast<float4*>(Vp_out)[d4] = accV;
    } else {
        long total = (long)B * H * KK * D;
        if (idx >= total) return;

        int d = (int)(idx % D);
        long tmp = idx / D;
        int kk = (int)(tmp % KK);
        tmp /= KK;
        int h = (int)(tmp % H);
        int b = (int)(tmp / H);

        float accK = 0.f;
        float accV = 0.f;

        const float* Kbase = K + (((b * H + h) * S) * D);
        const float* Vbase = V + (((b * H + h) * S) * D);

        #pragma unroll 1
        for (int s = 0; s < S; ++s) {
            float e = __ldg(E + s * KK + kk);
            float f = __ldg(Fm + s * KK + kk);
            float kval = Kbase[s * D + d];
            float vval = Vbase[s * D + d];
            accK = fmaf(e, kval, accK);
            accV = fmaf(f, vval, accV);
        }

        Kp[(((b * H + h) * KK + kk) * D + d)] = accK;
        Vp[(((b * H + h) * KK + kk) * D + d)] = accV;
    }
}

// -----------------------------
// Attention kernel (optimized): q-tiling by 2 for D=64
// One warp computes two query rows (q0 and q1=q0+1).
// K/V are streamed once per kk and reused for both queries.
// Vectorize K/V global loads via float4.
// -----------------------------
__global__ __launch_bounds__(64, 6)
void linformer_attn_warp64_q2_kernel(
    const float* __restrict__ Q,   // [B,H,S,64]
    const float* __restrict__ Kp,  // [B,H,KK,64]
    const float* __restrict__ Vp,  // [B,H,KK,64]
    float* __restrict__ Out,       // [B,H,S,64]
    int B, int H, int S, int KK,
    float scale
) {
    // 2 warps per block (64 threads)
    int tid = (int)threadIdx.x;
    int warp_in_block = tid >> 5;         // 0..1
    int lane = tid & 31;

    int warp_global = ((int)blockIdx.x << 1) + warp_in_block; // each block holds 2 warps
    int base_q_idx = warp_global << 1; // each warp handles 2 queries
    int total_q = B * H * S;

    if (base_q_idx >= total_q) return;

    // Decode q0
    int idx0 = base_q_idx;
    int q0 = idx0 % S;
    int tmp0 = idx0 / S;
    int h = tmp0 % H;
    int b = tmp0 / H;

    int idx1 = base_q_idx + 1;
    int q1_valid = (idx1 < total_q) ? 1 : 0;
    int q1 = q0 + 1;
    if (q1 >= S) q1_valid = 0; // since idx1 maps to next (b,h) when q0==S-1

    const float* qptr0 = Q + (((b * H + h) * S + q0) * 64);
    const float* qptr1 = q1_valid ? (Q + (((b * H + h) * S + q1) * 64)) : nullptr;

    const float* kbase = Kp + (((b * H + h) * KK) * 64);
    const float* vbase = Vp + (((b * H + h) * KK) * 64);

    float* outptr0 = Out + (((b * H + h) * S + q0) * 64);
    float* outptr1 = q1_valid ? (Out + (((b * H + h) * S + q1) * 64)) : nullptr;

    int d0 = (lane << 1);
    float q00 = __ldg(qptr0 + d0);
    float q01 = __ldg(qptr0 + d0 + 1);

    float q10 = 0.f, q11 = 0.f;
    if (q1_valid) {
        q10 = __ldg(qptr1 + d0);
        q11 = __ldg(qptr1 + d0 + 1);
    }

    // Online softmax state for both queries
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;

    float o00 = 0.f, o01 = 0.f;
    float o10 = 0.f, o11 = 0.f;

    // float4 views for K/V for vectorized loads
    const float4* k4 = reinterpret_cast<const float4*>(kbase);
    const float4* v4 = reinterpret_cast<const float4*>(vbase);
    int d4 = lane >> 1;         // 0..15
    int off_in4 = (lane & 1) << 1; // 0 or 2 (select xy or zw)

    #pragma unroll 1
    for (int kk = 0; kk < KK; ++kk) {
        float4 kv = __ldg(k4 + kk * 16 + d4);
        float4 vv = __ldg(v4 + kk * 16 + d4);

        float k0 = (off_in4 == 0) ? kv.x : kv.z;
        float k1 = (off_in4 == 0) ? kv.y : kv.w;

        // dot for q0
        float part0 = fmaf(q00, k0, q01 * k1);
        float dot0 = warp_sum(part0);
        float s0 = __shfl_sync(0xffffffffu, dot0, 0) * scale;

        float m0_new = fmaxf(m0, s0);
        float a0 = __expf(m0 - m0_new);
        float p0 = __expf(s0 - m0_new);
        float l0_new = l0 * a0 + p0;

        float v0 = (off_in4 == 0) ? vv.x : vv.z;
        float v1 = (off_in4 == 0) ? vv.y : vv.w;

        o00 = o00 * a0 + p0 * v0;
        o01 = o01 * a0 + p0 * v1;

        m0 = m0_new;
        l0 = l0_new;

        // dot for q1 (if valid)
        if (q1_valid) {
            float part1 = fmaf(q10, k0, q11 * k1);
            float dot1 = warp_sum(part1);
            float s1 = __shfl_sync(0xffffffffu, dot1, 0) * scale;

            float m1_new = fmaxf(m1, s1);
            float a1 = __expf(m1 - m1_new);
            float p1 = __expf(s1 - m1_new);
            float l1_new = l1 * a1 + p1;

            o10 = o10 * a1 + p1 * v0;
            o11 = o11 * a1 + p1 * v1;

            m1 = m1_new;
            l1 = l1_new;
        }
    }

    float inv_l0 = 1.f / (l0 + 1e-9f);
    outptr0[d0] = o00 * inv_l0;
    outptr0[d0 + 1] = o01 * inv_l0;

    if (q1_valid) {
        float inv_l1 = 1.f / (l1 + 1e-9f);
        outptr1[d0] = o10 * inv_l1;
        outptr1[d0 + 1] = o11 * inv_l1;
    }
}

// Generic fallback attention (online softmax), one thread per (b,h,q)
__global__ void linformer_attn_generic_kernel(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ Kp,  // [B,H,KK,D]
    const float* __restrict__ Vp,  // [B,H,KK,D]
    float* __restrict__ Out,       // [B,H,S,D]
    int B, int H, int S, int KK, int D,
    float scale
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * H * S;
    if (idx >= total) return;

    int q = (int)(idx % S);
    long tmp = idx / S;
    int h = (int)(tmp % H);
    int b = (int)(tmp / H);

    const float* qptr = Q + (((b * H + h) * S + q) * D);
    const float* kbase = Kp + (((b * H + h) * KK) * D);
    const float* vbase = Vp + (((b * H + h) * KK) * D);
    float* outptr = Out + (((b * H + h) * S + q) * D);

    float m = -INFINITY;
    float l = 0.f;

    for (int d = 0; d < D; ++d) outptr[d] = 0.f;

    #pragma unroll 1
    for (int kk = 0; kk < KK; ++kk) {
        const float* kptr = kbase + kk * D;
        const float* vptr = vbase + kk * D;

        float dot = 0.f;
        for (int d = 0; d < D; ++d) dot = fmaf(qptr[d], kptr[d], dot);
        float s = dot * scale;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p = __expf(s - m_new);
        float l_new = l * alpha + p;

        for (int d = 0; d < D; ++d) {
            outptr[d] = outptr[d] * alpha + p * vptr[d];
        }

        m = m_new;
        l = l_new;
    }

    float inv_l = 1.f / (l + 1e-9f);
    for (int d = 0; d < D; ++d) outptr[d] *= inv_l;
}

torch::Tensor linformer_attention_cuda(
    torch::Tensor Q, // [B,H,S,D] float32
    torch::Tensor K, // [B,H,S,D] float32
    torch::Tensor V, // [B,H,S,D] float32
    torch::Tensor E, // [S,KK] float32
    torch::Tensor Fm // [S,KK] float32
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V); CHECK_CUDA(E); CHECK_CUDA(Fm);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V); CHECK_CONTIGUOUS(E); CHECK_CONTIGUOUS(Fm);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V); CHECK_FLOAT(E); CHECK_FLOAT(Fm);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,S,D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");
    TORCH_CHECK(E.dim() == 2, "E must be [S,KK]");
    TORCH_CHECK(Fm.dim() == 2, "F must be [S,KK]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);
    int KK = (int)E.size(1);

    TORCH_CHECK((int)E.size(0) == S, "E first dim must equal S");
    TORCH_CHECK((int)Fm.size(0) == S, "F first dim must equal S");
    TORCH_CHECK((int)Fm.size(1) == KK, "F second dim must equal E second dim");

    auto opts = Q.options();
    auto Kp = torch::empty({B, H, KK, D}, opts);
    auto Vp = torch::empty({B, H, KK, D}, opts);
    auto Out = torch::empty({B, H, S, D}, opts);

    auto is_aligned_16_host = [](const void* p) -> bool {
        return ((uintptr_t)p & 0xF) == 0;
    };
    int vec4 = 0;
    if ((D % 4) == 0 &&
        is_aligned_16_host(Q.data_ptr()) &&
        is_aligned_16_host(K.data_ptr()) &&
        is_aligned_16_host(V.data_ptr()) &&
        is_aligned_16_host(Kp.data_ptr()) &&
        is_aligned_16_host(Vp.data_ptr())) {
        vec4 = 1;
    }

    // Projection launch
    const int proj_threads = 256;
    long proj_total = vec4 ? (long)B * H * KK * (D >> 2) : (long)B * H * KK * D;
    int proj_blocks = (int)((proj_total + proj_threads - 1) / proj_threads);

    linformer_project_kernel_v2<<<proj_blocks, proj_threads>>>(
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (const float*)E.data_ptr<float>(),
        (const float*)Fm.data_ptr<float>(),
        (float*)Kp.data_ptr<float>(),
        (float*)Vp.data_ptr<float>(),
        B, H, S, KK, D, vec4
    );

    float scale = 1.0f / sqrtf((float)D);

    if (D == 64) {
        // 2 warps/block, each warp handles 2 queries => 4 queries per block
        int total_rows = B * H * S;
        int warps_needed = (total_rows + 1) / 2;      // each warp covers 2 rows
        int blocks = (warps_needed + 1) / 2;          // 2 warps per block
        linformer_attn_warp64_q2_kernel<<<blocks, 64>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)Kp.data_ptr<float>(),
            (const float*)Vp.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, KK, scale
        );
    } else {
        const int attn_threads = 256;
        long attn_total = (long)B * H * S;
        int attn_blocks = (int)((attn_total + attn_threads - 1) / attn_threads);
        linformer_attn_generic_kernel<<<attn_blocks, attn_threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)Kp.data_ptr<float>(),
            (const float*)Vp.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, KK, D, scale
        );
    }

    return Out;
}
"""

cpp_src = r"""
torch::Tensor linformer_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor E,
    torch::Tensor Fm
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_linformer_attention_opt_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["linformer_attention_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Linformer Attention with custom CUDA:
    - Vectorized projection kernel (float4) when aligned
    - D=64 fast path: warp-level online softmax with q-tiling=2 and float4 K/V loads
    """

    def __init__(self, d_model, n_heads, seq_len, k):
        super().__init__()
        assert d_model % n_heads == 0
        assert k < seq_len

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

        self.custom_ops = custom_ops_lib

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        if seq_len != self.seq_len:
            E = F.adaptive_avg_pool1d(self.E.unsqueeze(0).transpose(1, 2), seq_len).transpose(1, 2).squeeze(0)
            F_proj = F.adaptive_avg_pool1d(self.F.unsqueeze(0).transpose(1, 2), seq_len).transpose(1, 2).squeeze(0)
        else:
            E = self.E
            F_proj = self.F

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        if not Q.is_cuda:
            Kp = torch.matmul(E.T.unsqueeze(0).unsqueeze(0), K)
            Vp = torch.matmul(F_proj.T.unsqueeze(0).unsqueeze(0), V)
            scores = torch.matmul(Q, Kp.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, Vp)
        else:
            Qf = Q.to(torch.float32)
            Kf = K.to(torch.float32)
            Vf = V.to(torch.float32)
            Ef = E.contiguous().to(torch.float32)
            Ff = F_proj.contiguous().to(torch.float32)

            out = self.custom_ops.linformer_attention_cuda(Qf.contiguous(), Kf.contiguous(), Vf.contiguous(), Ef, Ff)
            if out.dtype != Q.dtype:
                out = out.to(Q.dtype)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out