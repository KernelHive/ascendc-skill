import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    // assumes full warp participation
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

static __device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

// -------------------- Fast path: 2-query-per-warp, Dk=Dv=64, Nk==49 --------------------
// Each warp computes two rows (qpos, qpos+1) within same (b,h) to reuse K/V loads.
// Uses double-buffer prefetch of K/V to overlap memory latency.
__global__ __launch_bounds__(128, 3) void sdpa_online_warp2_d64_nk49_prefetch(
    const float* __restrict__ q,   // [B,H,Nq,64]
    const float* __restrict__ k,   // [B,H,Nk,64]
    const float* __restrict__ v,   // [B,H,Nk,64]
    float* __restrict__ out,       // [B,H,Nq,64]
    int B, int H, int Nq, int Nk,
    float scale
) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    // each warp handles a "pair-row": (b,h,qpos_pair) where qpos_pair steps by 2
    int pair_row = (int)blockIdx.x * warps_per_block + warp;
    int total_pairs = B * H * ((Nq + 1) >> 1);
    if (pair_row >= total_pairs) return;

    int qpair = pair_row % ((Nq + 1) >> 1);
    int tmp   = pair_row / ((Nq + 1) >> 1);
    int hidx  = tmp % H;
    int bidx  = tmp / H;

    int qpos0 = qpair * 2;
    int qpos1 = qpos0 + 1;
    bool has1 = (qpos1 < Nq);

    const float* q_ptr0 = q + (((bidx * H + hidx) * Nq + qpos0) * 64);
    const float* q_ptr1 = has1 ? (q + (((bidx * H + hidx) * Nq + qpos1) * 64)) : nullptr;

    const float* k_ptr = k + ((bidx * H + hidx) * Nk * 64);
    const float* v_ptr = v + ((bidx * H + hidx) * Nk * 64);

    float* o_ptr0 = out + (((bidx * H + hidx) * Nq + qpos0) * 64);
    float* o_ptr1 = has1 ? (out + (((bidx * H + hidx) * Nq + qpos1) * 64)) : nullptr;

    // Load Q for both rows into registers (2 elems per lane each)
    float q00 = ldg_f32(q_ptr0 + lane);
    float q01 = ldg_f32(q_ptr0 + lane + 32);
    float q10 = 0.f, q11 = 0.f;
    if (has1) {
        q10 = ldg_f32(q_ptr1 + lane);
        q11 = ldg_f32(q_ptr1 + lane + 32);
    }

    // Online softmax state for both rows
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;
    float o00 = 0.f, o01 = 0.f;
    float o10 = 0.f, o11 = 0.f;

    // Double-buffered prefetch for K/V (scalar per lane, two halves)
    const float* k_cur = k_ptr;
    const float* v_cur = v_ptr;

    float k0a = ldg_f32(k_cur + lane);
    float k0b = ldg_f32(k_cur + lane + 32);
    float v0a = ldg_f32(v_cur + lane);
    float v0b = ldg_f32(v_cur + lane + 32);

    // Prefetch next
    const float* k_nxt = k_cur + 64;
    const float* v_nxt = v_cur + 64;

    float k1a = 0.f, k1b = 0.f, v1a = 0.f, v1b = 0.f;
    if (1 < Nk) {
        k1a = ldg_f32(k_nxt + lane);
        k1b = ldg_f32(k_nxt + lane + 32);
        v1a = ldg_f32(v_nxt + lane);
        v1b = ldg_f32(v_nxt + lane + 32);
    }

    #pragma unroll
    for (int key = 0; key < 49; key++) {
        if (key >= Nk) break;

        // current buffered values are (k0*, v0*). next buffered are (k1*, v1*)
        // prefetch key+2 into (k2*, v2*) for next iteration swap
        float k2a = 0.f, k2b = 0.f, v2a = 0.f, v2b = 0.f;
        if (key + 2 < Nk) {
            const float* k2p = k_ptr + (key + 2) * 64;
            const float* v2p = v_ptr + (key + 2) * 64;
            k2a = ldg_f32(k2p + lane);
            k2b = ldg_f32(k2p + lane + 32);
            v2a = ldg_f32(v2p + lane);
            v2b = ldg_f32(v2p + lane + 32);
        }

        // Dot for row0
        float partial0 = fmaf(q00, k0a, q01 * k0b);
        float dot0 = warp_reduce_sum(partial0);
        dot0 = __shfl_sync(0xffffffff, dot0, 0) * scale;

        float m0_new = fmaxf(m0, dot0);
        float a0 = __expf(m0 - m0_new);
        float b0 = __expf(dot0 - m0_new);
        float l0_new = fmaf(a0, l0, b0);
        o00 = fmaf(b0, v0a, o00 * a0);
        o01 = fmaf(b0, v0b, o01 * a0);
        m0 = m0_new;
        l0 = l0_new;

        // Dot for row1 (if present)
        if (has1) {
            float partial1 = fmaf(q10, k0a, q11 * k0b);
            float dot1 = warp_reduce_sum(partial1);
            dot1 = __shfl_sync(0xffffffff, dot1, 0) * scale;

            float m1_new = fmaxf(m1, dot1);
            float a1 = __expf(m1 - m1_new);
            float b1 = __expf(dot1 - m1_new);
            float l1_new = fmaf(a1, l1, b1);
            o10 = fmaf(b1, v0a, o10 * a1);
            o11 = fmaf(b1, v0b, o11 * a1);
            m1 = m1_new;
            l1 = l1_new;
        }

        // advance pipeline: (k1,v1) becomes current, (k2,v2) becomes next
        k0a = k1a; k0b = k1b; v0a = v1a; v0b = v1b;
        k1a = k2a; k1b = k2b; v1a = v2a; v1b = v2b;
    }

    float inv0 = 1.0f / (l0 + 1e-9f);
    o_ptr0[lane]      = o00 * inv0;
    o_ptr0[lane + 32] = o01 * inv0;

    if (has1) {
        float inv1 = 1.0f / (l1 + 1e-9f);
        o_ptr1[lane]      = o10 * inv1;
        o_ptr1[lane + 32] = o11 * inv1;
    }
}

// -------------------- Fast path: 1-query-per-warp, Dk=Dv=64 --------------------
__global__ __launch_bounds__(256, 2) void sdpa_online_warp_d64_nk49_prefetch(
    const float* __restrict__ q,   // [B,H,Nq,64]
    const float* __restrict__ k,   // [B,H,Nk,64]
    const float* __restrict__ v,   // [B,H,Nk,64]
    float* __restrict__ out,       // [B,H,Nq,64]
    int B, int H, int Nq, int Nk,
    float scale
) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp;
    int total_rows = B * H * Nq;
    if (row >= total_rows) return;

    int qpos = row % Nq;
    int tmp  = row / Nq;
    int hidx = tmp % H;
    int bidx = tmp / H;

    const float* q_ptr = q + (((bidx * H + hidx) * Nq + qpos) * 64);
    const float* k_ptr = k + ((bidx * H + hidx) * Nk * 64);
    const float* v_ptr = v + ((bidx * H + hidx) * Nk * 64);
    float* o_ptr       = out + (((bidx * H + hidx) * Nq + qpos) * 64);

    float q0 = ldg_f32(q_ptr + lane);
    float q1 = ldg_f32(q_ptr + lane + 32);

    float o0 = 0.f, o1 = 0.f;
    float m  = -INFINITY;
    float l  = 0.f;

    const float* k_cur = k_ptr;
    const float* v_cur = v_ptr;
    float k0  = ldg_f32(k_cur + lane);
    float k1  = ldg_f32(k_cur + lane + 32);
    float vv0 = ldg_f32(v_cur + lane);
    float vv1 = ldg_f32(v_cur + lane + 32);

    #pragma unroll
    for (int key = 0; key < 49; key++) {
        if (key >= Nk) break;

        float nk0 = 0.f, nk1 = 0.f, nvv0 = 0.f, nvv1 = 0.f;
        const float* k_next = k_cur + 64;
        const float* v_next = v_cur + 64;
        if (key + 1 < Nk && key + 1 < 49) {
            nk0  = ldg_f32(k_next + lane);
            nk1  = ldg_f32(k_next + lane + 32);
            nvv0 = ldg_f32(v_next + lane);
            nvv1 = ldg_f32(v_next + lane + 32);
        }

        float partial = fmaf(q0, k0, q1 * k1);
        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = __expf(m - m_new);
        float beta  = __expf(dot - m_new);
        float l_new = fmaf(alpha, l, beta);

        o0 = fmaf(beta, vv0, o0 * alpha);
        o1 = fmaf(beta, vv1, o1 * alpha);

        m = m_new;
        l = l_new;

        k_cur = k_next;
        v_cur = v_next;
        k0 = nk0; k1 = nk1;
        vv0 = nvv0; vv1 = nvv1;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane]      = o0 * inv_l;
    o_ptr[lane + 32] = o1 * inv_l;
}

__global__ __launch_bounds__(256, 2) void sdpa_online_warp_d64(
    const float* __restrict__ q,   // [B,H,Nq,64]
    const float* __restrict__ k,   // [B,H,Nk,64]
    const float* __restrict__ v,   // [B,H,Nk,64]
    float* __restrict__ out,       // [B,H,Nq,64]
    int B, int H, int Nq, int Nk,
    float scale
) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp;
    int total_rows = B * H * Nq;
    if (row >= total_rows) return;

    int qpos = row % Nq;
    int tmp  = row / Nq;
    int hidx = tmp % H;
    int bidx = tmp / H;

    const float* q_ptr = q + (((bidx * H + hidx) * Nq + qpos) * 64);
    const float* k_ptr = k + ((bidx * H + hidx) * Nk * 64);
    const float* v_ptr = v + ((bidx * H + hidx) * Nk * 64);
    float* o_ptr       = out + (((bidx * H + hidx) * Nq + qpos) * 64);

    float q0 = ldg_f32(q_ptr + lane);
    float q1 = ldg_f32(q_ptr + lane + 32);

    float o0 = 0.f, o1 = 0.f;
    float m  = -INFINITY;
    float l  = 0.f;

    #pragma unroll 1
    for (int key = 0; key < Nk; key++) {
        const float* k_row = k_ptr + key * 64;
        const float* v_row = v_ptr + key * 64;

        float k0 = ldg_f32(k_row + lane);
        float k1 = ldg_f32(k_row + lane + 32);

        float partial = fmaf(q0, k0, q1 * k1);
        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = __expf(m - m_new);
        float beta  = __expf(dot - m_new);
        float l_new = fmaf(alpha, l, beta);

        float vv0 = ldg_f32(v_row + lane);
        float vv1 = ldg_f32(v_row + lane + 32);

        o0 = fmaf(beta, vv0, o0 * alpha);
        o1 = fmaf(beta, vv1, o1 * alpha);

        m = m_new;
        l = l_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane]      = o0 * inv_l;
    o_ptr[lane + 32] = o1 * inv_l;
}

// -------------------- Generic fallback: 3-pass warp kernel, expects k [B,H,Dk,Nk] --------------------
__global__ void sdpa_warp_fwd_generic(
    const float* __restrict__ q,   // [B,H,Nq,Dk]
    const float* __restrict__ k,   // [B,H,Dk,Nk]
    const float* __restrict__ v,   // [B,H,Nk,Dv]
    float* __restrict__ out,       // [B,H,Nq,Dv]
    int B, int H, int Nq, int Nk, int Dk, int Dv,
    float scale
) {
    constexpr int WARPS_PER_BLOCK = 4;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    int total_rows = B * H * Nq;
    if (row >= total_rows) return;

    int q_idx = row % Nq;
    int tmp   = row / Nq;
    int h_idx = tmp % H;
    int b_idx = tmp / H;

    const float* q_ptr = q + (((b_idx * H + h_idx) * Nq + q_idx) * Dk);
    const float* k_ptr = k + ((b_idx * H + h_idx) * Dk * Nk); // [Dk, Nk]
    const float* v_ptr = v + ((b_idx * H + h_idx) * Nk * Dv);
    float* out_ptr     = out + (((b_idx * H + h_idx) * Nq + q_idx) * Dv);

    float local_max = -INFINITY;
    for (int key = lane; key < Nk; key += 32) {
        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d * Nk + key), dot);
        float logit = dot * scale;
        local_max = fmaxf(local_max, logit);
    }
    float wmax = warp_reduce_max(local_max);
    wmax = __shfl_sync(0xffffffff, wmax, 0);

    float local_sum = 0.0f;
    for (int key = lane; key < Nk; key += 32) {
        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d * Nk + key), dot);
        float logit = dot * scale;
        local_sum += __expf(logit - wmax);
    }
    float wsum = warp_reduce_sum(local_sum);
    wsum = __shfl_sync(0xffffffff, wsum, 0) + 1e-9f;
    float inv_sum = 1.0f / wsum;

    for (int dv = lane; dv < Dv; dv += 32) {
        float acc = 0.0f;
        for (int key = 0; key < Nk; key++) {
            float w = 0.0f;
            int owner = key & 31;
            if (owner == lane) {
                float dot = 0.0f;
                #pragma unroll 1
                for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d * Nk + key), dot);
                float logit = dot * scale;
                w = __expf(logit - wmax) * inv_sum;
            }
            w = __shfl_sync(0xffffffff, w, owner);
            acc = fmaf(w, ldg_f32(v_ptr + key * Dv + dv), acc);
        }
        out_ptr[dv] = acc;
    }
}

torch::Tensor sdpa_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Nq,Dk]");
    TORCH_CHECK(k.dim() == 4, "k must be [B,H,?,?]");
    TORCH_CHECK(v.dim() == 4, "v must be [B,H,Nk,Dv]");

    int B  = (int)q.size(0);
    int H  = (int)q.size(1);
    int Nq = (int)q.size(2);
    int Dk = (int)q.size(3);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H, "k must match q in B and H");
    TORCH_CHECK(v.size(0) == B && v.size(1) == H, "v must match q in B and H");

    int Dv = (int)v.size(3);
    int Nk = (int)v.size(2);

    auto out = torch::empty({B, H, Nq, Dv}, q.options());
    float scale = 1.0f / sqrtf((float)Dk);

    // Fast path assumes: Dk=64, Dv=64, k is [B,H,Nk,64]
    if (Dk == 64 && Dv == 64 && k.size(2) == Nk && k.size(3) == 64) {
        if (Nk == 49) {
            // Use tiled warp2 kernel when Nq is non-trivial; it is for ViT (49).
            int total_pairs = B * H * ((Nq + 1) >> 1);
            constexpr int WARPS_PER_BLOCK = 4; // smaller block to mitigate reg pressure
            int threads = WARPS_PER_BLOCK * 32;
            int blocks = (total_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            sdpa_online_warp2_d64_nk49_prefetch<<<blocks, threads>>>(
                (const float*)q.data_ptr<float>(),
                (const float*)k.data_ptr<float>(),
                (const float*)v.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, H, Nq, Nk, scale
            );
        } else {
            int total_rows = B * H * Nq;
            constexpr int WARPS_PER_BLOCK = 8;
            int threads = WARPS_PER_BLOCK * 32;
            int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            sdpa_online_warp_d64<<<blocks, threads>>>(
                (const float*)q.data_ptr<float>(),
                (const float*)k.data_ptr<float>(),
                (const float*)v.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, H, Nq, Nk, scale
            );
        }
        return out;
    }

    // Generic fallback expects baseline k layout [B,H,Dk,Nk]
    TORCH_CHECK((int)k.size(2) == Dk, "generic path expects k as [B,H,Dk,Nk] with matching Dk");
    TORCH_CHECK((int)k.size(3) == Nk, "generic path expects k Nk to match v Nk");

    int total_rows = B * H * Nq;
    constexpr int WARPS_PER_BLOCK = 4;
    int threads = WARPS_PER_BLOCK * 32;
    int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    sdpa_warp_fwd_generic<<<blocks, threads>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, H, Nq, Nk, Dk, Dv, scale
    );
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor sdpa_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_sdpa_modular_opt11_warp2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sdpa_forward_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Modular Q/K/V projections + fused SDPA CUDA kernel.
    Fast path uses K layout [B,H,Nk,64]. For Nk==49 uses a 2-query-per-warp kernel to reuse K/V.
    """

    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.custom_ops_lib = custom_ops_lib

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = (
            self.fc_q(queries)
            .view(b_s, nq, self.h, self.d_k)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,H,Nq,Dk]

        k = (
            self.fc_k(keys)
            .view(b_s, nk, self.h, self.d_k)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,H,Nk,Dk]

        v = (
            self.fc_v(values)
            .view(b_s, nk, self.h, self.d_v)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,H,Nk,Dv]

        out = self.custom_ops_lib.sdpa_forward_cuda(q, k, v)  # [B,H,Nq,Dv]
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out