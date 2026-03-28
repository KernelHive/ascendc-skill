import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static __device__ __forceinline__ float softcap_fn(float x, float softcap) {
    if (softcap <= 0.f) return x;
    return softcap * tanhf(x / softcap);
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
    float o;
    o = __shfl_down_sync(0xffffffff, v, 16); v = fmaxf(v, o);
    o = __shfl_down_sync(0xffffffff, v, 8);  v = fmaxf(v, o);
    o = __shfl_down_sync(0xffffffff, v, 4);  v = fmaxf(v, o);
    o = __shfl_down_sync(0xffffffff, v, 2);  v = fmaxf(v, o);
    o = __shfl_down_sync(0xffffffff, v, 1);  v = fmaxf(v, o);
    return v;
}

template<bool USE_SOFTCAP>
__global__ void flash_attention_fwd_warp_d128(
    const __nv_bfloat16* __restrict__ Q, // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ K, // [B,Hkv,Sk,128]
    const __nv_bfloat16* __restrict__ V, // [B,Hkv,Sk,128]
    __nv_bfloat16* __restrict__ O,       // [B,Hq,Sq,128]
    int total_rows,
    int B, int Hq, int Hkv, int Sq, int Sk,
    int n_groups,
    float scale,
    int causal,
    int window_left, int window_right,
    float softcap
) {
    // One warp computes one (b,hq,iq) row
    int lane = threadIdx.x & 31;
    int warp_id_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_id_in_block;
    if (row >= total_rows) return;

    int iq = row % Sq;
    int tmp = row / Sq;
    int hq = tmp % Hq;
    int b  = tmp / Hq;
    int hkv = hq / n_groups;

    const int D = 128;

    const __nv_bfloat16* Q_bh = Q + (((b * Hq + hq) * Sq) * D);
    const __nv_bfloat16* K_bh = K + (((b * Hkv + hkv) * Sk) * D);
    const __nv_bfloat16* V_bh = V + (((b * Hkv + hkv) * Sk) * D);
    __nv_bfloat16* O_bh = O + (((b * Hq + hq) * Sq) * D);

    // mask interval
    int shift = Sk - Sq;
    int j_lo = 0;
    int j_hi = Sk - 1;

    if (causal) {
        int max_j_causal = iq + shift;
        j_hi = min(j_hi, max_j_causal);
    }
    if (window_left >= 0) {
        j_lo = max(j_lo, iq + shift - window_left);
    }
    if (window_right >= 0) {
        j_hi = min(j_hi, iq + shift + window_right);
    }

    if (j_lo > j_hi) {
        // write zeros
        int d = lane * 4;
        if (d < D) {
            O_bh[iq * D + d + 0] = __float2bfloat16(0.f);
            O_bh[iq * D + d + 1] = __float2bfloat16(0.f);
            O_bh[iq * D + d + 2] = __float2bfloat16(0.f);
            O_bh[iq * D + d + 3] = __float2bfloat16(0.f);
        }
        return;
    }

    // each lane owns 4 dims
    int d0 = lane * 4;
    const __nv_bfloat16* q_ptr = Q_bh + iq * D + d0;

    // Load Q as two bf16x2
    __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(q_ptr + 0);
    __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(q_ptr + 2);

    float q0 = __bfloat162float(q01.x);
    float q1 = __bfloat162float(q01.y);
    float q2 = __bfloat162float(q23.x);
    float q3 = __bfloat162float(q23.y);

    // Online softmax state
    float m = -INFINITY;
    float l = 0.f;

    // Output accumulators for this lane's 4 dims
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;

    for (int j = j_lo; j <= j_hi; j++) {
        const __nv_bfloat16* k_ptr = K_bh + j * D + d0;

        __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(k_ptr + 0);
        __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(k_ptr + 2);

        float part = 0.f;
        part += q0 * __bfloat162float(k01.x);
        part += q1 * __bfloat162float(k01.y);
        part += q2 * __bfloat162float(k23.x);
        part += q3 * __bfloat162float(k23.y);

        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        if (USE_SOFTCAP) dot = softcap_fn(dot, softcap);

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;

        o0 *= alpha; o1 *= alpha; o2 *= alpha; o3 *= alpha;

        const __nv_bfloat16* v_ptr = V_bh + j * D + d0;
        __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(v_ptr + 0);
        __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(v_ptr + 2);

        o0 += p * __bfloat162float(v01.x);
        o1 += p * __bfloat162float(v01.y);
        o2 += p * __bfloat162float(v23.x);
        o3 += p * __bfloat162float(v23.y);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);

    __nv_bfloat16* o_ptr = O_bh + iq * D + d0;
    *reinterpret_cast<__nv_bfloat162*>(o_ptr + 0) = __floats2bfloat162_rn(o0 * inv_l, o1 * inv_l);
    *reinterpret_cast<__nv_bfloat162*>(o_ptr + 2) = __floats2bfloat162_rn(o2 * inv_l, o3 * inv_l);
}

// Generic warp-per-row for any D multiple of 2 (slower fallback, still single-pass)
template<bool USE_SOFTCAP>
__global__ void flash_attention_fwd_warp_generic(
    const __nv_bfloat16* __restrict__ Q, // [B,Hq,Sq,D]
    const __nv_bfloat16* __restrict__ K, // [B,Hkv,Sk,D]
    const __nv_bfloat16* __restrict__ V, // [B,Hkv,Sk,D]
    __nv_bfloat16* __restrict__ O,       // [B,Hq,Sq,D]
    int total_rows,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    int n_groups,
    float scale,
    int causal,
    int window_left, int window_right,
    float softcap
) {
    int lane = threadIdx.x & 31;
    int warp_id_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_id_in_block;
    if (row >= total_rows) return;

    int iq = row % Sq;
    int tmp = row / Sq;
    int hq = tmp % Hq;
    int b  = tmp / Hq;
    int hkv = hq / n_groups;

    const __nv_bfloat16* Q_bh = Q + (((b * Hq + hq) * Sq) * D);
    const __nv_bfloat16* K_bh = K + (((b * Hkv + hkv) * Sk) * D);
    const __nv_bfloat16* V_bh = V + (((b * Hkv + hkv) * Sk) * D);
    __nv_bfloat16* O_bh = O + (((b * Hq + hq) * Sq) * D);

    int shift = Sk - Sq;
    int j_lo = 0;
    int j_hi = Sk - 1;

    if (causal) j_hi = min(j_hi, iq + shift);
    if (window_left >= 0) j_lo = max(j_lo, iq + shift - window_left);
    if (window_right >= 0) j_hi = min(j_hi, iq + shift + window_right);

    if (j_lo > j_hi) {
        for (int d = lane; d < D; d += 32) O_bh[iq * D + d] = __float2bfloat16(0.f);
        return;
    }

    // each lane owns a strided set of dims (2 at a time if possible)
    float m = -INFINITY;
    float l = 0.f;

    // we accumulate output in fp32 per owned dimension
    // for simplicity, keep two dims per iteration; if odd D, last handled scalar
    // (given typical head dims, D is even)
    // We'll store partials in registers for up to 8 dims per lane max if D large? -> no, strided loop updates per key, no need store whole vector.
    // Instead: do separate pass per d is too slow. So generic fallback does per-lane strided vector update (still fine for uncommon Ds).

    // We'll maintain up to 4 dims per lane via repeated loop each key; that is expensive but fallback only.
    // Choose 1 dim per lane per step (lane owns d=lane, lane+32, lane+64,...)
    // Accumulate those dims; still single-pass and no sync.
    // Note: This writes O in-place after loop.
    // We need per-owned-d accumulator; number of owned dims = ceil((D - lane)/32).
    // We'll cap at 8 to keep registers reasonable; beyond that, fallback may be slow.
    float out_acc[8];
    int out_idx[8];
    int cnt = 0;
    for (int d = lane; d < D && cnt < 8; d += 32) { out_idx[cnt] = d; out_acc[cnt] = 0.f; cnt++; }

    for (int j = j_lo; j <= j_hi; j++) {
        float part = 0.f;
        const __nv_bfloat16* q_ptr = Q_bh + iq * D;
        const __nv_bfloat16* k_ptr = K_bh + j * D;

        // dot product: each lane sums over its strided dims
        for (int d = lane; d < D; d += 32) {
            part += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        if (USE_SOFTCAP) dot = softcap_fn(dot, softcap);

        float m_new = fmaxf(m, dot);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float p = __expf(dot - m_new);

        l = l * alpha + p;
        for (int t = 0; t < cnt; t++) out_acc[t] *= alpha;

        const __nv_bfloat16* v_ptr = V_bh + j * D;
        for (int t = 0; t < cnt; t++) out_acc[t] += p * __bfloat162float(v_ptr[out_idx[t]]);

        m = m_new;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    for (int t = 0; t < cnt; t++) O_bh[iq * D + out_idx[t]] = __float2bfloat16(out_acc[t] * inv_l);
}

torch::Tensor flash_attention_fwd_cuda(
    torch::Tensor Q,  // [B,Hq,Sq,D] bf16
    torch::Tensor K,  // [B,Hkv,Sk,D] bf16
    torch::Tensor V,  // [B,Hkv,Sk,D] bf16
    double scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,Hq,Sq,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,Hkv,Sk,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,Hkv,Sk,D]");
    TORCH_CHECK(Q.scalar_type() == at::ScalarType::BFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.scalar_type() == at::ScalarType::BFloat16, "K must be bfloat16");
    TORCH_CHECK(V.scalar_type() == at::ScalarType::BFloat16, "V must be bfloat16");

    const int B   = (int)Q.size(0);
    const int Hq  = (int)Q.size(1);
    const int Sq  = (int)Q.size(2);
    const int D   = (int)Q.size(3);

    const int Bk  = (int)K.size(0);
    const int Hkv = (int)K.size(1);
    const int Sk  = (int)K.size(2);
    const int Dk  = (int)K.size(3);

    TORCH_CHECK(B == Bk, "Batch mismatch between Q and K");
    TORCH_CHECK(D == Dk, "Headdim mismatch between Q and K");
    TORCH_CHECK(V.size(0) == B && (int)V.size(1) == Hkv && (int)V.size(2) == Sk && (int)V.size(3) == D, "V must be [B,Hkv,Sk,D]");
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");

    int n_groups = Hq / Hkv;

    auto O = torch::empty({B, Hq, Sq, D}, Q.options());

    int total_rows = B * Hq * Sq;

    // 4 warps per block tends to balance occupancy/overhead.
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    float fscale = (float)scale;
    int ic = causal ? 1 : 0;
    int wl = (int)window_left;
    int wr = (int)window_right;
    float fsoftcap = (float)softcap;

    const __nv_bfloat16* qptr = (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>();
    const __nv_bfloat16* kptr = (const __nv_bfloat16*)K.data_ptr<at::BFloat16>();
    const __nv_bfloat16* vptr = (const __nv_bfloat16*)V.data_ptr<at::BFloat16>();
    __nv_bfloat16* optr = (__nv_bfloat16*)O.data_ptr<at::BFloat16>();

    bool use_softcap = (fsoftcap > 0.f);

    if (D == 128) {
        if (use_softcap) {
            flash_attention_fwd_warp_d128<true><<<blocks, THREADS>>>(
                qptr, kptr, vptr, optr,
                total_rows, B, Hq, Hkv, Sq, Sk, n_groups,
                fscale, ic, wl, wr, fsoftcap
            );
        } else {
            flash_attention_fwd_warp_d128<false><<<blocks, THREADS>>>(
                qptr, kptr, vptr, optr,
                total_rows, B, Hq, Hkv, Sq, Sk, n_groups,
                fscale, ic, wl, wr, fsoftcap
            );
        }
    } else {
        if (use_softcap) {
            flash_attention_fwd_warp_generic<true><<<blocks, THREADS>>>(
                qptr, kptr, vptr, optr,
                total_rows, B, Hq, Hkv, Sq, Sk, D, n_groups,
                fscale, ic, wl, wr, fsoftcap
            );
        } else {
            flash_attention_fwd_warp_generic<false><<<blocks, THREADS>>>(
                qptr, kptr, vptr, optr,
                total_rows, B, Hq, Hkv, Sq, Sk, D, n_groups,
                fscale, ic, wl, wr, fsoftcap
            );
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_attention_fwd_cuda launch failed: ", cudaGetErrorString(err));
    return O;
}
"""

cpp_src = r"""
torch::Tensor flash_attention_fwd_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_flash_attention_fwd_optwarp",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["flash_attention_fwd_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Custom-kernel replacement for flash_attention_fwd.

    Expects:
      q: (B, Sq, Hq, D) bfloat16
      k: (B, Sk, Hkv, D) bfloat16
      v: (B, Sk, Hkv, D) bfloat16
    Returns:
      out: (B, Sq, Hq, D) bfloat16
    """

    def __init__(self, nheads, nheads_kv, headdim, causal=True, window_size=(-1, -1), softcap=0.0):
        super().__init__()
        self.nheads = int(nheads)
        self.nheads_kv = int(nheads_kv)
        self.headdim = int(headdim)
        self.causal = bool(causal)
        self.window_size = window_size
        self.softcap = float(softcap)
        self.scale = 1.0 / math.sqrt(self.headdim)
        assert self.nheads % self.nheads_kv == 0

    def forward(self, q, k, v):
        if not q.is_cuda:
            q = q.cuda()
        if not k.is_cuda:
            k = k.cuda()
        if not v.is_cuda:
            v = v.cuda()

        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        B, Sq, Hq, D = q.shape
        Bk, Sk, Hkv, Dk = k.shape
        assert B == Bk and D == Dk
        assert Hq == self.nheads and Hkv == self.nheads_kv and D == self.headdim
        assert v.shape == (B, Sk, Hkv, D)

        # [B,S,H,D] -> [B,H,S,D] contiguous for kernel
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()

        wl, wr = self.window_size
        out_bhsd = custom_ops_lib.flash_attention_fwd_cuda(
            q_bhsd, k_bhsd, v_bhsd,
            float(self.scale),
            bool(self.causal),
            int(wl),
            int(wr),
            float(self.softcap),
        )
        return out_bhsd.transpose(1, 2).contiguous()