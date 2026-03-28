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

static __device__ __forceinline__ float softcap_apply(float x, float softcap) {
    if (softcap <= 0.f) return x;
    return softcap * tanhf(x / softcap);
}
static __device__ __forceinline__ float softcap_deriv(float x_raw, float softcap) {
    if (softcap <= 0.f) return 1.f;
    float t = tanhf(x_raw / softcap);
    return 1.f - t * t;
}

static __device__ __forceinline__ void compute_j_bounds(
    int iq, int Sq, int Sk,
    int causal, int window_left, int window_right,
    int &j_lo, int &j_hi
) {
    int shift = Sk - Sq;
    int base = iq + shift;
    j_lo = 0;
    j_hi = Sk - 1;
    if (causal) j_hi = min(j_hi, base);
    if (window_left >= 0)  j_lo = max(j_lo, base - window_left);
    if (window_right >= 0) j_hi = min(j_hi, base + window_right);
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

static __device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 x) {
    float2 out;
    out.x = __bfloat162float(x.x);
    out.y = __bfloat162float(x.y);
    return out;
}

// ---------------- dQ kernel (unchanged baseline) ----------------
__global__ __launch_bounds__(128, 4) void flash_bwd_dq_warp_d128(
    const __nv_bfloat16* __restrict__ dO, // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ Q,  // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ K,  // [B,Hkv,Sk,128]
    const __nv_bfloat16* __restrict__ V,  // [B,Hkv,Sk,128]
    __nv_bfloat16* __restrict__ dQ,       // [B,Hq,Sq,128]
    int total_rows, int Hq, int Hkv, int Sq, int Sk,
    int n_groups, float scale,
    int causal, int window_left, int window_right,
    float softcap
) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (row >= total_rows) return;

    int iq = row % Sq;
    int tmp = row / Sq;
    int hq = tmp % Hq;
    int b  = tmp / Hq;
    int hk = hq / n_groups;

    int j_lo, j_hi;
    compute_j_bounds(iq, Sq, Sk, causal, window_left, window_right, j_lo, j_hi);

    __nv_bfloat16* dQ_bh = dQ + (((b * Hq + hq) * Sq) * 128);
    if (j_lo > j_hi) {
        int d = lane * 4;
        int off = iq * 128 + d;
        if (d < 128) {
            dQ_bh[off + 0] = __float2bfloat16(0.f);
            dQ_bh[off + 1] = __float2bfloat16(0.f);
            dQ_bh[off + 2] = __float2bfloat16(0.f);
            dQ_bh[off + 3] = __float2bfloat16(0.f);
        }
        return;
    }

    const __nv_bfloat16* Q_bh  = Q  + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* dO_bh = dO + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* K_bk  = K  + (((b * Hkv + hk) * Sk) * 128);
    const __nv_bfloat16* V_bk  = V  + (((b * Hkv + hk) * Sk) * 128);

    const __nv_bfloat16* q_ptr  = Q_bh  + iq * 128;
    const __nv_bfloat16* do_ptr = dO_bh + iq * 128;

    int d = lane * 4;
    const __nv_bfloat162* q2  = (const __nv_bfloat162*)(q_ptr + d);
    const __nv_bfloat162* do2 = (const __nv_bfloat162*)(do_ptr + d);
    float2 qf0 = bf162_to_float2(q2[0]);
    float2 qf1 = bf162_to_float2(q2[1]);
    float q0=qf0.x, q1=qf0.y, q2f=qf1.x, q3=qf1.y;
    float2 dof0 = bf162_to_float2(do2[0]);
    float2 dof1 = bf162_to_float2(do2[1]);
    float do0=dof0.x, do1=dof0.y, do2f=dof1.x, do3=dof1.y;

    float m_lane = -INFINITY;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        if (lane == 0) m_lane = fmaxf(m_lane, s);
    }
    float m = __shfl_sync(0xffffffff, m_lane, 0);

    float l0 = 0.f;
    float delta0 = 0.f;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);
        float part_dot = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part_dot);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        float p_unnorm = __expf(s - m);

        const __nv_bfloat16* v_ptr = V_bk + j * 128;
        const __nv_bfloat162* v2 = (const __nv_bfloat162*)(v_ptr + d);
        float2 vf0 = bf162_to_float2(v2[0]);
        float2 vf1 = bf162_to_float2(v2[1]);
        float part_g = do0*vf0.x + do1*vf0.y + do2f*vf1.x + do3*vf1.y;
        float g = warp_reduce_sum(part_g);
        g = __shfl_sync(0xffffffff, g, 0);

        if (lane == 0) {
            l0 += p_unnorm;
            delta0 += p_unnorm * g;
        }
    }
    float l = __shfl_sync(0xffffffff, l0, 0) + 1e-9f;
    float inv_l = 1.0f / l;
    float Delta = __shfl_sync(0xffffffff, delta0, 0) * inv_l;

    float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);

        float part_dot = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float s_raw = warp_reduce_sum(part_dot);
        s_raw = __shfl_sync(0xffffffff, s_raw, 0) * scale;
        float s = softcap_apply(s_raw, softcap);
        float p = __expf(s - m) * inv_l;

        const __nv_bfloat16* v_ptr = V_bk + j * 128;
        const __nv_bfloat162* v2 = (const __nv_bfloat162*)(v_ptr + d);
        float2 vf0 = bf162_to_float2(v2[0]);
        float2 vf1 = bf162_to_float2(v2[1]);
        float part_g = do0*vf0.x + do1*vf0.y + do2f*vf1.x + do3*vf1.y;
        float g = warp_reduce_sum(part_g);
        g = __shfl_sync(0xffffffff, g, 0);

        float scd = softcap_deriv(s_raw, softcap);
        float dS = p * (g - Delta) * scd;
        float w = dS * scale;

        acc0 += w * kf0.x;
        acc1 += w * kf0.y;
        acc2 += w * kf1.x;
        acc3 += w * kf1.y;
    }

    int off = iq * 128 + d;
    dQ_bh[off + 0] = __float2bfloat16(acc0);
    dQ_bh[off + 1] = __float2bfloat16(acc1);
    dQ_bh[off + 2] = __float2bfloat16(acc2);
    dQ_bh[off + 3] = __float2bfloat16(acc3);
}

// ---------------- dV kernel (keep current baseline: warp-per-row atomic fp32) ----------------
__global__ __launch_bounds__(32, 8) void flash_bwd_dv_row_atomic_d128(
    const __nv_bfloat16* __restrict__ dO, // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ Q,  // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ K,  // [B,Hkv,Sk,128]
    float* __restrict__ dV_accum,         // [B,Hkv,Sk,128] fp32 (atomic)
    int total_rows, int Hq, int Hkv, int Sq, int Sk,
    int n_groups, float scale,
    int causal, int window_left, int window_right,
    float softcap
) {
    int lane = threadIdx.x & 31;
    int row = (int)blockIdx.x;
    if (row >= total_rows) return;

    int iq = row % Sq;
    int tmp = row / Sq;
    int hq = tmp % Hq;
    int b  = tmp / Hq;
    int hk = hq / n_groups;

    int j_lo, j_hi;
    compute_j_bounds(iq, Sq, Sk, causal, window_left, window_right, j_lo, j_hi);
    if (j_lo > j_hi) return;

    const __nv_bfloat16* Q_bh  = Q  + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* dO_bh = dO + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* K_bk  = K  + (((b * Hkv + hk) * Sk) * 128);

    const __nv_bfloat16* q_ptr  = Q_bh  + iq * 128;
    const __nv_bfloat16* do_ptr = dO_bh + iq * 128;

    int d4 = lane * 4;
    const __nv_bfloat162* q2  = (const __nv_bfloat162*)(q_ptr + d4);
    const __nv_bfloat162* do2 = (const __nv_bfloat162*)(do_ptr + d4);
    float2 qf0 = bf162_to_float2(q2[0]);
    float2 qf1 = bf162_to_float2(q2[1]);
    float q0=qf0.x, q1=qf0.y, q2f=qf1.x, q3=qf1.y;
    float2 dof0 = bf162_to_float2(do2[0]);
    float2 dof1 = bf162_to_float2(do2[1]);
    float do0=dof0.x, do1=dof0.y, do2v=dof1.x, do3=dof1.y;

    float m_lane = -INFINITY;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        if (lane == 0) m_lane = fmaxf(m_lane, s);
    }
    float m = __shfl_sync(0xffffffff, m_lane, 0);

    float l0 = 0.f;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        float p_unnorm = __expf(s - m);
        if (lane == 0) l0 += p_unnorm;
    }
    float l = __shfl_sync(0xffffffff, l0, 0) + 1e-9f;
    float inv_l = 1.f / l;

    long long dv_base = (((long long)b * Hkv + hk) * Sk) * 128;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2[0]);
        float2 kf1 = bf162_to_float2(k2[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        float p = __expf(s - m) * inv_l;

        long long out = dv_base + (long long)j * 128 + d4;
        atomicAdd(dV_accum + out + 0, p * do0);
        atomicAdd(dV_accum + out + 1, p * do1);
        atomicAdd(dV_accum + out + 2, p * do2v);
        atomicAdd(dV_accum + out + 3, p * do3);
    }
}

// ---------------- New dK kernel: warp-per-row streaming + per-warp smem accumulation, then atomic flush ----------------
// One warp per (b,hq,iq). For each key j, each lane accumulates 4 dims of dK[j,:] into smem,
// then flushes exactly 4 atomics per lane (128 atomics total per j per row), but avoids 128-thread block,
// eliminates __syncthreads, and avoids staging K/V tiles with barriers.
__global__ __launch_bounds__(32, 12) void flash_bwd_dk_warp_smemflush_atomic_d128(
    const __nv_bfloat16* __restrict__ dO, // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ Q,  // [B,Hq,Sq,128]
    const __nv_bfloat16* __restrict__ K,  // [B,Hkv,Sk,128]
    const __nv_bfloat16* __restrict__ V,  // [B,Hkv,Sk,128]
    float* __restrict__ dK_accum,         // [B,Hkv,Sk,128] fp32 (atomic)
    int total_rows, int Hq, int Hkv, int Sq, int Sk,
    int n_groups, float scale,
    int causal, int window_left, int window_right,
    float softcap
) {
    int lane = threadIdx.x & 31;
    int row = (int)blockIdx.x;
    if (row >= total_rows) return;

    int iq = row % Sq;
    int tmp = row / Sq;
    int hq = tmp % Hq;
    int b  = tmp / Hq;
    int hk = hq / n_groups;

    int j_lo, j_hi;
    compute_j_bounds(iq, Sq, Sk, causal, window_left, window_right, j_lo, j_hi);
    if (j_lo > j_hi) return;

    const __nv_bfloat16* Q_bh  = Q  + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* dO_bh = dO + (((b * Hq  + hq) * Sq) * 128);
    const __nv_bfloat16* K_bk  = K  + (((b * Hkv + hk) * Sk) * 128);
    const __nv_bfloat16* V_bk  = V  + (((b * Hkv + hk) * Sk) * 128);

    const __nv_bfloat16* q_ptr  = Q_bh  + iq * 128;
    const __nv_bfloat16* do_ptr = dO_bh + iq * 128;

    int d4 = lane * 4;
    const __nv_bfloat162* q2  = (const __nv_bfloat162*)(q_ptr + d4);
    const __nv_bfloat162* do2 = (const __nv_bfloat162*)(do_ptr + d4);

    float2 qf0 = bf162_to_float2(q2[0]);
    float2 qf1 = bf162_to_float2(q2[1]);
    float q0=qf0.x, q1=qf0.y, q2f=qf1.x, q3=qf1.y;

    float2 dof0 = bf162_to_float2(do2[0]);
    float2 dof1 = bf162_to_float2(do2[1]);
    float do0=dof0.x, do1=dof0.y, do2f=dof1.x, do3=dof1.y;

    // Pass1: m
    float m_lane = -INFINITY;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2j = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2j[0]);
        float2 kf1 = bf162_to_float2(k2j[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        if (lane == 0) m_lane = fmaxf(m_lane, s);
    }
    float m = __shfl_sync(0xffffffff, m_lane, 0);

    // Pass2: l and delta (delta = sum p_unnorm * g, g = dO·V)
    float l0 = 0.f;
    float delta0 = 0.f;
    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2j = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2j[0]);
        float2 kf1 = bf162_to_float2(k2j[1]);
        float part = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float dot = warp_reduce_sum(part);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;
        float s = softcap_apply(dot, softcap);
        float p_unnorm = __expf(s - m);

        const __nv_bfloat16* v_ptr = V_bk + (long long)j * 128;
        const __nv_bfloat162* v2j = (const __nv_bfloat162*)(v_ptr + d4);
        float2 vf0 = bf162_to_float2(v2j[0]);
        float2 vf1 = bf162_to_float2(v2j[1]);
        float part_g = do0*vf0.x + do1*vf0.y + do2f*vf1.x + do3*vf1.y;
        float g = warp_reduce_sum(part_g);
        g = __shfl_sync(0xffffffff, g, 0);

        if (lane == 0) { l0 += p_unnorm; delta0 += p_unnorm * g; }
    }
    float l = __shfl_sync(0xffffffff, l0, 0) + 1e-9f;
    float inv_l = 1.0f / l;
    float Delta = __shfl_sync(0xffffffff, delta0, 0) * inv_l;

    long long dk_base = (((long long)b * Hkv + hk) * Sk) * 128;

    // Per-warp smem for 128 floats (one j at a time)
    __shared__ float sm_dk[128];

    #pragma unroll 1
    for (int j = j_lo; j <= j_hi; ++j) {
        // compute coeff = dS * scale, where dS = p*(g-Delta)*softcap'
        const __nv_bfloat16* k_ptr = K_bk + (long long)j * 128;
        const __nv_bfloat162* k2j = (const __nv_bfloat162*)(k_ptr + d4);
        float2 kf0 = bf162_to_float2(k2j[0]);
        float2 kf1 = bf162_to_float2(k2j[1]);

        float part_dot = q0*kf0.x + q1*kf0.y + q2f*kf1.x + q3*kf1.y;
        float s_raw = warp_reduce_sum(part_dot);
        s_raw = __shfl_sync(0xffffffff, s_raw, 0) * scale;
        float s = softcap_apply(s_raw, softcap);
        float p = __expf(s - m) * inv_l;

        const __nv_bfloat16* v_ptr = V_bk + (long long)j * 128;
        const __nv_bfloat162* v2j = (const __nv_bfloat162*)(v_ptr + d4);
        float2 vf0 = bf162_to_float2(v2j[0]);
        float2 vf1 = bf162_to_float2(v2j[1]);
        float part_g = do0*vf0.x + do1*vf0.y + do2f*vf1.x + do3*vf1.y;
        float g = warp_reduce_sum(part_g);
        g = __shfl_sync(0xffffffff, g, 0);

        float scd = softcap_deriv(s_raw, softcap);
        float dS = p * (g - Delta) * scd;
        float coeff = dS * scale;

        // Load q[d] for 4 dims
        float qd0 = __bfloat162float(q_ptr[d4 + 0]);
        float qd1 = __bfloat162float(q_ptr[d4 + 1]);
        float qd2 = __bfloat162float(q_ptr[d4 + 2]);
        float qd3 = __bfloat162float(q_ptr[d4 + 3]);

        // Write contributions into shared memory (no race: each lane owns 4 unique indices)
        int base = d4;
        sm_dk[base + 0] = coeff * qd0;
        sm_dk[base + 1] = coeff * qd1;
        sm_dk[base + 2] = coeff * qd2;
        sm_dk[base + 3] = coeff * qd3;

        // Warp-synchronous: now flush to global atomics
        long long out = dk_base + (long long)j * 128 + d4;
        atomicAdd(dK_accum + out + 0, sm_dk[base + 0]);
        atomicAdd(dK_accum + out + 1, sm_dk[base + 1]);
        atomicAdd(dK_accum + out + 2, sm_dk[base + 2]);
        atomicAdd(dK_accum + out + 3, sm_dk[base + 3]);
    }
}

__global__ void cast_accum_to_bf16(
    const float* __restrict__ accum,
    __nv_bfloat16* __restrict__ out,
    int total
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2bfloat16(accum[idx]);
}

// Host wrapper
std::vector<torch::Tensor> flash_attention_bwd_cuda(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap
) {
    CHECK_INPUT(dO);
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(dO.dim() == 4 && Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "All inputs must be 4D");
    TORCH_CHECK(dO.scalar_type() == at::ScalarType::BFloat16, "dO must be bfloat16");
    TORCH_CHECK(Q.scalar_type()  == at::ScalarType::BFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.scalar_type()  == at::ScalarType::BFloat16, "K must be bfloat16");
    TORCH_CHECK(V.scalar_type()  == at::ScalarType::BFloat16, "V must be bfloat16");

    int B  = (int)Q.size(0);
    int Hq = (int)Q.size(1);
    int Sq = (int)Q.size(2);
    int D  = (int)Q.size(3);
    TORCH_CHECK(D == 128, "Optimized kernel supports D==128 only");

    TORCH_CHECK(dO.sizes() == Q.sizes(), "dO and Q must have same shape [B,Hq,Sq,D]");
    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "Batch mismatch");
    int Hkv = (int)K.size(1);
    int Sk  = (int)K.size(2);
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "Head dim mismatch");
    TORCH_CHECK(V.size(1) == Hkv && V.size(2) == Sk, "V shape mismatch");
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");
    int n_groups = Hq / Hkv;

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);

    int total_rows = B * Hq * Sq;

    // dQ warp-per-row (4 warps/block)
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_DQ = WARPS_PER_BLOCK * 32;
    int blocks_dq = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    flash_bwd_dq_warp_d128<<<blocks_dq, THREADS_DQ>>>(
        (const __nv_bfloat16*)dO.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)V.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)dQ.data_ptr<at::BFloat16>(),
        total_rows, Hq, Hkv, Sq, Sk,
        n_groups, (float)scale,
        causal ? 1 : 0, (int)window_left, (int)window_right,
        (float)softcap
    );

    // dV: row streaming + atomic fp32 accumulation + cast
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto dV_accum = torch::zeros({B, Hkv, Sk, 128}, opts_f);
    flash_bwd_dv_row_atomic_d128<<<total_rows, 32>>>(
        (const __nv_bfloat16*)dO.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
        (float*)dV_accum.data_ptr<float>(),
        total_rows, Hq, Hkv, Sq, Sk,
        n_groups, (float)scale,
        causal ? 1 : 0, (int)window_left, (int)window_right,
        (float)softcap
    );

    int total_dv = B * Hkv * Sk * 128;
    int t = 256;
    int grid_dv_cast = (total_dv + t - 1) / t;
    cast_accum_to_bf16<<<grid_dv_cast, t>>>(
        (const float*)dV_accum.data_ptr<float>(),
        (__nv_bfloat16*)dV.data_ptr<at::BFloat16>(),
        total_dv
    );

    // dK: warp streaming + fp32 atomic accumulation + cast (NEW)
    auto dK_accum = torch::zeros({B, Hkv, Sk, 128}, opts_f);
    flash_bwd_dk_warp_smemflush_atomic_d128<<<total_rows, 32>>>(
        (const __nv_bfloat16*)dO.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)V.data_ptr<at::BFloat16>(),
        (float*)dK_accum.data_ptr<float>(),
        total_rows, Hq, Hkv, Sq, Sk,
        n_groups, (float)scale,
        causal ? 1 : 0, (int)window_left, (int)window_right,
        (float)softcap
    );

    int total_dk = B * Hkv * Sk * 128;
    int grid_dk_cast = (total_dk + t - 1) / t;
    cast_accum_to_bf16<<<grid_dk_cast, t>>>(
        (const float*)dK_accum.data_ptr<float>(),
        (__nv_bfloat16*)dK.data_ptr<at::BFloat16>(),
        total_dk
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_attention_bwd_cuda launch failed: ", cudaGetErrorString(err));
    return {dQ, dK, dV};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_bwd_cuda(
    torch::Tensor dO,
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
    name="custom_ops_lib_flash_attention_bwd_optv9_dkwarp_smemflush",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["flash_attention_bwd_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, nheads, nheads_kv, headdim, causal=True, window_size=(-1, -1), softcap=0.0):
        super().__init__()
        self.nheads = int(nheads)
        self.nheads_kv = int(nheads_kv)
        self.headdim = int(headdim)
        self.causal = bool(causal)
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.softcap = float(softcap)
        self.scale = 1.0 / math.sqrt(self.headdim)
        assert self.nheads % self.nheads_kv == 0

    def forward(self, dout, q, k, v):
        if not dout.is_cuda:
            dout = dout.cuda()
        if not q.is_cuda:
            q = q.cuda()
        if not k.is_cuda:
            k = k.cuda()
        if not v.is_cuda:
            v = v.cuda()

        if dout.dtype != torch.bfloat16:
            dout = dout.to(torch.bfloat16)
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        B, Sq, Hq, D = q.shape
        Bk, Sk, Hkv, Dk = k.shape
        assert (B, D) == (Bk, Dk)
        assert Hq == self.nheads and Hkv == self.nheads_kv and D == self.headdim
        assert dout.shape == (B, Sq, Hq, D)
        assert v.shape == (B, Sk, Hkv, D)

        if D != 128:
            raise RuntimeError("custom flash_attention_bwd_optv9_dkwarp_smemflush only supports headdim == 128")

        dO_bhsd = dout.transpose(1, 2).contiguous()
        Q_bhsd = q.transpose(1, 2).contiguous()
        K_bhsd = k.transpose(1, 2).contiguous()
        V_bhsd = v.transpose(1, 2).contiguous()

        wl, wr = self.window_size
        dQ_bhsd, dK_bhsd, dV_bhsd = custom_ops_lib.flash_attention_bwd_cuda(
            dO_bhsd,
            Q_bhsd,
            K_bhsd,
            V_bhsd,
            float(self.scale),
            bool(self.causal),
            int(wl),
            int(wr),
            float(self.softcap),
        )

        dq = dQ_bhsd.transpose(1, 2).contiguous()
        dk = dK_bhsd.transpose(1, 2).contiguous()
        dv = dV_bhsd.transpose(1, 2).contiguous()
        return dq, dk, dv