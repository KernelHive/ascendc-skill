import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Fast path specialized for Dh=64 and NK==49.
// One warp computes one (b, head, qpos) row with online softmax.
// Tuned to reduce register pressure vs prior baseline:
// - fewer warps/block (128 threads) to encourage lower reg allocation
// - tighter launch bounds
// - no extra prefetch temporaries (keep simple streaming)
__global__ __launch_bounds__(128, 3) void ssa_online_warp_d64_nk49_regopt(
    const float* __restrict__ q,   // [B,NQ,D]
    const float* __restrict__ k,   // [B,NK,D]
    const float* __restrict__ v,   // [B,NK,D]
    float* __restrict__ out,       // [B,NQ,D]
    int B, int NQ, int NK, int H, int D,
    float scale
) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp;
    int total_rows = B * H * NQ;
    if (row >= total_rows) return;

    int qpos = row % NQ;
    int tmp  = row / NQ;
    int head = tmp % H;
    int b    = tmp / H;

    constexpr int Dh = 64;
    const float* __restrict__ q_ptr  = q + ((b * NQ + qpos) * D + head * Dh);
    const float* __restrict__ k_ptr0 = k + (b * NK * D + head * Dh);
    const float* __restrict__ v_ptr0 = v + (b * NK * D + head * Dh);
    float* __restrict__ o_ptr        = out + ((b * NQ + qpos) * D + head * Dh);

    // Load Q once (each lane owns 2 channels)
    float q0 = ldg_f32(q_ptr + lane);
    float q1 = ldg_f32(q_ptr + lane + 32);

    // Online softmax state and output accumulator (2 channels per lane)
    float o0 = 0.0f, o1 = 0.0f;
    float m  = -INFINITY;
    float l  = 0.0f;

    const float* __restrict__ k_ptr = k_ptr0;
    const float* __restrict__ v_ptr = v_ptr0;

    // Fully unrolled for NK=49; safe guard for NK<49
    #pragma unroll
    for (int key = 0; key < 49; key++) {
        if (key >= NK) break;

        float k0  = ldg_f32(k_ptr + lane);
        float k1  = ldg_f32(k_ptr + lane + 32);

        float partial = fmaf(q0, k0, q1 * k1);
        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = __expf(m - m_new);
        float beta  = __expf(dot - m_new);
        float l_new = fmaf(alpha, l, beta);

        float vv0 = ldg_f32(v_ptr + lane);
        float vv1 = ldg_f32(v_ptr + lane + 32);

        o0 = fmaf(beta, vv0, o0 * alpha);
        o1 = fmaf(beta, vv1, o1 * alpha);

        m = m_new;
        l = l_new;

        k_ptr += D;
        v_ptr += D;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane]      = o0 * inv_l;
    o_ptr[lane + 32] = o1 * inv_l;
}

// General Dh=64, NK<=64 online kernel (kept for other NK values).
__global__ __launch_bounds__(128, 3) void ssa_online_warp_d64_nk64_regopt(
    const float* __restrict__ q,   // [B,NQ,D]
    const float* __restrict__ k,   // [B,NK,D]
    const float* __restrict__ v,   // [B,NK,D]
    float* __restrict__ out,       // [B,NQ,D]
    int B, int NQ, int NK, int H, int D,
    float scale
) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    int row = (int)blockIdx.x * warps_per_block + warp;
    int total_rows = B * H * NQ;
    if (row >= total_rows) return;

    int qpos = row % NQ;
    int tmp  = row / NQ;
    int head = tmp % H;
    int b    = tmp / H;

    constexpr int Dh = 64;
    const float* __restrict__ q_ptr = q + ((b * NQ + qpos) * D + head * Dh);
    const float* __restrict__ k_ptr = k + (b * NK * D + head * Dh);
    const float* __restrict__ v_ptr = v + (b * NK * D + head * Dh);
    float* __restrict__ o_ptr = out + ((b * NQ + qpos) * D + head * Dh);

    float q0 = ldg_f32(q_ptr + lane);
    float q1 = ldg_f32(q_ptr + lane + 32);

    float o0 = 0.0f, o1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    #pragma unroll 1
    for (int key = 0; key < 64; key++) {
        if (key >= NK) break;

        float k0 = ldg_f32(k_ptr + lane);
        float k1 = ldg_f32(k_ptr + lane + 32);

        float partial = fmaf(q0, k0, q1 * k1);
        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0) * scale;

        float m_new = fmaxf(m, dot);
        float alpha = __expf(m - m_new);
        float beta  = __expf(dot - m_new);
        float l_new = fmaf(alpha, l, beta);

        float vv0 = ldg_f32(v_ptr + lane);
        float vv1 = ldg_f32(v_ptr + lane + 32);

        o0 = fmaf(beta, vv0, o0 * alpha);
        o1 = fmaf(beta, vv1, o1 * alpha);

        m = m_new;
        l = l_new;

        k_ptr += D;
        v_ptr += D;
    }

    float inv_l = 1.0f / (l + 1e-9f);
    o_ptr[lane] = o0 * inv_l;
    o_ptr[lane + 32] = o1 * inv_l;
}

// Generic fallback: 3-pass warp kernel (correct for any Dk/Dv, but slower).
static __device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void ssa_warp_fwd_generic(
    const float* __restrict__ q,   // [B,NQ,D]
    const float* __restrict__ k,   // [B,NK,D]
    const float* __restrict__ v,   // [B,NK,D]
    float* __restrict__ out,       // [B,NQ,D]
    int B, int NQ, int NK, int H, int D, int Dk, int Dv,
    float scale
) {
    constexpr int WARPS_PER_BLOCK = 4;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int row = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    int total_rows = B * H * NQ;
    if (row >= total_rows) return;

    int qpos = row % NQ;
    int tmp  = row / NQ;
    int head = tmp % H;
    int b    = tmp / H;

    const float* __restrict__ q_ptr = q + ((b * NQ + qpos) * D + head * Dk);
    const float* __restrict__ k_base = k + (b * NK * D + head * Dk);
    const float* __restrict__ v_base = v + (b * NK * D + head * Dv);
    float* __restrict__ o_ptr = out + ((b * NQ + qpos) * D + head * Dv);

    float local_max = -INFINITY;
    for (int key = lane; key < NK; key += 32) {
        const float* __restrict__ k_ptr = k_base + key * D;
        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d), dot);
        float logit = dot * scale;
        local_max = fmaxf(local_max, logit);
    }
    float wmax = warp_reduce_max(local_max);
    wmax = __shfl_sync(0xffffffff, wmax, 0);

    float local_sum = 0.0f;
    for (int key = lane; key < NK; key += 32) {
        const float* __restrict__ k_ptr = k_base + key * D;
        float dot = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d), dot);
        float logit = dot * scale;
        local_sum += __expf(logit - wmax);
    }
    float wsum = warp_reduce_sum(local_sum);
    wsum = __shfl_sync(0xffffffff, wsum, 0) + 1e-9f;
    float inv_sum = 1.0f / wsum;

    for (int dv = lane; dv < Dv; dv += 32) {
        float acc = 0.0f;
        for (int key = 0; key < NK; key++) {
            float w = 0.0f;
            int owner = key & 31;
            if (owner == lane) {
                const float* __restrict__ k_ptr = k_base + key * D;
                float dot = 0.0f;
                #pragma unroll 1
                for (int d = 0; d < Dk; d++) dot = fmaf(q_ptr[d], ldg_f32(k_ptr + d), dot);
                float logit = dot * scale;
                w = __expf(logit - wmax) * inv_sum;
            }
            w = __shfl_sync(0xffffffff, w, owner);
            const float* __restrict__ v_ptr = v_base + key * D;
            acc = fmaf(w, ldg_f32(v_ptr + dv), acc);
        }
        o_ptr[dv] = acc;
    }
}

torch::Tensor simplified_self_attention_cuda(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, int64_t h) {
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(values);

    at::cuda::CUDAGuard device_guard(queries.device());

    TORCH_CHECK(queries.dim() == 3, "queries must be (B, NQ, D)");
    TORCH_CHECK(keys.dim() == 3, "keys must be (B, NK, D)");
    TORCH_CHECK(values.dim() == 3, "values must be (B, NK, D)");
    TORCH_CHECK(queries.size(0) == keys.size(0) && queries.size(0) == values.size(0), "batch sizes must match");
    TORCH_CHECK(keys.size(1) == values.size(1), "nk must match between keys and values");
    TORCH_CHECK(queries.size(2) == keys.size(2) && queries.size(2) == values.size(2), "d_model must match");

    int B  = (int)queries.size(0);
    int NQ = (int)queries.size(1);
    int NK = (int)keys.size(1);
    int D  = (int)queries.size(2);
    int H  = (int)h;

    TORCH_CHECK(H > 0, "h must be > 0");
    TORCH_CHECK(D % H == 0, "d_model must be divisible by h");
    int Dk = D / H;
    int Dv = Dk;

    auto out = torch::empty({B, NQ, D}, queries.options());
    float scale = 1.0f / sqrtf((float)Dk);

    if (Dk == 64 && Dv == 64 && NK <= 64) {
        constexpr int WARPS_PER_BLOCK = 4; // 128 threads to reduce reg pressure
        int threads = WARPS_PER_BLOCK * 32;
        int total_rows = B * H * NQ;
        int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        if (NK == 49) {
            ssa_online_warp_d64_nk49_regopt<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                (const float*)queries.data_ptr<float>(),
                (const float*)keys.data_ptr<float>(),
                (const float*)values.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, NQ, NK, H, D, scale
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            ssa_online_warp_d64_nk64_regopt<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                (const float*)queries.data_ptr<float>(),
                (const float*)keys.data_ptr<float>(),
                (const float*)values.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, NQ, NK, H, D, scale
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        constexpr int WARPS_PER_BLOCK = 4;
        int threads = WARPS_PER_BLOCK * 32;
        int total_rows = B * H * NQ;
        int blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        ssa_warp_fwd_generic<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            (const float*)queries.data_ptr<float>(),
            (const float*)keys.data_ptr<float>(),
            (const float*)values.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, NQ, NK, H, D, Dk, Dv, scale
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor simplified_self_attention_cuda(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, int64_t h);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_simplified_self_attention_opt9",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["simplified_self_attention_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Drop-in replacement using a fused CUDA kernel for simplified attention,
    then applies the output projection fc_o.
    """
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)  # parity; not used in fused kernel
        self.custom_ops_lib = custom_ops_lib

    def forward(self, queries, keys, values):
        if not queries.is_cuda:
            b_s, nq = queries.shape[:2]
            nk = keys.shape[1]
            q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
            k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
            v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
            att = torch.matmul(q, k) / math.sqrt(self.d_k)
            att = torch.softmax(att, -1)
            att = self.dropout(att)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
            return self.fc_o(out)

        q = queries.contiguous()
        k = keys.contiguous()
        v = values.contiguous()

        if q.dtype != torch.float32:
            q = q.float()
            k = k.float()
            v = v.float()

        attn_out = self.custom_ops_lib.simplified_self_attention_cuda(q, k, v, self.h)  # (B, NQ, D)
        return self.fc_o(attn_out)