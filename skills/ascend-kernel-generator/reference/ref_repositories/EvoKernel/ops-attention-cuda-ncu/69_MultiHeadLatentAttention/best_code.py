import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static inline __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static inline __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

static inline __device__ __nv_bfloat16 f32_to_bf16(float v) { return __float2bfloat16_rn(v); }

static inline __device__ __nv_bfloat16 ldg_bf16(const __nv_bfloat16* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline __device__ __nv_bfloat162 ldg_bf162(const __nv_bfloat162* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// ------------------------------ Fallback kernel (kept for generality) ------------------------------
static inline __device__ float block_reduce_sum(float val) {
    __shared__ float sh[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float w = warp_reduce_sum(val);
    if (lane == 0) sh[warp] = w;
    __syncthreads();
    if (warp == 0) {
        float v = (tid < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
        float ww = warp_reduce_sum(v);
        if (lane == 0) sh[0] = ww;
    }
    __syncthreads();
    return sh[0];
}

static inline __device__ __nv_bfloat162 load_bf162(const __nv_bfloat16* p) {
    return *reinterpret_cast<const __nv_bfloat162*>(p);
}

__global__ void mla_paged_fwd_online_kernel_fallback(
    const void* __restrict__ q_ptr_void,
    const void* __restrict__ kv_ptr_void,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    void* __restrict__ out_ptr_void,
    int B, int Sq, int H, int Dqk, int Dv,
    int page_block_size, int max_blocks_per_seq,
    float scale, bool causal
) {
    const __nv_bfloat16* __restrict__ q = (const __nv_bfloat16*)q_ptr_void;
    const __nv_bfloat16* __restrict__ kv = (const __nv_bfloat16*)kv_ptr_void;
    __nv_bfloat16* __restrict__ out = (__nv_bfloat16*)out_ptr_void;

    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    int tid = threadIdx.x;

    int cache_len = cache_seqlens[b];
    if (cache_len <= 0) return;

    int max_j = cache_len - 1;
    if (causal) {
        max_j = i + cache_len - Sq;
        if (max_j < -1) max_j = -1;
        if (max_j > cache_len - 1) max_j = cache_len - 1;
    }
    int effective_len = max_j + 1;
    if (effective_len <= 0) {
        for (int d = tid; d < Dv; d += blockDim.x) {
            int out_idx = (((b * Sq + i) * H + h) * Dv + d);
            out[out_idx] = f32_to_bf16(0.0f);
        }
        return;
    }

    int blocks_needed = (effective_len + page_block_size - 1) / page_block_size;

    extern __shared__ unsigned char smem[];
    __nv_bfloat16* q_sh = reinterpret_cast<__nv_bfloat16*>(smem);
    int* bt_sh = reinterpret_cast<int*>(q_sh + Dqk);

    const __nv_bfloat16* qvec = q + (((b * Sq + i) * H + h) * Dqk);
    for (int d = tid; d < Dqk; d += blockDim.x) q_sh[d] = qvec[d];

    const int* bt_row = block_table + b * max_blocks_per_seq;
    for (int t = tid; t < blocks_needed; t += blockDim.x) bt_sh[t] = bt_row[t];
    __syncthreads();

    float m = -FLT_MAX;
    float s = 0.0f;

    float acc0 = 0.0f, acc1 = 0.0f;
    int d0 = tid;
    int d1 = tid + blockDim.x;
    bool has_d0 = (d0 < Dv);
    bool has_d1 = (d1 < Dv);

    for (int j = 0; j < effective_len; ++j) {
        int block_logical = j / page_block_size;
        int offset = j - block_logical * page_block_size;
        int phys = bt_sh[block_logical];

        const __nv_bfloat16* kvec = kv + (((phys * page_block_size + offset) * 1) * Dqk);
        const __nv_bfloat16* vvec = kvec;

        float partial = 0.0f;
        int d = tid * 2;
        int stride = blockDim.x * 2;

        for (; d + 1 < Dqk; d += stride) {
            __nv_bfloat162 q2 = load_bf162(q_sh + d);
            __nv_bfloat162 k2 = load_bf162(kvec + d);
            float2 qf = __bfloat1622float2(q2);
            float2 kf = __bfloat1622float2(k2);
            partial = fmaf(qf.x, kf.x, partial);
            partial = fmaf(qf.y, kf.y, partial);
        }
        if ((Dqk & 1) && (d == Dqk - 1)) {
            partial = fmaf(__bfloat162float(q_sh[d]), __bfloat162float(kvec[d]), partial);
        }

        float dot = block_reduce_sum(partial);
        float score = dot * scale;

        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float beta  = __expf(score - m_new);
        s = s * alpha + beta;

        if (has_d0) {
            float v0 = __bfloat162float(vvec[d0]);
            acc0 = acc0 * alpha + beta * v0;
        }
        if (has_d1) {
            float v1 = __bfloat162float(vvec[d1]);
            acc1 = acc1 * alpha + beta * v1;
        }

        m = m_new;
        __syncthreads();
    }

    float inv_s = 1.0f / (s + 1e-9f);
    if (has_d0) out[(((b * Sq + i) * H + h) * Dv + d0)] = f32_to_bf16(acc0 * inv_s);
    if (has_d1) out[(((b * Sq + i) * H + h) * Dv + d1)] = f32_to_bf16(acc1 * inv_s);
}

// ------------------------------ Fast-path: Dqk=576, Dv=512, 2 warps/CTA, page-wise loop ------------------------------
// Key properties:
// - Correct warp-uniform online softmax (explicit broadcasts)
// - Lane0 loads phys block id; avoids shared memory
// - Iterates per page (block) and unrolls within page_block_size (typically 16)
// - BF16x2 vector loads/stores for V/output
__global__ __launch_bounds__(64, 4)
void mla_paged_fwd_rowblock576_v512_opt3(
    const __nv_bfloat16* __restrict__ q,            // [B,Sq,H,576]
    const __nv_bfloat16* __restrict__ kv,           // [NB,PBS,1,576]
    const int* __restrict__ block_table,            // [B,MB]
    const int* __restrict__ cache_seqlens,          // [B]
    __nv_bfloat16* __restrict__ out,                // [B,Sq,H,512]
    int B, int Sq, int H,
    int page_block_size, int max_blocks_per_seq,
    float scale, bool causal
) {
    int row = (int)blockIdx.x;
    int h = row % H;
    int tmp = row / H;
    int i = tmp % Sq;
    int b = tmp / Sq;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0 or 1
    if (warp > 1) return;

    int cache_len = __ldg(cache_seqlens + b);
    int max_j = cache_len - 1;
    if (cache_len <= 0) max_j = -1;
    if (causal && max_j >= 0) {
        int mj = i + cache_len - Sq;
        mj = mj < -1 ? -1 : mj;
        mj = mj > (cache_len - 1) ? (cache_len - 1) : mj;
        max_j = mj;
    }
    int effective_len = max_j + 1;

    int out_base = (((b * Sq + i) * H + h) * 512);
    if (effective_len <= 0) {
        int pair_base = warp * 128; // 128 bf16x2 pairs per warp
        for (int p = lane; p < 128; p += 32) {
            int dv = (pair_base + p) * 2;
            *reinterpret_cast<__nv_bfloat162*>(out + out_base + dv) =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
        return;
    }

    int blocks_needed = (effective_len + page_block_size - 1) / page_block_size;

    const int* bt_row = block_table + b * max_blocks_per_seq;
    const __nv_bfloat16* qvec = q + (((b * Sq + i) * H + h) * 576);

    float m = -FLT_MAX;
    float s = 0.0f;

    // V accumulators: each warp covers 256 dims => 128 bf16x2 pairs => 4 pairs per lane
    float2 vacc[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) vacc[t] = make_float2(0.0f, 0.0f);

    int dv_pair_base = warp * 128;

    // Iterate per page to avoid div/mod per token.
    int remaining = effective_len;
    #pragma unroll 1
    for (int bl = 0; bl < 64; ++bl) { // hard safety cap; blocks_needed <= max_blocks_per_seq (<=64 typical)
        if (bl >= blocks_needed) break;

        // Lane0 loads physical block id, broadcast
        int phys = 0;
        if (lane == 0) phys = __ldg(bt_row + bl);
        phys = __shfl_sync(0xffffffff, phys, 0);

        // Base pointer for this page (points to token0 in this page)
        const __nv_bfloat16* page_base = kv + ((phys * page_block_size) * 576);

        // tokens in this page
        int tok_in_page = remaining > page_block_size ? page_block_size : remaining;
        remaining -= tok_in_page;

        // Unroll for common page_block_size=16; guarded by tok_in_page
        #pragma unroll 16
        for (int tkn = 0; tkn < 16; ++tkn) {
            if (tkn >= tok_in_page) break;

            const __nv_bfloat16* kvec = page_base + tkn * 576;
            const __nv_bfloat16* vvec = kvec;

            // Dot over 576: 288 bf16x2; each lane handles 9 bf16x2 (same mapping as baseline)
            float part = 0.0f;
            int d2 = lane * 2;
            #pragma unroll
            for (int it = 0; it < 9; ++it) {
                int d = d2 + it * 64;
                __nv_bfloat162 q2 = ldg_bf162(reinterpret_cast<const __nv_bfloat162*>(qvec + d));
                __nv_bfloat162 k2 = ldg_bf162(reinterpret_cast<const __nv_bfloat162*>(kvec + d));
                float2 qf = __bfloat1622float2(q2);
                float2 kf = __bfloat1622float2(k2);
                part = fmaf(qf.x, kf.x, part);
                part = fmaf(qf.y, kf.y, part);
            }

            float dot0 = warp_reduce_sum(part);
            float dot = __shfl_sync(0xffffffff, dot0, 0); // broadcast sum to all lanes
            float score = dot * scale;

            // Online softmax: compute in lane0, broadcast alpha/beta/m_new/s_new
            float m_new, alpha, beta, s_new;
            if (lane == 0) {
                m_new = fmaxf(m, score);
                alpha = __expf(m - m_new);
                beta  = __expf(score - m_new);
                s_new = s * alpha + beta;
            }
            m_new = __shfl_sync(0xffffffff, m_new, 0);
            alpha = __shfl_sync(0xffffffff, alpha, 0);
            beta  = __shfl_sync(0xffffffff, beta, 0);
            s_new = __shfl_sync(0xffffffff, s_new, 0);

            // Accumulate V (bf16x2 vector loads)
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int pair_idx = dv_pair_base + lane + u * 32;
                int dv = pair_idx * 2;
                __nv_bfloat162 v2 = ldg_bf162(reinterpret_cast<const __nv_bfloat162*>(vvec + dv));
                float2 vf = __bfloat1622float2(v2);
                vacc[u].x = vacc[u].x * alpha + beta * vf.x;
                vacc[u].y = vacc[u].y * alpha + beta * vf.y;
            }

            m = m_new;
            s = s_new;
        }
    }

    float inv_s = 1.0f / (s + 1e-9f);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int pair_idx = dv_pair_base + lane + u * 32;
        int dv = pair_idx * 2;
        float2 v = vacc[u];
        __nv_bfloat162 o2 = __floats2bfloat162_rn(v.x * inv_s, v.y * inv_s);
        *reinterpret_cast<__nv_bfloat162*>(out + out_base + dv) = o2;
    }
}

torch::Tensor mla_paged_attention_fwd_cuda(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    double scale,
    int64_t headdim_v,
    bool causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(kv_cache);
    CHECK_INPUT(block_table);
    CHECK_INPUT(cache_seqlens);

    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bfloat16");
    TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::BFloat16, "kv_cache must be bfloat16");
    TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int, "block_table must be int32");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::ScalarType::Int, "cache_seqlens must be int32");

    TORCH_CHECK(q.dim() == 4, "q must be [B, Sq, H, Dqk]");
    TORCH_CHECK(kv_cache.dim() == 4, "kv_cache must be [NB, PBS, 1, Dqk]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, MB]");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be [B]");

    int B = (int)q.size(0);
    int Sq = (int)q.size(1);
    int H  = (int)q.size(2);
    int Dqk = (int)q.size(3);

    int page_block_size = (int)kv_cache.size(1);
    TORCH_CHECK((int)kv_cache.size(2) == 1, "kv_cache third dim must be 1");
    TORCH_CHECK((int)kv_cache.size(3) == Dqk, "kv_cache last dim must match q Dqk");

    int max_blocks_per_seq = (int)block_table.size(1);
    TORCH_CHECK((int)block_table.size(0) == B, "block_table batch dim mismatch");

    int Dv = (int)headdim_v;
    TORCH_CHECK(Dv > 0 && Dv <= Dqk, "headdim_v must be in (0, Dqk]");

    auto out = torch::empty({B, Sq, H, Dv}, q.options());
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (Dqk == 576 && Dv == 512) {
        int64_t total_rows = (int64_t)B * Sq * H;
        dim3 blocks((unsigned int)total_rows);
        dim3 threads(64);

        mla_paged_fwd_rowblock576_v512_opt3<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)q.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)kv_cache.data_ptr<at::BFloat16>(),
            (const int*)block_table.data_ptr<int>(),
            (const int*)cache_seqlens.data_ptr<int>(),
            (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
            B, Sq, H,
            page_block_size, max_blocks_per_seq,
            (float)scale, (bool)causal
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    int threads_fb = 256;
    dim3 block(threads_fb);
    dim3 grid(B, H, Sq);
    size_t shmem_fb = (size_t)Dqk * sizeof(__nv_bfloat16) + (size_t)max_blocks_per_seq * sizeof(int);

    mla_paged_fwd_online_kernel_fallback<<<grid, block, shmem_fb, stream>>>(
        q.data_ptr(),
        kv_cache.data_ptr(),
        (const int*)block_table.data_ptr<int>(),
        (const int*)cache_seqlens.data_ptr<int>(),
        out.data_ptr(),
        B, Sq, H, Dqk, Dv,
        page_block_size, max_blocks_per_seq,
        (float)scale, (bool)causal
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor mla_paged_attention_fwd_cuda(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    double scale,
    int64_t headdim_v,
    bool causal
);
"""

custom_ops_lib = load_inline(
    name="custom_mla_ops_opt7_rowblock_pagewise",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mla_paged_attention_fwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "-lineinfo",
        # keep some control over reg growth on different architectures
        "-maxrregcount=128",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Multi-head Latent Attention (MLA) using an optimized fused custom CUDA op over paged KV cache.
    Forward-only.
    """

    def __init__(self, nheads_q, headdim_qk, headdim_v, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = int(nheads_q)
        self.headdim_qk = int(headdim_qk)
        self.headdim_v = int(headdim_v)
        self.page_block_size = int(page_block_size)
        self.causal = bool(causal)
        self.scale = 1.0 / math.sqrt(self.headdim_qk)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, q, kv_cache, block_table, cache_seqlens):
        q = q.contiguous()
        kv_cache = kv_cache.contiguous()
        block_table = block_table.contiguous()
        cache_seqlens = cache_seqlens.contiguous()
        return self.custom_ops_lib.mla_paged_attention_fwd_cuda(
            q, kv_cache, block_table, cache_seqlens,
            float(self.scale), int(self.headdim_v), bool(self.causal)
        )