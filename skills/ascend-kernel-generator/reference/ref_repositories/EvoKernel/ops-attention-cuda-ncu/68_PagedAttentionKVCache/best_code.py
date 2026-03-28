import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)
#define CHECK_INPUT_I32(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x)

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ int32_t ldg_i32(const int32_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

__device__ __forceinline__ uint32_t ldg_u32(const uint32_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat16 bx;
  memcpy(&bx, &bits, sizeof(uint16_t));
  return __bfloat162float(bx);
#else
  uint32_t u = ((uint32_t)bits) << 16;
  float f;
  memcpy(&f, &u, sizeof(uint32_t));
  return f;
#endif
}

__device__ __forceinline__ float bf16_to_float(at::BFloat16 x) {
  uint16_t bits;
  memcpy(&bits, &x, sizeof(uint16_t));
  return bf16_bits_to_float(bits);
}

__device__ __forceinline__ at::BFloat16 float_to_bf16(float f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat16 bx = __float2bfloat16_rn(f);
  at::BFloat16 out;
  memcpy(&out, &bx, sizeof(uint16_t));
  return out;
#else
  uint32_t u;
  memcpy(&u, &f, sizeof(uint32_t));
  uint16_t bits = (uint16_t)(u >> 16);
  at::BFloat16 out;
  memcpy(&out, &bits, sizeof(uint16_t));
  return out;
#endif
}

// Load 2 bf16 packed in one u32 from address p (bf16 pointer), index i_pair = i/2
__device__ __forceinline__ float ldg_bf16_elem_as_float(const at::BFloat16* __restrict__ p, int i) {
  const uint32_t* p32 = reinterpret_cast<const uint32_t*>(p);
  uint32_t w = ldg_u32(p32 + (i >> 1));
  uint16_t bits = (i & 1) ? (uint16_t)(w >> 16) : (uint16_t)(w & 0xFFFFu);
  return bf16_bits_to_float(bits);
}

// L2 prefetch hint (best-effort)
__device__ __forceinline__ void prefetch_L2(const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
  (void)p;
#endif
}

// 1 warp handles one (b, hq). grid=(B,Hq).
__global__ void paged_attn_decode_warp_bf16_kernel(
    const at::BFloat16* __restrict__ Q,          // [B,1,Hq,D]
    const at::BFloat16* __restrict__ Kc,         // [NB,PBS,Hkv,D]
    const at::BFloat16* __restrict__ Vc,         // [NB,PBS,Hkv,D]
    const int32_t* __restrict__ cache_seqlens,   // [B]
    const int32_t* __restrict__ page_table,      // [B,max_blocks]
    at::BFloat16* __restrict__ Out,              // [B,1,Hq,D]
    int Hq, int Hkv, int D,
    int page_block_size,
    int max_blocks_per_seq,
    float scale
) {
  int lane = (int)threadIdx.x & 31;
  int b = (int)blockIdx.x;
  int hq = (int)blockIdx.y;

  int S = (int)ldg_i32(cache_seqlens + b);
  if (hq >= Hq) return;

  at::BFloat16* out_ptr = Out + (((b * 1 + 0) * Hq + hq) * D);
  if (S <= 0) {
    for (int d = lane; d < D; d += 32) out_ptr[d] = float_to_bf16(0.0f);
    return;
  }

  int n_groups = Hq / Hkv;
  int hkv = hq / n_groups;

  const at::BFloat16* q_ptr = Q + (((b * 1 + 0) * Hq + hq) * D);

  float m = -FLT_MAX;
  float l = 0.0f;

  // D==128 fast path: 4 elements per lane for Q and output
  if (D == 128) {
    float q0 = ldg_bf16_elem_as_float(q_ptr, lane);
    float q1 = ldg_bf16_elem_as_float(q_ptr, lane + 32);
    float q2 = ldg_bf16_elem_as_float(q_ptr, lane + 64);
    float q3 = ldg_bf16_elem_as_float(q_ptr, lane + 96);

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    int num_blocks = (S + page_block_size - 1) / page_block_size;
    const int32_t* pt_row = page_table + b * max_blocks_per_seq;

    // Software pipeline: precompute next pointers, prefetch
    const at::BFloat16* k_ptr_next = nullptr;
    const at::BFloat16* v_ptr_next = nullptr;
    bool has_next = false;

    // Helper lambda macro-ish for pointer calc
#define PTRS_FOR(lb, off, kptr, vptr) do { \
      int32_t phys = ldg_i32(pt_row + (lb)); \
      int token_base = phys * page_block_size + (off); \
      (kptr) = Kc + ((token_base * Hkv + hkv) * 128); \
      (vptr) = Vc + ((token_base * Hkv + hkv) * 128); \
    } while(0)

    // Prime next if exists
    if (num_blocks > 0) {
      PTRS_FOR(0, 0, k_ptr_next, v_ptr_next);
      has_next = true;
      prefetch_L2((const void*)k_ptr_next);
      prefetch_L2((const void*)v_ptr_next);
    }

    for (int lb = 0; lb < num_blocks; ++lb) {
      int remaining = S - lb * page_block_size;
      int tokens_in_block = remaining < page_block_size ? remaining : page_block_size;

      for (int off = 0; off < tokens_in_block; ++off) {
        const at::BFloat16* k_ptr = k_ptr_next;
        const at::BFloat16* v_ptr = v_ptr_next;

        // Stage prefetch for next token
        int n_lb = lb;
        int n_off = off + 1;
        if (n_off >= tokens_in_block) { n_lb = lb + 1; n_off = 0; }
        if (n_lb < num_blocks) {
          PTRS_FOR(n_lb, n_off, k_ptr_next, v_ptr_next);
          prefetch_L2((const void*)k_ptr_next);
          prefetch_L2((const void*)v_ptr_next);
        } else {
          k_ptr_next = nullptr; v_ptr_next = nullptr;
        }

        // Load K (packed u32 -> bf16 -> float)
        float k0 = ldg_bf16_elem_as_float(k_ptr, lane);
        float k1 = ldg_bf16_elem_as_float(k_ptr, lane + 32);
        float k2 = ldg_bf16_elem_as_float(k_ptr, lane + 64);
        float k3 = ldg_bf16_elem_as_float(k_ptr, lane + 96);

        float dot_partial = fmaf(q0, k0, 0.f);
        dot_partial = fmaf(q1, k1, dot_partial);
        dot_partial = fmaf(q2, k2, dot_partial);
        dot_partial = fmaf(q3, k3, dot_partial);

        float dot = warp_reduce_sum(dot_partial);
        dot = __shfl_sync(0xffffffff, dot, 0);
        float s = dot * scale;

        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * alpha + p;

        float vv0 = ldg_bf16_elem_as_float(v_ptr, lane);
        float vv1 = ldg_bf16_elem_as_float(v_ptr, lane + 32);
        float vv2 = ldg_bf16_elem_as_float(v_ptr, lane + 64);
        float vv3 = ldg_bf16_elem_as_float(v_ptr, lane + 96);

        acc0 = acc0 * alpha + vv0 * p;
        acc1 = acc1 * alpha + vv1 * p;
        acc2 = acc2 * alpha + vv2 * p;
        acc3 = acc3 * alpha + vv3 * p;

        m = m_new;
      }
    }

#undef PTRS_FOR

    float inv_l = 1.0f / (l + 1e-9f);
    out_ptr[lane]      = float_to_bf16(acc0 * inv_l);
    out_ptr[lane + 32] = float_to_bf16(acc1 * inv_l);
    out_ptr[lane + 64] = float_to_bf16(acc2 * inv_l);
    out_ptr[lane + 96] = float_to_bf16(acc3 * inv_l);
    return;
  }

  // Generic path (warp streaming), still 1 warp per (b,hq)
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  bool has0 = (lane < D);
  bool has1 = (lane + 32) < D;
  bool has2 = (lane + 64) < D;
  bool has3 = (lane + 96) < D;

  int num_blocks = (S + page_block_size - 1) / page_block_size;
  const int32_t* pt_row = page_table + b * max_blocks_per_seq;

  for (int lb = 0; lb < num_blocks; ++lb) {
    int32_t phys_block = ldg_i32(pt_row + lb);
    int remaining = S - lb * page_block_size;
    int tokens_in_block = remaining < page_block_size ? remaining : page_block_size;
    int token_base = phys_block * page_block_size;

    for (int off = 0; off < tokens_in_block; ++off) {
      const at::BFloat16* k_ptr = Kc + ((((token_base + off) * Hkv + hkv) * D));
      const at::BFloat16* v_ptr = Vc + ((((token_base + off) * Hkv + hkv) * D));
      // best-effort prefetch next cache lines
      prefetch_L2((const void*)(k_ptr + 0));
      prefetch_L2((const void*)(v_ptr + 0));

      float dot_partial = 0.f;
      for (int d = lane; d < D; d += 32) {
        float qf = bf16_to_float(q_ptr[d]);
        float kf = bf16_to_float(k_ptr[d]);
        dot_partial = fmaf(qf, kf, dot_partial);
      }
      float dot = warp_reduce_sum(dot_partial);
      dot = __shfl_sync(0xffffffff, dot, 0);
      float s = dot * scale;

      float m_new = fmaxf(m, s);
      float alpha = __expf(m - m_new);
      float p = __expf(s - m_new);
      l = l * alpha + p;

      if (has0) acc0 = acc0 * alpha + bf16_to_float(v_ptr[lane]) * p;
      if (has1) acc1 = acc1 * alpha + bf16_to_float(v_ptr[lane + 32]) * p;
      if (has2) acc2 = acc2 * alpha + bf16_to_float(v_ptr[lane + 64]) * p;
      if (has3) acc3 = acc3 * alpha + bf16_to_float(v_ptr[lane + 96]) * p;

      m = m_new;
    }
  }

  float inv_l = 1.0f / (l + 1e-9f);
  if (has0) out_ptr[lane] = float_to_bf16(acc0 * inv_l);
  if (has1) out_ptr[lane + 32] = float_to_bf16(acc1 * inv_l);
  if (has2) out_ptr[lane + 64] = float_to_bf16(acc2 * inv_l);
  if (has3) out_ptr[lane + 96] = float_to_bf16(acc3 * inv_l);
}

torch::Tensor paged_attention_kv_cache_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_seqlens,
    torch::Tensor page_table,
    double scale,
    bool causal
) {
  (void)causal; // decode seqlen_q==1: reference causal mask allows all keys

  CHECK_INPUT_BF16(q);
  CHECK_INPUT_BF16(k_cache);
  CHECK_INPUT_BF16(v_cache);
  CHECK_INPUT_I32(cache_seqlens);
  CHECK_INPUT_I32(page_table);

  TORCH_CHECK(q.dim() == 4, "q must be [B,1,Hq,D]");
  TORCH_CHECK(q.size(1) == 1, "decode kernel supports seqlen_q==1 only");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be [NB,PBS,Hkv,D]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must have same shape");
  TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be [B]");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be [B,max_blocks]");

  int B = (int)q.size(0);
  int Hq = (int)q.size(2);
  int D  = (int)q.size(3);

  int PBS = (int)k_cache.size(1);
  int Hkv = (int)k_cache.size(2);
  int Dk  = (int)k_cache.size(3);

  TORCH_CHECK(Dk == D, "headdim mismatch between q and cache");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK((int)page_table.size(0) == B, "page_table batch mismatch");
  int max_blocks = (int)page_table.size(1);

  auto out = torch::empty_like(q);

  dim3 block(32, 1, 1);      // 1 warp
  dim3 grid(B, Hq, 1);       // one warp per (b,hq)

  paged_attn_decode_warp_bf16_kernel<<<grid, block, 0>>>(
      (const at::BFloat16*)q.data_ptr<at::BFloat16>(),
      (const at::BFloat16*)k_cache.data_ptr<at::BFloat16>(),
      (const at::BFloat16*)v_cache.data_ptr<at::BFloat16>(),
      (const int32_t*)cache_seqlens.data_ptr<int32_t>(),
      (const int32_t*)page_table.data_ptr<int32_t>(),
      (at::BFloat16*)out.data_ptr<at::BFloat16>(),
      Hq, Hkv, D,
      PBS,
      max_blocks,
      (float)scale
  );

  return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor paged_attention_kv_cache_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_seqlens,
    torch::Tensor page_table,
    double scale,
    bool causal
);
"""

custom_ops_lib = load_inline(
    name="custom_paged_attn_ops_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["paged_attention_kv_cache_decode_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Paged KV Cache Attention optimized for decoding (seqlen_q==1) using a custom CUDA op.
    Falls back to a PyTorch reference path for unsupported cases.
    """

    def __init__(self, nheads_q, nheads_kv, headdim, page_block_size, causal=True):
        super().__init__()
        self.nheads_q = int(nheads_q)
        self.nheads_kv = int(nheads_kv)
        self.headdim = int(headdim)
        self.page_block_size = int(page_block_size)
        self.causal = bool(causal)
        self.scale = 1.0 / math.sqrt(self.headdim)
        assert self.nheads_q % self.nheads_kv == 0
        self.n_groups = self.nheads_q // self.nheads_kv
        self.custom_ops_lib = custom_ops_lib

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table):
        if (
            q.is_cuda
            and k_cache.is_cuda
            and v_cache.is_cuda
            and q.dtype == torch.bfloat16
            and k_cache.dtype == torch.bfloat16
            and v_cache.dtype == torch.bfloat16
            and cache_seqlens.dtype == torch.int32
            and page_table.dtype == torch.int32
            and q.is_contiguous()
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
            and cache_seqlens.is_contiguous()
            and page_table.is_contiguous()
            and q.dim() == 4
            and q.size(1) == 1
        ):
            return self.custom_ops_lib.paged_attention_kv_cache_decode_cuda(
                q, k_cache, v_cache, cache_seqlens, page_table, float(self.scale), bool(self.causal)
            )

        # Fallback: reference implementation
        batch = q.shape[0]
        seqlen_q = q.shape[1]
        outputs = []
        for b in range(batch):
            seq_len = int(cache_seqlens[b].item())
            num_blocks_needed = (seq_len + self.page_block_size - 1) // self.page_block_size

            kvk_parts = []
            kvv_parts = []
            for block_idx in range(num_blocks_needed):
                phys = int(page_table[b, block_idx].item())
                if block_idx == num_blocks_needed - 1:
                    remaining = seq_len - block_idx * self.page_block_size
                    kvk_parts.append(k_cache[phys, :remaining])
                    kvv_parts.append(v_cache[phys, :remaining])
                else:
                    kvk_parts.append(k_cache[phys])
                    kvv_parts.append(v_cache[phys])

            k = torch.cat(kvk_parts, dim=0)  # (seq_len, Hkv, D)
            v = torch.cat(kvv_parts, dim=0)  # (seq_len, Hkv, D)

            k = k.unsqueeze(0).transpose(1, 2).float()  # (1, Hkv, seq_len, D)
            v = v.unsqueeze(0).transpose(1, 2).float()

            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)

            q_b = q[b : b + 1].transpose(1, 2).float()  # (1, Hq, seqlen_q, D)
            scores = torch.matmul(q_b, k.transpose(-2, -1)) * self.scale

            if self.causal:
                row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                causal_mask = col_idx > (row_idx + seq_len - seqlen_q)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(attn, v).transpose(1, 2).to(q.dtype)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)