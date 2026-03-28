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

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
    return x;
}
static __device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, o));
    return x;
}
static __device__ __forceinline__ float warp_allreduce_sum(float x) {
    x = warp_reduce_sum(x);
    return __shfl_sync(0xffffffff, x, 0);
}
static __device__ __forceinline__ float warp_allreduce_max(float x) {
    x = warp_reduce_max(x);
    return __shfl_sync(0xffffffff, x, 0);
}

template<int HS, int TILE>
__global__ __launch_bounds__(32, 8)
void causal_attn_fwd_warp_row_hs_kernel(
    const float* __restrict__ q,   // [B,H,T,HS]
    const float* __restrict__ k,   // [B,H,T,HS]
    const float* __restrict__ v,   // [B,H,T,HS]
    float* __restrict__ y,         // [B,H,T,HS]
    int B, int H, int T,
    float scale,
    bool causal
) {
    // One warp computes one (b,h,t) row.
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x;

    const int64_t bh = row / T;
    const int t = (int)(row - bh * T);
    const int b = (int)(bh / H);
    const int h = (int)(bh - (int64_t)b * H);
    if (b >= B || h >= H || t >= T) return;

    const int j_max = causal ? (t + 1) : T;

    const int64_t base_bh = ((int64_t)b * H + h) * (int64_t)T * (int64_t)HS;
    const int64_t q_base  = base_bh + (int64_t)t * HS;

    constexpr int HS4 = HS / 4;
    const float4* q4p = reinterpret_cast<const float4*>(q + q_base);

    // Load Q into registers (float4 chunks owned by this lane in a strided manner).
    float4 qreg[ (HS4 + 31) / 32 ];
    #pragma unroll
    for (int i = 0; i < (HS4 + 31) / 32; ++i) qreg[i] = make_float4(0,0,0,0);

    #pragma unroll
    for (int i4 = lane, ridx = 0; i4 < HS4; i4 += 32, ++ridx) {
        qreg[ridx] = q4p[i4];
    }

    extern __shared__ float smem[];
    float* sK = smem;                  // TILE * HS floats
    float* sV = smem + TILE * HS;      // TILE * HS floats

    float row_max = -INFINITY;

    // Pass 1: compute max over logits
    for (int jb = 0; jb < j_max; jb += TILE) {
        const int tile_len = (jb + TILE <= j_max) ? TILE : (j_max - jb);
        const int vecs = tile_len * HS4; // float4 count

        // Cooperative load K and V to shared (float4 stores)
        for (int idx = lane; idx < vecs; idx += 32) {
            const int tj = idx / HS4;
            const int i4 = idx - tj * HS4;
            const int j = jb + tj;

            const int64_t kv_base = base_bh + (int64_t)j * HS;
            const float4* k4p = reinterpret_cast<const float4*>(k + kv_base);
            const float4* v4p = reinterpret_cast<const float4*>(v + kv_base);

            float4 kk = __ldg(k4p + i4);
            float4 vv = __ldg(v4p + i4);

            // Store to shared; shared is float-aligned; float4 ok if HS multiple of 4.
            reinterpret_cast<float4*>(sK + tj * HS)[i4] = kk;
            reinterpret_cast<float4*>(sV + tj * HS)[i4] = vv;
        }
        __syncwarp();

        // Each lane computes logits for keys tj = lane, lane+32, ... within this tile.
        for (int tj = lane; tj < tile_len; tj += 32) {
            const float4* k4s = reinterpret_cast<const float4*>(sK + tj * HS);

            float acc = 0.0f;
            #pragma unroll
            for (int i4 = 0, qidx = 0; i4 < HS4; i4 += 32, ++qidx) {
                const int kk_i4 = i4 + lane;
                // We need full dot; each lane only has a subset of q. Instead, do lane-local dot for its subset
                // and then warp-reduce to get full dot for this key.
                // We'll compute partial over q chunks owned by this lane:
                float part = 0.0f;
                #pragma unroll
                for (int p = 0; p < (HS4 + 31) / 32; ++p) {
                    int my_i4 = lane + p * 32;
                    if (my_i4 < HS4) {
                        float4 qv = qreg[p];
                        float4 kv = k4s[my_i4];
                        part += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                    }
                }
                acc = part;
                break; // acc already computed
            }

            float dot = warp_allreduce_sum(acc);
            float logit = dot * scale;
            row_max = fmaxf(row_max, logit);
        }
        row_max = warp_allreduce_max(row_max);
        __syncwarp();
    }

    // Pass 2: denom + output accumulation
    float denom = 0.0f;

    // Output accumulator: each lane owns d = lane, lane+32, lane+64 ...
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f; // HS=96 => 3 scalars per lane
    // For generic HS, we keep an array.
    float out_arr[8]; // supports up to HS=256 (8*32)
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_arr[i] = 0.0f;

    for (int jb = 0; jb < j_max; jb += TILE) {
        const int tile_len = (jb + TILE <= j_max) ? TILE : (j_max - jb);
        const int vecs = tile_len * HS4;

        for (int idx = lane; idx < vecs; idx += 32) {
            const int tj = idx / HS4;
            const int i4 = idx - tj * HS4;
            const int j = jb + tj;

            const int64_t kv_base = base_bh + (int64_t)j * HS;
            const float4* k4p = reinterpret_cast<const float4*>(k + kv_base);
            const float4* v4p = reinterpret_cast<const float4*>(v + kv_base);

            float4 kk = __ldg(k4p + i4);
            float4 vv = __ldg(v4p + i4);

            reinterpret_cast<float4*>(sK + tj * HS)[i4] = kk;
            reinterpret_cast<float4*>(sV + tj * HS)[i4] = vv;
        }
        __syncwarp();

        // Compute per-key weights and accumulate denom and output
        for (int tj = lane; tj < tile_len; tj += 32) {
            const float4* k4s = reinterpret_cast<const float4*>(sK + tj * HS);
            const float4* v4s = reinterpret_cast<const float4*>(sV + tj * HS);

            float part = 0.0f;
            #pragma unroll
            for (int p = 0; p < (HS4 + 31) / 32; ++p) {
                int my_i4 = lane + p * 32;
                if (my_i4 < HS4) {
                    float4 qv = qreg[p];
                    float4 kv = k4s[my_i4];
                    part += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            }
            float dot = warp_allreduce_sum(part);
            float logit = dot * scale;
            float w = __expf(logit - row_max);
            denom += w;

            // Accumulate into owned output dims for this lane
            // Use scalar loads from shared for simplicity; HS small (96) but generic up to 256.
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = lane + r * 32;
                if (d < HS) {
                    out_arr[r] = fmaf(w, ((const float*)(v4s))[d], out_arr[r]);
                }
            }
        }
        __syncwarp();
    }

    denom = warp_allreduce_sum(denom);
    float inv = 1.0f / denom;

    const int64_t y_base = q_base;
    // Write output
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = lane + r * 32;
        if (d < HS) {
            y[y_base + d] = out_arr[r] * inv;
        }
    }
}

template<int TILE>
__global__ __launch_bounds__(32, 8)
void causal_attn_fwd_warp_row_generic_kernel(
    const float* __restrict__ q,   // [B,H,T,HS]
    const float* __restrict__ k,   // [B,H,T,HS]
    const float* __restrict__ v,   // [B,H,T,HS]
    float* __restrict__ y,         // [B,H,T,HS]
    int B, int H, int T, int HS,
    float scale,
    bool causal
) {
    // HS must be divisible by 4 for this vectorized path.
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x;

    const int64_t bh = row / T;
    const int t = (int)(row - bh * T);
    const int b = (int)(bh / H);
    const int h = (int)(bh - (int64_t)b * H);
    if (b >= B || h >= H || t >= T) return;

    const int j_max = causal ? (t + 1) : T;

    const int HS4 = HS >> 2;

    const int64_t base_bh = ((int64_t)b * H + h) * (int64_t)T * (int64_t)HS;
    const int64_t q_base  = base_bh + (int64_t)t * HS;

    const float4* q4p = reinterpret_cast<const float4*>(q + q_base);

    // Q in registers: up to HS4=64 when HS=256; store strided per lane.
    float4 qreg[2]; // for common HS<=96, but handle larger with loop (loaded on demand)
    // We'll just load from global with __ldg when needed in dot to reduce register pressure.

    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = smem + TILE * HS;

    float row_max = -INFINITY;

    for (int jb = 0; jb < j_max; jb += TILE) {
        const int tile_len = (jb + TILE <= j_max) ? TILE : (j_max - jb);
        const int vecs = tile_len * HS4;
        for (int idx = lane; idx < vecs; idx += 32) {
            const int tj = idx / HS4;
            const int i4 = idx - tj * HS4;
            const int j = jb + tj;

            const int64_t kv_base = base_bh + (int64_t)j * HS;
            const float4* k4p = reinterpret_cast<const float4*>(k + kv_base);
            const float4* v4p = reinterpret_cast<const float4*>(v + kv_base);

            float4 kk = __ldg(k4p + i4);
            float4 vv = __ldg(v4p + i4);
            reinterpret_cast<float4*>(sK + tj * HS)[i4] = kk;
            reinterpret_cast<float4*>(sV + tj * HS)[i4] = vv;
        }
        __syncwarp();

        for (int tj = lane; tj < tile_len; tj += 32) {
            const float4* k4s = reinterpret_cast<const float4*>(sK + tj * HS);
            float part = 0.0f;
            for (int i4 = lane; i4 < HS4; i4 += 32) {
                float4 qv = __ldg(q4p + i4);
                float4 kv = k4s[i4];
                part += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
            float dot = warp_allreduce_sum(part);
            float logit = dot * scale;
            row_max = fmaxf(row_max, logit);
        }
        row_max = warp_allreduce_max(row_max);
        __syncwarp();
    }

    float denom = 0.0f;
    // output accumulators for up to HS=256 => 8 scalars per lane
    float out_arr[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_arr[i] = 0.0f;

    for (int jb = 0; jb < j_max; jb += TILE) {
        const int tile_len = (jb + TILE <= j_max) ? TILE : (j_max - jb);
        const int vecs = tile_len * HS4;
        for (int idx = lane; idx < vecs; idx += 32) {
            const int tj = idx / HS4;
            const int i4 = idx - tj * HS4;
            const int j = jb + tj;

            const int64_t kv_base = base_bh + (int64_t)j * HS;
            const float4* k4p = reinterpret_cast<const float4*>(k + kv_base);
            const float4* v4p = reinterpret_cast<const float4*>(v + kv_base);

            float4 kk = __ldg(k4p + i4);
            float4 vv = __ldg(v4p + i4);
            reinterpret_cast<float4*>(sK + tj * HS)[i4] = kk;
            reinterpret_cast<float4*>(sV + tj * HS)[i4] = vv;
        }
        __syncwarp();

        for (int tj = lane; tj < tile_len; tj += 32) {
            const float4* k4s = reinterpret_cast<const float4*>(sK + tj * HS);
            const float*  vfs = reinterpret_cast<const float*>(sV + tj * HS);

            float part = 0.0f;
            for (int i4 = lane; i4 < HS4; i4 += 32) {
                float4 qv = __ldg(q4p + i4);
                float4 kv = k4s[i4];
                part += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
            float dot = warp_allreduce_sum(part);
            float logit = dot * scale;
            float w = __expf(logit - row_max);
            denom += w;

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = lane + r * 32;
                if (d < HS) out_arr[r] = fmaf(w, vfs[d], out_arr[r]);
            }
        }
        __syncwarp();
    }

    denom = warp_allreduce_sum(denom);
    float inv = 1.0f / denom;
    const int64_t y_base = q_base;

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int d = lane + r * 32;
        if (d < HS) y[y_base + d] = out_arr[r] * inv;
    }
}

torch::Tensor min_gpt_causal_attention_cuda(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 &&
                k.scalar_type() == torch::kFloat32 &&
                v.scalar_type() == torch::kFloat32, "q,k,v must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q,k,v must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be [B,H,T,HS]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v must have same shape");

    int B  = (int)q.size(0);
    int H  = (int)q.size(1);
    int T  = (int)q.size(2);
    int HS = (int)q.size(3);

    TORCH_CHECK(HS > 0 && HS <= 256, "HS must be in (0,256]");
    TORCH_CHECK((HS % 4) == 0, "HS must be divisible by 4 for this kernel");
    TORCH_CHECK(T > 0 && T <= 1024, "T must be in (0,1024] for this kernel");

    auto y = torch::empty_like(q);
    float scale = 1.0f / sqrtf((float)HS);

    int rows = B * H * T;
    dim3 grid(rows);
    dim3 block(32); // 1 warp

    // tile size: trade-off shared memory vs fewer global re-reads; 64 usually good for T<=1024
    constexpr int TILE = 64;

    size_t smem = (size_t)2 * TILE * HS * sizeof(float); // K and V tiles

    if (HS == 96) {
        causal_attn_fwd_warp_row_hs_kernel<96, TILE><<<grid, block, smem>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, H, T, scale, causal
        );
    } else {
        causal_attn_fwd_warp_row_generic_kernel<TILE><<<grid, block, smem>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, H, T, HS, scale, causal
        );
    }
    return y;
}
"""

cpp_src = r"""
torch::Tensor min_gpt_causal_attention_cuda(torch::Tensor q,
                                           torch::Tensor k,
                                           torch::Tensor v,
                                           bool causal);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mingpt_causal_attention_opt5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["min_gpt_causal_attention_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    minGPT causal self-attention with custom fused CUDA op for (q,k,v)->y (inference, dropout=0).
    Projections remain in PyTorch.
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, nh, hs).transpose(1, 2).contiguous()
        q = q.view(B, T, nh, hs).transpose(1, 2).contiguous()
        v = v.view(B, T, nh, hs).transpose(1, 2).contiguous()

        if self.attn_dropout.p != 0.0 or self.training:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            y = self.custom_ops.min_gpt_causal_attention_cuda(q, k, v, True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y