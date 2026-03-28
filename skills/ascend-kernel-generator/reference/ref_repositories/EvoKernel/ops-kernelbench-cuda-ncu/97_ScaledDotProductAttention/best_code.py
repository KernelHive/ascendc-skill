import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA Scaled Dot Product Attention (forward only)
# - Specializes for D == 1024 (given workload), falls back to a generic kernel otherwise
# - Uses register-tiled output accumulation (float4 per thread) instead of shared out_acc[D]
# - Uses warp-level reductions for dot products to reduce syncthreads and shared traffic
# Assumptions:
# - Q, K, V are CUDA float32 contiguous
# - Shapes: [B, H, S, D]
# - No mask, no dropout, non-causal
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __forceinline__ __device__ float warp_sum(float v) {
    // Full mask for all current CUDA arch
    unsigned mask = 0xffffffffu;
    // Reduce within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template<int THREADS>
static __forceinline__ __device__ float block_sum(float v) {
    // Sum across block using warp sums + shared for warp leaders
    __shared__ float warp_sums[THREADS / 32]; // THREADS must be multiple of 32
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    float w = warp_sum(v);
    if (lane == 0) warp_sums[warp] = w;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        out = (lane < (THREADS / 32)) ? warp_sums[lane] : 0.0f;
        out = warp_sum(out);
    }
    // Broadcast final sum
    out = __shfl_sync(0xffffffffu, out, 0);
    return out;
}

template<int THREADS>
__global__ void sdpa_fwd_kernel_d1024(
    const float* __restrict__ Q,   // [B,H,S,D]
    const float* __restrict__ K,   // [B,H,S,D]
    const float* __restrict__ V,   // [B,H,S,D]
    float* __restrict__ Out,       // [B,H,S,D]
    int B, int H, int S,
    float scale
) {
    constexpr int D = 1024;
    constexpr int VEC = 4;            // float4
    constexpr int DV = D / VEC;       // 256 float4s per row
    static_assert(THREADS % 32 == 0, "THREADS must be warp-multiple");

    int qs = blockIdx.x;           // 0..S-1
    int bh = blockIdx.y;           // 0..(B*H-1)
    int b = bh / H;
    int h = bh - b * H;

    int tid = threadIdx.x;

    int64_t head_stride = (int64_t)S * (int64_t)D;
    int64_t base = ((int64_t)b * H + h) * head_stride;

    const float* q_ptr = Q + base + (int64_t)qs * D;
    const float* k_ptr = K + base;
    const float* v_ptr = V + base;
    float* out_ptr = Out + base + (int64_t)qs * D;

    // Register-tiled output accumulator: each thread owns multiple float4 lanes.
    // Total float4 elements = 256. With THREADS=256, each thread owns exactly 1 float4.
    // With THREADS=128, each thread owns 2 float4, etc.
    constexpr int V4_PER_THREAD = DV / THREADS; // assumes DV divisible by THREADS
    static_assert(DV % THREADS == 0, "DV must be divisible by THREADS for this specialization");

    float4 acc[V4_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < V4_PER_THREAD; ++i) {
        acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float m = -INFINITY;
    float l = 0.0f;

    // Use float4 vector loads for Q/K/V
    const float4* q4 = reinterpret_cast<const float4*>(q_ptr);

    for (int ks = 0; ks < S; ++ks) {
        const float* k_row = k_ptr + (int64_t)ks * D;
        const float* v_row = v_ptr + (int64_t)ks * D;

        const float4* k4 = reinterpret_cast<const float4*>(k_row);

        // Dot(q, k): each thread processes its float4(s)
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < V4_PER_THREAD; ++i) {
            int idx4 = tid + i * THREADS;
            float4 qv = q4[idx4];
            float4 kv = k4[idx4];
            partial = fmaf(qv.x, kv.x, partial);
            partial = fmaf(qv.y, kv.y, partial);
            partial = fmaf(qv.z, kv.z, partial);
            partial = fmaf(qv.w, kv.w, partial);
        }

        float dot = block_sum<THREADS>(partial);
        float logit = dot * scale;

        // Online softmax update
        float m_new = fmaxf(m, logit);
        float alpha = __expf(m - m_new);
        float p = __expf(logit - m_new);

        // Update vector accumulator: acc = acc*alpha + p*V
        const float4* v4 = reinterpret_cast<const float4*>(v_row);
        #pragma unroll
        for (int i = 0; i < V4_PER_THREAD; ++i) {
            int idx4 = tid + i * THREADS;
            float4 vv = v4[idx4];
            float4 a = acc[i];
            a.x = a.x * alpha + p * vv.x;
            a.y = a.y * alpha + p * vv.y;
            a.z = a.z * alpha + p * vv.z;
            a.w = a.w * alpha + p * vv.w;
            acc[i] = a;
        }

        l = l * alpha + p;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    float4* out4 = reinterpret_cast<float4*>(out_ptr);
    #pragma unroll
    for (int i = 0; i < V4_PER_THREAD; ++i) {
        int idx4 = tid + i * THREADS;
        float4 a = acc[i];
        a.x *= inv_l;
        a.y *= inv_l;
        a.z *= inv_l;
        a.w *= inv_l;
        out4[idx4] = a;
    }
}

template<int THREADS>
__global__ void sdpa_fwd_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int S, int D,
    float scale
) {
    // Generic but still improved vs baseline:
    // - warp/block reduction for dot
    // - register accumulation for a small tile per thread (float4 if possible)
    int qs = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh - b * H;
    int tid = threadIdx.x;

    int64_t head_stride = (int64_t)S * (int64_t)D;
    int64_t base = ((int64_t)b * H + h) * head_stride;

    const float* q_ptr = Q + base + (int64_t)qs * D;
    const float* k_ptr = K + base;
    const float* v_ptr = V + base;
    float* out_ptr = Out + base + (int64_t)qs * D;

    float m = -INFINITY;
    float l = 0.0f;

    // We'll accumulate output in chunks of 4 floats when possible; remainder scalar.
    int D4 = D / 4;
    int rem = D - D4 * 4;

    // Each thread owns multiple float4s in a strided manner; keep small register tile by looping at write time.
    // We cannot keep full D in registers for arbitrary D, so we do a two-pass:
    // 1) compute m and l (online softmax stats) using dot products
    // 2) recompute p and accumulate Out in a streaming way (no shared out_acc)
    // This trades compute for memory reduction and avoids huge shared memory.
    for (int ks = 0; ks < S; ++ks) {
        const float* k_row = k_ptr + (int64_t)ks * D;

        float partial = 0.0f;
        // vector part
        const float4* q4 = reinterpret_cast<const float4*>(q_ptr);
        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        for (int i4 = tid; i4 < D4; i4 += THREADS) {
            float4 qv = q4[i4];
            float4 kv = k4[i4];
            partial = fmaf(qv.x, kv.x, partial);
            partial = fmaf(qv.y, kv.y, partial);
            partial = fmaf(qv.z, kv.z, partial);
            partial = fmaf(qv.w, kv.w, partial);
        }
        // remainder
        int start = D4 * 4 + tid;
        for (int d = start; d < D; d += THREADS) {
            partial = fmaf(q_ptr[d], k_row[d], partial);
        }

        float dot = block_sum<THREADS>(partial);
        float logit = dot * scale;

        float m_new = fmaxf(m, logit);
        float alpha = __expf(m - m_new);
        float p = __expf(logit - m_new);
        l = l * alpha + p;
        m = m_new;
    }

    // Now accumulate output: Out = sum_i softmax_i * V_i
    // streaming online with rescaling again
    // Initialize local out to 0
    // We'll write directly to out_ptr by first zeroing it.
    for (int d = tid; d < D; d += THREADS) out_ptr[d] = 0.0f;
    __syncthreads();

    float m2 = -INFINITY;
    float l2 = 0.0f;

    for (int ks = 0; ks < S; ++ks) {
        const float* k_row = k_ptr + (int64_t)ks * D;
        const float* v_row = v_ptr + (int64_t)ks * D;

        float partial = 0.0f;
        const float4* q4 = reinterpret_cast<const float4*>(q_ptr);
        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        for (int i4 = tid; i4 < D4; i4 += THREADS) {
            float4 qv = q4[i4];
            float4 kv = k4[i4];
            partial = fmaf(qv.x, kv.x, partial);
            partial = fmaf(qv.y, kv.y, partial);
            partial = fmaf(qv.z, kv.z, partial);
            partial = fmaf(qv.w, kv.w, partial);
        }
        int start = D4 * 4 + tid;
        for (int d = start; d < D; d += THREADS) {
            partial = fmaf(q_ptr[d], k_row[d], partial);
        }

        float dot = block_sum<THREADS>(partial);
        float logit = dot * scale;

        float m_new = fmaxf(m2, logit);
        float alpha = __expf(m2 - m_new);
        float p = __expf(logit - m_new);

        // out = out*alpha + p*v
        // vector part
        float4* out4 = reinterpret_cast<float4*>(out_ptr);
        const float4* v4 = reinterpret_cast<const float4*>(v_row);
        for (int i4 = tid; i4 < D4; i4 += THREADS) {
            float4 o = out4[i4];
            float4 vv = v4[i4];
            o.x = o.x * alpha + p * vv.x;
            o.y = o.y * alpha + p * vv.y;
            o.z = o.z * alpha + p * vv.z;
            o.w = o.w * alpha + p * vv.w;
            out4[i4] = o;
        }
        // remainder
        for (int d = start; d < D; d += THREADS) {
            out_ptr[d] = out_ptr[d] * alpha + p * v_row[d];
        }

        l2 = l2 * alpha + p;
        m2 = m_new;
        __syncthreads();
    }

    float inv_l = 1.0f / l2;
    for (int d = tid; d < D; d += THREADS) out_ptr[d] *= inv_l;
}

torch::Tensor sdpa_fwd_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "sdpa_fwd_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 &&
                K.scalar_type() == torch::kFloat32 &&
                V.scalar_type() == torch::kFloat32, "sdpa_fwd_cuda: only float32 supported");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "sdpa_fwd_cuda: inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "sdpa_fwd_cuda: expected 4D tensors [B,H,S,D]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D, "sdpa_fwd_cuda: K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D, "sdpa_fwd_cuda: V shape mismatch");

    auto Out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    // Launch configuration: one block per (qs, b*h)
    dim3 grid(S, B * H, 1);

    // Use 256 threads for D=1024 specialization (perfect mapping: 1 float4 per thread)
    // For generic, keep 256 for reasonable reduction granularity.
    constexpr int THREADS = 256;
    dim3 block(THREADS, 1, 1);

    if (D == 1024) {
        // Ensure alignment assumptions for float4 are reasonable: contiguous float tensors are at least 4-byte aligned;
        // float4 vectorization benefits even without guaranteed 16B alignment, but usually tensors are 256B aligned.
        sdpa_fwd_kernel_d1024<THREADS><<<grid, block>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, scale
        );
    } else {
        sdpa_fwd_kernel_generic<THREADS><<<grid, block>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, S, D, scale
        );
    }

    return Out;
}
"""

cpp_src = r"""
torch::Tensor sdpa_fwd_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_sdpa_opt2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sdpa_fwd_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_ops = custom_ops_lib

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise RuntimeError("ModelNew expects CUDA tensor inputs")
        if Q.dtype != torch.float32:
            Q = Q.float()
        if K.dtype != torch.float32:
            K = K.float()
        if V.dtype != torch.float32:
            V = V.float()
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()
        return self.custom_ops.sdpa_fwd_cuda(Q, K, V)


batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device="cuda", dtype=torch.float32)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device="cuda", dtype=torch.float32)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device="cuda", dtype=torch.float32)
    return [Q, K, V]

def get_init_inputs():
    return []