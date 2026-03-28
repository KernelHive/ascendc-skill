import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from operator import itemgetter

# -------------------------
# Helper functions (same behavior as original)
# -------------------------

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations

# -------------------------
# Custom CUDA extension
#   Fast paths:
#     - E==64 and T==7: warp, q-tile=4, smem-stage K/V per key (float4)
#     - E==64 and T<=64: warp, q-tile=4, smem-stage K/V per key (float4)
#   Fallback:
#     - generic 3-pass warp-per-row kernel
# Forward-only.
# -------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {                           \
  cudaError_t err = cudaGetLastError();                               \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err)); \
} while (0)
#endif

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}
__device__ __forceinline__ float4 ldg_f4(const float4* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Fast path: E=64, qtile=4. Each block=1 warp handles 4 query rows for one bh:
// i0=quad*4 + {0,1,2,3}. K/V staged into smem each key step via float4.
template<int T_FIXED>
__global__ __launch_bounds__(32, 4) void fused_axial_attn_fwd_warp64_qtile4_smem_kv_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int BH, int T_runtime,
    float scale
) {
    int lane = threadIdx.x & 31;

    const int T = (T_FIXED > 0) ? T_FIXED : T_runtime;
    int quad_per_bh = (T + 3) >> 2; // ceil(T/4)

    int quad_row = (int)blockIdx.x; // ranges [0, BH*quad_per_bh)
    int bh = quad_row / quad_per_bh;
    int quad = quad_row - bh * quad_per_bh;
    if (bh >= BH) return;

    int i0 = quad * 4 + 0;
    int i1 = quad * 4 + 1;
    int i2 = quad * 4 + 2;
    int i3 = quad * 4 + 3;

    bool v0q = (i0 < T);
    bool v1q = (i1 < T);
    bool v2q = (i2 < T);
    bool v3q = (i3 < T);

    const float* __restrict__ k_base = K + (bh * T * 64);
    const float* __restrict__ v_base = V + (bh * T * 64);

    const float* __restrict__ q0_ptr = Q + ((bh * T + i0) * 64);
    const float* __restrict__ q1_ptr = Q + ((bh * T + i1) * 64);
    const float* __restrict__ q2_ptr = Q + ((bh * T + i2) * 64);
    const float* __restrict__ q3_ptr = Q + ((bh * T + i3) * 64);

    float* __restrict__ o0_ptr = Out + ((bh * T + i0) * 64);
    float* __restrict__ o1_ptr = Out + ((bh * T + i1) * 64);
    float* __restrict__ o2_ptr = Out + ((bh * T + i2) * 64);
    float* __restrict__ o3_ptr = Out + ((bh * T + i3) * 64);

    // Q in regs: each lane owns dims lane and lane+32
    float q00=0.f,q01=0.f,q10=0.f,q11=0.f,q20=0.f,q21=0.f,q30=0.f,q31=0.f;
    if (v0q) { q00 = ldg_f32(q0_ptr + lane); q01 = ldg_f32(q0_ptr + lane + 32); }
    if (v1q) { q10 = ldg_f32(q1_ptr + lane); q11 = ldg_f32(q1_ptr + lane + 32); }
    if (v2q) { q20 = ldg_f32(q2_ptr + lane); q21 = ldg_f32(q2_ptr + lane + 32); }
    if (v3q) { q30 = ldg_f32(q3_ptr + lane); q31 = ldg_f32(q3_ptr + lane + 32); }

    // online softmax states and output accumulators per query
    float m0=-INFINITY,l0=0.f,a00=0.f,a01=0.f;
    float m1=-INFINITY,l1=0.f,a10=0.f,a11=0.f;
    float m2=-INFINITY,l2=0.f,a20=0.f,a21=0.f;
    float m3=-INFINITY,l3=0.f,a30=0.f,a31=0.f;

    extern __shared__ float smem[];
    float* smK = smem;        // [64]
    float* smV = smem + 64;   // [64]

    bool kv_vec4_ok = ((((uintptr_t)k_base | (uintptr_t)v_base) & 0xF) == 0);

    #pragma unroll 1
    for (int j = 0; j < T; ++j) {
        const float* k_ptr = k_base + j * 64;
        const float* v_ptr = v_base + j * 64;

        if (kv_vec4_ok) {
            if (lane < 16) {
                const float4* k4 = (const float4*)k_ptr;
                const float4* v4 = (const float4*)v_ptr;
                float4 kk = ldg_f4(k4 + lane);
                float4 vv = ldg_f4(v4 + lane);
                ((float4*)smK)[lane] = kk;
                ((float4*)smV)[lane] = vv;
            }
        } else {
            smK[lane]      = ldg_f32(k_ptr + lane);
            smK[lane + 32] = ldg_f32(k_ptr + lane + 32);
            smV[lane]      = ldg_f32(v_ptr + lane);
            smV[lane + 32] = ldg_f32(v_ptr + lane + 32);
        }

        __syncwarp();

        float k0 = smK[lane];
        float k1 = smK[lane + 32];
        float vv0 = smV[lane];
        float vv1 = smV[lane + 32];

        float pd0 = v0q ? fmaf(q00, k0, q01 * k1) : 0.f;
        float pd1 = v1q ? fmaf(q10, k0, q11 * k1) : 0.f;
        float pd2 = v2q ? fmaf(q20, k0, q21 * k1) : 0.f;
        float pd3 = v3q ? fmaf(q30, k0, q31 * k1) : 0.f;

        float dot0 = warp_reduce_sum(pd0);
        float dot1 = warp_reduce_sum(pd1);
        float dot2 = warp_reduce_sum(pd2);
        float dot3 = warp_reduce_sum(pd3);

        dot0 = __shfl_sync(0xffffffff, dot0, 0);
        dot1 = __shfl_sync(0xffffffff, dot1, 0);
        dot2 = __shfl_sync(0xffffffff, dot2, 0);
        dot3 = __shfl_sync(0xffffffff, dot3, 0);

        float x0 = dot0 * scale;
        float x1 = dot1 * scale;
        float x2 = dot2 * scale;
        float x3 = dot3 * scale;

        if (v0q) {
            float m_new = fmaxf(m0, x0);
            float alpha = (m0 == -INFINITY) ? 0.f : __expf(m0 - m_new);
            float beta  = __expf(x0 - m_new);
            l0  = l0 * alpha + beta;
            a00 = a00 * alpha + beta * vv0;
            a01 = a01 * alpha + beta * vv1;
            m0 = m_new;
        }
        if (v1q) {
            float m_new = fmaxf(m1, x1);
            float alpha = (m1 == -INFINITY) ? 0.f : __expf(m1 - m_new);
            float beta  = __expf(x1 - m_new);
            l1  = l1 * alpha + beta;
            a10 = a10 * alpha + beta * vv0;
            a11 = a11 * alpha + beta * vv1;
            m1 = m_new;
        }
        if (v2q) {
            float m_new = fmaxf(m2, x2);
            float alpha = (m2 == -INFINITY) ? 0.f : __expf(m2 - m_new);
            float beta  = __expf(x2 - m_new);
            l2  = l2 * alpha + beta;
            a20 = a20 * alpha + beta * vv0;
            a21 = a21 * alpha + beta * vv1;
            m2 = m_new;
        }
        if (v3q) {
            float m_new = fmaxf(m3, x3);
            float alpha = (m3 == -INFINITY) ? 0.f : __expf(m3 - m_new);
            float beta  = __expf(x3 - m_new);
            l3  = l3 * alpha + beta;
            a30 = a30 * alpha + beta * vv0;
            a31 = a31 * alpha + beta * vv1;
            m3 = m_new;
        }

        __syncwarp();
    }

    if (v0q) {
        float inv = 1.0f / (l0 + 1e-9f);
        o0_ptr[lane]      = a00 * inv;
        o0_ptr[lane + 32] = a01 * inv;
    }
    if (v1q) {
        float inv = 1.0f / (l1 + 1e-9f);
        o1_ptr[lane]      = a10 * inv;
        o1_ptr[lane + 32] = a11 * inv;
    }
    if (v2q) {
        float inv = 1.0f / (l2 + 1e-9f);
        o2_ptr[lane]      = a20 * inv;
        o2_ptr[lane + 32] = a21 * inv;
    }
    if (v3q) {
        float inv = 1.0f / (l3 + 1e-9f);
        o3_ptr[lane]      = a30 * inv;
        o3_ptr[lane + 32] = a31 * inv;
    }
}

// Generic fallback kernel (same as baseline)
__global__ void fused_axial_attn_fwd_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int BH, int T, int E,
    float scale
) {
    int warp_id_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x / 32;

    int row = (int)blockIdx.x * warps_per_block + warp_id_in_block;
    int total_rows = BH * T;
    if (row >= total_rows) return;

    int i = row % T;
    int bh = row / T;

    const float* q_ptr = Q + ((bh * T + i) * E);
    const float* k_base = K + (bh * T * E);
    const float* v_base = V + (bh * T * E);
    float* out_ptr = Out + ((bh * T + i) * E);

    float local_max = -INFINITY;
    for (int j = lane; j < T; j += 32) {
        const float* k_ptr = k_base + j * E;
        float dot = 0.f;
        #pragma unroll 1
        for (int e = 0; e < E; ++e) dot += q_ptr[e] * k_ptr[e];
        float logit = dot * scale;
        local_max = fmaxf(local_max, logit);
    }
    float max_logit = warp_reduce_max(local_max);
    max_logit = __shfl_sync(0xffffffff, max_logit, 0);

    float local_sum = 0.f;
    for (int j = lane; j < T; j += 32) {
        const float* k_ptr = k_base + j * E;
        float dot = 0.f;
        #pragma unroll 1
        for (int e = 0; e < E; ++e) dot += q_ptr[e] * k_ptr[e];
        float logit = dot * scale;
        local_sum += __expf(logit - max_logit);
    }
    float sumexp = warp_reduce_sum(local_sum);
    sumexp = __shfl_sync(0xffffffff, sumexp, 0);
    float inv_denom = 1.0f / (sumexp + 1e-9f);

    for (int e = lane; e < E; e += 32) {
        float acc = 0.f;
        for (int j = 0; j < T; ++j) {
            float w = 0.f;
            if (lane == 0) {
                const float* k_ptr = k_base + j * E;
                float dot = 0.f;
                #pragma unroll 1
                for (int ee = 0; ee < E; ++ee) dot += q_ptr[ee] * k_ptr[ee];
                float logit = dot * scale;
                w = __expf(logit - max_logit) * inv_denom;
            }
            w = __shfl_sync(0xffffffff, w, 0);
            const float* v_ptr = v_base + j * E;
            acc += w * v_ptr[e];
        }
        out_ptr[e] = acc;
    }
}

torch::Tensor fused_axial_attn_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 3, "Q must be [BH, T, E]");
    TORCH_CHECK(K.dim() == 3, "K must be [BH, T, E]");
    TORCH_CHECK(V.dim() == 3, "V must be [BH, T, E]");

    int64_t BH = Q.size(0);
    int64_t T  = Q.size(1);
    int64_t E  = Q.size(2);

    TORCH_CHECK(K.size(0) == BH && K.size(1) == T && K.size(2) == E, "K shape mismatch");
    TORCH_CHECK(V.size(0) == BH && V.size(1) == T && V.size(2) == E, "V shape mismatch");

    auto Out = torch::empty({BH, T, E}, Q.options());
    float scale = 1.0f / sqrtf((float)E);

    if (E == 64 && T == 7) {
        int quad_per_bh = (7 + 3) >> 2; // 2
        int blocks = (int)BH * quad_per_bh;
        dim3 threads(32);
        size_t shmem = (size_t)(128 * sizeof(float)); // K(64) + V(64)
        fused_axial_attn_fwd_warp64_qtile4_smem_kv_kernel<7><<<blocks, threads, shmem>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            (int)BH, (int)T, scale
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return Out;
    }

    if (E == 64 && T <= 64) {
        int quad_per_bh = (((int)T) + 3) >> 2;
        int blocks = (int)BH * quad_per_bh;
        dim3 threads(32);
        size_t shmem = (size_t)(128 * sizeof(float));
        fused_axial_attn_fwd_warp64_qtile4_smem_kv_kernel<-1><<<blocks, threads, shmem>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            (int)BH, (int)T, scale
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return Out;
    }

    int total_rows = (int)(BH * T);
    int threads = 256; // 8 warps
    int warps_per_block = threads / 32;
    int blocks = (total_rows + warps_per_block - 1) / warps_per_block;

    fused_axial_attn_fwd_kernel_generic<<<blocks, threads>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        (int)BH, (int)T, (int)E,
        scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Out;
}
"""

cpp_src = r"""
torch::Tensor fused_axial_attn_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_axial_attention_opt_qtile4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_axial_attn_forward_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# -------------------------
# Modules: fused SelfAttention + PermuteToFrom + ModelNew
# -------------------------

class SelfAttentionFused(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)
        self.custom_ops = custom_ops_lib

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q = self.to_q(x)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        b, t, _ = q.shape
        h, e = self.heads, self.dim_heads

        def merge_heads(z):
            return (
                z.reshape(b, t, h, e)
                 .transpose(1, 2)
                 .contiguous()
                 .reshape(b * h, t, e)
                 .contiguous()
            )

        q = merge_heads(q)
        k = merge_heads(k)
        v = merge_heads(v)

        out = self.custom_ops.fused_axial_attn_forward_cuda(q, k, v)  # [b*h, t, e]
        out = out.reshape(b, h, t, e).transpose(1, 2).contiguous().reshape(b, t, h * e)
        out = self.to_out(out)
        return out


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class ModelNew(nn.Module):
    """
    Drop-in replacement for the provided Axial Attention Model.
    Uses a q-tiled (4 queries/warp) online-softmax fused CUDA attention core for E=64, T<=64,
    with a dedicated T=7 specialization (common for 7x7 axial attention).
    Forward-only.
    """
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, "hidden dimension must be divisible by number of heads"
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, self.dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttentionFused(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, "input tensor does not have the correct number of dimensions"
        assert x.shape[self.dim_index] == self.dim, "input tensor does not have the correct input dimension"

        if self.sum_axial_out:
            out = None
            for axial_attn in self.axial_attentions:
                y = axial_attn(x)
                out = y if out is None else (out + y)
            return out

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out