import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

static inline __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static inline __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// -------------------------------------------
// D==64 optimized kernel:
// - 2 queries per warp (i0, i1=i0+stride)
// - unroll keys by 2 (process j and j+stride per iter)
// - double-buffer K/V scalars
// - multi-warp blocks for occupancy (no inter-warp sync)
// -------------------------------------------
__global__ __launch_bounds__(128, 4)
void strided_attention_fwd_d64_qtile2_warp_unroll2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int S,
    int stride,
    float inv_sqrt_d
) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    long long linear = (long long)blockIdx.x * warps_per_block + warp_id;

    // Each warp mapped to (b,h,residue,tile) where tile steps by (2*stride) in i-space
    int residue = (int)(linear % stride);
    long long tmp = linear / stride;

    int tiles_per_res = (S - residue + (2 * stride) - 1) / (2 * stride); // ceil
    if (tiles_per_res <= 0) return;

    long long bh = tmp / tiles_per_res;
    int tile = (int)(tmp - bh * tiles_per_res);

    if (bh >= (long long)B * H) return;
    int b = (int)(bh / H);
    int h = (int)(bh - (long long)b * H);

    int i0 = residue + tile * (2 * stride);
    int i1 = i0 + stride;

    if (i0 >= S) return;
    bool has_i1 = (i1 < S);

    const float* q0_ptr = Q + (((b * H + h) * S + i0) * 64);
    const float* q1_ptr = has_i1 ? (Q + (((b * H + h) * S + i1) * 64)) : nullptr;

    // Each lane owns 2 dims (lane and lane+32)
    float q0_0 = ldg_f32(q0_ptr + lane);
    float q0_1 = ldg_f32(q0_ptr + lane + 32);

    float q1_0 = 0.f, q1_1 = 0.f;
    if (has_i1) {
        q1_0 = ldg_f32(q1_ptr + lane);
        q1_1 = ldg_f32(q1_ptr + lane + 32);
    }

    // Online softmax state for two queries
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;

    float acc0_0 = 0.f, acc0_1 = 0.f;
    float acc1_0 = 0.f, acc1_1 = 0.f;

    // If residue >= S, only self exists but residue is i%stride so residue < stride;
    // still handle stride > S:
    if (residue >= S) {
        float* o0_ptr = O + (((b * H + h) * S + i0) * 64);
        o0_ptr[lane] = 0.f; o0_ptr[lane + 32] = 0.f;
        if (has_i1) {
            float* o1_ptr = O + (((b * H + h) * S + i1) * 64);
            o1_ptr[lane] = 0.f; o1_ptr[lane + 32] = 0.f;
        }
        return;
    }

    // Initialize double buffer for first key
    int j = residue;

    const float* k_ptr0 = K + (((b * H + h) * S + j) * 64);
    const float* v_ptr0 = V + (((b * H + h) * S + j) * 64);
    float k0_0 = ldg_f32(k_ptr0 + lane);
    float k0_1 = ldg_f32(k_ptr0 + lane + 32);
    float v0_0 = ldg_f32(v_ptr0 + lane);
    float v0_1 = ldg_f32(v_ptr0 + lane + 32);

    // Prefetch second (j+stride) if exists
    int j1 = j + stride;
    float k1_0 = 0.f, k1_1 = 0.f, v1_0 = 0.f, v1_1 = 0.f;
    bool has_j1 = (j1 < S);
    if (has_j1) {
        const float* k_ptr1 = K + (((b * H + h) * S + j1) * 64);
        const float* v_ptr1 = V + (((b * H + h) * S + j1) * 64);
        k1_0 = ldg_f32(k_ptr1 + lane);
        k1_1 = ldg_f32(k_ptr1 + lane + 32);
        v1_0 = ldg_f32(v_ptr1 + lane);
        v1_1 = ldg_f32(v_ptr1 + lane + 32);
    }

    // Process two keys per loop iteration: j and j+stride.
    for (; j < S; j += 2 * stride) {
        // ---- key at j (buffer 0) ----
        float partial0 = q0_0 * k0_0 + q0_1 * k0_1;
        float dot0 = warp_reduce_sum(partial0);
        dot0 = __shfl_sync(0xffffffff, dot0, 0) * inv_sqrt_d;

        float m0_new = fmaxf(m0, dot0);
        float a0 = __expf(m0 - m0_new);
        float bw0 = __expf(dot0 - m0_new);
        m0 = m0_new;
        l0 = l0 * a0 + bw0;
        acc0_0 = acc0_0 * a0 + v0_0 * bw0;
        acc0_1 = acc0_1 * a0 + v0_1 * bw0;

        if (has_i1) {
            float partial1 = q1_0 * k0_0 + q1_1 * k0_1;
            float dot1 = warp_reduce_sum(partial1);
            dot1 = __shfl_sync(0xffffffff, dot1, 0) * inv_sqrt_d;

            float m1_new = fmaxf(m1, dot1);
            float a1 = __expf(m1 - m1_new);
            float bw1 = __expf(dot1 - m1_new);
            m1 = m1_new;
            l1 = l1 * a1 + bw1;
            acc1_0 = acc1_0 * a1 + v0_0 * bw1;
            acc1_1 = acc1_1 * a1 + v0_1 * bw1;
        }

        // ---- key at j+stride (buffer 1), if exists ----
        if (has_j1) {
            float partial0b = q0_0 * k1_0 + q0_1 * k1_1;
            float dot0b = warp_reduce_sum(partial0b);
            dot0b = __shfl_sync(0xffffffff, dot0b, 0) * inv_sqrt_d;

            float m0_newb = fmaxf(m0, dot0b);
            float a0b = __expf(m0 - m0_newb);
            float bw0b = __expf(dot0b - m0_newb);
            m0 = m0_newb;
            l0 = l0 * a0b + bw0b;
            acc0_0 = acc0_0 * a0b + v1_0 * bw0b;
            acc0_1 = acc0_1 * a0b + v1_1 * bw0b;

            if (has_i1) {
                float partial1b = q1_0 * k1_0 + q1_1 * k1_1;
                float dot1b = warp_reduce_sum(partial1b);
                dot1b = __shfl_sync(0xffffffff, dot1b, 0) * inv_sqrt_d;

                float m1_newb = fmaxf(m1, dot1b);
                float a1b = __expf(m1 - m1_newb);
                float bw1b = __expf(dot1b - m1_newb);
                m1 = m1_newb;
                l1 = l1 * a1b + bw1b;
                acc1_0 = acc1_0 * a1b + v1_0 * bw1b;
                acc1_1 = acc1_1 * a1b + v1_1 * bw1b;
            }
        }

        // Prefetch next pair: (j+2*stride) and (j+3*stride)
        int j_next0 = j + 2 * stride;
        int j_next1 = j + 3 * stride;

        bool has_next0 = (j_next0 < S);
        bool has_next1 = (j_next1 < S);

        float nk0_0 = 0.f, nk0_1 = 0.f, nv0_0 = 0.f, nv0_1 = 0.f;
        float nk1_0 = 0.f, nk1_1 = 0.f, nv1_0 = 0.f, nv1_1 = 0.f;

        if (has_next0) {
            const float* kp = K + (((b * H + h) * S + j_next0) * 64);
            const float* vp = V + (((b * H + h) * S + j_next0) * 64);
            nk0_0 = ldg_f32(kp + lane);
            nk0_1 = ldg_f32(kp + lane + 32);
            nv0_0 = ldg_f32(vp + lane);
            nv0_1 = ldg_f32(vp + lane + 32);
        }
        if (has_next1) {
            const float* kp = K + (((b * H + h) * S + j_next1) * 64);
            const float* vp = V + (((b * H + h) * S + j_next1) * 64);
            nk1_0 = ldg_f32(kp + lane);
            nk1_1 = ldg_f32(kp + lane + 32);
            nv1_0 = ldg_f32(vp + lane);
            nv1_1 = ldg_f32(vp + lane + 32);
        }

        // Rotate buffers
        k0_0 = nk0_0; k0_1 = nk0_1; v0_0 = nv0_0; v0_1 = nv0_1;
        k1_0 = nk1_0; k1_1 = nk1_1; v1_0 = nv1_0; v1_1 = nv1_1;
        has_j1 = has_next1;
    }

    float invl0 = 1.0f / fmaxf(l0, 1e-20f);
    float* o0_ptr = O + (((b * H + h) * S + i0) * 64);
    o0_ptr[lane] = acc0_0 * invl0;
    o0_ptr[lane + 32] = acc0_1 * invl0;

    if (has_i1) {
        float invl1 = 1.0f / fmaxf(l1, 1e-20f);
        float* o1_ptr = O + (((b * H + h) * S + i1) * 64);
        o1_ptr[lane] = acc1_0 * invl1;
        o1_ptr[lane + 32] = acc1_1 * invl1;
    }
}

// --------------------
// Generic fallback: any D
// - 4 warps/block for better occupancy
// --------------------
__global__ __launch_bounds__(128, 4)
void strided_attention_fwd_generic_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int S, int D,
    int stride,
    float inv_sqrt_d
) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    long long q_linear = (long long)blockIdx.x * warps_per_block + warp_id;
    long long total_q = (long long)B * H * S;
    if (q_linear >= total_q) return;

    int i = (int)(q_linear % S);
    long long tmp = q_linear / S;
    int h = (int)(tmp % H);
    int b = (int)(tmp / H);

    int residue = i % stride;
    const float* q_ptr = Q + (((b * H + h) * S + i) * D);

    float m = -INFINITY;
    float l = 0.0f;

    // up to D<=256 typical; keep small fixed acc stripe (8*32=256)
    float acc[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) acc[u] = 0.0f;

    for (int j = residue; j < S; j += stride) {
        const float* k_ptr = K + (((b * H + h) * S + j) * D);
        const float* v_ptr = V + (((b * H + h) * S + j) * D);

        float partial = 0.0f;
        for (int d = lane; d < D; d += WARP) {
            partial += ldg_f32(q_ptr + d) * ldg_f32(k_ptr + d);
        }

        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0);
        float x = dot * inv_sqrt_d;

        float m_new = fmaxf(m, x);
        float a = __expf(m - m_new);
        float bw = __expf(x - m_new);
        m = m_new;
        l = l * a + bw;

        int base = lane;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            int d = base + u * WARP;
            if (d < D) {
                float vv = ldg_f32(v_ptr + d);
                acc[u] = acc[u] * a + vv * bw;
            }
        }
    }

    float inv_l = 1.0f / fmaxf(l, 1e-20f);
    float* o_ptr = O + (((b * H + h) * S + i) * D);

    int base = lane;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int d = base + u * WARP;
        if (d < D) o_ptr[d] = acc[u] * inv_l;
    }
}

torch::Tensor strided_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int64_t stride
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q,K,V must be [B,H,S,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q,K,V must have same shape");
    TORCH_CHECK(stride > 0, "stride must be > 0");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    auto O = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    if (D == 64) {
        // warps: B*H*stride*tiles_per_res
        int tiles_per_res_max = (S + (2 * (int)stride) - 1) / (2 * (int)stride);
        long long warps_total = (long long)B * H * (long long)stride * (long long)tiles_per_res_max;

        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (int)((warps_total + warps_per_block - 1) / warps_per_block);

        strided_attention_fwd_d64_qtile2_warp_unroll2_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, S,
            (int)stride,
            inv_sqrt_d
        );
    } else {
        long long total_q = (long long)B * H * S;
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (int)((total_q + warps_per_block - 1) / warps_per_block);

        strided_attention_fwd_generic_kernel<<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)O.data_ptr<float>(),
            B, H, S, D,
            (int)stride,
            inv_sqrt_d
        );
    }

    return O;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor strided_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t stride);
"""

custom_ops_lib = load_inline(
    name="custom_strided_attention_ops_v7",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["strided_attention_forward_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Strided (Dilated) Attention mechanism with a custom fused CUDA operator
    replacing masked QK^T + softmax + AV (forward only).
    """

    def __init__(self, d_model, n_heads, stride):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.stride = stride
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.0)
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        B, S, _ = x.size()

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        O = self.custom_ops.strided_attention_forward_cuda(Q, K, V, int(self.stride))

        O = O.transpose(1, 2).contiguous().view(B, S, self.d_model)
        O = self.W_o(O)
        return O