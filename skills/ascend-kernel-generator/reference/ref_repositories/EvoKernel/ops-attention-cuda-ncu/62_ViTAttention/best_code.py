import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ----------------------------
# Custom CUDA extension: fused attention core
# ----------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ float warp_broadcast(float v, int src_lane) {
    return __shfl_sync(0xffffffff, v, src_lane);
}

// One warp computes one output row (b,h,i) for head_dim=64.
// Each lane owns 2 output channels: d=lane and d=lane+32.
// Uses online softmax to avoid materializing attention.
template<bool UNROLL49>
__global__ __launch_bounds__(32, 8)
void vit_attn_fwd_d64_warp_kernel(
    const float* __restrict__ Q,    // [B,H,N,64] contiguous
    const float* __restrict__ K,    // [B,H,N,64] contiguous
    const float* __restrict__ V,    // [B,H,N,64] contiguous
    float* __restrict__ Out,        // [B,H,N,64] contiguous
    int B, int H, int N,
    float scale
) {
    int idx = (int)blockIdx.x;
    int i = idx % N;
    int tmp = idx / N;
    int h = tmp % H;
    int b = tmp / H;
    int bh = b * H + h;

    int lane = (int)threadIdx.x; // 0..31

    const float* __restrict__ q_row = Q + ((size_t)bh * N + i) * 64;
    const float* __restrict__ k_base = K + ((size_t)bh * N) * 64;
    const float* __restrict__ v_base = V + ((size_t)bh * N) * 64;
    float* __restrict__ out_row = Out + ((size_t)bh * N + i) * 64;

    float q0 = ro_load_f32(q_row + lane);
    float q1 = ro_load_f32(q_row + 32 + lane);

    float m = -INFINITY;
    float l = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;

    // loop over keys j
    int jmax = UNROLL49 ? 49 : N;
#pragma unroll
    for (int j = 0; j < (UNROLL49 ? 49 : 1); ++j) { /* dummy for unroll control */ }

    if (UNROLL49) {
#pragma unroll
        for (int j = 0; j < 49; ++j) {
            const float* __restrict__ k_row = k_base + (size_t)j * 64;
            const float* __restrict__ v_row = v_base + (size_t)j * 64;

            float p = q0 * ro_load_f32(k_row + lane) + q1 * ro_load_f32(k_row + 32 + lane);
            float dot = warp_reduce_sum(p);
            dot = warp_broadcast(dot, 0) * scale;

            float m_new = fmaxf(m, dot);
            float alpha = __expf(m - m_new);
            float beta  = __expf(dot - m_new);
            float l_new = l * alpha + beta;

            float vv0 = ro_load_f32(v_row + lane);
            float vv1 = ro_load_f32(v_row + 32 + lane);
            o0 = o0 * alpha + beta * vv0;
            o1 = o1 * alpha + beta * vv1;

            m = m_new;
            l = l_new;
        }
    } else {
#pragma unroll 1
        for (int j = 0; j < N; ++j) {
            const float* __restrict__ k_row = k_base + (size_t)j * 64;
            const float* __restrict__ v_row = v_base + (size_t)j * 64;

            float p = q0 * ro_load_f32(k_row + lane) + q1 * ro_load_f32(k_row + 32 + lane);
            float dot = warp_reduce_sum(p);
            dot = warp_broadcast(dot, 0) * scale;

            float m_new = fmaxf(m, dot);
            float alpha = __expf(m - m_new);
            float beta  = __expf(dot - m_new);
            float l_new = l * alpha + beta;

            float vv0 = ro_load_f32(v_row + lane);
            float vv1 = ro_load_f32(v_row + 32 + lane);
            o0 = o0 * alpha + beta * vv0;
            o1 = o1 * alpha + beta * vv1;

            m = m_new;
            l = l_new;
        }
    }

    float inv_l = 1.0f / (l + 1e-9f);
    out_row[lane] = o0 * inv_l;
    out_row[32 + lane] = o1 * inv_l;
}

torch::Tensor vi_t_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,Dh]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,N,Dh]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,N,Dh]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int Dh = (int)Q.size(3);

    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == N && K.size(3) == Dh, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == N && V.size(3) == Dh, "V shape mismatch");

    TORCH_CHECK(Dh == 64, "Optimized CUDA path supports head_dim==64 only (got Dh=", Dh, ")");
    TORCH_CHECK(N <= 64, "Optimized CUDA path supports N<=64 only (got N=", N, ")");

    auto Out = torch::empty_like(Q);

    int blocks = B * H * N;
    dim3 threads(32, 1, 1);

    if (N == 49) {
        vit_attn_fwd_d64_warp_kernel<true><<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, N,
            (float)scale
        );
    } else {
        vit_attn_fwd_d64_warp_kernel<false><<<blocks, threads>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H, N,
            (float)scale
        );
    }

    return Out;
}
"""

cpp_src = r"""
torch::Tensor vi_t_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vi_t_attention_d64_n64_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vi_t_attention_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


# ----------------------------
# Model using custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    ViT Attention with fused CUDA attention core (QK^T-softmax-AV) for inference.
    Fast path: CUDA fp32, contiguous, head_dim==64, N<=64, dropout disabled.
    Falls back to PyTorch reference otherwise.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_drop_p = float(attn_drop)
        self.proj_drop_p = float(proj_drop)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B,H,N,Dh]

        use_fast = (
            x.is_cuda and
            x.dtype == torch.float32 and
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
            self.attn_drop_p == 0.0 and self.proj_drop_p == 0.0 and
            self.head_dim == 64 and
            N <= 64
        )

        if use_fast:
            out = self.custom_ops_lib.vi_t_attention_cuda(q, k, v, float(self.scale))  # [B,H,N,Dh]
            x = out.transpose(1, 2).reshape(B, N, C).contiguous()
            x = self.proj(x)
            return x

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x