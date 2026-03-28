import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA: fused scaled-dot-product attention forward (inference-oriented)
# Optimized fast path for float32, contiguous, Q/K/V [B,H,S,D] with D=64, S=512.
#
# v3 improvements over baseline:
# - q-tiling increased to 4 queries per warp (reuse streamed K/V more)
# - 2 warps per block (64 threads) to improve SM residency/latency hiding
# - warp-private work: no shared memory tiles, no __syncthreads()
# - vectorized float4 loads for V and output stores
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

struct __align__(16) float4a { float x, y, z, w; };

__device__ __forceinline__ float4a ld_float4(const float* p) {
    return *reinterpret_cast<const float4a*>(p);
}
__device__ __forceinline__ void st_float4(float* p, const float4a& v) {
    *reinterpret_cast<float4a*>(p) = v;
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Each warp computes 4 query rows: i0=4*qgroup + {0,1,2,3}
// grid.x = B*H*(S/4) items; block has 64 threads = 2 warps; each warp handles one item.
__global__ void sdpa_fwd_d64_s512_warp_q4_2warps(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H,
    float inv_sqrt_d
) {
    constexpr int S = 512;
    constexpr int D = 64;
    constexpr int VEC = 4;
    constexpr int D4 = D / VEC; // 16

    int warp_in_block = (int)(threadIdx.x >> 5); // 0..1
    int lane = (int)(threadIdx.x & 31);
    int item = (int)blockIdx.x * 2 + warp_in_block; // one item per warp

    int qgroup = item % (S / 4);
    int tmp = item / (S / 4);
    int h = tmp % H;
    int b = tmp / H;
    int bh = b * H + h;

    int i0 = qgroup * 4;
    int i1 = i0 + 1;
    int i2 = i0 + 2;
    int i3 = i0 + 3;

    // Load Q for 4 rows into registers:
    // Each lane loads 2 floats per row; 32 lanes cover 64 dims.
    int d = lane * 2; // 0..62
    const float* q0_ptr = Q + ((bh * S + i0) * D);
    const float* q1_ptr = Q + ((bh * S + i1) * D);
    const float* q2_ptr = Q + ((bh * S + i2) * D);
    const float* q3_ptr = Q + ((bh * S + i3) * D);

    float q0a = ldg_f32(q0_ptr + d + 0);
    float q0b = ldg_f32(q0_ptr + d + 1);
    float q1a = ldg_f32(q1_ptr + d + 0);
    float q1b = ldg_f32(q1_ptr + d + 1);
    float q2a = ldg_f32(q2_ptr + d + 0);
    float q2b = ldg_f32(q2_ptr + d + 1);
    float q3a = ldg_f32(q3_ptr + d + 0);
    float q3b = ldg_f32(q3_ptr + d + 1);

    // Online softmax state for 4 rows
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;
    float m2 = -INFINITY, l2 = 0.f;
    float m3 = -INFINITY, l3 = 0.f;

    // Output accumulators: lanes 0..15 each own one float4 (64 dims)
    float4a out0 = {0.f, 0.f, 0.f, 0.f};
    float4a out1 = {0.f, 0.f, 0.f, 0.f};
    float4a out2 = {0.f, 0.f, 0.f, 0.f};
    float4a out3 = {0.f, 0.f, 0.f, 0.f};
    int d4 = lane; // 0..31; only <16 used

    // stream over keys
    #pragma unroll 1
    for (int j = 0; j < S; j++) {
        const float* k_ptr = K + ((bh * S + j) * D);

        // Prefetch K scalars for this lane
        float k0 = ldg_f32(k_ptr + d + 0);
        float k1 = ldg_f32(k_ptr + d + 1);

        float part0 = fmaf(q0a, k0, q0b * k1);
        float part1 = fmaf(q1a, k0, q1b * k1);
        float part2 = fmaf(q2a, k0, q2b * k1);
        float part3 = fmaf(q3a, k0, q3b * k1);

        float dot0 = warp_reduce_sum(part0);
        float dot1 = warp_reduce_sum(part1);
        float dot2 = warp_reduce_sum(part2);
        float dot3 = warp_reduce_sum(part3);

        dot0 = __shfl_sync(0xffffffff, dot0, 0);
        dot1 = __shfl_sync(0xffffffff, dot1, 0);
        dot2 = __shfl_sync(0xffffffff, dot2, 0);
        dot3 = __shfl_sync(0xffffffff, dot3, 0);

        float s0 = dot0 * inv_sqrt_d;
        float s1 = dot1 * inv_sqrt_d;
        float s2 = dot2 * inv_sqrt_d;
        float s3 = dot3 * inv_sqrt_d;

        // Update online softmax for 4 rows
        float m0_new = fmaxf(m0, s0);
        float a0 = __expf(m0 - m0_new);
        float p0 = __expf(s0 - m0_new);
        l0 = l0 * a0 + p0;
        m0 = m0_new;

        float m1_new = fmaxf(m1, s1);
        float a1 = __expf(m1 - m1_new);
        float p1 = __expf(s1 - m1_new);
        l1 = l1 * a1 + p1;
        m1 = m1_new;

        float m2_new = fmaxf(m2, s2);
        float a2 = __expf(m2 - m2_new);
        float p2 = __expf(s2 - m2_new);
        l2 = l2 * a2 + p2;
        m2 = m2_new;

        float m3_new = fmaxf(m3, s3);
        float a3 = __expf(m3 - m3_new);
        float p3 = __expf(s3 - m3_new);
        l3 = l3 * a3 + p3;
        m3 = m3_new;

        if (d4 < D4) {
            const float* v_ptr = V + ((bh * S + j) * D) + d4 * VEC;
            float4a v4 = ld_float4(v_ptr);

            // scale accumulators by a*
            out0.x *= a0; out0.y *= a0; out0.z *= a0; out0.w *= a0;
            out1.x *= a1; out1.y *= a1; out1.z *= a1; out1.w *= a1;
            out2.x *= a2; out2.y *= a2; out2.z *= a2; out2.w *= a2;
            out3.x *= a3; out3.y *= a3; out3.z *= a3; out3.w *= a3;

            // fma with p*V
            out0.x = fmaf(p0, v4.x, out0.x);
            out0.y = fmaf(p0, v4.y, out0.y);
            out0.z = fmaf(p0, v4.z, out0.z);
            out0.w = fmaf(p0, v4.w, out0.w);

            out1.x = fmaf(p1, v4.x, out1.x);
            out1.y = fmaf(p1, v4.y, out1.y);
            out1.z = fmaf(p1, v4.z, out1.z);
            out1.w = fmaf(p1, v4.w, out1.w);

            out2.x = fmaf(p2, v4.x, out2.x);
            out2.y = fmaf(p2, v4.y, out2.y);
            out2.z = fmaf(p2, v4.z, out2.z);
            out2.w = fmaf(p2, v4.w, out2.w);

            out3.x = fmaf(p3, v4.x, out3.x);
            out3.y = fmaf(p3, v4.y, out3.y);
            out3.z = fmaf(p3, v4.z, out3.z);
            out3.w = fmaf(p3, v4.w, out3.w);
        }
    }

    float inv_l0 = 1.f / (l0 + 1e-9f);
    float inv_l1 = 1.f / (l1 + 1e-9f);
    float inv_l2 = 1.f / (l2 + 1e-9f);
    float inv_l3 = 1.f / (l3 + 1e-9f);

    if (d4 < D4) {
        out0.x *= inv_l0; out0.y *= inv_l0; out0.z *= inv_l0; out0.w *= inv_l0;
        out1.x *= inv_l1; out1.y *= inv_l1; out1.z *= inv_l1; out1.w *= inv_l1;
        out2.x *= inv_l2; out2.y *= inv_l2; out2.z *= inv_l2; out2.w *= inv_l2;
        out3.x *= inv_l3; out3.y *= inv_l3; out3.z *= inv_l3; out3.w *= inv_l3;

        float* o0_ptr = Out + ((bh * S + i0) * D) + d4 * VEC;
        float* o1_ptr = Out + ((bh * S + i1) * D) + d4 * VEC;
        float* o2_ptr = Out + ((bh * S + i2) * D) + d4 * VEC;
        float* o3_ptr = Out + ((bh * S + i3) * D) + d4 * VEC;

        st_float4(o0_ptr, out0);
        st_float4(o1_ptr, out1);
        st_float4(o2_ptr, out2);
        st_float4(o3_ptr, out3);
    }
}

torch::Tensor sdpa_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B, H, S, D]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(D == 64, "Only D=64 supported by optimized kernel");
    TORCH_CHECK(S == 512, "Only S=512 supported by optimized kernel");
    TORCH_CHECK((S % 4) == 0, "S must be divisible by 4 for q-tiling=4");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D, "V shape mismatch");

    auto Out = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // items = B*H*(S/4); each block has 2 warps => grid has ceil(items/2) blocks
    int items = B * H * (S / 4);
    int blocks = (items + 1) / 2;
    int threads = 64;

    sdpa_fwd_d64_s512_warp_q4_2warps<<<blocks, threads>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        B, H, inv_sqrt_d
    );

    return Out;
}
"""

cpp_src = r"""
torch::Tensor sdpa_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_mha_sdpa_s512_d64_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sdpa_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
)

# -----------------------------------------------------------------------------
# ModelNew: replaces attention subgraph with custom CUDA SDPA.
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_k = self.d_model // self.n_heads
        self.dropout_p = float(dropout)

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(key).view(batch_size, key.size(1), self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(value).view(batch_size, value.size(1), self.n_heads, self.d_k).transpose(1, 2).contiguous()

        use_fast = (
            (not self.training) and self.dropout_p == 0.0 and
            Q.is_cuda and K.is_cuda and V.is_cuda and
            Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32 and
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and
            Q.dim() == 4 and K.dim() == 4 and V.dim() == 4 and
            self.d_k == 64 and seq_len == 512 and
            (seq_len % 4) == 0 and
            Q.size(0) == batch_size and Q.size(1) == self.n_heads and Q.size(2) == seq_len and Q.size(3) == self.d_k
        )

        if use_fast:
            attn_output = custom_ops_lib.sdpa_forward_cuda(Q, K, V)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout_p != 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)
            attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(attn_output)
        out = self.dropout(out)
        return out