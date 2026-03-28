import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA: fused scaled-dot-product attention forward (inference-oriented)
# Fast path for float32, contiguous, Q/K/V [B,H,S,D] with D=64, S=512.
# Warp-level online softmax and accumulation in one pass.
#
# v2 improvements over baseline:
# - Increase q-tiling per warp from 4 -> 8 queries to amortize K/V streaming.
# - Lightweight software pipelining: prefetch next K (2 scalars/lane).
# - Keep vectorized float4 accumulation for V/Out when 16B-aligned (guarded).
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

struct __align__(16) float4a { float x, y, z, w; };

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float4a ld_float4a(const float* p) {
    return *reinterpret_cast<const float4a*>(p);
}
__device__ __forceinline__ void st_float4a(float* p, const float4a& v) {
    *reinterpret_cast<float4a*>(p) = v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Each warp computes 8 query rows (q-tiling=8).
// Block has 64 threads = 2 warps; each warp handles one (b,h,qgroup) item.
// Q/K dot uses scalar loads (2 floats per lane).
// V/Out use float4 vectorization (lanes 0..15) when aligned, else scalar (all lanes).
__global__ void sdpa_fwd_d64_s512_warp_q8_2warps(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H,
    float inv_sqrt_d,
    int vec4_ok
) {
    constexpr int S = 512;
    constexpr int D = 64;
    constexpr int VEC = 4;
    constexpr int D4 = D / VEC; // 16

    int warp_in_block = (int)(threadIdx.x >> 5); // 0..1
    int lane = (int)(threadIdx.x & 31);
    int item = (int)blockIdx.x * 2 + warp_in_block;

    int qgroup = item % (S / 8);
    int tmp = item / (S / 8);
    int h = tmp % H;
    int b = tmp / H;
    int bh = b * H + h;

    int i0 = qgroup * 8;
    int i1 = i0 + 1;
    int i2 = i0 + 2;
    int i3 = i0 + 3;
    int i4 = i0 + 4;
    int i5 = i0 + 5;
    int i6 = i0 + 6;
    int i7 = i0 + 7;

    // Q scalar loads: each lane loads 2 dims, 32 lanes cover 64 dims.
    int d = lane * 2; // 0..62
    const float* q0_ptr = Q + ((bh * S + i0) * D);
    const float* q1_ptr = Q + ((bh * S + i1) * D);
    const float* q2_ptr = Q + ((bh * S + i2) * D);
    const float* q3_ptr = Q + ((bh * S + i3) * D);
    const float* q4_ptr = Q + ((bh * S + i4) * D);
    const float* q5_ptr = Q + ((bh * S + i5) * D);
    const float* q6_ptr = Q + ((bh * S + i6) * D);
    const float* q7_ptr = Q + ((bh * S + i7) * D);

    float q0a = ldg_f32(q0_ptr + d + 0);
    float q0b = ldg_f32(q0_ptr + d + 1);
    float q1a = ldg_f32(q1_ptr + d + 0);
    float q1b = ldg_f32(q1_ptr + d + 1);
    float q2a = ldg_f32(q2_ptr + d + 0);
    float q2b = ldg_f32(q2_ptr + d + 1);
    float q3a = ldg_f32(q3_ptr + d + 0);
    float q3b = ldg_f32(q3_ptr + d + 1);
    float q4a = ldg_f32(q4_ptr + d + 0);
    float q4b = ldg_f32(q4_ptr + d + 1);
    float q5a = ldg_f32(q5_ptr + d + 0);
    float q5b = ldg_f32(q5_ptr + d + 1);
    float q6a = ldg_f32(q6_ptr + d + 0);
    float q6b = ldg_f32(q6_ptr + d + 1);
    float q7a = ldg_f32(q7_ptr + d + 0);
    float q7b = ldg_f32(q7_ptr + d + 1);

    // Online softmax state
    float m0=-INFINITY,l0=0.f, m1=-INFINITY,l1=0.f, m2=-INFINITY,l2=0.f, m3=-INFINITY,l3=0.f;
    float m4=-INFINITY,l4=0.f, m5=-INFINITY,l5=0.f, m6=-INFINITY,l6=0.f, m7=-INFINITY,l7=0.f;

    // Accumulators: vec4 path lanes 0..15 each own one float4.
    float4a o0_4{0,0,0,0}, o1_4{0,0,0,0}, o2_4{0,0,0,0}, o3_4{0,0,0,0};
    float4a o4_4{0,0,0,0}, o5_4{0,0,0,0}, o6_4{0,0,0,0}, o7_4{0,0,0,0};

    // Scalar path: each lane owns 2 dims (d,d+1)
    float o0s0=0,o0s1=0,o1s0=0,o1s1=0,o2s0=0,o2s1=0,o3s0=0,o3s1=0;
    float o4s0=0,o4s1=0,o5s0=0,o5s1=0,o6s0=0,o6s1=0,o7s0=0,o7s1=0;

    int d4 = lane; // 0..31 (vec4 uses <16)

    // Prefetch first K
    const float* k_ptr0 = K + ((bh * S + 0) * D);
    float k0_next = ldg_f32(k_ptr0 + d + 0);
    float k1_next = ldg_f32(k_ptr0 + d + 1);

    #pragma unroll 1
    for (int j = 0; j < S; j++) {
        // use prefetched K
        float k0 = k0_next;
        float k1 = k1_next;

        // prefetch next K (software pipeline)
        if (j + 1 < S) {
            const float* k_ptr1 = K + ((bh * S + (j + 1)) * D);
            k0_next = ldg_f32(k_ptr1 + d + 0);
            k1_next = ldg_f32(k_ptr1 + d + 1);
        }

        float part0 = fmaf(q0a, k0, q0b * k1);
        float part1 = fmaf(q1a, k0, q1b * k1);
        float part2 = fmaf(q2a, k0, q2b * k1);
        float part3 = fmaf(q3a, k0, q3b * k1);
        float part4 = fmaf(q4a, k0, q4b * k1);
        float part5 = fmaf(q5a, k0, q5b * k1);
        float part6 = fmaf(q6a, k0, q6b * k1);
        float part7 = fmaf(q7a, k0, q7b * k1);

        float dot0 = warp_reduce_sum(part0);
        float dot1 = warp_reduce_sum(part1);
        float dot2 = warp_reduce_sum(part2);
        float dot3 = warp_reduce_sum(part3);
        float dot4 = warp_reduce_sum(part4);
        float dot5 = warp_reduce_sum(part5);
        float dot6 = warp_reduce_sum(part6);
        float dot7 = warp_reduce_sum(part7);

        dot0 = __shfl_sync(0xffffffff, dot0, 0);
        dot1 = __shfl_sync(0xffffffff, dot1, 0);
        dot2 = __shfl_sync(0xffffffff, dot2, 0);
        dot3 = __shfl_sync(0xffffffff, dot3, 0);
        dot4 = __shfl_sync(0xffffffff, dot4, 0);
        dot5 = __shfl_sync(0xffffffff, dot5, 0);
        dot6 = __shfl_sync(0xffffffff, dot6, 0);
        dot7 = __shfl_sync(0xffffffff, dot7, 0);

        float s0 = dot0 * inv_sqrt_d;
        float s1 = dot1 * inv_sqrt_d;
        float s2 = dot2 * inv_sqrt_d;
        float s3 = dot3 * inv_sqrt_d;
        float s4 = dot4 * inv_sqrt_d;
        float s5 = dot5 * inv_sqrt_d;
        float s6 = dot6 * inv_sqrt_d;
        float s7 = dot7 * inv_sqrt_d;

        float m0n = fmaxf(m0, s0); float a0 = __expf(m0 - m0n); float p0 = __expf(s0 - m0n); l0 = l0*a0 + p0; m0 = m0n;
        float m1n = fmaxf(m1, s1); float a1 = __expf(m1 - m1n); float p1 = __expf(s1 - m1n); l1 = l1*a1 + p1; m1 = m1n;
        float m2n = fmaxf(m2, s2); float a2 = __expf(m2 - m2n); float p2 = __expf(s2 - m2n); l2 = l2*a2 + p2; m2 = m2n;
        float m3n = fmaxf(m3, s3); float a3 = __expf(m3 - m3n); float p3 = __expf(s3 - m3n); l3 = l3*a3 + p3; m3 = m3n;
        float m4n = fmaxf(m4, s4); float a4 = __expf(m4 - m4n); float p4 = __expf(s4 - m4n); l4 = l4*a4 + p4; m4 = m4n;
        float m5n = fmaxf(m5, s5); float a5 = __expf(m5 - m5n); float p5 = __expf(s5 - m5n); l5 = l5*a5 + p5; m5 = m5n;
        float m6n = fmaxf(m6, s6); float a6 = __expf(m6 - m6n); float p6 = __expf(s6 - m6n); l6 = l6*a6 + p6; m6 = m6n;
        float m7n = fmaxf(m7, s7); float a7 = __expf(m7 - m7n); float p7 = __expf(s7 - m7n); l7 = l7*a7 + p7; m7 = m7n;

        const float* v_ptr = V + ((bh * S + j) * D);

        if (vec4_ok) {
            if (d4 < D4) {
                const float* v4_ptr = v_ptr + d4 * VEC;
                float4a v4 = ld_float4a(v4_ptr);

                // rescale
                o0_4.x*=a0; o0_4.y*=a0; o0_4.z*=a0; o0_4.w*=a0;
                o1_4.x*=a1; o1_4.y*=a1; o1_4.z*=a1; o1_4.w*=a1;
                o2_4.x*=a2; o2_4.y*=a2; o2_4.z*=a2; o2_4.w*=a2;
                o3_4.x*=a3; o3_4.y*=a3; o3_4.z*=a3; o3_4.w*=a3;
                o4_4.x*=a4; o4_4.y*=a4; o4_4.z*=a4; o4_4.w*=a4;
                o5_4.x*=a5; o5_4.y*=a5; o5_4.z*=a5; o5_4.w*=a5;
                o6_4.x*=a6; o6_4.y*=a6; o6_4.z*=a6; o6_4.w*=a6;
                o7_4.x*=a7; o7_4.y*=a7; o7_4.z*=a7; o7_4.w*=a7;

                // fma
                o0_4.x=fmaf(p0,v4.x,o0_4.x); o0_4.y=fmaf(p0,v4.y,o0_4.y); o0_4.z=fmaf(p0,v4.z,o0_4.z); o0_4.w=fmaf(p0,v4.w,o0_4.w);
                o1_4.x=fmaf(p1,v4.x,o1_4.x); o1_4.y=fmaf(p1,v4.y,o1_4.y); o1_4.z=fmaf(p1,v4.z,o1_4.z); o1_4.w=fmaf(p1,v4.w,o1_4.w);
                o2_4.x=fmaf(p2,v4.x,o2_4.x); o2_4.y=fmaf(p2,v4.y,o2_4.y); o2_4.z=fmaf(p2,v4.z,o2_4.z); o2_4.w=fmaf(p2,v4.w,o2_4.w);
                o3_4.x=fmaf(p3,v4.x,o3_4.x); o3_4.y=fmaf(p3,v4.y,o3_4.y); o3_4.z=fmaf(p3,v4.z,o3_4.z); o3_4.w=fmaf(p3,v4.w,o3_4.w);
                o4_4.x=fmaf(p4,v4.x,o4_4.x); o4_4.y=fmaf(p4,v4.y,o4_4.y); o4_4.z=fmaf(p4,v4.z,o4_4.z); o4_4.w=fmaf(p4,v4.w,o4_4.w);
                o5_4.x=fmaf(p5,v4.x,o5_4.x); o5_4.y=fmaf(p5,v4.y,o5_4.y); o5_4.z=fmaf(p5,v4.z,o5_4.z); o5_4.w=fmaf(p5,v4.w,o5_4.w);
                o6_4.x=fmaf(p6,v4.x,o6_4.x); o6_4.y=fmaf(p6,v4.y,o6_4.y); o6_4.z=fmaf(p6,v4.z,o6_4.z); o6_4.w=fmaf(p6,v4.w,o6_4.w);
                o7_4.x=fmaf(p7,v4.x,o7_4.x); o7_4.y=fmaf(p7,v4.y,o7_4.y); o7_4.z=fmaf(p7,v4.z,o7_4.z); o7_4.w=fmaf(p7,v4.w,o7_4.w);
            }
        } else {
            float v0 = ldg_f32(v_ptr + d + 0);
            float v1 = ldg_f32(v_ptr + d + 1);

            o0s0 = o0s0*a0 + p0*v0; o0s1 = o0s1*a0 + p0*v1;
            o1s0 = o1s0*a1 + p1*v0; o1s1 = o1s1*a1 + p1*v1;
            o2s0 = o2s0*a2 + p2*v0; o2s1 = o2s1*a2 + p2*v1;
            o3s0 = o3s0*a3 + p3*v0; o3s1 = o3s1*a3 + p3*v1;
            o4s0 = o4s0*a4 + p4*v0; o4s1 = o4s1*a4 + p4*v1;
            o5s0 = o5s0*a5 + p5*v0; o5s1 = o5s1*a5 + p5*v1;
            o6s0 = o6s0*a6 + p6*v0; o6s1 = o6s1*a6 + p6*v1;
            o7s0 = o7s0*a7 + p7*v0; o7s1 = o7s1*a7 + p7*v1;
        }
    }

    float inv_l0 = 1.f / (l0 + 1e-9f);
    float inv_l1 = 1.f / (l1 + 1e-9f);
    float inv_l2 = 1.f / (l2 + 1e-9f);
    float inv_l3 = 1.f / (l3 + 1e-9f);
    float inv_l4 = 1.f / (l4 + 1e-9f);
    float inv_l5 = 1.f / (l5 + 1e-9f);
    float inv_l6 = 1.f / (l6 + 1e-9f);
    float inv_l7 = 1.f / (l7 + 1e-9f);

    if (vec4_ok) {
        if (d4 < D4) {
            o0_4.x*=inv_l0; o0_4.y*=inv_l0; o0_4.z*=inv_l0; o0_4.w*=inv_l0;
            o1_4.x*=inv_l1; o1_4.y*=inv_l1; o1_4.z*=inv_l1; o1_4.w*=inv_l1;
            o2_4.x*=inv_l2; o2_4.y*=inv_l2; o2_4.z*=inv_l2; o2_4.w*=inv_l2;
            o3_4.x*=inv_l3; o3_4.y*=inv_l3; o3_4.z*=inv_l3; o3_4.w*=inv_l3;
            o4_4.x*=inv_l4; o4_4.y*=inv_l4; o4_4.z*=inv_l4; o4_4.w*=inv_l4;
            o5_4.x*=inv_l5; o5_4.y*=inv_l5; o5_4.z*=inv_l5; o5_4.w*=inv_l5;
            o6_4.x*=inv_l6; o6_4.y*=inv_l6; o6_4.z*=inv_l6; o6_4.w*=inv_l6;
            o7_4.x*=inv_l7; o7_4.y*=inv_l7; o7_4.z*=inv_l7; o7_4.w*=inv_l7;

            float* o0_ptr = Out + ((bh * S + i0) * D) + d4 * VEC;
            float* o1_ptr = Out + ((bh * S + i1) * D) + d4 * VEC;
            float* o2_ptr = Out + ((bh * S + i2) * D) + d4 * VEC;
            float* o3_ptr = Out + ((bh * S + i3) * D) + d4 * VEC;
            float* o4_ptr = Out + ((bh * S + i4) * D) + d4 * VEC;
            float* o5_ptr = Out + ((bh * S + i5) * D) + d4 * VEC;
            float* o6_ptr = Out + ((bh * S + i6) * D) + d4 * VEC;
            float* o7_ptr = Out + ((bh * S + i7) * D) + d4 * VEC;

            st_float4a(o0_ptr, o0_4);
            st_float4a(o1_ptr, o1_4);
            st_float4a(o2_ptr, o2_4);
            st_float4a(o3_ptr, o3_4);
            st_float4a(o4_ptr, o4_4);
            st_float4a(o5_ptr, o5_4);
            st_float4a(o6_ptr, o6_4);
            st_float4a(o7_ptr, o7_4);
        }
    } else {
        float* o0_ptr = Out + ((bh * S + i0) * D);
        float* o1_ptr = Out + ((bh * S + i1) * D);
        float* o2_ptr = Out + ((bh * S + i2) * D);
        float* o3_ptr = Out + ((bh * S + i3) * D);
        float* o4_ptr = Out + ((bh * S + i4) * D);
        float* o5_ptr = Out + ((bh * S + i5) * D);
        float* o6_ptr = Out + ((bh * S + i6) * D);
        float* o7_ptr = Out + ((bh * S + i7) * D);

        o0_ptr[d+0]=o0s0*inv_l0; o0_ptr[d+1]=o0s1*inv_l0;
        o1_ptr[d+0]=o1s0*inv_l1; o1_ptr[d+1]=o1s1*inv_l1;
        o2_ptr[d+0]=o2s0*inv_l2; o2_ptr[d+1]=o2s1*inv_l2;
        o3_ptr[d+0]=o3s0*inv_l3; o3_ptr[d+1]=o3s1*inv_l3;
        o4_ptr[d+0]=o4s0*inv_l4; o4_ptr[d+1]=o4s1*inv_l4;
        o5_ptr[d+0]=o5s0*inv_l5; o5_ptr[d+1]=o5s1*inv_l5;
        o6_ptr[d+0]=o6s0*inv_l6; o6_ptr[d+1]=o6s1*inv_l6;
        o7_ptr[d+0]=o7s0*inv_l7; o7_ptr[d+1]=o7s1*inv_l7;
    }
}

static inline int is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
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
    TORCH_CHECK((S % 8) == 0, "S must be divisible by 8 for q-tiling=8");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D, "V shape mismatch");

    auto Out = torch::empty_like(Q);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    int vec4_ok = is_aligned_16(V.data_ptr<float>()) && is_aligned_16(Out.data_ptr<float>());

    int items = B * H * (S / 8);
    int blocks = (items + 1) / 2; // 2 warps per block
    int threads = 64;

    sdpa_fwd_d64_s512_warp_q8_2warps<<<blocks, threads>>>(
        (const float*)Q.data_ptr<float>(),
        (const float*)K.data_ptr<float>(),
        (const float*)V.data_ptr<float>(),
        (float*)Out.data_ptr<float>(),
        B, H, inv_sqrt_d, vec4_ok
    );

    return Out;
}
"""

cpp_src = r"""
torch::Tensor sdpa_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

custom_ops_lib = load_inline(
    name="custom_self_attn_sdpa_s512_d64_alignsafe_v2_q8",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sdpa_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
)

# -----------------------------------------------------------------------------
# ModelNew
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

    def forward(self, x):
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        use_fast = (
            (not self.training) and self.dropout_p == 0.0 and
            Q.is_cuda and K.is_cuda and V.is_cuda and
            Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32 and
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and
            Q.dim() == 4 and K.dim() == 4 and V.dim() == 4 and
            self.d_k == 64 and S == 512
        )

        if use_fast:
            attn_out = custom_ops_lib.sdpa_forward_cuda(Q, K, V)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout_p != 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)
            attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(attn_out)
        out = self.dropout(out)
        return out