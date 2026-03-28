import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# AFT-Full fused forward (float32) - v6:
# - 4-query tiling per warp: each warp computes 4 query rows (i0..i3)
#   and reuses each streamed K/V load across those 4 queries.
# - Bias rows for (i0..i3) staged into shared memory once per block.
# - Block = 4 warps (128 threads) to improve latency hiding.
# - Vectorized float4 path with alignment checks; scalar fallback otherwise.
# - Single-pass online stable exp accumulation per query (FlashAttention-style).
# ------------------------------------------------------------

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float4 load_f4(const float4* p) { return *p; }
__device__ __forceinline__ void store_f4(float4* p, const float4 v) { *p = v; }

__device__ __forceinline__ float4 f4_max(float4 a, float4 b) {
    a.x = fmaxf(a.x, b.x);
    a.y = fmaxf(a.y, b.y);
    a.z = fmaxf(a.z, b.z);
    a.w = fmaxf(a.w, b.w);
    return a;
}
__device__ __forceinline__ float4 f4_add_scalar(float4 a, float b) {
    a.x += b; a.y += b; a.z += b; a.w += b;
    return a;
}
__device__ __forceinline__ float4 f4_mul(float4 a, float4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
    return a;
}
__device__ __forceinline__ float4 f4_add(float4 a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}
__device__ __forceinline__ float4 f4_div(float4 a, float4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
    return a;
}

__global__ void aft_full_i4_warp_vec4(
    const float* __restrict__ q,    // [B,N,D]
    const float* __restrict__ k,    // [B,N,D]
    const float* __restrict__ v,    // [B,N,D]
    const float* __restrict__ bias, // [N,N]
    float* __restrict__ out,        // [B,N,D]
    int B, int N, int D
) {
    // grid.x = ceil(N/4), grid.y = B
    int b = (int)blockIdx.y;
    int itile = (int)blockIdx.x * 4;
    if (b >= B || itile >= N) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5; // 0..3
    int warps_per_block = (int)(blockDim.x >> 5); // 4

    // shared bias tile: 4 rows x N cols
    extern __shared__ float sh_bias[];

    // Load bias rows for i0..i3 into shared (cooperative across block)
    for (int t = tid; t < 4 * N; t += (int)blockDim.x) {
        int r = t / N;   // 0..3
        int j = t - r * N;
        int i = itile + r;
        float val = 0.f;
        if (i < N) val = ldg_f32(bias + i * N + j);
        sh_bias[r * N + j] = val;
    }
    __syncthreads();

    int D4 = D >> 2;

    // Each warp covers disjoint d4 chunks: inter-warp striping
    for (int d4 = lane + warp_in_block * 32; d4 < D4; d4 += 32 * warps_per_block) {
        int d = d4 << 2;

        // Load Q for each of up to 4 query rows
        float4 q0 = make_float4(0.f,0.f,0.f,0.f);
        float4 q1 = make_float4(0.f,0.f,0.f,0.f);
        float4 q2 = make_float4(0.f,0.f,0.f,0.f);
        float4 q3 = make_float4(0.f,0.f,0.f,0.f);

        int i0 = itile + 0;
        int i1 = itile + 1;
        int i2 = itile + 2;
        int i3 = itile + 3;

        if (i0 < N) q0 = load_f4((const float4*)(q + ((b * N + i0) * D + d)));
        if (i1 < N) q1 = load_f4((const float4*)(q + ((b * N + i1) * D + d)));
        if (i2 < N) q2 = load_f4((const float4*)(q + ((b * N + i2) * D + d)));
        if (i3 < N) q3 = load_f4((const float4*)(q + ((b * N + i3) * D + d)));

        // Online stable exp accumulators for 4 queries
        float4 m0 = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float4 m1 = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float4 m2 = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float4 m3 = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);

        float4 den0 = make_float4(0.f,0.f,0.f,0.f);
        float4 den1 = make_float4(0.f,0.f,0.f,0.f);
        float4 den2 = make_float4(0.f,0.f,0.f,0.f);
        float4 den3 = make_float4(0.f,0.f,0.f,0.f);

        float4 num0 = make_float4(0.f,0.f,0.f,0.f);
        float4 num1 = make_float4(0.f,0.f,0.f,0.f);
        float4 num2 = make_float4(0.f,0.f,0.f,0.f);
        float4 num3 = make_float4(0.f,0.f,0.f,0.f);

        #pragma unroll 1
        for (int j = 0; j < N; ++j) {
            // Stream K/V once
            float4 kv = load_f4((const float4*)(k + ((b * N + j) * D + d)));
            float4 vv = load_f4((const float4*)(v + ((b * N + j) * D + d)));

            // Biases from shared
            float b0 = sh_bias[0 * N + j];
            float b1 = sh_bias[1 * N + j];
            float b2 = sh_bias[2 * N + j];
            float b3 = sh_bias[3 * N + j];

            // Query 0
            if (i0 < N) {
                float4 x0 = f4_add_scalar(kv, b0);
                float4 mn = f4_max(m0, x0);
                float4 scale = make_float4(__expf(m0.x - mn.x), __expf(m0.y - mn.y), __expf(m0.z - mn.z), __expf(m0.w - mn.w));
                den0 = f4_mul(den0, scale);
                num0 = f4_mul(num0, scale);
                float4 e = make_float4(__expf(x0.x - mn.x), __expf(x0.y - mn.y), __expf(x0.z - mn.z), __expf(x0.w - mn.w));
                den0 = f4_add(den0, e);
                num0 = f4_add(num0, make_float4(e.x * vv.x, e.y * vv.y, e.z * vv.z, e.w * vv.w));
                m0 = mn;
            }

            // Query 1
            if (i1 < N) {
                float4 x1 = f4_add_scalar(kv, b1);
                float4 mn = f4_max(m1, x1);
                float4 scale = make_float4(__expf(m1.x - mn.x), __expf(m1.y - mn.y), __expf(m1.z - mn.z), __expf(m1.w - mn.w));
                den1 = f4_mul(den1, scale);
                num1 = f4_mul(num1, scale);
                float4 e = make_float4(__expf(x1.x - mn.x), __expf(x1.y - mn.y), __expf(x1.z - mn.z), __expf(x1.w - mn.w));
                den1 = f4_add(den1, e);
                num1 = f4_add(num1, make_float4(e.x * vv.x, e.y * vv.y, e.z * vv.z, e.w * vv.w));
                m1 = mn;
            }

            // Query 2
            if (i2 < N) {
                float4 x2 = f4_add_scalar(kv, b2);
                float4 mn = f4_max(m2, x2);
                float4 scale = make_float4(__expf(m2.x - mn.x), __expf(m2.y - mn.y), __expf(m2.z - mn.z), __expf(m2.w - mn.w));
                den2 = f4_mul(den2, scale);
                num2 = f4_mul(num2, scale);
                float4 e = make_float4(__expf(x2.x - mn.x), __expf(x2.y - mn.y), __expf(x2.z - mn.z), __expf(x2.w - mn.w));
                den2 = f4_add(den2, e);
                num2 = f4_add(num2, make_float4(e.x * vv.x, e.y * vv.y, e.z * vv.z, e.w * vv.w));
                m2 = mn;
            }

            // Query 3
            if (i3 < N) {
                float4 x3 = f4_add_scalar(kv, b3);
                float4 mn = f4_max(m3, x3);
                float4 scale = make_float4(__expf(m3.x - mn.x), __expf(m3.y - mn.y), __expf(m3.z - mn.z), __expf(m3.w - mn.w));
                den3 = f4_mul(den3, scale);
                num3 = f4_mul(num3, scale);
                float4 e = make_float4(__expf(x3.x - mn.x), __expf(x3.y - mn.y), __expf(x3.z - mn.z), __expf(x3.w - mn.w));
                den3 = f4_add(den3, e);
                num3 = f4_add(num3, make_float4(e.x * vv.x, e.y * vv.y, e.z * vv.z, e.w * vv.w));
                m3 = mn;
            }
        }

        // Write out with sigmoid gate
        if (i0 < N) {
            float4 mix = f4_div(num0, den0);
            float4 gate = make_float4(sigmoidf_fast(q0.x), sigmoidf_fast(q0.y), sigmoidf_fast(q0.z), sigmoidf_fast(q0.w));
            float4 o = make_float4(gate.x * mix.x, gate.y * mix.y, gate.z * mix.z, gate.w * mix.w);
            store_f4((float4*)(out + ((b * N + i0) * D + d)), o);
        }
        if (i1 < N) {
            float4 mix = f4_div(num1, den1);
            float4 gate = make_float4(sigmoidf_fast(q1.x), sigmoidf_fast(q1.y), sigmoidf_fast(q1.z), sigmoidf_fast(q1.w));
            float4 o = make_float4(gate.x * mix.x, gate.y * mix.y, gate.z * mix.z, gate.w * mix.w);
            store_f4((float4*)(out + ((b * N + i1) * D + d)), o);
        }
        if (i2 < N) {
            float4 mix = f4_div(num2, den2);
            float4 gate = make_float4(sigmoidf_fast(q2.x), sigmoidf_fast(q2.y), sigmoidf_fast(q2.z), sigmoidf_fast(q2.w));
            float4 o = make_float4(gate.x * mix.x, gate.y * mix.y, gate.z * mix.z, gate.w * mix.w);
            store_f4((float4*)(out + ((b * N + i2) * D + d)), o);
        }
        if (i3 < N) {
            float4 mix = f4_div(num3, den3);
            float4 gate = make_float4(sigmoidf_fast(q3.x), sigmoidf_fast(q3.y), sigmoidf_fast(q3.z), sigmoidf_fast(q3.w));
            float4 o = make_float4(gate.x * mix.x, gate.y * mix.y, gate.z * mix.z, gate.w * mix.w);
            store_f4((float4*)(out + ((b * N + i3) * D + d)), o);
        }
    }
}

__global__ void aft_full_i2_warp_scalar(
    const float* __restrict__ q,    // [B,N,D]
    const float* __restrict__ k,    // [B,N,D]
    const float* __restrict__ v,    // [B,N,D]
    const float* __restrict__ bias, // [N,N]
    float* __restrict__ out,        // [B,N,D]
    int B, int N, int D
) {
    // scalar fallback: 2-query tiling, no shared bias (keep it simple)
    int b = (int)blockIdx.y;
    int itile = (int)blockIdx.x * 2;
    if (b >= B || itile >= N) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;

    int i0 = itile;
    int i1 = itile + 1;

    for (int d = lane; d < D; d += 32) {
        float q0 = (i0 < N) ? ldg_f32(q + ((b * N + i0) * D + d)) : 0.f;
        float q1 = (i1 < N) ? ldg_f32(q + ((b * N + i1) * D + d)) : 0.f;

        float m0 = -INFINITY, den0 = 0.f, num0 = 0.f;
        float m1 = -INFINITY, den1 = 0.f, num1 = 0.f;

        #pragma unroll 1
        for (int j = 0; j < N; ++j) {
            float kv = ldg_f32(k + ((b * N + j) * D + d));
            float vv = ldg_f32(v + ((b * N + j) * D + d));

            if (i0 < N) {
                float x = kv + ldg_f32(bias + i0 * N + j);
                float mn = fmaxf(m0, x);
                float scale = __expf(m0 - mn);
                den0 *= scale; num0 *= scale;
                float e = __expf(x - mn);
                den0 += e; num0 += e * vv;
                m0 = mn;
            }
            if (i1 < N) {
                float x = kv + ldg_f32(bias + i1 * N + j);
                float mn = fmaxf(m1, x);
                float scale = __expf(m1 - mn);
                den1 *= scale; num1 *= scale;
                float e = __expf(x - mn);
                den1 += e; num1 += e * vv;
                m1 = mn;
            }
        }

        if (i0 < N) out[((b * N + i0) * D + d)] = sigmoidf_fast(q0) * (num0 / den0);
        if (i1 < N) out[((b * N + i1) * D + d)] = sigmoidf_fast(q1) * (num1 / den1);
    }
}

torch::Tensor aft_full_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && bias.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float, "q must be float32");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Float, "k must be float32");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Float, "v must be float32");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float, "bias must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && bias.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be 3D [B,N,D]");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D [N,N]");

    int B = (int)q.size(0);
    int N = (int)q.size(1);
    int D = (int)q.size(2);

    TORCH_CHECK((int)k.size(0) == B && (int)k.size(1) == N && (int)k.size(2) == D, "k shape mismatch");
    TORCH_CHECK((int)v.size(0) == B && (int)v.size(1) == N && (int)v.size(2) == D, "v shape mismatch");
    TORCH_CHECK((int)bias.size(0) == N && (int)bias.size(1) == N, "bias must be [N,N]");

    auto out = torch::empty({B, N, D}, torch::TensorOptions().dtype(q.dtype()).device(q.device()));

    // vec4 guard: D%4 and pointer alignment for q/k/v/out
    bool vec4_ok = false;
    if ((D & 3) == 0) {
        uintptr_t qa = (uintptr_t)q.data_ptr<float>();
        uintptr_t ka = (uintptr_t)k.data_ptr<float>();
        uintptr_t va = (uintptr_t)v.data_ptr<float>();
        uintptr_t oa = (uintptr_t)out.data_ptr<float>();
        if (((qa | ka | va | oa) & 0xF) == 0) vec4_ok = true;
    }

    if (vec4_ok) {
        dim3 grid((unsigned)((N + 3) / 4), (unsigned)B, 1);
        dim3 block(128, 1, 1); // 4 warps
        size_t shmem = (size_t)(4 * N) * sizeof(float);
        aft_full_i4_warp_vec4<<<grid, block, shmem>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, N, D
        );
    } else {
        dim3 grid((unsigned)((N + 1) / 2), (unsigned)B, 1);
        dim3 block(32, 1, 1);
        aft_full_i2_warp_scalar<<<grid, block>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)k.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, N, D
        );
    }

    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor aft_full_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias);
"""

custom_ops_lib = load_inline(
    name="custom_aft_full_ops_v6",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["aft_full_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    AFT-Full with further optimized fused CUDA kernel:
    - 4-query tiling per warp to reuse streamed K/V loads
    - shared-memory staged bias rows per i-tile
    - 4-warp blocks for better latency hiding
    - single-pass online stable exp accumulation
    """
    def __init__(self, d_model, n=49):
        super(ModelNew, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.custom_ops_lib = custom_ops_lib

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q = self.fc_q(input)
        k = self.fc_k(input)
        v = self.fc_v(input)

        if not q.is_cuda:
            bs, n, dim = input.shape
            k4 = k.view(1, bs, n, dim)
            v4 = v.view(1, bs, n, dim)
            numerator = torch.sum(torch.exp(k4 + self.position_biases.view(n, 1, -1, 1)) * v4, dim=2)
            denominator = torch.sum(torch.exp(k4 + self.position_biases.view(n, 1, -1, 1)), dim=2)
            out = numerator / denominator
            return torch.sigmoid(q) * out.permute(1, 0, 2)

        # Kernel requires contiguous float32 CUDA
        if q.dtype != torch.float32:
            q = q.float()
            k = k.float()
            v = v.float()
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        bias = self.position_biases
        if bias.dtype != torch.float32:
            bias = bias.float()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        return self.custom_ops_lib.aft_full_cuda(q, k, v, bias)