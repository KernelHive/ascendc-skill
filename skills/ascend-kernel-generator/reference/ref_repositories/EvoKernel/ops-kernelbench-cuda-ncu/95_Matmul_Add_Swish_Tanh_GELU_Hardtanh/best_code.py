import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused (add bias + swish + tanh + gelu + hardtanh) forward ----

matmul_add_swish_tanh_gelu_hardtanh_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float clamp_m1_p1(float x) {
    return fminf(fmaxf(x, -1.0f), 1.0f);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// GELU tanh approximation:
// 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
__device__ __forceinline__ float gelu_tanh_approx(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = kAlpha * (x + kBeta * x3);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float act_fused(float v) {
    v = v * sigmoidf_fast(v);   // swish
    v = tanhf(v);               // tanh
    v = gelu_tanh_approx(v);    // gelu
    v = clamp_m1_p1(v);         // hardtanh [-1,1]
    return v;
}

static inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

template<int TILE>
__global__ __launch_bounds__(128, 4)
void add_swish_tanh_gelu_hardtanh_fwd_rowwise_bias_smem_vec(
    const float* __restrict__ x,     // [B, O]
    const float* __restrict__ bias,  // [O]
    float* __restrict__ out,         // [B, O]
    int B,
    int O
) {
    // Stage a contiguous bias tile into shared memory: bias[col0 : col0+TILE)
    __shared__ float sbias[TILE];

    // 2D mapping:
    // - blockIdx.y picks a row
    // - blockIdx.x picks a column tile group (TILE columns per block in x dimension)
    int row = (int)blockIdx.y;
    int col0 = (int)blockIdx.x * TILE;
    if (row >= B || col0 >= O) return;

    int t = (int)threadIdx.x;

    // cooperative load bias tile
    #pragma unroll
    for (int i = t; i < TILE; i += 128) {
        int c = col0 + i;
#if __CUDA_ARCH__ >= 350
        sbias[i] = (c < O) ? __ldg(bias + c) : 0.0f;
#else
        sbias[i] = (c < O) ? bias[c] : 0.0f;
#endif
    }
    __syncthreads();

    const float* __restrict__ xrow = x + (int64_t)row * (int64_t)O + (int64_t)col0;
    float* __restrict__ orow = out + (int64_t)row * (int64_t)O + (int64_t)col0;

    // Vectorized path: process float4 where possible
    int tileO = min(TILE, O - col0);
    int vec4_elems = tileO >> 2;          // number of float4
    int rem = tileO & 3;

    const float4* __restrict__ x4 = (const float4*)xrow;
    float4* __restrict__ o4 = (float4*)orow;

    // Each thread handles multiple float4s within the tile.
    // Small unroll to raise ILP and allow more in-flight memory ops.
    for (int i4 = t; i4 < vec4_elems; i4 += 128) {
        // unroll by 2 within bounds
        int i4_0 = i4;
        int i4_1 = i4 + 128;

        // first
        {
            float4 xv = x4[i4_0];
            int j = (i4_0 << 2);
            float b0 = sbias[j + 0];
            float b1 = sbias[j + 1];
            float b2 = sbias[j + 2];
            float b3 = sbias[j + 3];
            float o0 = act_fused(xv.x + b0);
            float o1 = act_fused(xv.y + b1);
            float o2 = act_fused(xv.z + b2);
            float o3 = act_fused(xv.w + b3);
            o4[i4_0] = make_float4(o0, o1, o2, o3);
        }

        if (i4_1 < vec4_elems) {
            float4 xv = x4[i4_1];
            int j = (i4_1 << 2);
            float b0 = sbias[j + 0];
            float b1 = sbias[j + 1];
            float b2 = sbias[j + 2];
            float b3 = sbias[j + 3];
            float o0 = act_fused(xv.x + b0);
            float o1 = act_fused(xv.y + b1);
            float o2 = act_fused(xv.z + b2);
            float o3 = act_fused(xv.w + b3);
            o4[i4_1] = make_float4(o0, o1, o2, o3);
        }
    }

    // Tail (<=3 floats) handled by first few threads only
    if (rem) {
        int base = (vec4_elems << 2);
        for (int r = t; r < rem; r += 128) {
            int j = base + r;
            float v = xrow[j] + sbias[j];
            orow[j] = act_fused(v);
        }
    }
}

__global__ __launch_bounds__(256, 2)
void add_swish_tanh_gelu_hardtanh_fwd_scalar_generic(
    const float* __restrict__ x,     // [n]
    const float* __restrict__ bias,  // [O]
    float* __restrict__ out,         // [n]
    int64_t n,
    int32_t O
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        int32_t j = (int32_t)(idx % (int64_t)O);
#if __CUDA_ARCH__ >= 350
        float b = __ldg(bias + j);
#else
        float b = bias[j];
#endif
        float v = x[idx] + b;
        out[idx] = act_fused(v);
    }
}

torch::Tensor add_swish_tanh_gelu_hardtanh_forward_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double /*hardtanh_min*/,
    double /*hardtanh_max*/
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported for x");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "only float32 supported for bias");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, O]");
    TORCH_CHECK(x.size(1) == bias.size(0), "x.size(1) must match bias.size(0)");

    auto out = torch::empty_like(x);

    int B = (int)x.size(0);
    int O = (int)x.size(1);

    // Fast path: row-wise tiles with shared-memory bias staging.
    // TILE=1024 (4KB smem) balances reuse without hurting occupancy much.
    // Use it when O is reasonably large (true here) and contiguous.
    constexpr int TILE = 1024;

    if (O >= TILE) {
        dim3 block(128, 1, 1);
        dim3 grid((O + TILE - 1) / TILE, B, 1);
        add_swish_tanh_gelu_hardtanh_fwd_rowwise_bias_smem_vec<TILE><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, O
        );
        return out;
    }

    // Fallback generic 1D kernel
    const int64_t n = x.numel();
    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = clamp_int(blocks, 1, 8192);

    add_swish_tanh_gelu_hardtanh_fwd_scalar_generic<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)bias.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n, (int32_t)O
    );
    return out;
}
"""

matmul_add_swish_tanh_gelu_hardtanh_cpp_source = r"""
torch::Tensor add_swish_tanh_gelu_hardtanh_forward_cuda(torch::Tensor x, torch::Tensor bias, double hardtanh_min, double hardtanh_max);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_add_swish_tanh_gelu_hardtanh_v5",
    cpp_sources=matmul_add_swish_tanh_gelu_hardtanh_cpp_source,
    cuda_sources=matmul_add_swish_tanh_gelu_hardtanh_cuda_source,
    functions=["add_swish_tanh_gelu_hardtanh_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)

# ---- Model wrapper using the custom op ----

class ModelNew(nn.Module):
    """
    Model that performs GEMM (nn.Linear) followed by a fused (add bias + swish + tanh + gelu + hardtanh) custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.custom_ops_lib = custom_ops_lib
        self.hardtanh_min = -1.0
        self.hardtanh_max = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.matmul(x)  # cuBLAS GEMM
        if not x.is_contiguous():
            x = x.contiguous()
        bias = self.add_value
        if not bias.is_contiguous():
            bias = bias.contiguous()
        return self.custom_ops_lib.add_swish_tanh_gelu_hardtanh_forward_cuda(
            x, bias, float(self.hardtanh_min), float(self.hardtanh_max)
        )

# Keep the same input helpers for compatibility with the original harness.
batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]