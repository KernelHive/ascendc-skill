import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: optimized fused post-op (min + bias + scale) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float fminf_fast(float a, float b) { return a < b ? a : b; }

static inline bool is_aligned_uint(uintptr_t p, uintptr_t a) { return (p & (a - 1)) == 0; }

// ---------------------------------------------
// bias_numel == 1 : vectorized grid-stride
// ---------------------------------------------

__global__ __launch_bounds__(256, 4) void postop_bias1_vec4(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t total,
    float constant_value,
    float scale
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const int64_t total4 = total >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    const float b = ldg_f32(bias);
    const float alpha = scale;
    const float beta = b * scale;

    for (int64_t i = tid; i < total4; i += stride) {
        float4 v = x4[i];
        v.x = fminf_fast(v.x, constant_value) * alpha + beta;
        v.y = fminf_fast(v.y, constant_value) * alpha + beta;
        v.z = fminf_fast(v.z, constant_value) * alpha + beta;
        v.w = fminf_fast(v.w, constant_value) * alpha + beta;
        out4[i] = v;
    }
}

__global__ __launch_bounds__(256, 4) void postop_bias1_scalar(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t total,
    float constant_value,
    float scale
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float b = ldg_f32(bias);
    const float alpha = scale;
    const float beta = b * scale;

    for (int64_t i = tid; i < total; i += stride) {
        float v = x[i];
        out[i] = fminf_fast(v, constant_value) * alpha + beta;
    }
}

// ---------------------------------------------
// bias_numel == C broadcast over HW.
// Specialized 2D mapping: blocks over (C, NHW)
// Layout is contiguous NCHW.
// index = (n*C + c)*HW + hw
// We avoid div/mod per element and reuse per-block bias.
// ---------------------------------------------

template<int ILP4>
__global__ __launch_bounds__(256, 3) void postop_biasC_2d_vec4_ilp(
    const float* __restrict__ x,
    const float* __restrict__ bias, // (C)
    float* __restrict__ out,
    int N, int C, int HW,
    float constant_value,
    float scale
) {
    // blockIdx.y selects channel
    int c = (int)blockIdx.y;
    if (c >= C) return;

    // stage bias for the block
    __shared__ float s_beta;
    if (threadIdx.x == 0) {
        float b = ldg_f32(bias + c);
        s_beta = b * scale;
    }
    __syncthreads();

    const float alpha = scale;
    const float beta = s_beta;

    // Each blockIdx.x covers a tile of NHW in float4 units.
    // We treat NHW = N*HW.
    const int64_t NHW = (int64_t)N * (int64_t)HW;

    // vector length 4 over the HW dimension only; to preserve contiguity we vectorize over hw.
    // We require HW % 4 == 0 and base address aligned.
    const int hw4 = HW >> 2; // HW in float4 units
    const int64_t nhw4 = (int64_t)N * (int64_t)hw4; // NHW in float4 units

    // Starting element in float4 units for this block
    int64_t tile_start = (int64_t)blockIdx.x * (int64_t)(blockDim.x * ILP4);
    int64_t tid = (int64_t)threadIdx.x;

    // Base pointer for this channel in float4 units:
    // For each n, channel c starts at (n*C + c)*HW, which in float4 is ((n*C + c)*HW)/4
    // We iterate over (n, hw4) linear index j4 in [0, N*hw4)
    for (int ii = 0; ii < ILP4; ++ii) {
        int64_t j4 = tile_start + tid + (int64_t)ii * (int64_t)blockDim.x;
        if (j4 >= nhw4) continue;

        int n = (int)(j4 / (int64_t)hw4);
        int hw_idx4 = (int)(j4 - (int64_t)n * (int64_t)hw4);

        int64_t base_elem = ((int64_t)n * (int64_t)C + (int64_t)c) * (int64_t)HW;
        int64_t off4 = (base_elem >> 2) + (int64_t)hw_idx4;

        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

        float4 v = x4[off4];
        v.x = fminf_fast(v.x, constant_value) * alpha + beta;
        v.y = fminf_fast(v.y, constant_value) * alpha + beta;
        v.z = fminf_fast(v.z, constant_value) * alpha + beta;
        v.w = fminf_fast(v.w, constant_value) * alpha + beta;
        o4[off4] = v;
    }
}

__global__ __launch_bounds__(256, 3) void postop_biasC_2d_scalar(
    const float* __restrict__ x,
    const float* __restrict__ bias, // (C)
    float* __restrict__ out,
    int64_t total,
    int N, int C, int HW,
    float constant_value,
    float scale
) {
    // 1D fallback (still improved arithmetic form)
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float alpha = scale;
    for (int64_t i = tid; i < total; i += stride) {
        // i -> c = (i/HW) % C
        int64_t t = i / (int64_t)HW;
        int c = (int)(t - (t / (int64_t)C) * (int64_t)C);
        float beta = ldg_f32(bias + c) * alpha;
        out[i] = fminf_fast(x[i], constant_value) * alpha + beta;
    }
}

// --------- Host launcher ---------

torch::Tensor conv2d_min_add_multiply_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!bias.is_contiguous()) bias = bias.contiguous();

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int HW = H * W;

    const int bias_numel = (int)bias.numel();
    TORCH_CHECK(bias_numel == 1 || bias_numel == C,
                "bias must have numel 1 or C (supports shapes like (C), (C,1,1), (1,C,1,1))");

    auto out = torch::empty_like(x);

    const int64_t total = (int64_t)N * (int64_t)C * (int64_t)H * (int64_t)W;

    const float* xp = (const float*)x.data_ptr<float>();
    const float* bp = (const float*)bias.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    const float cval = (float)constant_value;
    const float scale = (float)scaling_factor;

    int dev = -1;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    const uintptr_t xaddr = (uintptr_t)xp;
    const uintptr_t oaddr = (uintptr_t)op;

    const int threads = 256;

    if (bias_numel == 1) {
        int max_blocks = sm_count > 0 ? sm_count * 20 : 4096;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;

        const bool can_vec4 = ((total & 3LL) == 0LL) && is_aligned_uint(xaddr, 16) && is_aligned_uint(oaddr, 16);
        if (can_vec4) {
            postop_bias1_vec4<<<blocks, threads>>>(xp, bp, op, total, cval, scale);
        } else {
            postop_bias1_scalar<<<blocks, threads>>>(xp, bp, op, total, cval, scale);
        }
        return out;
    }

    // bias_numel == C
    // Prefer specialized 2D kernel when HW divisible by 4 and addresses aligned.
    const bool can_vec4_hw = ((HW & 3) == 0) && is_aligned_uint(xaddr, 16) && is_aligned_uint(oaddr, 16);

    if (can_vec4_hw) {
        // 2D grid: x dimension tiles NHW in float4 units, y dimension channels
        const int hw4 = HW >> 2;
        const int64_t nhw4 = (int64_t)N * (int64_t)hw4;

        // ILP: each thread handles ILP float4s
        constexpr int ILP = 4;
        int64_t tile = (int64_t)threads * (int64_t)ILP;
        int grid_x = (int)((nhw4 + tile - 1) / tile);

        // cap grid_x to keep scheduling efficient
        int max_grid_x = sm_count > 0 ? sm_count * 8 : 4096;
        if (grid_x < 1) grid_x = 1;
        if (grid_x > max_grid_x) grid_x = max_grid_x;

        dim3 grid((unsigned)grid_x, (unsigned)C, 1);
        postop_biasC_2d_vec4_ilp<ILP><<<grid, threads>>>(xp, bp, op, N, C, HW, cval, scale);
    } else {
        int max_blocks = sm_count > 0 ? sm_count * 20 : 4096;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (blocks > max_blocks) blocks = max_blocks;
        postop_biasC_2d_scalar<<<blocks, threads>>>(xp, bp, op, total, N, C, HW, cval, scale);
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor conv2d_min_add_multiply_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_min_add_multiply_opt5_2d",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_min_add_multiply_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Convolution followed by fused post-processing:
      y = (min(conv(x), constant_value) + bias) * scaling_factor
    using an optimized custom CUDA kernel for the post-processing.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        return self.custom_ops_lib.conv2d_min_add_multiply_cuda(
            x, bias, self.constant_value, self.scaling_factor
        )

# Keep original input helpers for compatibility with the provided scaffold.
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]