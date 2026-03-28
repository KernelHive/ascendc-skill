import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused (multiply + leaky_relu + gelu) forward ----

conv_mul_leaky_relu_gelu_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__)
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// Conservative constant-memory capacity for typical C up to 4096.
// If C larger, we fall back to global memory.
#ifndef CONST_MULT_MAX
#define CONST_MULT_MAX 4096
#endif

__constant__ float c_multiplier[CONST_MULT_MAX];

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    // GELU(x) ~= 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = kAlpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Mostly-branchless LeakyReLU: v = max(x,0) + slope*min(x,0)
__device__ __forceinline__ float leaky_relu_nb(float x, float negative_slope) {
    float pos = fmaxf(x, 0.0f);
    float neg = x - pos;              // <= 0
    return pos + negative_slope * neg;
}

__device__ __forceinline__ float act(float x, float m, float negative_slope) {
    float v = x * m;
    v = leaky_relu_nb(v, negative_slope);
    return gelu_tanh_approx(v);
}

template <bool UseConstMult, int UNROLL4>
__global__ __launch_bounds__(128, 3) void mul_leaky_relu_gelu_vec4_2d_kernel(
    const float4* __restrict__ x4,    // [planes * HW4]
    const float*  __restrict__ mult,  // [C] (used when !UseConstMult)
    float4* __restrict__ out4,        // [planes * HW4]
    int32_t HW4,
    int32_t C,
    int32_t planes,
    float negative_slope
) {
    // 2D grid:
    // blockIdx.y selects plane (N*C)
    // blockIdx.x selects tile within HW4
    int32_t plane = (int32_t)blockIdx.y;
    if (plane >= planes) return;

    int32_t c = plane - (plane / C) * C; // plane % C, faster than % on some arch
    float m = UseConstMult ? c_multiplier[c] : LDG(&mult[c]);

    int64_t base4 = (int64_t)plane * (int64_t)HW4;

    // Each block covers a contiguous segment: tile_start .. tile_start+tile_elems
    // Threads process float4 indices in a coalesced pattern; each thread handles UNROLL4 elements.
    int32_t tile_elems = (int32_t)blockDim.x * UNROLL4;
    int32_t tile_start = (int32_t)blockIdx.x * tile_elems;

    int32_t t = (int32_t)threadIdx.x;
    int32_t idx = tile_start + t;

    #pragma unroll
    for (int u = 0; u < UNROLL4; u++) {
        int32_t i4 = idx + u * (int32_t)blockDim.x;
        if (i4 < HW4) {
            float4 a = x4[base4 + (int64_t)i4];
            a.x = act(a.x, m, negative_slope);
            a.y = act(a.y, m, negative_slope);
            a.z = act(a.z, m, negative_slope);
            a.w = act(a.w, m, negative_slope);
            out4[base4 + (int64_t)i4] = a;
        }
    }
}

template <bool UseConstMult, int UNROLL>
__global__ __launch_bounds__(128, 3) void mul_leaky_relu_gelu_scalar_2d_kernel(
    const float* __restrict__ x,      // [planes * HW]
    const float* __restrict__ mult,   // [C] (used when !UseConstMult)
    float* __restrict__ out,          // [planes * HW]
    int32_t HW,
    int32_t C,
    int32_t planes,
    float negative_slope
) {
    int32_t plane = (int32_t)blockIdx.y;
    if (plane >= planes) return;

    int32_t c = plane - (plane / C) * C;
    float m = UseConstMult ? c_multiplier[c] : LDG(&mult[c]);

    int64_t base = (int64_t)plane * (int64_t)HW;

    int32_t tile_elems = (int32_t)blockDim.x * UNROLL;
    int32_t tile_start = (int32_t)blockIdx.x * tile_elems;

    int32_t t = (int32_t)threadIdx.x;
    int32_t idx = tile_start + t;

    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        int32_t i = idx + u * (int32_t)blockDim.x;
        if (i < HW) {
            float v = x[base + (int64_t)i];
            out[base + (int64_t)i] = act(v, m, negative_slope);
        }
    }
}

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

torch::Tensor mul_leaky_relu_gelu_forward_cuda(torch::Tensor x, torch::Tensor multiplier, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "mul_leaky_relu_gelu_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "mul_leaky_relu_gelu_forward_cuda: multiplier must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "mul_leaky_relu_gelu_forward_cuda: only float32 supported");
    TORCH_CHECK(multiplier.scalar_type() == torch::kFloat32, "mul_leaky_relu_gelu_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "mul_leaky_relu_gelu_forward_cuda: x must be contiguous (NCHW)");
    TORCH_CHECK(multiplier.is_contiguous(), "mul_leaky_relu_gelu_forward_cuda: multiplier must be contiguous");
    TORCH_CHECK(x.dim() == 4, "mul_leaky_relu_gelu_forward_cuda: x must be 4D NCHW");
    TORCH_CHECK(multiplier.dim() == 1, "mul_leaky_relu_gelu_forward_cuda: multiplier must be 1D of shape [C]");

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    TORCH_CHECK(C64 <= INT32_MAX, "mul_leaky_relu_gelu_forward_cuda: C too large");
    TORCH_CHECK(H64 * W64 <= INT32_MAX, "mul_leaky_relu_gelu_forward_cuda: HW too large");
    TORCH_CHECK(multiplier.size(0) == C64, "mul_leaky_relu_gelu_forward_cuda: multiplier.size(0) must equal x.size(1)");

    const int32_t C = (int32_t)C64;
    const int32_t HW = (int32_t)(H64 * W64);
    const int64_t planes64 = N64 * C64;
    TORCH_CHECK(planes64 <= INT32_MAX, "mul_leaky_relu_gelu_forward_cuda: N*C too large");
    const int32_t planes = (int32_t)planes64;

    auto out = torch::empty_like(x);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Optionally use constant memory for multiplier.
    // Correctness note: during training multiplier values can change each step even if storage pointer stays.
    // We conservatively refresh every call if multiplier.requires_grad() (training scenario).
    bool can_use_const = (C <= CONST_MULT_MAX);
    if (can_use_const) {
        // Refresh policy: always refresh if requires_grad, else cache by pointer.
        static void* last_ptr = nullptr;
        void* cur_ptr = (void*)multiplier.data_ptr<float>();
        bool refresh = multiplier.requires_grad() ? true : (cur_ptr != last_ptr);
        if (refresh) {
            cudaMemcpyToSymbolAsync(
                c_multiplier,
                multiplier.data_ptr<float>(),
                (size_t)C * sizeof(float),
                0,
                cudaMemcpyDeviceToDevice,
                stream
            );
            last_ptr = cur_ptr;
        }
    }

    // launch config tuned for lower registers + more blocks.
    const int threads = 128;

    // Prefer vec4 when aligned and HW divisible by 4
    const uintptr_t x_ptr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t o_ptr = (uintptr_t)out.data_ptr<float>();
    bool aligned16 = ((x_ptr | o_ptr) & 0xF) == 0;
    bool hw_div4 = ((HW & 3) == 0);

    // 2D grid:
    // y = planes
    // x = number of tiles over HW/HW4
    if (aligned16 && hw_div4) {
        const int32_t HW4 = HW >> 2;
        constexpr int UNROLL4 = 4; // each thread handles 4 float4 => 16 floats
        int tiles_x = ceil_div_int(HW4, threads * UNROLL4);
        dim3 block(threads);
        dim3 grid(tiles_x, planes, 1);

        if (can_use_const) {
            mul_leaky_relu_gelu_vec4_2d_kernel<true, UNROLL4><<<grid, block, 0, stream>>>(
                (const float4*)x.data_ptr<float>(),
                (const float*)multiplier.data_ptr<float>(),
                (float4*)out.data_ptr<float>(),
                HW4, C, planes, (float)negative_slope
            );
        } else {
            mul_leaky_relu_gelu_vec4_2d_kernel<false, UNROLL4><<<grid, block, 0, stream>>>(
                (const float4*)x.data_ptr<float>(),
                (const float*)multiplier.data_ptr<float>(),
                (float4*)out.data_ptr<float>(),
                HW4, C, planes, (float)negative_slope
            );
        }
    } else {
        constexpr int UNROLL = 8; // each thread handles 8 scalars
        int tiles_x = ceil_div_int(HW, threads * UNROLL);
        dim3 block(threads);
        dim3 grid(tiles_x, planes, 1);

        if (can_use_const) {
            mul_leaky_relu_gelu_scalar_2d_kernel<true, UNROLL><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)multiplier.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                HW, C, planes, (float)negative_slope
            );
        } else {
            mul_leaky_relu_gelu_scalar_2d_kernel<false, UNROLL><<<grid, block, 0, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)multiplier.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                HW, C, planes, (float)negative_slope
            );
        }
    }

    return out;
}
"""

conv_mul_leaky_relu_gelu_cpp_source = r"""
torch::Tensor mul_leaky_relu_gelu_forward_cuda(torch::Tensor x, torch::Tensor multiplier, double negative_slope);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_multiply_leaky_relu_gelu_v5",
    cpp_sources=conv_mul_leaky_relu_gelu_cpp_source,
    cuda_sources=conv_mul_leaky_relu_gelu_cuda_source,
    functions=["mul_leaky_relu_gelu_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs Conv2d (cuDNN) followed by a fused custom CUDA kernel:
    (multiply by per-channel learnable scalar) + LeakyReLU + GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        mult = self.multiplier.view(self.multiplier.shape[0]).contiguous()
        neg_slope = float(self.leaky_relu.negative_slope)
        return self.custom_ops_lib.mul_leaky_relu_gelu_forward_cuda(x, mult, neg_slope)


# Keep the same input helpers for compatibility with the original harness.
batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]