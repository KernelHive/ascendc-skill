import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused divide + LeakyReLU forward (safer + occupancy-tuned) ----

conv2d_divide_leaky_relu_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float leaky_relu_f32(float v, float ns) {
    return (v >= 0.0f) ? v : (v * ns);
}

// Keep kernels intentionally simple to reduce register pressure.
// Vector path requires n % 4 == 0; no tail handling => no races.
__global__ __launch_bounds__(256, 4)
void div_leaky_relu_fwd_vec4_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t n,
                                   float inv_divisor,
                                   float negative_slope) {
    const int64_t n4 = n >> 2;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        v.x *= inv_divisor; v.y *= inv_divisor; v.z *= inv_divisor; v.w *= inv_divisor;
        v.x = leaky_relu_f32(v.x, negative_slope);
        v.y = leaky_relu_f32(v.y, negative_slope);
        v.z = leaky_relu_f32(v.z, negative_slope);
        v.w = leaky_relu_f32(v.w, negative_slope);
        out4[i] = v;
    }
}

__global__ __launch_bounds__(256, 4)
void div_leaky_relu_fwd_scalar_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int64_t n,
                                      float inv_divisor,
                                      float negative_slope) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        float v = ldg_f32(x + idx) * inv_divisor;
        out[idx] = leaky_relu_f32(v, negative_slope);
    }
}

static inline int get_sm_count(int device) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    return sm_count;
}

template <typename KernelPtr>
static inline int occupancy_blocks_per_sm(KernelPtr kptr, int threads) {
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kptr, threads, 0);
    return blocks_per_sm > 0 ? blocks_per_sm : 1;
}

torch::Tensor div_leaky_relu_forward_cuda(torch::Tensor x, double divisor, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "div_leaky_relu_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "div_leaky_relu_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "div_leaky_relu_forward_cuda: x must be contiguous");
    TORCH_CHECK(divisor != 0.0, "div_leaky_relu_forward_cuda: divisor must be non-zero");

    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return out;

    const float inv_div = 1.0f / (float)divisor;
    const float ns = (float)negative_slope;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Try a couple threadblock sizes; pick by occupancy heuristic (blocks/SM),
    // then cap total blocks to avoid over-launching.
    const int device = x.get_device();
    const int sm = get_sm_count(device);

    int threads_candidates[2] = {256, 128};
    int best_threads = 256;
    int best_bpsm = 1;

    // Prefer vec4 kernel's occupancy since it's the fast path.
    for (int i = 0; i < 2; ++i) {
        int t = threads_candidates[i];
        int bpsm = occupancy_blocks_per_sm(div_leaky_relu_fwd_vec4_kernel, t);
        if (bpsm > best_bpsm) {
            best_bpsm = bpsm;
            best_threads = t;
        }
    }

    int blocks = sm * best_bpsm;
    // Modest cap to avoid the "more blocks is better" pitfall for elementwise kernels.
    if (blocks > sm * 8) blocks = sm * 8;
    if (blocks < 1) blocks = 1;

    const uintptr_t xptr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t optr = (uintptr_t)out.data_ptr<float>();
    const bool aligned16 = ((xptr | optr) & 0xF) == 0;

    // Safe vectorization gate: require exact multiple of 4 to eliminate any tail code.
    if (aligned16 && ((n & 3LL) == 0)) {
        div_leaky_relu_fwd_vec4_kernel<<<blocks, best_threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            n,
            inv_div,
            ns
        );
    } else {
        // Scalar handles any n safely.
        // Recompute occupancy for scalar kernel if threads changed (minor but consistent).
        int bpsm = occupancy_blocks_per_sm(div_leaky_relu_fwd_scalar_kernel, best_threads);
        int sblocks = sm * bpsm;
        if (sblocks > sm * 8) sblocks = sm * 8;
        if (sblocks < 1) sblocks = 1;

        div_leaky_relu_fwd_scalar_kernel<<<sblocks, best_threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            n,
            inv_div,
            ns
        );
    }

    return out;
}
"""

conv2d_divide_leaky_relu_cpp_source = r"""
torch::Tensor div_leaky_relu_forward_cuda(torch::Tensor x, double divisor, double negative_slope);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_divide_leaky_relu_opt_v4",
    cpp_sources=conv2d_divide_leaky_relu_cpp_source,
    cuda_sources=conv2d_divide_leaky_relu_cuda_source,
    functions=["div_leaky_relu_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "-lineinfo",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, then applies an optimized fused divide + LeakyReLU via custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.negative_slope = 0.01
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.div_leaky_relu_forward_cuda(x, self.divisor, self.negative_slope)


# Keep the same input helpers for compatibility with the original harness.
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]