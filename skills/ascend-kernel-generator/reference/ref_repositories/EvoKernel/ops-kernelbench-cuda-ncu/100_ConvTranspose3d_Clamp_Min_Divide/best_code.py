import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Custom CUDA op: clamp_min + divide (implemented as multiply by inv_div)
# Improvements vs baseline:
#   - Avoid unconditional contiguous() in out-of-place path (only if needed)
#   - Add modest ILP via loop unrolling (2x) for both scalar and vec4 kernels
#   - Occupancy-aware grid sizing (blocks per SM) + reasonable cap
#   - Keep single-kernel epilogue (no tail kernels, no extra launches)
#   - Keep vec4 path only when (aligned && n%4==0) to remain safe and simple
# ---------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

__device__ __forceinline__ float clamp_min_f(float v, float lo) {
    return v < lo ? lo : v;
}

static inline bool is_aligned_16_host(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// ------------------------------
// Vec4 kernels (n%4==0, aligned)
// Unroll by 2 float4 per iteration
// ------------------------------
__global__ void clamp_min_mul_inplace_f4_u2_kernel(
    float* __restrict__ x,
    int64_t n,
    float clamp_min,
    float mul
){
    int64_t n4 = n >> 2; // number of float4
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float4* x4 = reinterpret_cast<float4*>(x);

    for (int64_t i4 = tid; i4 < n4; i4 += (stride * 2)) {
        int64_t j4 = i4 + stride;

        float4 v0 = x4[i4];
        v0.x = clamp_min_f(v0.x, clamp_min) * mul;
        v0.y = clamp_min_f(v0.y, clamp_min) * mul;
        v0.z = clamp_min_f(v0.z, clamp_min) * mul;
        v0.w = clamp_min_f(v0.w, clamp_min) * mul;
        x4[i4] = v0;

        if (j4 < n4) {
            float4 v1 = x4[j4];
            v1.x = clamp_min_f(v1.x, clamp_min) * mul;
            v1.y = clamp_min_f(v1.y, clamp_min) * mul;
            v1.z = clamp_min_f(v1.z, clamp_min) * mul;
            v1.w = clamp_min_f(v1.w, clamp_min) * mul;
            x4[j4] = v1;
        }
    }
}

__global__ void clamp_min_mul_out_f4_u2_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n,
    float clamp_min,
    float mul
){
    int64_t n4 = n >> 2;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int64_t i4 = tid; i4 < n4; i4 += (stride * 2)) {
        int64_t j4 = i4 + stride;

        float4 v0 = x4[i4];
        float4 o0;
        o0.x = clamp_min_f(v0.x, clamp_min) * mul;
        o0.y = clamp_min_f(v0.y, clamp_min) * mul;
        o0.z = clamp_min_f(v0.z, clamp_min) * mul;
        o0.w = clamp_min_f(v0.w, clamp_min) * mul;
        y4[i4] = o0;

        if (j4 < n4) {
            float4 v1 = x4[j4];
            float4 o1;
            o1.x = clamp_min_f(v1.x, clamp_min) * mul;
            o1.y = clamp_min_f(v1.y, clamp_min) * mul;
            o1.z = clamp_min_f(v1.z, clamp_min) * mul;
            o1.w = clamp_min_f(v1.w, clamp_min) * mul;
            y4[j4] = o1;
        }
    }
}

// ------------------------------
// Scalar kernels (generic)
// Unroll by 4 scalars per iteration
// ------------------------------
__global__ void clamp_min_mul_inplace_f1_u4_kernel(
    float* __restrict__ x,
    int64_t n,
    float clamp_min,
    float mul
){
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t i = tid;
    // process 4 iterations per loop to increase ILP
    for (; i + 3 * stride < n; i += 4 * stride) {
        float v0 = x[i];
        float v1 = x[i + stride];
        float v2 = x[i + 2 * stride];
        float v3 = x[i + 3 * stride];

        v0 = clamp_min_f(v0, clamp_min) * mul;
        v1 = clamp_min_f(v1, clamp_min) * mul;
        v2 = clamp_min_f(v2, clamp_min) * mul;
        v3 = clamp_min_f(v3, clamp_min) * mul;

        x[i] = v0;
        x[i + stride] = v1;
        x[i + 2 * stride] = v2;
        x[i + 3 * stride] = v3;
    }
    for (; i < n; i += stride) {
        float v = x[i];
        x[i] = clamp_min_f(v, clamp_min) * mul;
    }
}

__global__ void clamp_min_mul_out_f1_u4_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n,
    float clamp_min,
    float mul
){
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t i = tid;
    for (; i + 3 * stride < n; i += 4 * stride) {
        float v0 = x[i];
        float v1 = x[i + stride];
        float v2 = x[i + 2 * stride];
        float v3 = x[i + 3 * stride];

        y[i] = clamp_min_f(v0, clamp_min) * mul;
        y[i + stride] = clamp_min_f(v1, clamp_min) * mul;
        y[i + 2 * stride] = clamp_min_f(v2, clamp_min) * mul;
        y[i + 3 * stride] = clamp_min_f(v3, clamp_min) * mul;
    }
    for (; i < n; i += stride) {
        float v = x[i];
        y[i] = clamp_min_f(v, clamp_min) * mul;
    }
}

static inline int pick_threads(int64_t n){
    // Bandwidth kernels usually like 256; use 128 for smaller to reduce overhead.
    if (n < (1<<16)) return 128;
    return 256;
}

static inline int pick_blocks_occupancy(int threads){
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int blocks_per_sm = 0;
    // Choose a representative kernel for occupancy; scalar is safe lower bound.
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        clamp_min_mul_out_f1_u4_kernel,
        threads,
        0
    ));
    // Ensure at least a couple blocks/SM for latency hiding; cap to avoid huge grids.
    if (blocks_per_sm < 2) blocks_per_sm = 2;
    if (blocks_per_sm > 8) blocks_per_sm = 8;

    int blocks = blocks_per_sm * prop.multiProcessorCount;
    if (blocks < 1) blocks = 1;
    return blocks;
}

static inline int clamp_grid(int blocks){
    // Reasonable cap for 1D grids to avoid launch overhead.
    const int cap = 65535;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    return blocks;
}

torch::Tensor clamp_min_divide_cuda(torch::Tensor x, double clamp_min_d, double divisor_d) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(divisor_d != 0.0, "divisor must be non-zero");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    // Avoid unconditional contiguous(): if x is contiguous, use directly.
    // If not, make a contiguous copy once (still better than always copying).
    torch::Tensor xc = x.is_contiguous() ? x : x.contiguous();
    auto y = torch::empty_like(xc);

    int64_t n = xc.numel();
    if (n == 0) return y;

    float clamp_min = (float)clamp_min_d;
    float mul = (float)(1.0 / divisor_d);

    int threads = pick_threads(n);
    int blocks = pick_blocks_occupancy(threads);

    // For very large tensors, ensure enough CTAs to cover, but don't explode.
    int64_t need = (n + threads - 1) / threads;
    if (need > blocks) blocks = (int)need;
    blocks = clamp_grid(blocks);

    bool vec_ok = (n % 4 == 0) &&
                  is_aligned_16_host(xc.data_ptr<float>()) &&
                  is_aligned_16_host(y.data_ptr<float>());

    if (vec_ok) {
        clamp_min_mul_out_f4_u2_kernel<<<blocks, threads, 0, stream>>>(
            xc.data_ptr<float>(),
            y.data_ptr<float>(),
            n,
            clamp_min,
            mul
        );
    } else {
        clamp_min_mul_out_f1_u4_kernel<<<blocks, threads, 0, stream>>>(
            xc.data_ptr<float>(),
            y.data_ptr<float>(),
            n,
            clamp_min,
            mul
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor clamp_min_divide_inplace_cuda(torch::Tensor x, double clamp_min_d, double divisor_d) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(divisor_d != 0.0, "divisor must be non-zero");
    TORCH_CHECK(x.is_contiguous(), "inplace expects contiguous tensor");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    int64_t n = x.numel();
    if (n == 0) return x;

    float clamp_min = (float)clamp_min_d;
    float mul = (float)(1.0 / divisor_d);

    int threads = pick_threads(n);
    int blocks = pick_blocks_occupancy(threads);
    int64_t need = (n + threads - 1) / threads;
    if (need > blocks) blocks = (int)need;
    blocks = clamp_grid(blocks);

    bool vec_ok = (n % 4 == 0) && is_aligned_16_host(x.data_ptr<float>());

    if (vec_ok) {
        clamp_min_mul_inplace_f4_u2_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            n,
            clamp_min,
            mul
        );
    } else {
        clamp_min_mul_inplace_f1_u4_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            n,
            clamp_min,
            mul
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor clamp_min_divide_cuda(torch::Tensor x, double clamp_min_d, double divisor_d);
torch::Tensor clamp_min_divide_inplace_cuda(torch::Tensor x, double clamp_min_d, double divisor_d);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_clampmin_div_v5_ilp_occ_nocontig",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["clamp_min_divide_cuda", "clamp_min_divide_inplace_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) -> fused CUDA post-op: clamp(min=min_value) + divide(divisor)

    Improvements:
      - Keep in-place epilogue when conv output is contiguous float32 CUDA
      - Out-of-place epilogue avoids unconditional contiguous() copies
      - CUDA epilogue uses modest ILP unrolling + occupancy-aware grid sizing
      - Vec4 path only when fully safe (aligned && numel%4==0), no tail kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=True,
        )
        self.min_value = float(min_value)
        self.divisor = float(divisor)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_transpose(x)

        # In-place is safe because y is newly produced; require contiguous float32 CUDA.
        if y.is_cuda and y.dtype == torch.float32 and y.is_contiguous():
            return self.custom_ops.clamp_min_divide_inplace_cuda(y, self.min_value, self.divisor)

        # Otherwise use out-of-place; kernel will avoid contig() unless needed.
        return self.custom_ops.clamp_min_divide_cuda(y, self.min_value, self.divisor)