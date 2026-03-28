import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FP32
#define CHECK_FP32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

// Fast sigmoid using fast exp. With --use_fast_math, __expf is fast.
__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

static inline int get_sm_count() {
    int dev = -1;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return prop.multiProcessorCount;
}

// Vectorized kernel with ILP unrolling: processes 2x float4 per iteration when possible.
__global__ __launch_bounds__(128, 4) void sigmoid_scale_residual_add_vec4x2_kernel(
    const float* __restrict__ Y,
    float* __restrict__ Out,
    int64_t N,
    float scaling_factor
) {
    const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    const int64_t N4 = N >> 2;               // float4 items
    const int64_t N8 = N4 >> 1;              // pairs of float4 (8 floats)
    const float4* __restrict__ y4p = reinterpret_cast<const float4*>(Y);
    float4* __restrict__ o4p = reinterpret_cast<float4*>(Out);

    // Main loop: each iteration handles 2 float4
    for (int64_t i8 = tid; i8 < N8; i8 += stride) {
        const int64_t base = i8 << 1; // *2 float4

        float4 a = y4p[base + 0];
        float4 b = y4p[base + 1];

        // a
        float sa0 = sigmoidf_fast(a.x);
        float sa1 = sigmoidf_fast(a.y);
        float sa2 = sigmoidf_fast(a.z);
        float sa3 = sigmoidf_fast(a.w);
        float4 oa;
        oa.x = a.x + scaling_factor * sa0;
        oa.y = a.y + scaling_factor * sa1;
        oa.z = a.z + scaling_factor * sa2;
        oa.w = a.w + scaling_factor * sa3;

        // b
        float sb0 = sigmoidf_fast(b.x);
        float sb1 = sigmoidf_fast(b.y);
        float sb2 = sigmoidf_fast(b.z);
        float sb3 = sigmoidf_fast(b.w);
        float4 ob;
        ob.x = b.x + scaling_factor * sb0;
        ob.y = b.y + scaling_factor * sb1;
        ob.z = b.z + scaling_factor * sb2;
        ob.w = b.w + scaling_factor * sb3;

        o4p[base + 0] = oa;
        o4p[base + 1] = ob;
    }

    // Tail for odd float4 count (still N%4==0 but N4 may be odd)
    if ((N4 & 1LL) != 0) {
        const int64_t last = N4 - 1;
        // Only first stride "lane" handles it to avoid overlap.
        for (int64_t i4 = tid; i4 == 0; i4 += stride) {
            float4 y4 = y4p[last];
            float s0 = sigmoidf_fast(y4.x);
            float s1 = sigmoidf_fast(y4.y);
            float s2 = sigmoidf_fast(y4.z);
            float s3 = sigmoidf_fast(y4.w);
            float4 o4;
            o4.x = y4.x + scaling_factor * s0;
            o4.y = y4.y + scaling_factor * s1;
            o4.z = y4.z + scaling_factor * s2;
            o4.w = y4.w + scaling_factor * s3;
            o4p[last] = o4;
        }
    }
}

// float4 (no x2 unroll) as secondary fast-path.
__global__ __launch_bounds__(128, 4) void sigmoid_scale_residual_add_vec4_kernel(
    const float* __restrict__ Y,
    float* __restrict__ Out,
    int64_t N,
    float scaling_factor
) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    int64_t N4 = N >> 2;
    const float4* __restrict__ y4p = reinterpret_cast<const float4*>(Y);
    float4* __restrict__ o4p = reinterpret_cast<float4*>(Out);

    for (int64_t i4 = tid; i4 < N4; i4 += stride) {
        float4 y4 = y4p[i4];
        float s0 = sigmoidf_fast(y4.x);
        float s1 = sigmoidf_fast(y4.y);
        float s2 = sigmoidf_fast(y4.z);
        float s3 = sigmoidf_fast(y4.w);

        float4 o4;
        o4.x = y4.x + scaling_factor * s0;
        o4.y = y4.y + scaling_factor * s1;
        o4.z = y4.z + scaling_factor * s2;
        o4.w = y4.w + scaling_factor * s3;

        o4p[i4] = o4;
    }
}

__global__ __launch_bounds__(256, 2) void sigmoid_scale_residual_add_scalar_kernel(
    const float* __restrict__ Y,
    float* __restrict__ Out,
    int64_t N,
    float scaling_factor
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    // Light unroll by 2 to increase ILP for scalar path
    for (int64_t i = idx; i < N; i += (stride << 1)) {
        float y0 = ldg_f32(Y + i);
        float s0 = sigmoidf_fast(y0);
        Out[i] = y0 + scaling_factor * s0;

        int64_t j = i + stride;
        if (j < N) {
            float y1 = ldg_f32(Y + j);
            float s1 = sigmoidf_fast(y1);
            Out[j] = y1 + scaling_factor * s1;
        }
    }
}

static inline void launch_fused(
    const float* y,
    float* out,
    int64_t N,
    float scaling_factor
) {
    const int sm = get_sm_count();

    // For this kernel (exp + memory), 128 threads often reduces reg pressure and improves occupancy.
    const int threads_vec = 128;
    const int threads_sca = 256;

    // Blocks: allow more to increase MLP, but cap to avoid overhead.
    const int max_blocks = sm * 12;

    auto stream = at::cuda::getDefaultCUDAStream();

    // Vectorized paths require 16B alignment and N%4==0.
    const uintptr_t y_addr = (uintptr_t)y;
    const uintptr_t o_addr = (uintptr_t)out;
    const bool aligned16 = ((y_addr & 0xF) == 0) && ((o_addr & 0xF) == 0);
    const bool n_div4 = ((N & 3LL) == 0);

    if (aligned16 && n_div4) {
        const int64_t N4 = N >> 2;
        // Prefer vec4x2 when there are enough items to benefit (and handle odd tail).
        const bool can_x2 = (N4 >= 2);
        if (can_x2) {
            int64_t blocks64 = ( (N4 >> 1) + threads_vec - 1 ) / threads_vec; // based on N8
            int blocks = (int)blocks64;
            if (blocks > max_blocks) blocks = max_blocks;
            if (blocks < 1) blocks = 1;
            sigmoid_scale_residual_add_vec4x2_kernel<<<blocks, threads_vec, 0, stream>>>(
                y, out, N, scaling_factor
            );
        } else {
            int64_t blocks64 = (N + threads_vec - 1) / threads_vec;
            int blocks = (int)blocks64;
            if (blocks > max_blocks) blocks = max_blocks;
            if (blocks < 1) blocks = 1;
            sigmoid_scale_residual_add_vec4_kernel<<<blocks, threads_vec, 0, stream>>>(
                y, out, N, scaling_factor
            );
        }
    } else {
        int64_t blocks64 = (N + threads_sca - 1) / threads_sca;
        int blocks = (int)blocks64;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;
        sigmoid_scale_residual_add_scalar_kernel<<<blocks, threads_sca, 0, stream>>>(
            y, out, N, scaling_factor
        );
    }
}

torch::Tensor gemm_sigmoid_scaling_residual_add_cuda(torch::Tensor y, double scaling_factor) {
    CHECK_CUDA(y);
    CHECK_FP32(y);
    CHECK_CONTIGUOUS(y);
    TORCH_CHECK(y.dim() == 2, "y must be 2D (batch_size, hidden_size)");

    auto out = torch::empty_like(y);
    launch_fused(
        (const float*)y.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        y.numel(),
        (float)scaling_factor
    );
    return out;
}

torch::Tensor gemm_sigmoid_scaling_residual_add_out_cuda(torch::Tensor y, torch::Tensor out, double scaling_factor) {
    CHECK_CUDA(y);
    CHECK_CUDA(out);
    CHECK_FP32(y);
    CHECK_FP32(out);
    CHECK_CONTIGUOUS(y);
    CHECK_CONTIGUOUS(out);
    TORCH_CHECK(y.sizes() == out.sizes(), "out must have same shape as y");
    TORCH_CHECK(y.dim() == 2, "y must be 2D (batch_size, hidden_size)");

    launch_fused(
        (const float*)y.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        y.numel(),
        (float)scaling_factor
    );
    return out;
}
"""

cpp_source = r"""
torch::Tensor gemm_sigmoid_scaling_residual_add_cuda(torch::Tensor y, double scaling_factor);
torch::Tensor gemm_sigmoid_scaling_residual_add_out_cuda(torch::Tensor y, torch::Tensor out, double scaling_factor);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_sigmoid_scale_resadd_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "gemm_sigmoid_scaling_residual_add_cuda",
        "gemm_sigmoid_scaling_residual_add_out_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps GEMM as nn.Linear (cuBLAS) and fuses sigmoid + scaling + residual add
    with a more latency-hiding CUDA kernel (vectorized float4x2 fast path).
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = float(scaling_factor)
        self.custom_ops_lib = custom_ops_lib
        self._out_buf = None  # reusable FP32 buffer (CUDA only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gemm(x)

        if not y.is_cuda:
            orig = y
            z = torch.sigmoid(y)
            z = z * self.scaling_factor
            z = z + orig
            return z

        orig_dtype = y.dtype
        if y.dtype != torch.float32:
            y_fp32 = y.float()
        else:
            y_fp32 = y
        if not y_fp32.is_contiguous():
            y_fp32 = y_fp32.contiguous()

        out_fp32 = None
        if self._out_buf is not None:
            if self._out_buf.is_cuda and self._out_buf.dtype == torch.float32 and self._out_buf.is_contiguous():
                if tuple(self._out_buf.shape) == tuple(y_fp32.shape):
                    out_fp32 = self._out_buf

        if out_fp32 is None:
            out_fp32 = torch.empty_like(y_fp32)
            self._out_buf = out_fp32

        out_fp32 = self.custom_ops_lib.gemm_sigmoid_scaling_residual_add_out_cuda(
            y_fp32, out_fp32, float(self.scaling_factor)
        )

        if out_fp32.dtype != orig_dtype:
            return out_fp32.to(dtype=orig_dtype)
        return out_fp32