import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# -------------------- CUDA/C++ extension: fuse (Swish -> Multiply -> Swish) after GroupNorm --------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // With --use_fast_math, expf may map to __expf (toolchain-dependent)
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float swish(float x) {
    return x * sigmoidf_fast(x);
}

static inline __host__ __device__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

__global__ __launch_bounds__(256, 2)
void swish_mul_swish_fwd_scalar_kernel(
    const float* __restrict__ x,  // [B, C]
    const float* __restrict__ w,  // [C]
    float* __restrict__ out,      // [B, C]
    int B, int C)
{
    int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int row0 = (int)blockIdx.y;

    for (int row = row0; row < B; row += (int)gridDim.y) {
        if (c < C) {
            int64_t idx = (int64_t)row * (int64_t)C + (int64_t)c;
            float v = x[idx];
            float y = swish(v);
            y *= w[c];
            out[idx] = swish(y);
        }
    }
}

__global__ __launch_bounds__(256, 2)
void swish_mul_swish_fwd_vec4_kernel(
    const float* __restrict__ x,  // [B, C]
    const float* __restrict__ w,  // [C]
    float* __restrict__ out,      // [B, C]
    int B, int C)
{
    // Each thread handles one float4 (4 columns)
    int c4 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // 0..C4-1
    int row0 = (int)blockIdx.y;

    int C4 = C >> 2;
    if (c4 >= C4) return;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);

    // Load weights as scalar (w is [C], contiguous)
    int c_base = c4 << 2;
    float w0 = w[c_base + 0];
    float w1 = w[c_base + 1];
    float w2 = w[c_base + 2];
    float w3 = w[c_base + 3];

    for (int row = row0; row < B; row += (int)gridDim.y) {
        int64_t idx4 = (int64_t)row * (int64_t)C4 + (int64_t)c4;
        float4 xv = x4[idx4];

        float y0 = swish(xv.x) * w0;
        float y1 = swish(xv.y) * w1;
        float y2 = swish(xv.z) * w2;
        float y3 = swish(xv.w) * w3;

        float4 ov;
        ov.x = swish(y0);
        ov.y = swish(y1);
        ov.z = swish(y2);
        ov.w = swish(y3);

        o4[idx4] = ov;
    }
}

torch::Tensor swish_multiply_swish_forward_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "swish_multiply_swish_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "swish_multiply_swish_forward_cuda: w must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "swish_multiply_swish_forward_cuda: only float32 supported for x");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "swish_multiply_swish_forward_cuda: only float32 supported for w");
    TORCH_CHECK(x.is_contiguous(), "swish_multiply_swish_forward_cuda: x must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "swish_multiply_swish_forward_cuda: w must be contiguous");
    TORCH_CHECK(x.dim() == 2, "swish_multiply_swish_forward_cuda: x must be 2D [B, C]");
    TORCH_CHECK(w.dim() == 1, "swish_multiply_swish_forward_cuda: w must be 1D [C]");

    const int64_t B64 = x.size(0);
    const int64_t C64 = x.size(1);
    TORCH_CHECK(B64 <= INT32_MAX && C64 <= INT32_MAX, "B/C too large");
    const int B = (int)B64;
    const int C = (int)C64;
    TORCH_CHECK((int64_t)w.size(0) == (int64_t)C, "w.size(0) must equal x.size(1)");

    auto out = torch::empty_like(x);

    const int threads = 256;

    int grid_y = B;
    if (grid_y > 256) grid_y = 256;
    if (grid_y < 1) grid_y = 1;

    bool vec4_ok =
        ((C & 3) == 0) &&
        is_aligned_16(x.data_ptr()) &&
        is_aligned_16(out.data_ptr()) &&
        is_aligned_16(w.data_ptr());

    if (vec4_ok) {
        int C4 = C >> 2;
        int grid_x = (C4 + threads - 1) / threads;
        dim3 blocks((unsigned int)grid_x, (unsigned int)grid_y, 1);
        swish_mul_swish_fwd_vec4_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C
        );
    } else {
        int grid_x = (C + threads - 1) / threads;
        dim3 blocks((unsigned int)grid_x, (unsigned int)grid_y, 1);
        swish_mul_swish_fwd_scalar_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C
        );
    }

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor swish_multiply_swish_forward_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_group_norm_swish_multiply_swish_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["swish_multiply_swish_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized replacement for: gemm -> group_norm -> swish -> multiply -> swish

    - GEMM stays as nn.Linear (cuBLAS)
    - GroupNorm stays as nn.GroupNorm (PyTorch)
    - Post-GN elementwise chain fused into one CUDA kernel:
        y = swish(x); y *= w; out = swish(y)

    Fast path: CUDA + float32 + contiguous for both x and w.
    Otherwise: safe PyTorch fallback preserving dtype/autocast behavior.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = self.group_norm(x)

        w = self.multiply_weight

        # Fast CUDA path (match extension constraints; avoid forced dtype changes vs baseline)
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            if w.device != x.device:
                w = w.to(device=x.device)
            if w.dtype == torch.float32 and w.is_contiguous():
                return self.custom_ops_lib.swish_multiply_swish_forward_cuda(x, w)

        # Fallback (preserves dtype semantics)
        y = x * torch.sigmoid(x)
        y = y * w
        y = y * torch.sigmoid(y)
        return y