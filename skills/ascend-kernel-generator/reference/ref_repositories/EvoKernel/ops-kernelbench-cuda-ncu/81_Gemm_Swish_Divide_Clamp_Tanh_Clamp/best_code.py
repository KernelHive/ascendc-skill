import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused (swish + divide + clamp + tanh + clamp) forward ----

gemm_swish_divide_clamp_tanh_clamp_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // With --use_fast_math, expf is fast; keep simple.
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swish_divide_clamp_tanh_clamp_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n,
    float clamp1_min,
    float clamp1_max,
    float clamp2_min,
    float clamp2_max
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        // Swish: x * sigmoid(x)
        v = v * sigmoidf_fast(v);
        // divide by 2
        v = v * 0.5f;
        // clamp [-1,1]
        v = clampf(v, clamp1_min, clamp1_max);
        // tanh
        v = tanhf(v);
        // clamp [-1,1]
        v = clampf(v, clamp2_min, clamp2_max);
        out[idx] = v;
    }
}

torch::Tensor swish_divide_clamp_tanh_clamp_forward_cuda(
    torch::Tensor x,
    double clamp1_min,
    double clamp1_max,
    double clamp2_min,
    double clamp2_max
) {
    TORCH_CHECK(x.is_cuda(), "swish_divide_clamp_tanh_clamp_forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "swish_divide_clamp_tanh_clamp_forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "swish_divide_clamp_tanh_clamp_forward_cuda: x must be contiguous");

    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    const dim3 blocks((unsigned int)blocks64);

    swish_divide_clamp_tanh_clamp_fwd_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n,
        (float)clamp1_min,
        (float)clamp1_max,
        (float)clamp2_min,
        (float)clamp2_max
    );

    return out;
}
"""

gemm_swish_divide_clamp_tanh_clamp_cpp_source = r"""
torch::Tensor swish_divide_clamp_tanh_clamp_forward_cuda(
    torch::Tensor x,
    double clamp1_min,
    double clamp1_max,
    double clamp2_min,
    double clamp2_max
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_swish_divide_clamp_tanh_clamp",
    cpp_sources=gemm_swish_divide_clamp_tanh_clamp_cpp_source,
    cuda_sources=gemm_swish_divide_clamp_tanh_clamp_cuda_source,
    functions=["swish_divide_clamp_tanh_clamp_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)

# ---- Model wrapper using the custom op ----

class ModelNew(nn.Module):
    """
    Model that performs GEMM (nn.Linear) followed by a fused custom CUDA kernel:
    swish -> divide(2) -> clamp[-1,1] -> tanh -> clamp[-1,1]
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.custom_ops_lib = custom_ops_lib
        self.clamp1_min = -1.0
        self.clamp1_max = 1.0
        self.clamp2_min = -1.0
        self.clamp2_max = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)  # cuBLAS GEMM
        if not x.is_contiguous():
            x = x.contiguous()
        # This fused kernel exactly matches: swish -> /2 -> clamp -> tanh -> clamp
        return self.custom_ops_lib.swish_divide_clamp_tanh_clamp_forward_cuda(
            x,
            float(self.clamp1_min),
            float(self.clamp1_max),
            float(self.clamp2_min),
            float(self.clamp2_max),
        )

# Keep the same input helpers for compatibility with the original harness.
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]