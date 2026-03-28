import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused scaling + hardtanh + gelu forward (optimized) ----

gemm_scaling_hardtanh_gelu_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// Fast erf approximation (Abramowitz-Stegun 7.1.26 / common CUDA-friendly poly).
// Max error ~1e-4..1e-3 depending on range; suitable for GELU approximation in inference.
__device__ __forceinline__ float erf_approx(float x) {
    // Save the sign of x
    float s = copysignf(1.0f, x);
    x = fabsf(x);

    // Coefficients
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p  = 0.3275911f;

    float t = 1.0f / (1.0f + p * x);
    // Horner polynomial
    float y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    // exp(-x*x)
    float e = __expf(-x * x);
    float r = 1.0f - y * e;
    return s * r;
}

__device__ __forceinline__ float gelu_erf_approx(float x) {
    // GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
    const float inv_sqrt2 = 0.70710678118654752440f;
    float u = x * inv_sqrt2;
    float e = erf_approx(u);
    return 0.5f * x * (1.0f + e);
}

__global__ void scaling_hardtanh_gelu_fwd_kernel_i32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        v = fmaf(v, scaling_factor, 0.0f);          // v *= scaling_factor (FMA form)
        v = clampf(v, hardtanh_min, hardtanh_max);  // hardtanh
        out[idx] = gelu_erf_approx(v);              // gelu approx (erf-based)
    }
}

torch::Tensor scaling_hardtanh_gelu_forward_cuda_out(
    torch::Tensor x,
    torch::Tensor out,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max
) {
    TORCH_CHECK(x.is_cuda(), "scaling_hardtanh_gelu_forward_cuda_out: x must be CUDA");
    TORCH_CHECK(out.is_cuda(), "scaling_hardtanh_gelu_forward_cuda_out: out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.numel() == out.numel(), "x and out must have same numel");

    int64_t n64 = x.numel();
    TORCH_CHECK(n64 <= (int64_t)INT32_MAX, "tensor too large for i32 kernel");
    int n = (int)n64;

    // Keep launch simple; avoid per-call device queries.
    // 128 threads tends to reduce register pressure and increases residency for latency hiding.
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    scaling_hardtanh_gelu_fwd_kernel_i32<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n,
        (float)scaling_factor,
        (float)hardtanh_min,
        (float)hardtanh_max
    );

    return out;
}

torch::Tensor scaling_hardtanh_gelu_forward_cuda(
    torch::Tensor x,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max
) {
    TORCH_CHECK(x.is_cuda(), "scaling_hardtanh_gelu_forward_cuda: x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    auto out = torch::empty_like(x);
    return scaling_hardtanh_gelu_forward_cuda_out(x, out, scaling_factor, hardtanh_min, hardtanh_max);
}
"""

gemm_scaling_hardtanh_gelu_cpp_source = r"""
torch::Tensor scaling_hardtanh_gelu_forward_cuda(torch::Tensor x, double scaling_factor, double hardtanh_min, double hardtanh_max);
torch::Tensor scaling_hardtanh_gelu_forward_cuda_out(torch::Tensor x, torch::Tensor out, double scaling_factor, double hardtanh_min, double hardtanh_max);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_scaling_hardtanh_gelu_opt3",
    cpp_sources=gemm_scaling_hardtanh_gelu_cpp_source,
    cuda_sources=gemm_scaling_hardtanh_gelu_cuda_source,
    functions=[
        "scaling_hardtanh_gelu_forward_cuda",
        "scaling_hardtanh_gelu_forward_cuda_out",
    ],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)

# ---- Model wrapper using the custom op ----

class ModelNew(nn.Module):
    """
    Model that performs GEMM (nn.Linear) followed by a fused (scaling + HardTanh + GELU) custom CUDA kernel.
    Includes an optional persistent output buffer to reduce allocation overhead.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.custom_ops_lib = custom_ops_lib
        self._out_buf = None  # persistent buffer (inference-friendly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)  # cuBLAS GEMM
        if not x.is_contiguous():
            x = x.contiguous()

        # Persistent output buffer reuse (avoids empty_like allocation on hot path).
        # Safe because we shape-check every call.
        if self._out_buf is None or self._out_buf.numel() != x.numel() or self._out_buf.shape != x.shape or self._out_buf.device != x.device:
            self._out_buf = torch.empty_like(x)

        return self.custom_ops_lib.scaling_hardtanh_gelu_forward_cuda_out(
            x, self._out_buf, self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

# Keep the same input helpers for compatibility with the original harness.
batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]