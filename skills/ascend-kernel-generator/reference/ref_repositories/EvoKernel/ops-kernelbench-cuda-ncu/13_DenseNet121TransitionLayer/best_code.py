import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------
# Custom CUDA extensions
# -----------------------

cuda_src = r"""
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

__device__ __forceinline__ float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

template <typename T>
__device__ __forceinline__ T ldg_ptr(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(at::Half v) { return __half2float((__half)v); }

__device__ __forceinline__ void store_from_float(float* out, float v) { *out = v; }
__device__ __forceinline__ void store_from_float(at::Half* out, float v) { *out = (at::Half)__float2half_rn(v); }

// Pooled input for one ci at (n,ho,wo): 0.25*(relu(x00)+relu(x01)+relu(x10)+relu(x11))
template <typename scalar_t>
__device__ __forceinline__ float pooled_ci(
    const scalar_t* __restrict__ x,
    int64_t base, int W
) {
    float v00 = relu_f(to_float(ldg_ptr(x + base)));
    float v01 = relu_f(to_float(ldg_ptr(x + base + 1)));
    float v10 = relu_f(to_float(ldg_ptr(x + base + (int64_t)W)));
    float v11 = relu_f(to_float(ldg_ptr(x + base + (int64_t)W + 1)));
    return 0.25f * (v00 + v01 + v10 + v11);
}

// Generic scalar kernel (pool-before-conv) for float/half
template <typename scalar_t, int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void transition_poolconv_generic(
    const scalar_t* __restrict__ x,   // [N, Cin, H, W]
    const scalar_t* __restrict__ w,   // [Cout, Cin]
    scalar_t* __restrict__ y,         // [N, Cout, H2, W2]
    int N, int Cin, int H, int W, int Cout
) {
    int H2 = H >> 1;
    int W2 = W >> 1;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cout * H2 * W2;
    if (idx >= total) return;

    int wo = (int)(idx % W2);
    int64_t t1 = idx / W2;
    int ho = (int)(t1 % H2);
    int64_t t2 = t1 / H2;
    int co = (int)(t2 % Cout);
    int n  = (int)(t2 / Cout);

    int h0 = ho << 1;
    int w0 = wo << 1;

    float acc = 0.0f;

    // Pool-before-conv: one pooled scalar per ci, then dot with weights
    for (int ci = 0; ci < Cin; ++ci) {
        int64_t base = (((int64_t)n * Cin + ci) * H + h0) * W + w0;
        float p = pooled_ci<scalar_t>(x, base, W);
        float wf = to_float(ldg_ptr(w + (int64_t)co * Cin + ci));
        acc = fmaf(wf, p, acc);
    }

    store_from_float((scalar_t*)(y + idx), acc);
}

// Fast path: Cin==32, Cout even. Accumulate two output channels at once using float2/half2 weights.
// Each thread still computes one (n, co_pair, ho, wo) where co_pair = even channel index.
// Output is written to two channels (co, co+1).
template <typename scalar_t, int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void transition_poolconv_cin32_vec2(
    const scalar_t* __restrict__ x,   // [N, 32, H, W]
    const scalar_t* __restrict__ w,   // [Cout, 32]
    scalar_t* __restrict__ y,         // [N, Cout, H2, W2]
    int N, int H, int W, int Cout
) {
    int Cin = 32;
    int H2 = H >> 1;
    int W2 = W >> 1;

    // Total elements for co pairs: N * (Cout/2 rounded down) * H2 * W2
    int Cout2 = Cout >> 1; // number of pairs (ignore last odd channel if any, handled by generic fallback)
    int64_t idx2 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total2 = (int64_t)N * Cout2 * H2 * W2;
    if (idx2 >= total2) return;

    int wo = (int)(idx2 % W2);
    int64_t t1 = idx2 / W2;
    int ho = (int)(t1 % H2);
    int64_t t2 = t1 / H2;
    int cop = (int)(t2 % Cout2);  // pair index
    int n   = (int)(t2 / Cout2);

    int co0 = cop << 1; // even output channel
    int h0 = ho << 1;
    int w0 = wo << 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Unrolled Cin=32
#pragma unroll
    for (int ci = 0; ci < 32; ++ci) {
        int64_t base = (((int64_t)n * Cin + ci) * H + h0) * W + w0;
        float p = pooled_ci<scalar_t>(x, base, W);

        // Load weights for (co0,ci) and (co0+1,ci) as a vector when possible
        if constexpr (std::is_same<scalar_t, float>::value) {
            const float* wf0p = (const float*)(w + (int64_t)co0 * 32 + ci);
            const float* wf1p = (const float*)(w + (int64_t)(co0 + 1) * 32 + ci);
            float wf0 = ldg_ptr(wf0p);
            float wf1 = ldg_ptr(wf1p);
            acc0 = fmaf(wf0, p, acc0);
            acc1 = fmaf(wf1, p, acc1);
        } else {
            // half path: load as half, accumulate FP32
            const at::Half* wf0p = (const at::Half*)(w + (int64_t)co0 * 32 + ci);
            const at::Half* wf1p = (const at::Half*)(w + (int64_t)(co0 + 1) * 32 + ci);
            float wf0 = to_float(ldg_ptr(wf0p));
            float wf1 = to_float(ldg_ptr(wf1p));
            acc0 = fmaf(wf0, p, acc0);
            acc1 = fmaf(wf1, p, acc1);
        }
    }

    // Store two channels (coalescing is still good because threads walk contiguous idx2)
    int64_t out_base0 = (((int64_t)n * Cout + co0) * H2 + ho) * W2 + wo;
    store_from_float((scalar_t*)(y + out_base0), acc0);
    store_from_float((scalar_t*)(y + out_base0 + (int64_t)H2 * W2), acc1);
}

torch::Tensor transition_forward_cuda(torch::Tensor x, torch::Tensor w2d) {
    CHECK_CUDA(x);
    CHECK_CUDA(w2d);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w2d);

    TORCH_CHECK(x.is_floating_point(), "x must be floating point");
    TORCH_CHECK(w2d.is_floating_point(), "w must be floating point");
    TORCH_CHECK(x.scalar_type() == w2d.scalar_type(), "x and w must have same dtype");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half,
                "supported dtypes: float16/float32");

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w2d.dim() == 2, "w2d must be 2D [Cout, Cin]");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    TORCH_CHECK((H % 2) == 0 && (W % 2) == 0, "H and W must be divisible by 2");
    TORCH_CHECK((int)w2d.size(1) == Cin, "w2d Cin mismatch");
    int Cout = (int)w2d.size(0);

    int H2 = H / 2;
    int W2 = W / 2;

    auto y = torch::empty({N, Cout, H2, W2}, x.options());

    constexpr int THREADS = 256;

    // Prefer Cin==32 and even Cout fast path (common for the prompt).
    bool use_fast = (Cin == 32) && ((Cout & 1) == 0);

    if (use_fast) {
        int Cout2 = Cout >> 1;
        int64_t total2 = (int64_t)N * Cout2 * H2 * W2;
        int blocks = (int)((total2 + THREADS - 1) / THREADS);

        if (x.scalar_type() == at::ScalarType::Float) {
            transition_poolconv_cin32_vec2<float, THREADS><<<blocks, THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
                x.data_ptr<float>(),
                w2d.data_ptr<float>(),
                y.data_ptr<float>(),
                N, H, W, Cout
            );
        } else {
            transition_poolconv_cin32_vec2<at::Half, THREADS><<<blocks, THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
                (at::Half*)x.data_ptr<at::Half>(),
                (at::Half*)w2d.data_ptr<at::Half>(),
                (at::Half*)y.data_ptr<at::Half>(),
                N, H, W, Cout
            );
        }
        return y;
    }

    // Generic fallback
    int64_t total = (int64_t)N * Cout * H2 * W2;
    int blocks = (int)((total + THREADS - 1) / THREADS);

    if (x.scalar_type() == at::ScalarType::Float) {
        transition_poolconv_generic<float, THREADS><<<blocks, THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(),
            w2d.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cin, H, W, Cout
        );
    } else {
        transition_poolconv_generic<at::Half, THREADS><<<blocks, THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
            (at::Half*)x.data_ptr<at::Half>(),
            (at::Half*)w2d.data_ptr<at::Half>(),
            (at::Half*)y.data_ptr<at::Half>(),
            N, Cin, H, W, Cout
        );
    }

    return y;
}
"""

cpp_src = r"""
torch::Tensor transition_forward_cuda(torch::Tensor x, torch::Tensor w2d);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dense_net121_transition_layer_fused_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["transition_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -----------------------
# Model rewrite
# -----------------------

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv1x1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep BN in PyTorch for correct running-stat/affine semantics.
        x = self.bn(x)
        # Custom kernel expects contiguous NCHW and [Cout,Cin] weights.
        x = x.contiguous()
        w2d = self.conv1x1.weight.view(self.conv1x1.out_channels, self.conv1x1.in_channels).contiguous()
        y = custom_ops_lib.transition_forward_cuda(x, w2d)
        return y