import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA post-ops after Conv3d:
# ReLU -> LeakyReLU -> GELU(tanh approx) -> Sigmoid -> BiasAdd
# Input/output: contiguous NCDHW float32
# Bias: [C,1,1,1] float32 (broadcast), staged in constant memory when C <= 4096
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Correct stream access (avoid prior build failure)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int BIAS_C_MAX = 4096;
__constant__ float c_bias[BIAS_C_MAX];

__device__ __forceinline__ float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }
__device__ __forceinline__ float leaky_relu_f32(float x, float neg_slope) { return x >= 0.0f ? x : x * neg_slope; }

// tanh-based GELU approximation
__device__ __forceinline__ float gelu_tanh_f32(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float postops_f32(float v, float neg_slope) {
    v = relu_f32(v);
    v = leaky_relu_f32(v, neg_slope);
    v = gelu_tanh_f32(v);
    v = sigmoid_f32(v);
    return v;
}

__device__ __forceinline__ float bias_read(int c, const float* __restrict__ bias_c, bool use_const) {
#if __CUDA_ARCH__ >= 350
    if (use_const) return c_bias[c];
    return __ldg(bias_c + c);
#else
    return use_const ? c_bias[c] : bias_c[c];
#endif
}

// Vectorized kernel: processes float4; computes channel once per float4 and only
// corrects per-lane if the float4 crosses an inner boundary (rare when inner%4==0).
__global__ void fused_vec4_f32(
    const float4* __restrict__ x4,
    const float* __restrict__ bias_c,
    float4* __restrict__ out4,
    int64_t n4,
    int64_t inner,
    int64_t C,
    float neg_slope,
    bool use_const_bias
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i4 = tid; i4 < n4; i4 += stride) {
        int64_t base = i4 * 4;

        // channel for first lane
        int64_t t0 = base / inner;              // = n*C + c
        int c0 = (int)(t0 - (t0 / C) * C);      // %C (1 div)

        // detect if this float4 stays within the same inner region
        int64_t inner_pos = base - t0 * inner;  // base % inner without extra div/mod
        bool same_c = (inner_pos <= (inner - 4));

        float4 v = x4[i4];

        if (same_c) {
            float b = bias_read(c0, bias_c, use_const_bias);
            v.x = postops_f32(v.x, neg_slope) + b;
            v.y = postops_f32(v.y, neg_slope) + b;
            v.z = postops_f32(v.z, neg_slope) + b;
            v.w = postops_f32(v.w, neg_slope) + b;
            out4[i4] = v;
        } else {
            // Rare boundary case: compute channel per lane (up to 3 more divs, but rare)
            int64_t t1 = (base + 1) / inner;
            int64_t t2 = (base + 2) / inner;
            int64_t t3 = (base + 3) / inner;
            int c1 = (int)(t1 - (t1 / C) * C);
            int c2 = (int)(t2 - (t2 / C) * C);
            int c3 = (int)(t3 - (t3 / C) * C);

            float b0 = bias_read(c0, bias_c, use_const_bias);
            float b1 = (c1 == c0) ? b0 : bias_read(c1, bias_c, use_const_bias);
            float b2 = (c2 == c0) ? b0 : (c2 == c1 ? b1 : bias_read(c2, bias_c, use_const_bias));
            float b3 = (c3 == c0) ? b0 : (c3 == c1 ? b1 : (c3 == c2 ? b2 : bias_read(c3, bias_c, use_const_bias)));

            v.x = postops_f32(v.x, neg_slope) + b0;
            v.y = postops_f32(v.y, neg_slope) + b1;
            v.z = postops_f32(v.z, neg_slope) + b2;
            v.w = postops_f32(v.w, neg_slope) + b3;
            out4[i4] = v;
        }
    }
}

// Scalar tail kernel (also used when vec4 not possible)
__global__ void fused_scalar_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias_c,
    float* __restrict__ out,
    int64_t n_elem,
    int64_t inner,
    int64_t C,
    float neg_slope,
    bool use_const_bias
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < n_elem; idx += stride) {
        int64_t t = idx / inner;          // n*C + c
        int c = (int)(t - (t / C) * C);   // %C
        float v = x[idx];
        v = postops_f32(v, neg_slope) + bias_read(c, bias_c, use_const_bias);
        out[idx] = v;
    }
}

torch::Tensor fused_conv3d_postops_cuda(torch::Tensor x, torch::Tensor bias, double neg_slope) {
    TORCH_CHECK(x.is_cuda(), "fused_conv3d_postops_cuda: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "fused_conv3d_postops_cuda: bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fused_conv3d_postops_cuda: only float32 supported");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "fused_conv3d_postops_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "fused_conv3d_postops_cuda: x must be contiguous (NCDHW)");
    TORCH_CHECK(bias.is_contiguous(), "fused_conv3d_postops_cuda: bias must be contiguous");

    TORCH_CHECK(x.dim() == 5, "fused_conv3d_postops_cuda: x must be 5D [N,C,D,H,W]");
    TORCH_CHECK(bias.dim() == 4, "fused_conv3d_postops_cuda: bias must be 4D [C,1,1,1]");
    int64_t C = x.size(1);
    TORCH_CHECK(bias.size(0) == C, "fused_conv3d_postops_cuda: bias.size(0) must match x.size(1)");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
                "fused_conv3d_postops_cuda: bias must have shape [C,1,1,1]");

    auto bias_c = bias.view({C});
    auto out = torch::empty_like(x);

    int64_t N = x.size(0);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);
    int64_t inner = D * H * W;
    int64_t n_elem = N * C * inner;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Constant memory bias staging with pointer-change caching (avoid recopy every call)
    bool use_const_bias = (C <= BIAS_C_MAX);
    if (use_const_bias) {
        // Per-device cache of last bias pointer copied
        static thread_local void* last_bias_ptr = nullptr;
        void* cur_ptr = (void*)bias_c.data_ptr<float>();
        if (cur_ptr != last_bias_ptr) {
            cudaMemcpyToSymbolAsync(c_bias, cur_ptr, (size_t)C * sizeof(float), 0,
                                    cudaMemcpyDeviceToDevice, stream);
            last_bias_ptr = cur_ptr;
        }
    }

    // Launch config
    const int threads = 256;
    int64_t blocks64 = (n_elem + threads - 1) / threads;
    int blocks = (blocks64 > (int64_t)2147483647) ? 2147483647 : (int)blocks64;

    // Vectorized path prerequisites:
    // - pointers 16B aligned
    // - n_elem divisible by 4
    // - inner divisible by 4 (so float4 groups rarely cross channel boundary; enables fast same_c path)
    uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    uintptr_t op = (uintptr_t)out.data_ptr<float>();
    bool aligned16 = ((xp | op) & 0xF) == 0;
    bool ndiv4 = ((n_elem & 3LL) == 0);
    bool inner_div4 = ((inner & 3LL) == 0);

    if (aligned16 && ndiv4 && inner_div4) {
        int64_t n4 = n_elem / 4;
        int64_t blocks4_64 = (n4 + threads - 1) / threads;
        int blocks4 = (blocks4_64 > (int64_t)2147483647) ? 2147483647 : (int)blocks4_64;
        fused_vec4_f32<<<blocks4, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            bias_c.data_ptr<float>(),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            n4, inner, C, (float)neg_slope, use_const_bias
        );
    } else {
        fused_scalar_f32<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            out.data_ptr<float>(),
            n_elem, inner, C, (float)neg_slope, use_const_bias
        );
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor fused_conv3d_postops_cuda(torch::Tensor x, torch::Tensor bias, double neg_slope);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_postops_opt3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv3d_postops_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d via PyTorch/cuDNN, then fused CUDA post-ops:
    ReLU -> LeakyReLU -> GELU -> Sigmoid -> BiasAdd.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.custom_ops_lib = custom_ops_lib
        self.neg_slope = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or x.dtype != torch.float32:
            x = torch.relu(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = F.gelu(x)
            x = torch.sigmoid(x)
            return x + self.bias

        if not x.is_contiguous():
            x = x.contiguous()

        bias = self.bias
        if not bias.is_cuda:
            bias = bias.to(device=x.device)
        if bias.dtype != torch.float32:
            bias = bias.float()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        if bias.dim() != 4 or bias.size(1) != 1 or bias.size(2) != 1 or bias.size(3) != 1:
            x = torch.relu(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = F.gelu(x)
            x = torch.sigmoid(x)
            return x + self.bias

        return self.custom_ops_lib.fused_conv3d_postops_cuda(x, bias, float(self.neg_slope))