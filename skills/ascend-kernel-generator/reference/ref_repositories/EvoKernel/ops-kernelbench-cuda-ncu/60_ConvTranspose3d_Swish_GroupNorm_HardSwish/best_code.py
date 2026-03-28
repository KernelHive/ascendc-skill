import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Fused CUDA op:
#   y = HardSwish( GroupNorm( Swish(x) ) )
# where:
#   Swish(x) = x * sigmoid(x)
#   GroupNorm over (N, G) groups across K = (C/G)*D*H*W
#   HardSwish(z) = z * clamp(z+3,0,6)/6
#
# Notes:
# - ConvTranspose3d is kept in PyTorch/cuDNN (do not reimplement).
# - This custom op fuses Swish + GroupNorm (affine) + HardSwish.
# - Strict contract: x must be contiguous CUDA 5D [N,C,D,H,W], float16/float32.
# - weight/bias are 1D [C] tensors (GroupNorm affine).
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// -------------------- helpers --------------------
__device__ __forceinline__ float sigmoid_f(float x) {
    // fast enough; compiled with --use_fast_math for __expf
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float swish_f(float x) {
    return x * sigmoid_f(x);
}

__device__ __forceinline__ float clamp_relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

__device__ __forceinline__ float hardswish_f(float x) {
    float t = clamp_relu6(x + 3.0f);
    return x * (t * (1.0f / 6.0f));
}

// Welford update for numerical stability
__device__ __forceinline__ void welford_update(float x, float &mean, float &m2, int &count) {
    count += 1;
    float delta = x - mean;
    mean += delta / count;
    float delta2 = x - mean;
    m2 += delta * delta2;
}

__device__ __forceinline__ void welford_combine(
    float mean_b, float m2_b, int count_b,
    float &mean_a, float &m2_a, int &count_a
) {
    if (count_b == 0) return;
    if (count_a == 0) {
        mean_a = mean_b; m2_a = m2_b; count_a = count_b; return;
    }
    float delta = mean_b - mean_a;
    int count = count_a + count_b;
    mean_a += delta * (float)count_b / (float)count;
    m2_a += m2_b + delta * delta * (float)count_a * (float)count_b / (float)count;
    count_a = count;
}

// -------------------- kernels --------------------
// Block computes stats for one (n,g). We use Welford across K elements.
// Then a second pass applies norm+affine+hswish.
template <typename scalar_t>
__global__ void gn_swish_hswish_forward_kernel(
    const scalar_t* __restrict__ x,          // [N,C,D,H,W]
    const scalar_t* __restrict__ gamma,      // [C]
    const scalar_t* __restrict__ beta,       // [C]
    scalar_t* __restrict__ y,                // [N,C,D,H,W]
    int N, int C, int D, int H, int W,
    int G, float eps
) {
    int ng = (int)blockIdx.x; // 0..N*G-1
    int n = ng / G;
    int g = ng - n * G;
    int Cg = C / G;

    int64_t HW = (int64_t)D * H * W;
    int64_t K = (int64_t)Cg * HW;

    // base offsets
    int c_start = g * Cg;
    int64_t base_n = (int64_t)n * C * HW;

    // pass 1: stats over swish(x)
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    // each thread processes a strided chunk
    for (int64_t k = (int64_t)threadIdx.x; k < K; k += (int64_t)blockDim.x) {
        int64_t c_in_g = k / HW;           // 0..Cg-1
        int64_t s = k - c_in_g * HW;       // 0..HW-1
        int c = c_start + (int)c_in_g;
        int64_t idx = base_n + (int64_t)c * HW + s;

        float xv;
        if constexpr (std::is_same<scalar_t, __half>::value) {
            xv = __half2float(reinterpret_cast<const __half*>(x)[idx]);
        } else {
            xv = (float)reinterpret_cast<const float*>(x)[idx];
        }
        float sv = swish_f(xv);
        welford_update(sv, mean, m2, count);
    }

    // reduce within block using shared memory
    __shared__ float s_mean[256];
    __shared__ float s_m2[256];
    __shared__ int s_count[256];

    int t = threadIdx.x;
    if (t < 256) { // we launch 256 threads
        s_mean[t] = mean;
        s_m2[t] = m2;
        s_count[t] = count;
    }
    __syncthreads();

    // tree reduction
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (t < offset) {
            float mean_b = s_mean[t + offset];
            float m2_b = s_m2[t + offset];
            int count_b = s_count[t + offset];
            float mean_a = s_mean[t];
            float m2_a = s_m2[t];
            int count_a = s_count[t];
            welford_combine(mean_b, m2_b, count_b, mean_a, m2_a, count_a);
            s_mean[t] = mean_a;
            s_m2[t] = m2_a;
            s_count[t] = count_a;
        }
        __syncthreads();
    }

    float mu = s_mean[0];
    float var = (s_count[0] > 1) ? (s_m2[0] / (float)s_count[0]) : 0.0f;
    float inv_std = rsqrtf(var + eps);

    // pass 2: apply GN affine + hardswish, write y
    for (int64_t k = (int64_t)threadIdx.x; k < K; k += (int64_t)blockDim.x) {
        int64_t c_in_g = k / HW;
        int64_t s = k - c_in_g * HW;
        int c = c_start + (int)c_in_g;
        int64_t idx = base_n + (int64_t)c * HW + s;

        float xv;
        if constexpr (std::is_same<scalar_t, __half>::value) {
            xv = __half2float(reinterpret_cast<const __half*>(x)[idx]);
        } else {
            xv = (float)reinterpret_cast<const float*>(x)[idx];
        }
        float sv = swish_f(xv);

        float gma, bta;
        if constexpr (std::is_same<scalar_t, __half>::value) {
            gma = __half2float(reinterpret_cast<const __half*>(gamma)[c]);
            bta = __half2float(reinterpret_cast<const __half*>(beta)[c]);
        } else {
            gma = (float)reinterpret_cast<const float*>(gamma)[c];
            bta = (float)reinterpret_cast<const float*>(beta)[c];
        }

        float norm = (sv - mu) * inv_std;
        float z = norm * gma + bta;
        float out = hardswish_f(z);

        if constexpr (std::is_same<scalar_t, __half>::value) {
            reinterpret_cast<__half*>(y)[idx] = __float2half_rn(out);
        } else {
            reinterpret_cast<float*>(y)[idx] = out;
        }
    }
}

torch::Tensor swish_groupnorm_hardswish_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t groups,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be [C]");
    TORCH_CHECK(x.size(1) == gamma.size(0) && x.size(1) == beta.size(0), "gamma/beta must match C");
    TORCH_CHECK(groups > 0, "groups must be > 0");
    TORCH_CHECK(x.size(1) % groups == 0, "C must be divisible by groups");
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type() && x.scalar_type() == beta.scalar_type(),
                "x/gamma/beta must have same dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "Only float32/float16 supported");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int D = (int)x.size(2);
    int H = (int)x.size(3);
    int W = (int)x.size(4);
    int G = (int)groups;

    auto y = torch::empty_like(x);

    int blocks = N * G;
    int threads = 256;

    // Limit blocks to CUDA grid limits; N*G is typically small.
    TORCH_CHECK(blocks > 0, "invalid blocks");

    if (x.scalar_type() == torch::kFloat32) {
        gn_swish_hswish_forward_kernel<float><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, D, H, W, G, (float)eps
        );
    } else {
        gn_swish_hswish_forward_kernel<__half><<<blocks, threads>>>(
            (const __half*)x.data_ptr<at::Half>(),
            (const __half*)gamma.data_ptr<at::Half>(),
            (const __half*)beta.data_ptr<at::Half>(),
            (__half*)y.data_ptr<at::Half>(),
            N, C, D, H, W, G, (float)eps
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor swish_groupnorm_hardswish_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t groups,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_swish_group_norm_hard_swish_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["swish_groupnorm_hardswish_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps ConvTranspose3d in PyTorch/cuDNN, replaces:
        Swish -> GroupNorm -> HardSwish
    with a single fused custom CUDA op.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.groups = int(groups)
        self.eps = float(eps)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x_c = x.contiguous()
        w = self.group_norm.weight.contiguous()
        b = self.group_norm.bias.contiguous()
        return self.custom_ops.swish_groupnorm_hardswish_cuda(x_c, w, b, self.groups, self.eps)