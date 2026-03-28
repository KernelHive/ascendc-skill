import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused InstanceNorm2d forward (no affine, no running stats) + divide
# - Two-pass: (sum, sumsq) then normalize
# - float4 vectorized loads/stores when 16B-aligned
# - CUB BlockReduce for reductions (sum, sumsq)
# - Avoids unsupported stream APIs; uses default stream launch like the reference
# ------------------------------------------------------------

fused_inorm_div_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cub/block/block_reduce.cuh>

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void instancenorm2d_fwd_div_sum_sumsq_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps,
    float inv_divide_by
) {
    int nc = (int)blockIdx.x; // one block per (n,c)
    int n = nc / C;
    int c = nc - n * C;
    int HW = H * W;
    int base = ((n * C + c) * HW);

    const float* __restrict__ xb = x + base;
    float* __restrict__ yb = y + base;

    // Vectorization eligibility
    uintptr_t addr = (uintptr_t)xb;
    bool aligned16 = ((addr & 0xF) == 0);

    float sum = 0.0f;
    float sumsq = 0.0f;

    if (aligned16) {
        int hw4 = HW >> 2;
        int tail = HW & 3;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);

        for (int i4 = (int)threadIdx.x; i4 < hw4; i4 += BLOCK_THREADS) {
            float4 v = x4[i4];
            sum   += (v.x + v.y) + (v.z + v.w);
            sumsq = fmaf(v.x, v.x, sumsq);
            sumsq = fmaf(v.y, v.y, sumsq);
            sumsq = fmaf(v.z, v.z, sumsq);
            sumsq = fmaf(v.w, v.w, sumsq);
        }
        if (tail) {
            int start = hw4 * 4;
            for (int t = (int)threadIdx.x; t < tail; t += BLOCK_THREADS) {
                float v = xb[start + t];
                sum += v;
                sumsq = fmaf(v, v, sumsq);
            }
        }
    } else {
        for (int i = (int)threadIdx.x; i < HW; i += BLOCK_THREADS) {
            float v = xb[i];
            sum += v;
            sumsq = fmaf(v, v, sumsq);
        }
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_sum;
    __shared__ typename BlockReduce::TempStorage temp_sumsq;

    float bsum = BlockReduce(temp_sum).Sum(sum);
    float bsumsq = BlockReduce(temp_sumsq).Sum(sumsq);

    __shared__ float s_mean;
    __shared__ float s_invstd;

    if (threadIdx.x == 0) {
        float inv_denom = 1.0f / (float)HW;
        float mean = bsum * inv_denom;
        float ex2  = bsumsq * inv_denom;
        float var = ex2 - mean * mean;
        var = var < 0.0f ? 0.0f : var;
        s_mean = mean;
        s_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float invstd = s_invstd;
    float scale = invstd * inv_divide_by;

    // Pass 2: normalize + divide (fused) + store
    if (aligned16) {
        int hw4 = HW >> 2;
        int tail = HW & 3;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xb);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(yb);

        for (int i4 = (int)threadIdx.x; i4 < hw4; i4 += BLOCK_THREADS) {
            float4 v = x4[i4];
            float4 o;
            o.x = (v.x - mean) * scale;
            o.y = (v.y - mean) * scale;
            o.z = (v.z - mean) * scale;
            o.w = (v.w - mean) * scale;
            y4[i4] = o;
        }
        if (tail) {
            int start = hw4 * 4;
            for (int t = (int)threadIdx.x; t < tail; t += BLOCK_THREADS) {
                float v = xb[start + t];
                yb[start + t] = (v - mean) * scale;
            }
        }
    } else {
        for (int i = (int)threadIdx.x; i < HW; i += BLOCK_THREADS) {
            float v = xb[i];
            yb[i] = (v - mean) * scale;
        }
    }
}

torch::Tensor instancenorm2d_forward_div_cuda(torch::Tensor x, double eps, double divide_by) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW contiguous)");
    TORCH_CHECK(divide_by != 0.0, "divide_by must be non-zero");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto y = torch::empty_like(x);

    constexpr int THREADS = 256;
    const int blocks = N * C;

    float inv_div = (float)(1.0 / divide_by);

    instancenorm2d_fwd_div_sum_sumsq_kernel<THREADS><<<blocks, THREADS>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, C, H, W,
        (float)eps,
        inv_div
    );

    return y;
}
"""

fused_inorm_div_cpp_decl = r"""
torch::Tensor instancenorm2d_forward_div_cuda(torch::Tensor x, double eps, double divide_by);
"""

custom_ops_lib = load_inline(
    name="custom_conv2d_inorm_div_ops_v1",
    cpp_sources=fused_inorm_div_cpp_decl,
    cuda_sources=fused_inorm_div_cuda_src,
    functions=["instancenorm2d_forward_div_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Conv2d -> (fused InstanceNorm2d + divide) using a custom CUDA kernel.

    Fast path contract:
      - input after conv is CUDA float32 contiguous NCHW
      - InstanceNorm2d uses affine=False, track_running_stats=False
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by, eps: float = 1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Match the intended fast-path: no affine / no running stats
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        self.divide_by = float(divide_by)
        self.eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.is_contiguous()
            and (self.instance_norm.affine is False)
            and (self.instance_norm.track_running_stats is False)
        ):
            return self.custom_ops_lib.instancenorm2d_forward_div_cuda(x, self.eps, self.divide_by)

        # Conservative fallback
        x = self.instance_norm(x)
        return x / self.divide_by