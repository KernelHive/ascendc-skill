import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized RMSNorm over dim=1 for contiguous NCHW float32 CUDA tensors.
# Mapping: blocks cover (n,h) x w-tiles, threads cover contiguous w (float4 when possible).
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int C_UNROLL>  // if C_UNROLL>0, C must equal C_UNROLL and loop is unrolled
__global__ void rms_norm_nchw_wcontig_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps
) {
    int nh = (int)blockIdx.x; // 0 .. N*H-1
    int n = nh / H;
    int h = nh - n * H;

    // tile in W
    int tile = (int)blockIdx.y;

    // vectorized width: 4 floats per thread element
    constexpr int VEC = 4;
    int tid = (int)threadIdx.x;

    // each block covers VEC*blockDim.x elements in W
    int w0 = (tile * blockDim.x + tid) * VEC;

    if (w0 >= W) return;

    int stride_c = H * W;
    // base for (n,0,h,0)
    int base_nh0 = ((n * C) * H + h) * W;

    // For each of the 4 lanes, accumulate sumsq over C
    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;

    bool full_vec = (w0 + 3) < W;

    if (full_vec) {
        if constexpr (C_UNROLL > 0) {
#pragma unroll
            for (int c = 0; c < C_UNROLL; c++) {
                const float* ptr = x + base_nh0 + c * stride_c + w0;
                float4 v = *reinterpret_cast<const float4*>(ptr);
                sum0 = fmaf(v.x, v.x, sum0);
                sum1 = fmaf(v.y, v.y, sum1);
                sum2 = fmaf(v.z, v.z, sum2);
                sum3 = fmaf(v.w, v.w, sum3);
            }
        } else {
            for (int c = 0; c < C; c++) {
                const float* ptr = x + base_nh0 + c * stride_c + w0;
                float4 v = *reinterpret_cast<const float4*>(ptr);
                sum0 = fmaf(v.x, v.x, sum0);
                sum1 = fmaf(v.y, v.y, sum1);
                sum2 = fmaf(v.z, v.z, sum2);
                sum3 = fmaf(v.w, v.w, sum3);
            }
        }

        float inv0 = rsqrtf(sum0 * (1.0f / (float)C) + eps);
        float inv1 = rsqrtf(sum1 * (1.0f / (float)C) + eps);
        float inv2 = rsqrtf(sum2 * (1.0f / (float)C) + eps);
        float inv3 = rsqrtf(sum3 * (1.0f / (float)C) + eps);

        if constexpr (C_UNROLL > 0) {
#pragma unroll
            for (int c = 0; c < C_UNROLL; c++) {
                const float* px = x + base_nh0 + c * stride_c + w0;
                float* py = y + base_nh0 + c * stride_c + w0;
                float4 v = *reinterpret_cast<const float4*>(px);
                v.x *= inv0; v.y *= inv1; v.z *= inv2; v.w *= inv3;
                *reinterpret_cast<float4*>(py) = v;
            }
        } else {
            for (int c = 0; c < C; c++) {
                const float* px = x + base_nh0 + c * stride_c + w0;
                float* py = y + base_nh0 + c * stride_c + w0;
                float4 v = *reinterpret_cast<const float4*>(px);
                v.x *= inv0; v.y *= inv1; v.z *= inv2; v.w *= inv3;
                *reinterpret_cast<float4*>(py) = v;
            }
        }
    } else {
        // Scalar tail for up to 4 elements
        float sums[4] = {0.f, 0.f, 0.f, 0.f};
        int valid = W - w0;
        if (valid > 4) valid = 4;

        if constexpr (C_UNROLL > 0) {
#pragma unroll
            for (int c = 0; c < C_UNROLL; c++) {
                const float* ptr = x + base_nh0 + c * stride_c + w0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < valid) {
                        float v = ptr[i];
                        sums[i] = fmaf(v, v, sums[i]);
                    }
                }
            }
        } else {
            for (int c = 0; c < C; c++) {
                const float* ptr = x + base_nh0 + c * stride_c + w0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < valid) {
                        float v = ptr[i];
                        sums[i] = fmaf(v, v, sums[i]);
                    }
                }
            }
        }

        float inv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (i < valid) inv[i] = rsqrtf(sums[i] * (1.0f / (float)C) + eps);
        }

        if constexpr (C_UNROLL > 0) {
#pragma unroll
            for (int c = 0; c < C_UNROLL; c++) {
                const float* px = x + base_nh0 + c * stride_c + w0;
                float* py = y + base_nh0 + c * stride_c + w0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < valid) py[i] = px[i] * inv[i];
                }
            }
        } else {
            for (int c = 0; c < C; c++) {
                const float* px = x + base_nh0 + c * stride_c + w0;
                float* py = y + base_nh0 + c * stride_c + w0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (i < valid) py[i] = px[i] * inv[i];
                }
            }
        }
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "rms_norm_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rms_norm_cuda: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "rms_norm_cuda: x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "rms_norm_cuda: expected x to have shape (N, C, H, W)");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto y = torch::empty_like(x);

    // blocks: (N*H) rows and tiles of W
    // Each block covers blockDim.x*4 w-elements.
    int threads = 256;  // good balance; each thread handles 4 w's
    int elems_per_block = threads * 4;
    int tiles_w = (W + elems_per_block - 1) / elems_per_block;

    dim3 block((unsigned)threads, 1, 1);
    dim3 grid((unsigned)(N * H), (unsigned)tiles_w, 1);

    const float feps = (float)eps;

    // Specialize for common C=64 to enable unrolling and reduce loop overhead.
    if (C == 64) {
        rms_norm_nchw_wcontig_f32_kernel<64><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W, feps
        );
    } else {
        rms_norm_nchw_wcontig_f32_kernel<0><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W, feps
        );
    }

    return y;
}
"""

cpp_src = r"""
torch::Tensor rms_norm_cuda(torch::Tensor x, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_rms_norm_opt3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["rms_norm_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    RMSNorm using an optimized custom CUDA kernel over dim=1 (channels/features),
    optimized for contiguous NCHW float32 by streaming along contiguous W with float4 vectorization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.rms_norm_cuda(x, float(self.eps))