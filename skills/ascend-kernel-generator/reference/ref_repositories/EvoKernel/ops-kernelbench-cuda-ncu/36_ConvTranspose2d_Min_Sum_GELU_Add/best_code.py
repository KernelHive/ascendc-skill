import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA op (post ConvTranspose2d):
#   y[n,0,0,w] = gelu( sum_{h=0..H-1} min_{c=0..C-1} x[n,c,h,w] ) + bias
#
# Input:  x [N,C,H,W] float32 CUDA contiguous (NCHW)
# Bias:   bias numel==1 float32 CUDA
# Output: y [N,1,1,W] float32 CUDA contiguous
#
# Optimization vs baseline:
# - Each block owns (n, w_tile) where w_tile is 32-wide (or 32-wide of float indices).
# - Warps iterate h in a strided loop; for each h they compute min over C for their lane's w.
# - Reduce partial sums across warps in shared memory (no global atomics).
# - Optional float4 path for loads when *strictly* aligned and W%4==0.
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float gelu_approx(float x) {
    // tanh-based approximation, fast-math enabled by compile flags
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    float t = kAlpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

// Reduce sum across lanes in a warp (not used for final per-lane outputs, but handy)
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<bool VEC4>
__global__ __launch_bounds__(256, 3) void min_sum_gelu_add_tiled_kernel(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ bias,// scalar
    float* __restrict__ y,         // [N,1,1,W]
    int N, int C, int H, int W
) {
    // Block mapping:
    // grid.x = w_tile index (tile width is 32 scalars or 32 scalars even in vec4 mode; vec4 only changes how we load)
    // grid.y = n
    const int n = (int)blockIdx.y;
    if (n >= N) return;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;         // 0..7
    const int lane = tid & 31;         // 0..31
    const int warps_per_block = 8;

    const int w_tile = (int)blockIdx.x;
    const int w_base = w_tile * 32;
    const int w = w_base + lane;
    if (w >= W) return;

    // NCHW strides
    const int64_t stride_n = (int64_t)C * (int64_t)H * (int64_t)W;
    const int64_t stride_c = (int64_t)H * (int64_t)W;
    const int64_t stride_h = (int64_t)W;

    // Accumulate sum over H of min over C for this lane's w.
    float partial_sum = 0.0f;

    // Each warp processes h = warp, warp+warps_per_block, ...
    // This gives ILP across warps and improves cache behavior by keeping all warps on same w tile.
    for (int h = warp; h < H; h += warps_per_block) {
        float vmin = INFINITY;
        int64_t base_nhw = (int64_t)n * stride_n + (int64_t)h * stride_h + (int64_t)w;

        if constexpr (VEC4) {
            // vec4 path: load x as float4 along w, but each lane corresponds to one scalar.
            // We'll have each group of 4 lanes share one float4 load by reloading, but we keep it simple:
            // just do scalar loads here unless we can guarantee 16B alignment per (n,h,c,w_base) and w is in-bounds.
            // Since VEC4 is only enabled when w_base is 16B aligned and W%4==0, we can safely load via float4
            // for lanes that are at positions (0,4,8,...,28) and broadcast within 4-lane group.
            // This reduces memory transactions while keeping correctness.
            const int lane4 = lane & ~3;       // 0,4,8,...
            const int sub = lane & 3;          // 0..3
            const int w4 = w_base + lane4;
            if (w4 + 3 < W) {
                for (int c = 0; c < C; ++c) {
                    const float4* p4 = reinterpret_cast<const float4*>(
                        x + (int64_t)n * stride_n + (int64_t)c * stride_c + (int64_t)h * stride_h + (int64_t)w4
                    );
                    float4 vv = *p4; // aligned by gating
                    float val = (sub == 0 ? vv.x : (sub == 1 ? vv.y : (sub == 2 ? vv.z : vv.w)));
                    vmin = fminf(vmin, val);
                }
            } else {
                // tail fallback for last partial tile (should not happen when W is multiple of 4 and w is checked, but safe)
                for (int c = 0; c < C; ++c) {
                    float v = ldg_f32(x + base_nhw + (int64_t)c * stride_c);
                    vmin = fminf(vmin, v);
                }
            }
        } else {
            // scalar path
            #pragma unroll 1
            for (int c = 0; c < C; ++c) {
                float v = ldg_f32(x + base_nhw + (int64_t)c * stride_c);
                vmin = fminf(vmin, v);
            }
        }

        partial_sum += vmin;
    }

    // Reduce partial_sum across warps (over H dimension) for each lane independently.
    // shared layout: [warps_per_block][32]
    __shared__ float shmem[8 * 32];
    shmem[warp * 32 + lane] = partial_sum;
    __syncthreads();

    // First warp reduces across warps for each lane
    float sumv = 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int wpi = 0; wpi < warps_per_block; ++wpi) {
            sumv += shmem[wpi * 32 + lane];
        }
        float outv = gelu_approx(sumv) + bias[0];
        y[(int64_t)n * (int64_t)W + (int64_t)w] = outv;
    }
}

torch::Tensor min_sum_gelu_add_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.numel() == 1, "bias must have exactly 1 element");

    auto x_c = x.contiguous();
    auto b_c = bias.contiguous();

    int N = (int)x_c.size(0);
    int C = (int)x_c.size(1);
    int H = (int)x_c.size(2);
    int W = (int)x_c.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "invalid x shape");

    auto y = torch::empty({(int64_t)N, 1, 1, (int64_t)W}, x_c.options());

    // 32-wide tiles along W
    int tiles_w = (W + 31) / 32;

    dim3 block(256, 1, 1);       // 8 warps
    dim3 grid(tiles_w, N, 1);

    // Strict vec4 gating:
    // - W%4==0
    // - x pointer aligned to 16B
    // - y pointer aligned to 16B (not required for scalar store, but keep consistent)
    // - w_base*4 bytes alignment => w_base multiple-of-4; since w_base = tile*32, always true
    // - stride_h = W: for each row, base offset is h*W; W%4==0 ensures row starts 16B aligned
    uintptr_t xp = (uintptr_t)x_c.data_ptr<float>();
    uintptr_t yp = (uintptr_t)y.data_ptr<float>();
    bool vec4_ok = ((W & 3) == 0) && ((xp & 15) == 0) && ((yp & 15) == 0);

    if (vec4_ok) {
        min_sum_gelu_add_tiled_kernel<true><<<grid, block>>>(
            x_c.data_ptr<float>(),
            b_c.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        min_sum_gelu_add_tiled_kernel<false><<<grid, block>>>(
            x_c.data_ptr<float>(),
            b_c.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, H, W
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor min_sum_gelu_add_cuda(torch::Tensor x, torch::Tensor bias);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_min_sum_gelu_add_opt",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["min_sum_gelu_add_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps nn.ConvTranspose2d on cuDNN, fuses:
      min(dim=1, keepdim=True) -> sum(dim=2, keepdim=True) -> GELU -> +bias
    into an optimized custom CUDA kernel producing [N,1,1,W].
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.custom_ops = custom_ops_lib
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        return self.custom_ops.min_sum_gelu_add_cuda(x, self.bias)