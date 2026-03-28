import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# CUDA v5:
# - 256-thread CTA (8 warps) per (n,c) for more MLP vs warp-only
# - Optional float4 vectorized loads with strict alignment + tail
# - Warp-shuffle reduction + tiny shared scratch (8 floats)
# - Persistent grid-stride over NC; no atomics
# - Second mean is no-op, folded away
# - Explicit cudaGetLastError() checking (no undefined macros)
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ __launch_bounds__(256, 2)
void mul_gap_block256(
    const float* __restrict__ x,   // [N,C,H,W] contiguous
    float* __restrict__ y,         // [N*C] flattened
    int NC, int HW,
    float scale                    // multiplier / HW
) {
    // One CTA handles one (n,c), grid-stride over NC for persistence.
    int nc0 = (int)blockIdx.x;
    int tid = (int)threadIdx.x;              // 0..255
    int lane = tid & 31;                     // 0..31
    int warp = tid >> 5;                     // 0..7

    extern __shared__ float smem[];          // at least 8 floats
    float* warp_sums = smem;                 // [8]

    // Vectorization conditions:
    // - base address 16B aligned
    // - HW multiple of 4 (so float4 covers whole)
    // Note: base = x + nc*HW; if x is aligned, base aligned iff (nc*HW) is multiple of 4.
    // We'll check alignment at runtime by pointer value.
    for (int nc = nc0; nc < NC; nc += (int)gridDim.x) {
        const float* base_ptr = x + (int64_t)nc * (int64_t)HW;

        float sum = 0.0f;

        // Try float4 path when safe.
        uintptr_t addr = (uintptr_t)base_ptr;
        bool aligned16 = ((addr & 0xF) == 0);
        bool hw_mul4 = ((HW & 3) == 0);

        if (aligned16 && hw_mul4) {
            const float4* base4 = (const float4*)base_ptr;
            int HW4 = HW >> 2;  // number of float4s

            // grid-stride over float4 elements with 256 threads
            for (int i4 = tid; i4 < HW4; i4 += 256) {
#if __CUDA_ARCH__ >= 350
                float4 v = __ldg(base4 + i4);
#else
                float4 v = base4[i4];
#endif
                sum += (v.x + v.y) + (v.z + v.w);
            }
        } else {
            // Scalar path: grid-stride over HW
            for (int i = tid; i < HW; i += 256) {
#if __CUDA_ARCH__ >= 350
                sum += __ldg(base_ptr + i);
#else
                sum += base_ptr[i];
#endif
            }
        }

        // Reduce within warp
        sum = warp_reduce_sum(sum);

        // One lane writes warp sum
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();

        // Warp 0 reduces the 8 warp sums
        if (warp == 0) {
            float v = (lane < 8) ? warp_sums[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) {
                y[nc] = v * scale;
            }
        }
        __syncthreads();
    }
}

static inline void cuda_check_last_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

torch::Tensor multiply_global_avg_pool_cuda(torch::Tensor x, double multiplier) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");

    auto x_c = x.contiguous(); // NCHW contiguous
    const int64_t N64 = x_c.size(0);
    const int64_t C64 = x_c.size(1);
    const int64_t H64 = x_c.size(2);
    const int64_t W64 = x_c.size(3);
    TORCH_CHECK(N64 > 0 && C64 > 0 && H64 > 0 && W64 > 0, "invalid x shape");
    TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX && H64 <= INT_MAX && W64 <= INT_MAX, "shape too large");

    const int N = (int)N64;
    const int C = (int)C64;
    const int H = (int)H64;
    const int W = (int)W64;
    const int HW = H * W;
    const int NC = N * C;

    auto y = torch::empty({N, C, 1, 1}, x_c.options()); // contiguous NCHW
    float* y_ptr = y.data_ptr<float>();

    // blocks: one per NC, capped; persistent grid-stride covers all.
    int blocks = NC;
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    float scale = (float)multiplier * (1.0f / (float)HW);

    mul_gap_block256<<<blocks, 256, 8 * sizeof(float)>>>(
        x_c.data_ptr<float>(),
        y_ptr,
        NC, HW,
        scale
    );
    cuda_check_last_error();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor multiply_global_avg_pool_cuda(torch::Tensor x, double multiplier);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_v5",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["multiply_global_avg_pool_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps nn.ConvTranspose2d on cuDNN, fuses:
      x * multiplier -> mean(HW, keepdim=True) -> mean(HW, keepdim=True)
    into a single optimized CUDA kernel producing [N, C, 1, 1].
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.custom_ops = custom_ops_lib
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        return self.custom_ops.multiply_global_avg_pool_cuda(x, self.multiplier)