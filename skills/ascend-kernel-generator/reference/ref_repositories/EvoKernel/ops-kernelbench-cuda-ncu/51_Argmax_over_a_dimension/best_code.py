import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ extension: argmax over dim==1 on 3D float tensors (warp-parallel, multi-warp blocks)
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

#ifndef __CUDA_ARCH__
#define __ldg(x) (*(x))
#endif

#define WARP_SIZE 32

static __forceinline__ __device__ void warp_reduce_argmax(float &v, int &idx) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v2 = __shfl_down_sync(mask, v, offset);
        int   i2 = __shfl_down_sync(mask, idx, offset);
        if (v2 > v || (v2 == v && i2 < idx)) {
            v = v2;
            idx = i2;
        }
    }
}

template<int WARPS_PER_BLOCK, int UNROLL>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, 2)
void argmax_dim1_warp_kernel_i32(
    const float* __restrict__ x,
    int64_t* __restrict__ out,
    int B, int D1, int D2
) {
    int b = (int)blockIdx.y;
    int warp_id = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);

    int d2 = (int)blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (b >= B || d2 >= D2) return;

    int base = (b * D1) * D2 + d2;

    float maxv = -INFINITY;
    int max_i = 0;

    int step = WARP_SIZE * UNROLL;
    for (int d1 = lane; d1 < D1; d1 += step) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int dd1 = d1 + u * WARP_SIZE;
            if (dd1 < D1) {
                float v = __ldg(x + base + dd1 * D2);
                if (v > maxv) {
                    maxv = v;
                    max_i = dd1;
                }
            }
        }
    }

    warp_reduce_argmax(maxv, max_i);
    if (lane == 0) out[(int64_t)b * (int64_t)D2 + (int64_t)d2] = (int64_t)max_i;
}

template<int WARPS_PER_BLOCK, int UNROLL>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, 2)
void argmax_dim1_warp_kernel_i64(
    const float* __restrict__ x,
    int64_t* __restrict__ out,
    int64_t B, int64_t D1, int64_t D2
) {
    int64_t b = (int64_t)blockIdx.y;
    int warp_id = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);

    int64_t d2 = (int64_t)blockIdx.x * (int64_t)WARPS_PER_BLOCK + (int64_t)warp_id;
    if (b >= B || d2 >= D2) return;

    int64_t base = (b * D1) * D2 + d2;

    float maxv = -INFINITY;
    int max_i = 0;

    int64_t step = (int64_t)WARP_SIZE * (int64_t)UNROLL;
    for (int64_t d1 = (int64_t)lane; d1 < D1; d1 += step) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int64_t dd1 = d1 + (int64_t)u * (int64_t)WARP_SIZE;
            if (dd1 < D1) {
                float v = __ldg(x + base + dd1 * D2);
                if (v > maxv) {
                    maxv = v;
                    max_i = (int)dd1; // D1 typically fits in int; output is int64
                }
            }
        }
    }

    warp_reduce_argmax(maxv, max_i);
    if (lane == 0) out[b * D2 + d2] = (int64_t)max_i;
}

torch::Tensor argmax_over_dim_cuda(torch::Tensor x, int64_t dim) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);

    TORCH_CHECK(x.dim() == 3, "Only 3D tensors supported in this custom op");
    TORCH_CHECK(dim == 1, "This custom kernel is specialized for dim==1");

    const int64_t B64  = (int64_t)x.size(0);
    const int64_t D164 = (int64_t)x.size(1);
    const int64_t D264 = (int64_t)x.size(2);

    auto out = torch::empty({B64, D264}, torch::TensorOptions().device(x.device()).dtype(torch::kInt64));

    // Tunables (good default for large D2)
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads
    constexpr int UNROLL = 4;

    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 grid((unsigned int)((D264 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
              (unsigned int)B64);

    bool use_i32 =
        (B64  <= (int64_t)std::numeric_limits<int>::max()) &&
        (D164 <= (int64_t)std::numeric_limits<int>::max()) &&
        (D264 <= (int64_t)std::numeric_limits<int>::max()) &&
        (B64 * D164 <= (int64_t)std::numeric_limits<int>::max()) &&
        (B64 * D264 <= (int64_t)std::numeric_limits<int>::max()) &&
        (D164 * D264 <= (int64_t)std::numeric_limits<int>::max()) &&
        (B64 * D164 * D264 <= (int64_t)std::numeric_limits<int>::max());

    if (use_i32) {
        argmax_dim1_warp_kernel_i32<WARPS_PER_BLOCK, UNROLL><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (int64_t*)out.data_ptr<int64_t>(),
            (int)B64, (int)D164, (int)D264
        );
    } else {
        argmax_dim1_warp_kernel_i64<WARPS_PER_BLOCK, UNROLL><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (int64_t*)out.data_ptr<int64_t>(),
            B64, D164, D264
        );
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor argmax_over_dim_cuda(torch::Tensor x, int64_t dim);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["argmax_over_dim_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA kernel for argmax.
    Uses the custom kernel for the hot path: x is 3D, dim==1, float32 contiguous CUDA.
    Falls back to torch.argmax otherwise to preserve full semantics.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.argmax(x, dim=self.dim)

        # Preserve semantics for arbitrary dim/shape via fallback.
        if x.dim() != 3 or self.dim != 1:
            return torch.argmax(x, dim=self.dim)

        # Kernel requirements
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.argmax_over_dim_cuda(x, int(self.dim))