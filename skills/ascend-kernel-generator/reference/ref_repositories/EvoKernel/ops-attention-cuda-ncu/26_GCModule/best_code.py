import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

template<int CTILE, int WARPS_PER_BLOCK>
__global__ void gc_context_smem_hw49_multiwarp_kernel(
    const float* __restrict__ x,        // [B,C,49]
    const float* __restrict__ context,  // [B,49] (from [B,1,H,W])
    float* __restrict__ out,            // [B,C]
    int B, int C)
{
    // grid: x= B, y = number of groups of (WARPS_PER_BLOCK * CTILE) channels
    int b = (int)blockIdx.x;
    int group = (int)blockIdx.y;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..WARPS_PER_BLOCK-1

    extern __shared__ float sctx[]; // size 49 floats

    // Stage context into shared memory once per block
    const float* cb = context + b * 49;
    if (tid < 49) {
        sctx[tid] = __ldg(cb + tid);
    }
    __syncthreads();

    // Each warp handles one CTILE-sized channel tile within this group
    int tile_in_group = warp; // 0..WARPS_PER_BLOCK-1
    int c0 = (group * WARPS_PER_BLOCK + tile_in_group) * CTILE;

    float acc[CTILE];
    #pragma unroll
    for (int t = 0; t < CTILE; ++t) acc[t] = 0.0f;

    // Precompute per-channel base pointers for this (b, channel)
    const float* xbases[CTILE];
    #pragma unroll
    for (int t = 0; t < CTILE; ++t) {
        int c = c0 + t;
        xbases[t] = (c < C) ? (x + ((b * C + c) * 49)) : nullptr;
    }

    // lane covers i=lane and i=lane+32 (only if <49)
    if (lane < 49) {
        float w0 = sctx[lane];
        #pragma unroll
        for (int t = 0; t < CTILE; ++t) {
            const float* xb = xbases[t];
            if (xb) acc[t] = fmaf(__ldg(xb + lane), w0, acc[t]);
        }
    }

    int i1 = lane + 32;
    if (i1 < 49) {
        float w1 = sctx[i1];
        #pragma unroll
        for (int t = 0; t < CTILE; ++t) {
            const float* xb = xbases[t];
            if (xb) acc[t] = fmaf(__ldg(xb + i1), w1, acc[t]);
        }
    }

    // Reduce within warp and store
    #pragma unroll
    for (int t = 0; t < CTILE; ++t) {
        float s = warp_allreduce_sum(acc[t]);
        int c = c0 + t;
        if (lane == 0 && c < C) out[b * C + c] = s;
    }
}

template<int CTILE, int WARPS_PER_BLOCK>
__global__ void gc_context_smem_generic_multiwarp_kernel(
    const float* __restrict__ x,        // [B,C,HW]
    const float* __restrict__ context,  // [B,HW]
    float* __restrict__ out,            // [B,C]
    int B, int C, int HW)
{
    int b = (int)blockIdx.x;
    int group = (int)blockIdx.y;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    extern __shared__ float sctx[]; // size HW floats

    // Stage context into shared memory
    const float* cb = context + b * HW;
    for (int i = tid; i < HW; i += blockDim.x) {
        sctx[i] = __ldg(cb + i);
    }
    __syncthreads();

    int tile_in_group = warp;
    int c0 = (group * WARPS_PER_BLOCK + tile_in_group) * CTILE;

    float acc[CTILE];
    #pragma unroll
    for (int t = 0; t < CTILE; ++t) acc[t] = 0.0f;

    const float* xbases[CTILE];
    #pragma unroll
    for (int t = 0; t < CTILE; ++t) {
        int c = c0 + t;
        xbases[t] = (c < C) ? (x + ((b * C + c) * HW)) : nullptr;
    }

    for (int i = lane; i < HW; i += 32) {
        float w = sctx[i];
        #pragma unroll
        for (int t = 0; t < CTILE; ++t) {
            const float* xb = xbases[t];
            if (xb) acc[t] = fmaf(__ldg(xb + i), w, acc[t]);
        }
    }

    #pragma unroll
    for (int t = 0; t < CTILE; ++t) {
        float s = warp_allreduce_sum(acc[t]);
        int c = c0 + t;
        if (lane == 0 && c < C) out[b * C + c] = s;
    }
}

torch::Tensor gc_context_forward_cuda(torch::Tensor x, torch::Tensor context) {
    CHECK_INPUT(x);
    CHECK_INPUT(context);
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor (B,C,H,W)");
    TORCH_CHECK(context.dim() == 4, "context must be a 4D tensor (B,1,H,W)");
    TORCH_CHECK(context.size(0) == x.size(0), "batch mismatch");
    TORCH_CHECK(context.size(2) == x.size(2) && context.size(3) == x.size(3), "spatial mismatch");
    TORCH_CHECK(context.size(1) == 1, "context channel must be 1");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;

    auto out = torch::empty({B, C, 1, 1}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
    float* out_bc = out.view({B, C}).data_ptr<float>();

    const float* xptr = x.data_ptr<float>();
    const float* cptr = context.data_ptr<float>(); // treat as [B,HW]

    constexpr int CTILE = 8;
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 threads(32 * WARPS_PER_BLOCK, 1, 1);

    int tiles_per_group = WARPS_PER_BLOCK;
    int ch_per_group = tiles_per_group * CTILE;
    int groups = (C + ch_per_group - 1) / ch_per_group;
    dim3 blocks((unsigned)B, (unsigned)groups, 1);

    if (HW == 49) {
        size_t smem = 49 * sizeof(float);
        gc_context_smem_hw49_multiwarp_kernel<CTILE, WARPS_PER_BLOCK><<<blocks, threads, smem>>>(
            xptr, cptr, out_bc, B, C
        );
    } else {
        // Shared memory: HW floats
        size_t smem = (size_t)HW * sizeof(float);
        gc_context_smem_generic_multiwarp_kernel<CTILE, WARPS_PER_BLOCK><<<blocks, threads, smem>>>(
            xptr, cptr, out_bc, B, C, HW
        );
    }
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gc_context_forward_cuda(torch::Tensor x, torch::Tensor context);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gc_context_forward_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization", "-lineinfo"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """GC Module using an optimized custom CUDA kernel for context modeling matmul."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.LayerNorm([channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor input")
        if x.dtype != torch.float32:
            raise RuntimeError("ModelNew requires float32 input")
        if not x.is_contiguous():
            x = x.contiguous()

        context = self.conv(x)
        if not context.is_contiguous():
            context = context.contiguous()

        ctx = custom_ops_lib.gc_context_forward_cuda(x, context)
        y = self.transform(ctx)
        return x + y