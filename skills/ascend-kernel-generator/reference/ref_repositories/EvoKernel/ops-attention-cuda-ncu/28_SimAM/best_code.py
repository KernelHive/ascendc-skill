import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ extension: optimized fused SimAM forward
simam_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_broadcast0(float v) {
    return __shfl_sync(FULL_MASK, v, 0);
}

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// HW=49 specialized: 4 warps per block, each warp processes one (b,c) plane.
// This improves latency hiding vs. 1-warp blocks and keeps work mapping branch-light.
template<int WARPS_PER_BLOCK>
__global__ void simam_forward_kernel_hw49_warp4(
    const float* __restrict__ x,
    float* __restrict__ out,
    int BC,            // B*C
    float e_lambda)
{
    constexpr int HW = 49;
    constexpr float invHW = 1.0f / 49.0f;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..WARPS_PER_BLOCK-1

    // Each block handles WARPS_PER_BLOCK planes
    int bc = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
    if (bc >= BC) return;

    int base = bc * HW;

    // Alignment check for vector path (float4)
    uintptr_t addr_x   = (uintptr_t)(x + base);
    uintptr_t addr_out = (uintptr_t)(out + base);
    bool aligned16 = ((addr_x | addr_out) & 0xF) == 0;

    // Cache up to two scalars per lane for scalar path.
    float v0 = 0.f, v1 = 0.f;
    bool has0 = lane < HW;
    bool has1 = (lane + 32) < HW;

    float sum = 0.f, sumsq = 0.f;

    if (aligned16) {
        // Vector path: lanes 0..11 load float4 for indices [0..47], lane 12 loads tail 48.
        float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
        float tail = 0.f;

        if (lane < 12) {
            const float4* x4 = reinterpret_cast<const float4*>(x + base);
            v4 = x4[lane];
            sum   = (v4.x + v4.y) + (v4.z + v4.w);
            sumsq = (v4.x*v4.x + v4.y*v4.y) + (v4.z*v4.z + v4.w*v4.w);
        } else if (lane == 12) {
            tail = x[base + 48];
            sum = tail;
            sumsq = tail * tail;
        }

        float sum_all = warp_reduce_sum(sum);
        float sumsq_all = warp_reduce_sum(sumsq);
        sum_all = warp_broadcast0(sum_all);
        sumsq_all = warp_broadcast0(sumsq_all);

        float mean = sum_all * invHW;
        float sse = sumsq_all - 49.0f * mean * mean;
        sse = fmaxf(sse, 0.0f);

        float inv_denom = 1.0f / (4.0f * (sse / 48.0f + e_lambda));

        if (lane < 12) {
            float4 o;
            float dx = v4.x - mean; float yx = dx*dx*inv_denom + 0.5f; o.x = v4.x * sigmoidf_fast(yx);
            float dy = v4.y - mean; float yy = dy*dy*inv_denom + 0.5f; o.y = v4.y * sigmoidf_fast(yy);
            float dz = v4.z - mean; float yz = dz*dz*inv_denom + 0.5f; o.z = v4.z * sigmoidf_fast(yz);
            float dw = v4.w - mean; float yw = dw*dw*inv_denom + 0.5f; o.w = v4.w * sigmoidf_fast(yw);

            float4* out4 = reinterpret_cast<float4*>(out + base);
            out4[lane] = o;
        } else if (lane == 12) {
            float xv = x[base + 48];
            float d = xv - mean;
            float y = d * d * inv_denom + 0.5f;
            out[base + 48] = xv * sigmoidf_fast(y);
        }
        return;
    }

    // Scalar path: every lane participates; two elements per lane max
    if (has0) {
        v0 = ldg_f32(x + base + lane);
        sum += v0;
        sumsq += v0 * v0;
    }
    if (has1) {
        v1 = ldg_f32(x + base + lane + 32);
        sum += v1;
        sumsq += v1 * v1;
    }

    float sum_all = warp_reduce_sum(sum);
    float sumsq_all = warp_reduce_sum(sumsq);
    sum_all = warp_broadcast0(sum_all);
    sumsq_all = warp_broadcast0(sumsq_all);

    float mean = sum_all * invHW;
    float sse = sumsq_all - 49.0f * mean * mean;
    sse = fmaxf(sse, 0.0f);

    float inv_denom = 1.0f / (4.0f * (sse / 48.0f + e_lambda));

    if (has0) {
        float d = v0 - mean;
        float y = (d * d) * inv_denom + 0.5f;
        out[base + lane] = v0 * sigmoidf_fast(y);
    }
    if (has1) {
        float d = v1 - mean;
        float y = (d * d) * inv_denom + 0.5f;
        out[base + lane + 32] = v1 * sigmoidf_fast(y);
    }
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    // Assumes blockDim.x is a multiple of 32 and <= 256.
    __shared__ float warp_sums[8];
    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        float t = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) warp_sums[0] = t;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void simam_forward_kernel_generic(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    float e_lambda)
{
    int bc = (int)blockIdx.x;
    int b = bc / C;
    int c = bc - b * C;

    int HW = H * W;
    int base = (b * C + c) * HW;

    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < HW; i += (int)blockDim.x) {
        sum += x[base + i];
    }
    float block_sum = block_reduce_sum(sum);
    float mean = block_sum * (1.0f / (float)HW);

    float sum2 = 0.0f;
    for (int i = (int)threadIdx.x; i < HW; i += (int)blockDim.x) {
        float d = x[base + i] - mean;
        sum2 += d * d;
    }
    float sse = block_reduce_sum(sum2);

    float n = (float)(HW - 1);
    float inv_denom = 1.0f / (4.0f * (sse / n + e_lambda));

    for (int i = (int)threadIdx.x; i < HW; i += (int)blockDim.x) {
        float xv = x[base + i];
        float d = xv - mean;
        float y = (d * d) * inv_denom + 0.5f;
        out[base + i] = xv * sigmoidf_fast(y);
    }
}

torch::Tensor simam_forward_cuda(torch::Tensor x, double e_lambda) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B,C,H,W)");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto out = torch::empty_like(x);

    const int BC = B * C;

    if (H == 7 && W == 7) {
        constexpr int WARPS_PER_BLOCK = 4;
        const int threads = 32 * WARPS_PER_BLOCK;
        const int blocks = (BC + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        simam_forward_kernel_hw49_warp4<WARPS_PER_BLOCK><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            BC,
            (float)e_lambda
        );
    } else {
        const int blocks = BC;
        const int threads = 128;
        simam_forward_kernel_generic<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, H, W,
            (float)e_lambda
        );
    }

    return out;
}
"""

simam_cpp_source = r"""
torch::Tensor simam_forward_cuda(torch::Tensor x, double e_lambda);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_simam_opt6",
    cpp_sources=simam_cpp_source,
    cuda_sources=simam_cuda_source,
    functions=["simam_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Fused SimAM forward using an optimized custom CUDA kernel:
    out = x * sigmoid( ((x-mean)^2) / (4*(sum((x-mean)^2)/n + e_lambda)) + 0.5 )
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA input tensor")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops.simam_forward_cuda(x, self.e_lambda)