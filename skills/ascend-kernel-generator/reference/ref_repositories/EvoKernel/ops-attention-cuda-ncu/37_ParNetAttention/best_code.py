import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // --use_fast_math: expf -> __expf
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float siluf_fast(float x) {
    float s = sigmoidf_fast(x);
    return x * s;
}

__device__ __forceinline__ float warp_broadcast(float v, int src_lane = 0) {
    return __shfl_sync(0xffffffffu, v, src_lane);
}

template<int WARPS_PER_BLOCK, int TILES_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void par_net_attention_fused_hw49_warp_vec2_tiled(
    const float* __restrict__ x,   // (BC,49)
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ se,  // (BC)
    float* __restrict__ out,
    int BC
) {
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..WARPS_PER_BLOCK-1

    // Block handles a small fixed batch of tiles to increase ILP.
    int block_tile0 = ((int)blockIdx.x) * (WARPS_PER_BLOCK * TILES_PER_BLOCK);

    #pragma unroll
    for (int t = 0; t < TILES_PER_BLOCK; ++t) {
        int bc = block_tile0 + t * WARPS_PER_BLOCK + warp;
        if (bc >= BC) return;

        float gate = 0.0f;
        if (lane == 0) {
            gate = sigmoidf_fast(ldg_f32(se + bc));
        }
        gate = warp_broadcast(gate, 0);

        int64_t base = (int64_t)bc * 49;

        const float* px  = x  + base;
        const float* px1 = x1 + base;
        const float* px2 = x2 + base;
        float* pout = out + base;

        // float2 path is safe for contiguous float tensors: base is 49 floats, but tensor base pointer is aligned;
        // we still guard per-tile 8-byte alignment across all pointers.
        uintptr_t mask = ((uintptr_t)px) | ((uintptr_t)px1) | ((uintptr_t)px2) | ((uintptr_t)pout);
        bool aligned8 = (mask & 0x7) == 0;

        if (aligned8) {
            const float2* __restrict__ x2v  = reinterpret_cast<const float2*>(px);
            const float2* __restrict__ x12v = reinterpret_cast<const float2*>(px1);
            const float2* __restrict__ x22v = reinterpret_cast<const float2*>(px2);
            float2* __restrict__ o2v = reinterpret_cast<float2*>(pout);

            // 48 elems -> 24 float2
            for (int i2 = lane; i2 < 24; i2 += 32) {
                float2 vx  = x2v[i2];
                float2 vx1 = x12v[i2];
                float2 vx2 = x22v[i2];

                float2 vo;
                float s0 = vx1.x + vx2.x + vx.x * gate;
                float s1 = vx1.y + vx2.y + vx.y * gate;
                vo.x = siluf_fast(s0);
                vo.y = siluf_fast(s1);
                o2v[i2] = vo;
            }

            if (lane == 0) {
                float xv  = ldg_f32(px  + 48);
                float x1v = ldg_f32(px1 + 48);
                float x2v = ldg_f32(px2 + 48);
                float sum = x1v + x2v + xv * gate;
                pout[48] = siluf_fast(sum);
            }
        } else {
            for (int i = lane; i < 49; i += 32) {
                float xv  = ldg_f32(px  + i);
                float x1v = ldg_f32(px1 + i);
                float x2v = ldg_f32(px2 + i);
                float sum = x1v + x2v + xv * gate;
                pout[i] = siluf_fast(sum);
            }
        }
    }
}

template<int WARPS_PER_BLOCK, int TILES_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void par_net_attention_fused_generic_warp_scalar_tiled(
    const float* __restrict__ x,   // (BC,HW)
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ se,  // (BC)
    float* __restrict__ out,
    int BC,
    int HW
) {
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int block_tile0 = ((int)blockIdx.x) * (WARPS_PER_BLOCK * TILES_PER_BLOCK);

    #pragma unroll
    for (int t = 0; t < TILES_PER_BLOCK; ++t) {
        int bc = block_tile0 + t * WARPS_PER_BLOCK + warp;
        if (bc >= BC) return;

        float gate = 0.0f;
        if (lane == 0) {
            gate = sigmoidf_fast(ldg_f32(se + bc));
        }
        gate = warp_broadcast(gate, 0);

        int64_t base = (int64_t)bc * (int64_t)HW;
        const float* px  = x  + base;
        const float* px1 = x1 + base;
        const float* px2 = x2 + base;
        float* pout = out + base;

        for (int i = lane; i < HW; i += 32) {
            float xv  = ldg_f32(px  + i);
            float x1v = ldg_f32(px1 + i);
            float x2v = ldg_f32(px2 + i);
            float sum = x1v + x2v + xv * gate;
            pout[i] = siluf_fast(sum);
        }
    }
}

torch::Tensor par_net_attention_fused_cuda(
    torch::Tensor x,
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor se
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda() && se.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x1.scalar_type() == torch::kFloat32 && x2.scalar_type() == torch::kFloat32, "x1/x2 must be float32");
    TORCH_CHECK(se.scalar_type() == torch::kFloat32, "se must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous(), "x1/x2 must be contiguous");
    TORCH_CHECK(se.is_contiguous(), "se must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be (B,C,H,W)");
    TORCH_CHECK(x1.sizes() == x.sizes(), "x1 must match x shape");
    TORCH_CHECK(x2.sizes() == x.sizes(), "x2 must match x shape");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;

    TORCH_CHECK(se.dim() == 4 || se.dim() == 2, "se must be (B,C,1,1) or (B,C)");
    torch::Tensor se2 = se;
    if (se.dim() == 4) {
        TORCH_CHECK(se.size(0) == B && se.size(1) == C && se.size(2) == 1 && se.size(3) == 1,
                    "se (4D) must be (B,C,1,1)");
        se2 = se.view({B, C});
    } else {
        TORCH_CHECK(se.size(0) == B && se.size(1) == C, "se (2D) must be (B,C)");
    }

    auto out = torch::empty_like(x);
    int BC = B * C;

    // Tuned mapping:
    // - 4 warps/block (128 threads) gives good latency hiding on memory-bound elementwise.
    // - 2 tiles/block increases independent work per block without persistent grid-stride overhead.
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int TILES_PER_BLOCK = 2;
    int threads = WARPS_PER_BLOCK * 32;

    int tiles_per_block = WARPS_PER_BLOCK * TILES_PER_BLOCK;
    int blocks = (BC + tiles_per_block - 1) / tiles_per_block;

    // Cap grid to avoid huge launches; BC is large in typical usage anyway.
    if (blocks > 8192) blocks = 8192;
    if (blocks < 1) blocks = 1;

    if (HW == 49) {
        par_net_attention_fused_hw49_warp_vec2_tiled<WARPS_PER_BLOCK, TILES_PER_BLOCK><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)x1.data_ptr<float>(),
            (const float*)x2.data_ptr<float>(),
            (const float*)se2.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            BC
        );
    } else {
        par_net_attention_fused_generic_warp_scalar_tiled<WARPS_PER_BLOCK, TILES_PER_BLOCK><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)x1.data_ptr<float>(),
            (const float*)x2.data_ptr<float>(),
            (const float*)se2.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            BC, HW
        );
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor par_net_attention_fused_cuda(torch::Tensor x, torch::Tensor x1, torch::Tensor x2, torch::Tensor se);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_par_net_attention_opt3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["par_net_attention_fused_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """ParNet Attention with improved fused elementwise tail CUDA kernel."""
    def __init__(self, channel=512):
        super().__init__()
        self.sse_pool = nn.AdaptiveAvgPool2d(1)
        self.sse_conv = nn.Conv2d(channel, channel, kernel_size=1)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)

        se = self.sse_pool(x)
        se = self.sse_conv(se)  # pre-sigmoid; sigmoid applied in CUDA kernel

        if x.dtype != torch.float32:
            x = x.float()
        if x1.dtype != torch.float32:
            x1 = x1.float()
        if x2.dtype != torch.float32:
            x2 = x2.float()
        if se.dtype != torch.float32:
            se = se.float()

        x = x.contiguous()
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        se = se.contiguous()

        return self.custom_ops_lib.par_net_attention_fused_cuda(x, x1, x2, se)