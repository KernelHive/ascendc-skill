import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: fused scale + min-reduce over C (NCHW -> N1HW), optimized for coalescing along W ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float fminf_fast(float a, float b) { return a < b ? a : b; }

static inline __host__ __device__ bool is_aligned_uint(uintptr_t p, uintptr_t a) { return (p & (a - 1)) == 0; }

// Warp computes a contiguous W tile for one (n,h).
// Each lane computes 8 output pixels: two float4 segments.
// Mapping:
//   blockIdx.y -> nh = n*H + h
//   blockIdx.x, warp_in_block -> w_tile (multiple of 256 pixels per warp)
// For each channel c, load x[n,c,h,w:w+255] coalesced, update per-pixel minima.
__global__ __launch_bounds__(256, 2) void scale_min_reduce_coalesced_vec8_kernel(
    const float* __restrict__ x,   // [N,C,H,W]
    float* __restrict__ out,       // [N*H*W] (output flattened, corresponds to N,1,H,W)
    int N, int C, int H, int W,
    float scale
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;      // 0..7
    const int warps_per_block = blockDim.x >> 5;     // 8

    const int nh = (int)blockIdx.y;                  // 0..N*H-1
    if (nh >= N * H) return;
    const int n = nh / H;
    const int h = nh - n * H;

    // Each warp covers 256 columns at a time (8 floats per lane => 256 per warp).
    const int warp_global_x = (int)blockIdx.x * warps_per_block + warp_in_block;
    const int w_tile0 = warp_global_x * 256;

    if (w_tile0 >= W) return;

    // lane covers 8 contiguous pixels: [w_tile0 + lane*8 ... + lane*8+7]
    const int w_lane0 = w_tile0 + (lane << 3); // lane*8
    const int base_out = (n * H + h) * W;

    // Bounds: require w_lane0+7 < W for full vec8 path; otherwise fall back to scalar handling for tail.
    if (w_lane0 + 7 >= W) {
        // Tail scalar: each lane handles up to 8 pixels with bounds checks.
        float m[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) m[i] = INFINITY;

        const int hw = H * W;
        const int64_t HW = (int64_t)hw;
        const int64_t base = (int64_t)n * (int64_t)C * HW + (int64_t)h * (int64_t)W + (int64_t)w_lane0;

        for (int c = 0; c < C; ++c) {
            const float* xp = x + base + (int64_t)c * HW;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int w = w_lane0 + i;
                if (w < W) {
                    float v = ldg_f32(xp + i) * scale;
                    m[i] = fminf_fast(m[i], v);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int w = w_lane0 + i;
            if (w < W) out[base_out + w] = m[i];
        }
        return;
    }

    // Fast vec8 path: use two float4 loads per channel per lane.
    // Keep 8 minima in registers.
    float m0 = INFINITY, m1 = INFINITY, m2 = INFINITY, m3 = INFINITY;
    float m4 = INFINITY, m5 = INFINITY, m6 = INFINITY, m7 = INFINITY;

    const int hw = H * W;
    const int64_t HW = (int64_t)hw;
    const int64_t base = (int64_t)n * (int64_t)C * HW + (int64_t)h * (int64_t)W + (int64_t)w_lane0;

    // Pointers for vector loads; assume x is at least 4B aligned always, but float4 alignment checked on host for fast path.
    // Here we still use memcpy-like loads via reinterpret_cast because alignment is guaranteed by dispatch.
    for (int c = 0; c < C; ++c) {
        const float* xp = x + base + (int64_t)c * HW;

        // Prefetch next channel pointer to help ILP a bit (address calc overlap)
        // (compiler may schedule well even without, but helps reduce sync/pipeline gaps).
        const float* xp_next = (c + 1 < C) ? (xp + (int64_t)HW) : xp;

        const float4 v0 = *reinterpret_cast<const float4*>(xp);
        const float4 v1 = *reinterpret_cast<const float4*>(xp + 4);

        // Do some independent work before next iteration's loads can become dependent.
        const float a0 = v0.x * scale; m0 = fminf_fast(m0, a0);
        const float a1 = v0.y * scale; m1 = fminf_fast(m1, a1);
        const float a2 = v0.z * scale; m2 = fminf_fast(m2, a2);
        const float a3 = v0.w * scale; m3 = fminf_fast(m3, a3);
        const float a4 = v1.x * scale; m4 = fminf_fast(m4, a4);
        const float a5 = v1.y * scale; m5 = fminf_fast(m5, a5);
        const float a6 = v1.z * scale; m6 = fminf_fast(m6, a6);
        const float a7 = v1.w * scale; m7 = fminf_fast(m7, a7);

        // Touch xp_next to encourage keeping loop-carried dependency light (no functional effect).
        asm volatile("" :: "l"(xp_next));
    }

    // Store 8 outputs (contiguous)
    float4 o0; o0.x = m0; o0.y = m1; o0.z = m2; o0.w = m3;
    float4 o1; o1.x = m4; o1.y = m5; o1.z = m6; o1.w = m7;

    *reinterpret_cast<float4*>(out + base_out + w_lane0) = o0;
    *reinterpret_cast<float4*>(out + base_out + w_lane0 + 4) = o1;
}

// Scalar coalesced kernel for general alignment / small W.
// Warp computes 32 contiguous pixels for (n,h) per iteration (1 per lane), with grid-stride over w_tiles.
__global__ __launch_bounds__(256, 2) void scale_min_reduce_coalesced_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float scale
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int nh = (int)blockIdx.y;
    if (nh >= N * H) return;
    const int n = nh / H;
    const int h = nh - n * H;

    const int warp_global_x = (int)blockIdx.x * warps_per_block + warp_in_block;
    const int w0 = warp_global_x * 32 + lane;

    if (w0 >= W) return;

    const int hw = H * W;
    const int64_t HW = (int64_t)hw;
    const int64_t base = (int64_t)n * (int64_t)C * HW + (int64_t)h * (int64_t)W + (int64_t)w0;

    float m = INFINITY;
    for (int c = 0; c < C; ++c) {
        float v = ldg_f32(x + base + (int64_t)c * HW) * scale;
        m = fminf_fast(m, v);
    }
    out[(n * H + h) * W + w0] = m;
}

torch::Tensor conv2d_scaling_min_cuda(
    torch::Tensor x,  // [N,C,H,W]
    double scale_factor
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    if (!x.is_contiguous()) x = x.contiguous();

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto out_t = torch::empty({N, 1, H, W}, x.options());
    // Flatten output for simpler indexing
    auto out = out_t.view({N * H * W});

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    // Launch: 256 threads (8 warps). 2D grid: x over w_tiles, y over N*H.
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, x.get_device());
    const int threads = 256;
    const int warps_per_block = threads / 32;

    // Prefer enough blocks in x to cover W tiles, but cap total blocks for scheduling/launch overhead.
    // We'll cap blocks_x based on SMs; blocks_y is fixed to N*H.
    const int NH = N * H;

    // Vector8 path processes 256 columns per warp => 256*warps_per_block cols per block
    const int cols_per_block_vec8 = 256 * warps_per_block;
    const int blocks_x_needed_vec8 = (W + cols_per_block_vec8 - 1) / cols_per_block_vec8;

    // Scalar path processes 32 columns per warp => 32*warps_per_block cols per block
    const int cols_per_block_scalar = 32 * warps_per_block;
    const int blocks_x_needed_scalar = (W + cols_per_block_scalar - 1) / cols_per_block_scalar;

    const int blocks_x_cap = (sm_count > 0) ? (sm_count * 4) : 256; // moderate cap; y-dimension already large
    const float scale = (float)scale_factor;

    const uintptr_t xaddr = (uintptr_t)xp;
    const uintptr_t oaddr = (uintptr_t)op;

    // Vec8 requires float4 loads/stores alignment (16B) and W multiple of 8 for full tiles.
    const bool can_vec8 = (W >= 8) && ((W & 7) == 0) && is_aligned_uint(xaddr, 16) && is_aligned_uint(oaddr, 16);

    if (can_vec8) {
        int bx = blocks_x_needed_vec8;
        if (bx < 1) bx = 1;
        if (bx > blocks_x_cap) bx = blocks_x_cap;
        dim3 grid(bx, NH, 1);
        scale_min_reduce_coalesced_vec8_kernel<<<grid, threads>>>(xp, op, N, C, H, W, scale);
    } else {
        int bx = blocks_x_needed_scalar;
        if (bx < 1) bx = 1;
        if (bx > blocks_x_cap * 2) bx = blocks_x_cap * 2; // scalar needs more x-coverage
        dim3 grid(bx, NH, 1);
        scale_min_reduce_coalesced_scalar_kernel<<<grid, threads>>>(xp, op, N, C, H, W, scale);
    }

    return out_t;
}
"""

cpp_src = r"""
torch::Tensor conv2d_scaling_min_cuda(
    torch::Tensor x,
    double scale_factor
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_scaling_min_opt3_coalesced",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_scaling_min_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Convolution followed by fused scaling + min-reduction over channel dimension:
      y = min_c( conv(x)[n,c,h,w] * scale_factor ), keepdim=True
    Implemented with a custom CUDA kernel on the conv output (no custom convolution).
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if not x.is_cuda:
            x = x * self.scale_factor
            return torch.min(x, dim=1, keepdim=True)[0]
        if x.dtype != torch.float32:
            x = x.float()
        return self.custom_ops_lib.conv2d_scaling_min_cuda(x, self.scale_factor)


# Scaffold compatibility
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]