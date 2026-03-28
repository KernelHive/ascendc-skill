import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Fused CUDA: MaxPool3d(k=2,s=2) -> MaxPool3d(k=3,s=3) -> sum over C
# Input:  x [N, C, D, H, W] contiguous float32 CUDA
# Output: y [N, 1, D2, H2, W2]
# Matches PyTorch MaxPool3d defaults: stride=kernel_size, padding=0, dilation=1, ceil_mode=False
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <cmath>

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float neg_inf_f() { return -INFINITY; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    // Assumes blockDim.x is multiple of 32.
    static __shared__ float shared[8]; // up to 256 threads => 8 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

// Tile sizes (outputs per block): chosen to increase ILP and reduce per-output launch overhead.
// 2*2*2 = 8 outputs/block, 128 threads => 16 threads/output in "voxel-group" mapping.
#define TILE_W 2
#define TILE_H 2
#define TILE_D 2
#define VOXELS_PER_BLOCK (TILE_W*TILE_H*TILE_D)
#define THREADS 128

// Kernel maps each block to a tile of output voxels (n, od2.., oh2.., ow2..).
// Within each voxel, threads cooperate over C and reduce without atomics.
__global__ __launch_bounds__(THREADS, 3) void max2_max3_sumc_tiled_kernel(
    const float* __restrict__ x, // [N,C,D,H,W]
    float* __restrict__ y,       // [N,1,D2,H2,W2]
    int N, int C, int D, int H, int W,
    int D1, int H1, int W1,
    int D2, int H2, int W2
) {
    const int64_t HW  = (int64_t)H * (int64_t)W;
    const int64_t DHW = (int64_t)D * HW;

    // Flatten tiles over N*D2*H2*W2, but in tile units.
    const int tiles_w = (W2 + TILE_W - 1) / TILE_W;
    const int tiles_h = (H2 + TILE_H - 1) / TILE_H;
    const int tiles_d = (D2 + TILE_D - 1) / TILE_D;
    const int64_t tiles_per_n = (int64_t)tiles_d * tiles_h * tiles_w;
    const int64_t total_tiles = (int64_t)N * tiles_per_n;

    // Each output voxel uses a group of threads.
    const int group_size = THREADS / VOXELS_PER_BLOCK; // 16
    const int group_id = threadIdx.x / group_size;     // 0..7
    const int lane_in_group = threadIdx.x % group_size;

    // Decode group_id into (td,th,tw)
    const int tw = group_id % TILE_W;
    const int th = (group_id / TILE_W) % TILE_H;
    const int td = (group_id / (TILE_W*TILE_H));

    for (int64_t tile_idx = (int64_t)blockIdx.x; tile_idx < total_tiles; tile_idx += (int64_t)gridDim.x) {
        int64_t t = tile_idx;
        int n = (int)(t / tiles_per_n);
        int64_t rem = t - (int64_t)n * tiles_per_n;

        int tile_w = (int)(rem % tiles_w); rem /= tiles_w;
        int tile_h = (int)(rem % tiles_h); rem /= tiles_h;
        int tile_d = (int)(rem);

        const int ow2 = tile_w * TILE_W + tw;
        const int oh2 = tile_h * TILE_H + th;
        const int od2 = tile_d * TILE_D + td;

        if ((unsigned)ow2 >= (unsigned)W2 || (unsigned)oh2 >= (unsigned)H2 || (unsigned)od2 >= (unsigned)D2) {
            continue;
        }

        const int d1_base = od2 * 3;
        const int h1_base = oh2 * 3;
        const int w1_base = ow2 * 3;

        // Interior fast-path: composed window entirely within x.
        const bool interior =
            (d1_base >= 0) && (h1_base >= 0) && (w1_base >= 0) &&
            (d1_base + 2 < D1) && (h1_base + 2 < H1) && (w1_base + 2 < W1) &&
            (2 * d1_base + 5 < D) && (2 * h1_base + 5 < H) && (2 * w1_base + 5 < W);

        const int64_t n_base = (int64_t)n * (int64_t)C * DHW;

        float thread_sum = 0.0f;

        // Parallelize across channels within group: lane_in_group strides C by group_size.
        for (int c = lane_in_group; c < C; c += group_size) {
            const int64_t c_base = n_base + (int64_t)c * DHW;

            float max2 = neg_inf_f();

            if (interior) {
                const int d0_base = 2 * d1_base;
                const int h0_base = 2 * h1_base;
                const int w0_base = 2 * w1_base;

                // Vectorization: try float4 for 4 contiguous floats, else float2, else scalar.
                const uintptr_t base_ptr = (uintptr_t)(x + c_base);
                const bool vec4_ok = ((w0_base & 3) == 0) && ((W & 3) == 0) && ((base_ptr & 0xF) == 0);
                const bool vec2_ok = (!vec4_ok) && ((w0_base & 1) == 0) && ((W & 1) == 0) && ((base_ptr & 0x7) == 0);

                if (vec4_ok) {
#pragma unroll 1
                    for (int dd = 0; dd < 6; ++dd) {
                        const int64_t d_off = (int64_t)(d0_base + dd) * HW;
#pragma unroll 1
                        for (int hh = 0; hh < 6; ++hh) {
                            const int64_t dh_off = d_off + (int64_t)(h0_base + hh) * (int64_t)W;
                            const float4* p4 = reinterpret_cast<const float4*>(x + c_base + dh_off + (int64_t)w0_base);
                            // 6 floats => one float4 (4) + one float2 (2)
                            float4 v4 = __ldg(p4);
                            max2 = fmaxf(max2, v4.x);
                            max2 = fmaxf(max2, v4.y);
                            max2 = fmaxf(max2, v4.z);
                            max2 = fmaxf(max2, v4.w);
                            const float2* p2 = reinterpret_cast<const float2*>((const float*)(p4 + 1));
                            float2 v2 = __ldg(p2);
                            max2 = fmaxf(max2, v2.x);
                            max2 = fmaxf(max2, v2.y);
                        }
                    }
                } else if (vec2_ok) {
#pragma unroll 1
                    for (int dd = 0; dd < 6; ++dd) {
                        const int64_t d_off = (int64_t)(d0_base + dd) * HW;
#pragma unroll 1
                        for (int hh = 0; hh < 6; ++hh) {
                            const int64_t dh_off = d_off + (int64_t)(h0_base + hh) * (int64_t)W;
                            const float2* p2 = reinterpret_cast<const float2*>(x + c_base + dh_off + (int64_t)w0_base);
#pragma unroll
                            for (int k = 0; k < 3; ++k) {
                                float2 v2 = __ldg(p2 + k);
                                max2 = fmaxf(max2, v2.x);
                                max2 = fmaxf(max2, v2.y);
                            }
                        }
                    }
                } else {
#pragma unroll 1
                    for (int dd = 0; dd < 6; ++dd) {
                        const int64_t d_off = (int64_t)(d0_base + dd) * HW;
#pragma unroll 1
                        for (int hh = 0; hh < 6; ++hh) {
                            const int64_t dh_off = d_off + (int64_t)(h0_base + hh) * (int64_t)W;
#pragma unroll
                            for (int ww = 0; ww < 6; ++ww) {
                                float v = ldg_f(x + c_base + dh_off + (int64_t)(w0_base + ww));
                                max2 = fmaxf(max2, v);
                            }
                        }
                    }
                }
            } else {
                // Exact semantics for boundary tiles: maxpool2 over maxpool1
#pragma unroll
                for (int pd2 = 0; pd2 < 3; ++pd2) {
                    int d1 = d1_base + pd2;
                    if ((unsigned)d1 >= (unsigned)D1) continue;
                    int d_base = d1 * 2;
#pragma unroll
                    for (int ph2 = 0; ph2 < 3; ++ph2) {
                        int h1 = h1_base + ph2;
                        if ((unsigned)h1 >= (unsigned)H1) continue;
                        int h_base = h1 * 2;
#pragma unroll
                        for (int pw2 = 0; pw2 < 3; ++pw2) {
                            int w1 = w1_base + pw2;
                            if ((unsigned)w1 >= (unsigned)W1) continue;
                            int w_base = w1 * 2;

                            float max1 = neg_inf_f();
#pragma unroll
                            for (int kd = 0; kd < 2; ++kd) {
                                int d0 = d_base + kd;
                                if ((unsigned)d0 >= (unsigned)D) continue;
                                int64_t d_off = (int64_t)d0 * HW;
#pragma unroll
                                for (int kh = 0; kh < 2; ++kh) {
                                    int h0 = h_base + kh;
                                    if ((unsigned)h0 >= (unsigned)H) continue;
                                    int64_t dh_off = d_off + (int64_t)h0 * (int64_t)W;
#pragma unroll
                                    for (int kw = 0; kw < 2; ++kw) {
                                        int w0 = w_base + kw;
                                        if ((unsigned)w0 >= (unsigned)W) continue;
                                        float v = ldg_f(x + c_base + dh_off + (int64_t)w0);
                                        max1 = fmaxf(max1, v);
                                    }
                                }
                            }
                            max2 = fmaxf(max2, max1);
                        }
                    }
                }
                if (max2 == neg_inf_f()) max2 = 0.0f;
            }

            thread_sum += max2;
        }

        // Reduce within group_size threads (16). Use warp shuffle within the containing warp.
        // group_size=16 => within the same warp; safe. We need to only reduce among lanes that share the group.
        // Compute lane within warp:
        int lane = threadIdx.x & 31;
        // Mask for active lanes in this warp: all lanes active.
        // Do a butterfly reduction but only within 16-lane subgroup: use xor pattern with boundaries.
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, thread_sum, offset);
            // Ensure we don't mix groups: subgroup is contiguous of size 16.
            // Lanes 0-15 for group0+1? Actually groups are contiguous 16 threads: group0 lanes 0-15 in warp0,
            // group1 lanes 16-31 in warp0, etc. So subgroup boundary at lane%16.
            if ((lane & 15) < offset) {
                thread_sum += other;
            }
        }

        if ((lane_in_group == 0)) {
            const int64_t y_idx =
                (((int64_t)n * (int64_t)D2 + (int64_t)od2) * (int64_t)H2 + (int64_t)oh2) * (int64_t)W2 + (int64_t)ow2;
            y[y_idx] = thread_sum;
        }
    }
}

torch::Tensor max_max_sum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [N,C,D,H,W]");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_c = x.contiguous();

    const int N = (int)x_c.size(0);
    const int C = (int)x_c.size(1);
    const int D = (int)x_c.size(2);
    const int H = (int)x_c.size(3);
    const int W = (int)x_c.size(4);

    TORCH_CHECK(D >= 2 && H >= 2 && W >= 2, "Input too small for maxpool k=2");
    const int D1 = (D - 2) / 2 + 1;
    const int H1 = (H - 2) / 2 + 1;
    const int W1 = (W - 2) / 2 + 1;

    TORCH_CHECK(D1 >= 3 && H1 >= 3 && W1 >= 3, "After pool1, tensor too small for maxpool k=3");
    const int D2 = (D1 - 3) / 3 + 1;
    const int H2 = (H1 - 3) / 3 + 1;
    const int W2 = (W1 - 3) / 3 + 1;

    auto y = torch::empty({N, 1, D2, H2, W2}, x_c.options());

    const int tiles_w = (W2 + TILE_W - 1) / TILE_W;
    const int tiles_h = (H2 + TILE_H - 1) / TILE_H;
    const int tiles_d = (D2 + TILE_D - 1) / TILE_D;
    const int64_t tiles_per_n = (int64_t)tiles_d * tiles_h * tiles_w;
    const int64_t total_tiles = (int64_t)N * tiles_per_n;

    int blocks = (int)total_tiles;
    if (blocks < 2048) blocks = 2048;      // keep SMs busy for small outputs
    if (blocks > 262144) blocks = 262144;  // cap grid

    auto stream = at::cuda::getDefaultCUDAStream();
    max2_max3_sumc_tiled_kernel<<<blocks, THREADS, 0, stream>>>(
        (const float*)x_c.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        N, C, D, H, W,
        D1, H1, W1,
        D2, H2, W2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor max_max_sum_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_max_max_sum_fusedpool_v5_tiled",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["max_max_sum_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) -> fused CUDA: MaxPool3d(k=2) -> MaxPool3d(k=3) -> sum over C keepdim=True

    Constraints:
      - fused op supports float32 CUDA contiguous NCDHW
      - MaxPool3d uses default stride=kernel_size, padding=0, dilation=1, ceil_mode=False
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels), int(out_channels), int(kernel_size),
            stride=int(stride), padding=int(padding), bias=True
        )
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.custom_ops.max_max_sum_cuda(x)
        return x