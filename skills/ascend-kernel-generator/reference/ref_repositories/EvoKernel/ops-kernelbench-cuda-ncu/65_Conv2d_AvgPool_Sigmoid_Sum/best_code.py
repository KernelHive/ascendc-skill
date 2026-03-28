import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# CUDA extension: fused avg_pool2d -> sigmoid -> sum over C,H,W
# Input: NCHW float32 CUDA tensor (conv output)
# Output: [N] float32 CUDA tensor
#
# v4 improvements over v3:
# - Fix underutilization: use 2D grid (blockIdx.y tiles work for each n)
#   and atomicAdd one partial sum per block (amortized).
# - Hot path (k=4,s=4,p=0,count_include_pad=True):
#   * linear traversal over pooled plane for better locality/coalescing
#   * vectorized float4 loads for each 4-wide row (when aligned)
#   * 256-thread CTA + __launch_bounds__ to balance occupancy and ILP
# - General path also uses 2D grid + atomicAdd to avoid one-CTA-per-N underfill.
# ------------------------------------------------------------

fused_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template<int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    v = warp_reduce_sum(v);
    __shared__ float warp_sums[BLOCK_THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (threadIdx.x < (BLOCK_THREADS / 32)) ? warp_sums[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__device__ __forceinline__ float sum_float4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// Hot path: k=4,s=4,p=0,count_include_pad=True
// Grid: (N, tiles). Each block computes partial sum and atomicAdd to out[n].
__global__ __launch_bounds__(256, 2) void fused_avgpool4_sigmoid_sum_kernel_v4(
    const float* __restrict__ x, // [N,C,H,W]
    float* __restrict__ out,     // [N] (accumulated via atomic)
    int N, int C, int H, int W,
    int outH, int outW
) {
    constexpr int THREADS = 256;
    int n = (int)blockIdx.x;
    if (n >= N) return;

    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t base_n = (int64_t)n * (int64_t)C * HW;

    const int64_t total = (int64_t)C * (int64_t)outH * (int64_t)outW;

    // Tile the pooled-output linear space across blockIdx.y
    // Each block processes a contiguous chunk for better locality.
    const int64_t elems_per_block = (int64_t)THREADS * 8; // tune: 8 iters per thread
    int64_t start = (int64_t)blockIdx.y * elems_per_block;
    int64_t end   = start + elems_per_block;
    if (start >= total) return;
    if (end > total) end = total;

    float acc = 0.0f;

    // iterate linear pooled index p in [0,total)
    // layout: p = ((c*outH + oh)*outW + ow)
    for (int64_t p = start + threadIdx.x; p < end; p += THREADS) {
        int64_t t = p;
        int ow = (int)(t % outW);
        t /= outW;
        int oh = (int)(t % outH);
        int c  = (int)(t / outH);

        int ih0 = oh * 4;
        int iw0 = ow * 4;

        const float* ptr = x + base_n + (int64_t)c * HW + (int64_t)ih0 * (int64_t)W + (int64_t)iw0;

        float sum;
        // Vectorize each 4-wide row load if aligned and W multiple of 4.
        // ptr alignment implies iw0 is multiple of 4 (true because stride=4).
        if ((((uintptr_t)ptr) & 15u) == 0u && ((W & 3) == 0)) {
            const float4* r0 = reinterpret_cast<const float4*>(ptr + 0 * (int64_t)W);
            const float4* r1 = reinterpret_cast<const float4*>(ptr + 1 * (int64_t)W);
            const float4* r2 = reinterpret_cast<const float4*>(ptr + 2 * (int64_t)W);
            const float4* r3 = reinterpret_cast<const float4*>(ptr + 3 * (int64_t)W);
            float4 a0 = r0[0];
            float4 a1 = r1[0];
            float4 a2 = r2[0];
            float4 a3 = r3[0];
            sum = sum_float4(a0) + sum_float4(a1) + sum_float4(a2) + sum_float4(a3);
        } else {
            const float* r0 = ptr + 0 * (int64_t)W;
            const float* r1 = ptr + 1 * (int64_t)W;
            const float* r2 = ptr + 2 * (int64_t)W;
            const float* r3 = ptr + 3 * (int64_t)W;
            sum =
                (r0[0] + r0[1] + r0[2] + r0[3]) +
                (r1[0] + r1[1] + r1[2] + r1[3]) +
                (r2[0] + r2[1] + r2[2] + r2[3]) +
                (r3[0] + r3[1] + r3[2] + r3[3]);
        }

        float avg = sum * (1.0f / 16.0f);
        acc += sigmoid_fast(avg);
    }

    float block_sum = block_reduce_sum<THREADS>(acc);
    if (threadIdx.x == 0) atomicAdd(out + n, block_sum);
}

// General path: tiled over pooled outputs + atomicAdd partial sums
__global__ __launch_bounds__(256, 2) void fused_avgpool_sigmoid_sum_general_kernel_v4(
    const float* __restrict__ x,   // [N,C,H,W]
    float* __restrict__ out,       // [N] (accumulated via atomic)
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    bool count_include_pad
) {
    constexpr int THREADS = 256;
    int n = (int)blockIdx.x;
    if (n >= N) return;

    const int64_t HW = (int64_t)H * (int64_t)W;
    const int64_t base_n = (int64_t)n * (int64_t)C * HW;

    const int64_t total = (int64_t)C * (int64_t)outH * (int64_t)outW;

    const int64_t elems_per_block = (int64_t)THREADS * 4; // general path more expensive per elem
    int64_t start = (int64_t)blockIdx.y * elems_per_block;
    int64_t end   = start + elems_per_block;
    if (start >= total) return;
    if (end > total) end = total;

    float acc = 0.0f;

    for (int64_t p = start + threadIdx.x; p < end; p += THREADS) {
        int64_t t = p;
        int ow = (int)(t % outW);
        t /= outW;
        int oh = (int)(t % outH);
        int c  = (int)(t / outH);

        int hstart = oh * sH - pH;
        int wstart = ow * sW - pW;
        int hend   = hstart + kH;
        int wend   = wstart + kW;

        int h0 = hstart < 0 ? 0 : hstart;
        int w0 = wstart < 0 ? 0 : wstart;
        int h1 = hend > H ? H : hend;
        int w1 = wend > W ? W : wend;

        float sum = 0.0f;
        int count = 0;

        const float* base_nc = x + base_n + (int64_t)c * HW;
        for (int ih = h0; ih < h1; ++ih) {
            const float* row = base_nc + (int64_t)ih * (int64_t)W;
            // simple loop; compiler will unroll small kW sometimes
            for (int iw = w0; iw < w1; ++iw) {
                sum += row[iw];
                count++;
            }
        }

        float denom = count_include_pad ? (float)(kH * kW) : (count > 0 ? (float)count : 1.0f);
        float avg = sum / denom;
        acc += sigmoid_fast(avg);
    }

    float block_sum = block_reduce_sum<THREADS>(acc);
    if (threadIdx.x == 0) atomicAdd(out + n, block_sum);
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor fused_avgpool_sigmoid_sum_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW,
    bool ceil_mode,
    bool count_include_pad
) {
    TORCH_CHECK(x.is_cuda(), "fused_avgpool_sigmoid_sum_forward_cuda: x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "fused_avgpool_sigmoid_sum_forward_cuda: only float32 supported");
    TORCH_CHECK(x.dim() == 4, "fused_avgpool_sigmoid_sum_forward_cuda: expected NCHW 4D input");
    TORCH_CHECK(!ceil_mode, "fused_avgpool_sigmoid_sum_forward_cuda: ceil_mode=True not supported");
    TORCH_CHECK(x.is_contiguous(), "fused_avgpool_sigmoid_sum_forward_cuda: x must be contiguous NCHW");

    int64_t sH = sH_opt.has_value() ? sH_opt.value() : kH;
    int64_t sW = sW_opt.has_value() ? sW_opt.value() : kW;

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    TORCH_CHECK(kH > 0 && kW > 0, "kernel sizes must be > 0");
    TORCH_CHECK(sH > 0 && sW > 0, "strides must be > 0");
    TORCH_CHECK(pH >= 0 && pW >= 0, "paddings must be >= 0");

    const int64_t outH64 = (H64 + 2 * pH - kH) / sH + 1;
    const int64_t outW64 = (W64 + 2 * pW - kW) / sW + 1;
    TORCH_CHECK(outH64 >= 0 && outW64 >= 0, "computed output size is negative");

    // out will be atomically accumulated; initialize to 0
    auto out = torch::zeros({N64}, x.options().dtype(torch::kFloat32));

    bool fast4 = (kH == 4 && kW == 4 &&
                  sH == 4 && sW == 4 &&
                  pH == 0 && pW == 0 &&
                  count_include_pad);

    const int threads = 256;
    // Choose enough tiles to fill GPU; cap to avoid huge grids.
    // tiles ~ ceil(total / (threads*iters))
    int64_t total = C64 * outH64 * outW64;
    int64_t elems_per_block = fast4 ? (int64_t)threads * 8 : (int64_t)threads * 4;
    int64_t tiles = (total + elems_per_block - 1) / elems_per_block;

    // Heuristic cap: many GPUs saturate well with a few thousand CTAs per launch.
    if (tiles < 1) tiles = 1;
    if (tiles > 4096) tiles = 4096;

    dim3 blocks((unsigned int)N64, (unsigned int)tiles, 1);

    if (fast4) {
        fused_avgpool4_sigmoid_sum_kernel_v4<<<blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N64, (int)C64, (int)H64, (int)W64,
            (int)outH64, (int)outW64
        );
    } else {
        fused_avgpool_sigmoid_sum_general_kernel_v4<<<blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N64, (int)C64, (int)H64, (int)W64,
            (int)outH64, (int)outW64,
            (int)kH, (int)kW,
            (int)sH, (int)sW,
            (int)pH, (int)pW,
            count_include_pad
        );
    }

    check_cuda(cudaGetLastError(), "fused_avgpool_sigmoid_sum_forward_cuda kernel launch failed");
    return out;
}
"""

fused_cpp_src = r"""
torch::Tensor fused_avgpool_sigmoid_sum_forward_cuda(
    torch::Tensor x,
    int64_t kH, int64_t kW,
    c10::optional<int64_t> sH_opt, c10::optional<int64_t> sW_opt,
    int64_t pH, int64_t pW,
    bool ceil_mode,
    bool count_include_pad
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_avg_pool_sigmoid_sum_opt9",
    cpp_sources=fused_cpp_src,
    cuda_sources=fused_cuda_src,
    functions=["fused_avgpool_sigmoid_sum_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Keeps Conv2d as-is (cuDNN). Fuses:
      AvgPool2d(pool_kernel_size) -> Sigmoid -> Sum over [C,H,W]
    into a single CUDA op producing shape [N].

    Optimized hot path targets default AvgPool2d settings with pool_kernel_size=4.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        k = int(pool_kernel_size)
        self.kH = k
        self.kW = k
        self.sH = None
        self.sW = None
        self.pH = 0
        self.pW = 0
        self.ceil_mode = False
        self.count_include_pad = True

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4) or (not x.is_contiguous()):
            y = F.avg_pool2d(
                x,
                kernel_size=(self.kH, self.kW),
                stride=(self.kH if self.sH is None else self.sH,
                        self.kW if self.sW is None else self.sW),
                padding=(self.pH, self.pW),
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            y = torch.sigmoid(y)
            return torch.sum(y, dim=[1, 2, 3])

        return self.custom_ops.fused_avgpool_sigmoid_sum_forward_cuda(
            x,
            self.kH, self.kW,
            self.sH, self.sW,
            self.pH, self.pW,
            self.ceil_mode,
            self.count_include_pad,
        )