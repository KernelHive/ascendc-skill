import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA op: MaxPool2d (NCHW) -> HardTanh -> mean(H,W) -> tanh, output [N,C,1,1]
# Incremental performance improvements vs baseline:
# - Specialize dominant case k=2,s=2 with 1 warp per (n,c): warp-only reduction, no shared memory, no __syncthreads()
# - Many warps per CTA (default 8) + large grid + warp-granularity grid-stride over NC to maximize parallelism / latency hiding
# - Vectorized float2 loads for 2x2 pooling windows; clamp + accumulate in registers
# - Generic fallback keeps previous block-reduce (shared) path for arbitrary k,s

fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __forceinline__ __device__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// Generic block reduce sum (shared memory). Used only in fallback kernel.
static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);
    int nwarps = (int)((blockDim.x + 31) >> 5);

    float wsum = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = wsum;
    __syncthreads();

    float bsum = 0.0f;
    if (warp == 0) {
        bsum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
    }
    return bsum;
}

// Warp-centric k=2,s=2 kernel: 1 warp computes 1 (n,c) output.
// No shared memory, no __syncthreads(). Grid-stride at warp granularity.
template<int WARPS_PER_BLOCK>
__global__ void pool_hardtanh_mean_tanh_k2s2_warp_nc(
    const float* __restrict__ x,   // [N,C,H,W]
    float* __restrict__ out,       // [N*C]
    int N, int C, int H, int W,
    int Hout, int Wout,
    float ht_min, float ht_max
) {
    constexpr int WARP = 32;
    int lane = (int)(threadIdx.x & (WARP - 1));
    int warp_in_block = (int)(threadIdx.x >> 5);
    int warps_total = (int)(gridDim.x * WARPS_PER_BLOCK);

    int64_t NC = (int64_t)N * (int64_t)C;
    int64_t warp_global = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    int pooled_elems = Hout * Wout;
    // each pooled output reads a 2x2 -> 4 floats
    // pooled_elems is typically large; each lane walks strided.
    for (int64_t idx_nc = warp_global; idx_nc < NC; idx_nc += warps_total) {
        int n = (int)(idx_nc / C);
        int c = (int)(idx_nc - (int64_t)n * C);

        const float* base = x + ((int64_t)(n * C + c) * (int64_t)H * (int64_t)W);

        float local = 0.0f;
        for (int idx = lane; idx < pooled_elems; idx += WARP) {
            int ph = idx / Wout;
            int pw = idx - ph * Wout;
            int h0 = ph << 1;  // *2
            int w0 = pw << 1;  // *2

            const float* row0 = base + (int64_t)h0 * W + w0;
            const float* row1 = row0 + W;

            // Guarded by host: W even. w0 is even. row pointers 8B aligned for float2 loads.
            const float2* r0 = (const float2*)__builtin_assume_aligned((const void*)row0, 8);
            const float2* r1 = (const float2*)__builtin_assume_aligned((const void*)row1, 8);

            float2 a = __ldg(r0);
            float2 b = __ldg(r1);

            float m0 = a.x > a.y ? a.x : a.y;
            float m1 = b.x > b.y ? b.x : b.y;
            float m  = m0 > m1 ? m0 : m1;

            local += clampf(m, ht_min, ht_max);
        }

        float sum = warp_reduce_sum(local);
        if (lane == 0) {
            float mean = sum / (float)pooled_elems;
            out[(int64_t)n * C + c] = tanhf(mean);
        }
    }
}

// Generic fallback: one block computes one (n,c) at a time (grid-stride over NC).
__global__ void pool_hardtanh_mean_tanh_generic_nc(
    const float* __restrict__ x,   // [N,C,H,W]
    float* __restrict__ out,       // [N*C]
    int N, int C, int H, int W,
    int k, int s,
    int Hout, int Wout,
    float ht_min, float ht_max
) {
    int64_t NC = (int64_t)N * (int64_t)C;
    for (int64_t idx_nc = (int64_t)blockIdx.x; idx_nc < NC; idx_nc += (int64_t)gridDim.x) {
        int n = (int)(idx_nc / C);
        int c = (int)(idx_nc - (int64_t)n * C);

        const float* base = x + ((int64_t)(n * C + c) * (int64_t)H * (int64_t)W);
        int pooled_elems = Hout * Wout;

        float local = 0.0f;
        for (int idx = (int)threadIdx.x; idx < pooled_elems; idx += (int)blockDim.x) {
            int ph = idx / Wout;
            int pw = idx - ph * Wout;
            int hstart = ph * s;
            int wstart = pw * s;

            float m = -INFINITY;
            #pragma unroll 1
            for (int kh = 0; kh < k; ++kh) {
                int h0 = hstart + kh;
                const float* row = base + (int64_t)h0 * W + wstart;
                #pragma unroll 1
                for (int kw = 0; kw < k; ++kw) {
                    float v = __ldg(row + kw);
                    m = v > m ? v : m;
                }
            }
            local += clampf(m, ht_min, ht_max);
        }

        float sum = block_reduce_sum(local);
        if (threadIdx.x == 0) {
            float mean = sum / (float)pooled_elems;
            out[(int64_t)n * C + c] = tanhf(mean);
        }
        __syncthreads(); // protect shared warp_sums across grid-stride iterations
    }
}

torch::Tensor pool_hardtanh_mean_tanh_cuda(
    torch::Tensor x,
    int64_t maxpool_kernel_size,
    int64_t maxpool_stride,
    double hardtanh_min,
    double hardtanh_max
) {
    TORCH_CHECK(x.is_cuda(), "pool_hardtanh_mean_tanh_cuda: input must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "pool_hardtanh_mean_tanh_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "pool_hardtanh_mean_tanh_cuda: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "pool_hardtanh_mean_tanh_cuda: expected NCHW 4D tensor");

    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    int64_t H64 = x.size(2);
    int64_t W64 = x.size(3);

    int64_t k64 = maxpool_kernel_size;
    int64_t s64 = maxpool_stride;

    TORCH_CHECK(k64 > 0 && s64 > 0, "pool_hardtanh_mean_tanh_cuda: kernel/stride must be > 0");
    TORCH_CHECK(H64 >= k64 && W64 >= k64, "pool_hardtanh_mean_tanh_cuda: input smaller than kernel");

    int64_t Hout64 = (H64 - k64) / s64 + 1;
    int64_t Wout64 = (W64 - k64) / s64 + 1;
    TORCH_CHECK(Hout64 > 0 && Wout64 > 0, "pool_hardtanh_mean_tanh_cuda: invalid pooled output size");

    auto out = torch::empty({N64, C64, 1, 1}, x.options());

    int N = (int)N64;
    int C = (int)C64;
    int H = (int)H64;
    int W = (int)W64;
    int k = (int)k64;
    int s = (int)s64;
    int Hout = (int)Hout64;
    int Wout = (int)Wout64;

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    int64_t NC = (int64_t)N64 * (int64_t)C64;

    float htmin = (float)hardtanh_min;
    float htmax = (float)hardtanh_max;

    // Prefer warp-kernel for the common case k=2,s=2 and safe vector-load alignment (W even).
    bool fast = (k == 2 && s == 2 && ((W & 1) == 0));

    if (fast) {
        // 8 warps/block => 256 threads. Large grid to maximize parallelism; no need for persistent-CTA.
        constexpr int WARPS_PER_BLOCK = 8;
        dim3 block(WARPS_PER_BLOCK * 32);

        // Launch many blocks (cap at grid limit). Use a heuristic multiple of SMs, but do NOT clamp too low:
        // if NC is large, we want lots of warps in flight. Use min(max_grid, max( (NC + WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK, sm*something)).
        int dev = x.get_device();
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        int sms = prop.multiProcessorCount;

        int64_t blocks_from_nc = (NC + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int64_t blocks_min = (int64_t)sms * 20; // ensure plenty of CTAs to hide latency
        int64_t blocks64 = blocks_from_nc > blocks_min ? blocks_from_nc : blocks_min;
        if (blocks64 > 65535) blocks64 = 65535;
        if (blocks64 < 1) blocks64 = 1;

        pool_hardtanh_mean_tanh_k2s2_warp_nc<WARPS_PER_BLOCK><<< (int)blocks64, block >>>(
            xp, op, N, C, H, W, Hout, Wout, htmin, htmax
        );
    } else {
        // Fallback: keep baseline-like behavior.
        int pooled_elems = Hout * Wout;
        int threads = (pooled_elems >= 8192) ? 256 : 128;

        int blocks = (int)NC;
        if (blocks > 65535) blocks = 65535;
        if (blocks < 1) blocks = 1;

        pool_hardtanh_mean_tanh_generic_nc<<<blocks, threads>>>(
            xp, op, N, C, H, W, k, s, Hout, Wout, htmin, htmax
        );
    }

    return out;
}
"""

fused_cpp_source = r"""
torch::Tensor pool_hardtanh_mean_tanh_cuda(
    torch::Tensor x,
    int64_t maxpool_kernel_size,
    int64_t maxpool_stride,
    double hardtanh_min,
    double hardtanh_max
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["pool_hardtanh_mean_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Model using PyTorch ConvTranspose2d, then a fused CUDA op for:
    MaxPool2d -> HardTanh -> mean over (H,W) -> tanh.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.maxpool_kernel_size = int(maxpool_kernel_size)
        self.maxpool_stride = int(maxpool_stride)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.pool_hardtanh_mean_tanh_cuda(
            x,
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max,
        )