import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA/C++ extension for MaxPool1d forward (no indices), float32, dilation=1
# Adds a fast path specialized for kernel_size=8, stride=1, padding arbitrary (>=0).
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

static __device__ __forceinline__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Generic kernel: one thread per output (kept for correctness on non-specialized shapes).
__global__ void maxpool1d_forward_generic(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int L,
    int outL,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outL;
    if (idx >= total) return;

    int ox = idx % outL;
    int tmp = idx / outL;
    int c = tmp % C;
    int n = tmp / C;

    int base = (n * C + c) * L;
    int in_start = ox * stride - padding;

    float maxv = -INFINITY;

    // Small kernel sizes benefit from unrolling a bit, but keep generic.
    for (int k = 0; k < kernel_size; ++k) {
        int ix = in_start + k;
        if ((unsigned)ix < (unsigned)L) {
            float v = ld_ro(x + base + ix);
            maxv = fmaxf(maxv, v);
        }
    }

    y[idx] = maxv;
}

// Specialized fast kernel for kernel_size=8, stride=1.
// Mapping: one warp handles a tile of outputs for a single (n,c).
// Each lane computes OUTS_PER_LANE outputs (contiguous), using shared memory staging of the needed input segment.
//
// Grid: blocks over (n,c, tiles). blockDim can be (32, WARPS_PER_BLOCK, 1).
// Shared memory: per-warp buffer of (tile_out + 7) floats.
template<int OUTS_PER_LANE, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK, 2)
void maxpool1d_forward_k8s1_warp_tiled(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int L,
    int outL,
    int padding
) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp_in_block = threadIdx.y; // 0..WARPS_PER_BLOCK-1
    int warps_per_block = WARPS_PER_BLOCK;

    // tile of outputs covered by one warp
    constexpr int TILE_OUT = WARP * OUTS_PER_LANE;
    constexpr int SMEM_IN = TILE_OUT + 7; // need + (k-1) halo for k=8

    // 3D grid mapping packed into blockIdx.x:
    // total warps needed = N*C*ceil(outL/TILE_OUT)
    int tiles_per_row = (outL + TILE_OUT - 1) / TILE_OUT;
    int warp_global = (int)blockIdx.x * warps_per_block + warp_in_block;
    int total_warps = N * C * tiles_per_row;
    if (warp_global >= total_warps) return;

    int t = warp_global % tiles_per_row;
    int nc = warp_global / tiles_per_row;
    int c = nc % C;
    int n = nc / C;

    int out_tile_start = t * TILE_OUT;              // output index start (ox)
    int in_tile_start = out_tile_start - padding;   // input index start for ox=out_tile_start, since stride=1

    int base = (n * C + c) * L;
    int out_base = (n * C + c) * outL;

    extern __shared__ float smem[];
    // Each warp gets its own segment to avoid inter-warp sync.
    float* s = smem + warp_in_block * SMEM_IN;

    // Stage input segment [in_tile_start, in_tile_start + SMEM_IN) into shared memory.
    // Coalesced: lanes load consecutive elements.
    for (int i = lane; i < SMEM_IN; i += WARP) {
        int ix = in_tile_start + i;
        float v = -INFINITY;
        if ((unsigned)ix < (unsigned)L) v = ld_ro(x + base + ix);
        s[i] = v;
    }
    __syncwarp();

    // Each lane computes OUTS_PER_LANE outputs.
    #pragma unroll
    for (int j = 0; j < OUTS_PER_LANE; ++j) {
        int ox = out_tile_start + lane + j * WARP;
        if (ox < outL) {
            int si = (lane + j * WARP); // corresponds to input start for this ox within staged segment
            // Max over s[si + 0..7]
            float m0 = s[si + 0];
            float m1 = s[si + 1];
            float m2 = s[si + 2];
            float m3 = s[si + 3];
            float m4 = s[si + 4];
            float m5 = s[si + 5];
            float m6 = s[si + 6];
            float m7 = s[si + 7];
            float maxv = fmaxf(fmaxf(fmaxf(m0, m1), fmaxf(m2, m3)),
                               fmaxf(fmaxf(m4, m5), fmaxf(m6, m7)));
            y[out_base + ox] = maxv;
        }
    }
}

torch::Tensor maxpool1d_forward_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be > 0");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    auto xc = x.contiguous();
    int64_t N = xc.size(0);
    int64_t C = xc.size(1);
    int64_t L = xc.size(2);

    // Output length formula for MaxPool1d (dilation=1)
    int64_t outL = (L + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outL > 0, "computed outL must be > 0");

    auto y = torch::empty({N, C, outL}, torch::TensorOptions().dtype(xc.dtype()).device(xc.device()));

    const float* xp = (const float*)xc.data_ptr<float>();
    float* yp = (float*)y.data_ptr<float>();

    // Fast path: k=8, s=1. Padding arbitrary.
    if (kernel_size == 8 && stride == 1) {
        // Choose OUTS_PER_LANE to balance ILP vs registers.
        // OUTS_PER_LANE=4 => tile_out=128 outputs per warp.
        constexpr int OUTS_PER_LANE = 4;
        constexpr int WARPS_PER_BLOCK = 4; // 128 threads/block

        int tiles_per_row = (int)((outL + (32 * OUTS_PER_LANE) - 1) / (32 * OUTS_PER_LANE));
        int total_warps = (int)(N * C * tiles_per_row);
        int blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK, 1);
        size_t shmem = (size_t)WARPS_PER_BLOCK * (size_t)((32 * OUTS_PER_LANE) + 7) * sizeof(float);

        maxpool1d_forward_k8s1_warp_tiled<OUTS_PER_LANE, WARPS_PER_BLOCK>
            <<<blocks, block, shmem>>>(
                xp, yp,
                (int)N, (int)C, (int)L,
                (int)outL,
                (int)padding
            );
        return y;
    }

    // Generic fallback
    int total = (int)(N * C * outL);
    const int block = 256;
    const int grid = (total + block - 1) / block;

    maxpool1d_forward_generic<<<grid, block>>>(
        xp, yp,
        (int)N, (int)C, (int)L,
        (int)outL,
        (int)kernel_size,
        (int)stride,
        (int)padding
    );
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor maxpool1d_forward_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["maxpool1d_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Replacement model using a custom CUDA MaxPool1d forward kernel (no indices).
    Supports float32 CUDA input, 3D (N, C, L), dilation=1.
    Fast path specialized for kernel_size=8, stride=1.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        if stride is None:
            stride = kernel_size
        if dilation != 1:
            raise ValueError("This custom kernel supports dilation=1 only.")
        if return_indices:
            raise ValueError("This custom kernel does not support return_indices=True.")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.maxpool1d_forward_cuda(x, self.kernel_size, self.stride, self.padding)