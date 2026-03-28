import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA extension: optimized ECA attention (2-kernel: gate + apply) ----

eca_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

// Limit ECA kernel size for constant memory.
#ifndef ECA_MAX_K
#define ECA_MAX_K 33
#endif

__constant__ float c_conv_w[ECA_MAX_K];

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // Use expf; fast-math enabled via compilation flags.
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Full mask for <=32 threads warp
    unsigned mask = 0xffffffffu;
    // Reduce within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Kernel 1: compute gate[n, c] = sigmoid( conv1d_k( GAP(x)[n, :] )[c] )
// Input x: [N,C,H,W], output gate: [N,C]
__global__ void eca_gate_kernel(
    const float* __restrict__ x,
    float* __restrict__ gate,
    int N, int C, int H, int W,
    int k, int pad
) {
    // One block computes one (n,c) pooled value, then does conv over channels using pooled vector.
    // This is fine because C=512 and N=128 => 65536 blocks; block size small (e.g., 128).
    int nc = (int)blockIdx.x;
    int n = nc / C;
    int c = nc - n * C;

    int HW = H * W;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    float local = 0.0f;

    // Contiguous NCHW: for fixed (n,c), HW is contiguous.
    const float* xptr = x + ((n * C + c) * HW);
    for (int i = tid; i < HW; i += blockDim.x) local += xptr[i];

    // Warp reduce
    float wsum = warp_reduce_sum(local);

    // Reduce warps via shared memory (only nwarps floats)
    __shared__ float warp_sums[8]; // supports up to 8 warps = 256 threads
    if (lane == 0) warp_sums[warp] = wsum;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) total = v;
    }
    // Broadcast total within block using shared mem
    __shared__ float pooled;
    if (tid == 0) pooled = total * (1.0f / (float)HW);
    __syncthreads();

    // We need pooled values of neighbor channels for conv.
    // Avoid recompute: read pooled neighbors from gate buffer *temporarily* storing pooled values first.
    // Two-pass approach inside same kernel is not possible across different (n,c) blocks.
    // So instead, we compute pooled here and store it to a separate temporary buffer (gate used as temp),
    // and a second kernel will do conv+sigmoid? That would be 3 kernels.
    //
    // Better: do two kernels overall:
    //   (A) pooled_kernel -> pooled [N,C]
    //   (B) gate_kernel -> gate [N,C] from pooled using conv+sigmoid
    // But user asked incremental; still fine (2 kernels apply makes 3). We'll keep it 2 kernels total by
    // folding pooled+conv in one kernel that processes one (n, tileC) per block instead:
    // Compute a tile of pooled channels into shared, then conv for those channels.
    //
    // Therefore this per-(n,c) kernel is replaced below with a tiled kernel. Kept only as reference.
}

// Kernel A: compute pooled [N,C] = mean over HW.
// Grid: (N, tiles of C). Each block computes a tile of channels for one n.
template<int BLOCK_THREADS, int TILE_C>
__global__ void pooled_kernel_tiled(
    const float* __restrict__ x,
    float* __restrict__ pooled, // [N,C]
    int N, int C, int HW
) {
    int n = (int)blockIdx.y;
    int tile = (int)blockIdx.x;
    int c0 = tile * TILE_C;
    int tid = (int)threadIdx.x;

    // For each channel in tile, compute sum over HW using all threads (strided).
    // Accumulate per-thread partials for each channel.
    float acc[TILE_C];
    #pragma unroll
    for (int i = 0; i < TILE_C; i++) acc[i] = 0.0f;

    // Base pointer for this n, c0
    const float* base = x + ( (n * C + c0) * HW );

    // Iterate over HW; for each idx, load x for each channel in tile.
    // HW is small (49), TILE_C modest (e.g., 8): this is efficient and avoids extra syncs.
    for (int idx = tid; idx < HW; idx += BLOCK_THREADS) {
        const float* p = base + idx; // points at channel c0 element idx
        #pragma unroll
        for (int i = 0; i < TILE_C; i++) {
            int c = c0 + i;
            if (c < C) {
                acc[i] += p[i * HW];
            }
        }
    }

    // Reduce per-channel across threads using warp shuffles + small shared staging.
    // We do reduction per channel one by one (TILE_C is small).
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int MAX_WARPS = (BLOCK_THREADS + 31) / 32;

    __shared__ float warp_sums[TILE_C][MAX_WARPS];

    #pragma unroll
    for (int i = 0; i < TILE_C; i++) {
        float v = warp_reduce_sum(acc[i]);
        if (lane == 0) warp_sums[i][warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int i = 0; i < TILE_C; i++) {
            float v = (lane < MAX_WARPS) ? warp_sums[i][lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) {
                int c = c0 + i;
                if (c < C) pooled[n * C + c] = v * (1.0f / (float)HW);
            }
        }
    }
}

// Kernel B: gate from pooled via 1D conv + sigmoid, then apply to x -> out.
// We fuse conv+sigmoid+apply into one kernel to avoid reading gate twice.
// Grid: (N, tiles of C). Each block handles one n and TILE_C channels; threads cover HW and channels.
template<int TILE_C>
__global__ void gate_apply_kernel_tiled(
    const float* __restrict__ x,        // [N,C,HW]
    const float* __restrict__ pooled,   // [N,C]
    float* __restrict__ out,            // [N,C,HW]
    int N, int C, int HW,
    int k, int pad
) {
    int n = (int)blockIdx.y;
    int tile = (int)blockIdx.x;
    int c0 = tile * TILE_C;

    // Compute gates for TILE_C channels and keep in shared for broadcast to threads.
    __shared__ float gates[TILE_C];

    // Use first warp to compute gates (cheap: k*TILE_C loads from pooled)
    int tid = (int)threadIdx.x;
    if (tid < TILE_C) {
        int c = c0 + tid;
        float acc = 0.0f;
        if (c < C) {
            #pragma unroll
            for (int i = 0; i < ECA_MAX_K; i++) {
                // we'll guard by k below; loop unrolled to help ILP for small k.
                if (i < k) {
                    int cc = c + i - pad;
                    float y = 0.0f;
                    if (cc >= 0 && cc < C) y = pooled[n * C + cc];
                    acc += c_conv_w[i] * y;
                }
            }
        }
        gates[tid] = sigmoidf_fast(acc);
    }
    __syncthreads();

    // Apply gates to x for channels in tile.
    // Flattened layout for each (n,c): contiguous HW.
    // Threads iterate over (channel, idx) pairs for better parallelism.
    int total = TILE_C * HW;
    for (int linear = tid; linear < total; linear += blockDim.x) {
        int ci = linear / HW;
        int idx = linear - ci * HW;
        int c = c0 + ci;
        if (c < C) {
            float g = gates[ci];
            int off = (n * C + c) * HW + idx;
            out[off] = x[off] * g;
        }
    }
}

torch::Tensor eca_attention_cuda(torch::Tensor x, torch::Tensor conv_w, int64_t k) {
    CHECK_CUDA(x);
    CHECK_CUDA(conv_w);
    CHECK_F32(x);
    CHECK_F32(conv_w);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(conv_w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(conv_w.numel() == k, "conv_w must have k elements");
    TORCH_CHECK(k <= ECA_MAX_K, "k exceeds ECA_MAX_K constant memory limit");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;
    int pad = (int)((k - 1) / 2);

    auto out = torch::empty_like(x);
    // Temporary pooled buffer [N,C]
    auto pooled = torch::empty({N, C}, x.options());

    // Upload conv weights to constant memory (small copy per forward)
    cudaMemcpyToSymbol(c_conv_w, conv_w.data_ptr<float>(), k * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Kernel A launch
    constexpr int BLOCK_THREADS = 128;
    constexpr int TILE_C = 8; // balance between shared usage and reuse
    dim3 gridA((C + TILE_C - 1) / TILE_C, N, 1);
    pooled_kernel_tiled<BLOCK_THREADS, TILE_C><<<gridA, BLOCK_THREADS>>>(
        (const float*)x.data_ptr<float>(),
        (float*)pooled.data_ptr<float>(),
        N, C, HW
    );

    // Kernel B launch (fused gate+apply)
    // Threads cover TILE_C*HW = up to 8*49=392 elements => 256 threads is fine.
    dim3 gridB((C + TILE_C - 1) / TILE_C, N, 1);
    gate_apply_kernel_tiled<TILE_C><<<gridB, 256>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)pooled.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        N, C, HW, (int)k, pad
    );

    return out;
}
"""

eca_cpp_source = r"""
torch::Tensor eca_attention_cuda(torch::Tensor x, torch::Tensor conv_w, int64_t k);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_eca_attention_v2",
    cpp_sources=eca_cpp_source,
    cuda_sources=eca_cuda_source,
    functions=["eca_attention_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# ---- Model replacement ----

class ModelNew(nn.Module):
    """ECA attention using an optimized custom CUDA op (tiled GAP + fused gate/apply)."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.k = int(k)
        self.pad = (self.k - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.pad, bias=False)

    def forward(self, x):
        w = self.conv.weight.view(-1).contiguous()
        return custom_ops_lib.eca_attention_cuda(x.contiguous(), w, self.k)