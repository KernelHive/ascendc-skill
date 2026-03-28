import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# NetVLAD (no ghost clusters) forward optimized CUDA.
# BN remains in PyTorch; CUDA consumes BN logits (bn_out).
#
# Key improvement vs baseline:
# - Replace vlad_accum_{scalar,vec4} with a K-tiled kernel that reuses x via shared memory.
#   For fixed (b, d4-tile) block computes KTILE clusters (k) at once:
#     load x[b,n, d4vec] once -> shared
#     load p scalars for KTILE -> registers
#     update KTILE accumulators
#   This reduces redundant global reads of x by ~KTILE and improves cache locality.
# -----------------------------------------------------------------------------

netvlad_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

// ------------------ Warp/Block reductions ------------------
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    return v;
}

// Kernel 1: softmax probabilities per row (BN)
__global__ void softmax_probs_f32(
    const float* __restrict__ bn_out, // [BN,K]
    float* __restrict__ P,            // [BN,K]
    int BN, int K
) {
    int row = (int)blockIdx.x;
    if (row >= BN) return;
    const float* in = bn_out + (int64_t)row * K;
    float* out = P + (int64_t)row * K;

    float vmax = -INFINITY;
    for (int k = threadIdx.x; k < K; k += blockDim.x) vmax = fmaxf(vmax, in[k]);
    vmax = block_max(vmax);
    __shared__ float sh_m;
    if (threadIdx.x == 0) sh_m = vmax;
    __syncthreads();
    float m = sh_m;

    float vsum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) vsum += __expf(in[k] - m);
    vsum = block_sum(vsum);
    __shared__ float sh_inv;
    if (threadIdx.x == 0) sh_inv = 1.0f / vsum;
    __syncthreads();
    float inv = sh_inv;

    for (int k = threadIdx.x; k < K; k += blockDim.x) out[k] = __expf(in[k] - m) * inv;
}

// Kernel 2: a_sum[b,k] = sum_n P[b,n,k]
__global__ void asum_f32(
    const float* __restrict__ P, // [BN,K]
    float* __restrict__ a_sum,   // [B,K]
    int B, int N, int K
) {
    int bk = (int)blockIdx.x;
    int b = bk / K;
    int k = bk - b * K;
    if (b >= B) return;

    float acc = 0.0f;
    int tid = threadIdx.x;
    int base = b * N;
    for (int n = tid; n < N; n += blockDim.x) {
        acc += P[(int64_t)(base + n) * K + k];
    }
    acc = block_sum(acc);
    if (tid == 0) a_sum[b * K + k] = acc;
}

// Kernel 3: VLAD accumulation with K-tiling and shared-memory x reuse (vec4 path).
// Grid: (b, d4_tile, k_tile). Each CTA computes KTILE clusters for one (b, d4_tile).
template<int KTILE, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void vlad_accum_k_tiled_vec4_f32(
    const float* __restrict__ x,  // [B,N,D]
    const float* __restrict__ P,  // [BN,K]
    float* __restrict__ vlad,     // [B,K,D]
    int B, int N, int D, int K
) {
    int b = (int)blockIdx.x;
    int d4_tile = (int)blockIdx.y;
    int k_tile = (int)blockIdx.z;

    if (b >= B) return;

    int D4 = D >> 2;
    int idx4 = d4_tile * BLOCK_THREADS + threadIdx.x;
    bool in_range = (idx4 < D4);

    int k0 = k_tile * KTILE;
    if (k0 >= K) return;

    // shared x tile: one float4 per thread
    extern __shared__ float4 shx4[];
    // accumulators for KTILE clusters
    float4 acc[KTILE];
    #pragma unroll
    for (int t = 0; t < KTILE; ++t) {
        acc[t].x = acc[t].y = acc[t].z = acc[t].w = 0.0f;
    }

    const float4* x4 = (const float4*)(x + ((int64_t)b * N) * D);
    int baseBN = b * N;

    for (int n = 0; n < N; ++n) {
        // load x vector once per n
        float4 xv;
        if (in_range) {
            xv = x4[(int64_t)n * D4 + idx4];
        } else {
            xv.x = xv.y = xv.z = xv.w = 0.0f;
        }
        shx4[threadIdx.x] = xv;
        __syncthreads();

        float4 xvs = shx4[threadIdx.x];

        // update all clusters in the tile
        #pragma unroll
        for (int t = 0; t < KTILE; ++t) {
            int k = k0 + t;
            if (k < K && in_range) {
                float p = P[(int64_t)(baseBN + n) * K + k];
                acc[t].x = fmaf(p, xvs.x, acc[t].x);
                acc[t].y = fmaf(p, xvs.y, acc[t].y);
                acc[t].z = fmaf(p, xvs.z, acc[t].z);
                acc[t].w = fmaf(p, xvs.w, acc[t].w);
            }
        }
        __syncthreads();
    }

    if (in_range) {
        // store accumulators for each k
        #pragma unroll
        for (int t = 0; t < KTILE; ++t) {
            int k = k0 + t;
            if (k < K) {
                float4* v4 = (float4*)(vlad + (((int64_t)b * K + k) * D));
                v4[idx4] = acc[t];
            }
        }
    }
}

// Scalar fallback: original mapping (b,k,d-tile)
__global__ void vlad_accum_scalar_f32(
    const float* __restrict__ x,  // [B,N,D]
    const float* __restrict__ P,  // [BN,K]
    float* __restrict__ vlad,     // [B,K,D]
    int B, int N, int D, int K
) {
    int b = (int)blockIdx.x;
    int k = (int)blockIdx.y;
    int tile = (int)blockIdx.z;

    int d0 = tile * blockDim.x + threadIdx.x;
    if (b >= B || k >= K || d0 >= D) return;

    float acc = 0.0f;
    int64_t x_base = ((int64_t)b * N) * D;
    int baseBN = b * N;
    for (int n = 0; n < N; ++n) {
        float p = P[(int64_t)(baseBN + n) * K + k];
        acc = fmaf(p, x[x_base + (int64_t)n * D + d0], acc);
    }
    vlad[((int64_t)b * K + k) * D + d0] = acc;
}

// Kernel 4: subtract a_sum*clusters2 and intra-norm over D for each (b,k), in-place
__global__ void subtract_intra_f32(
    float* __restrict__ vlad,            // [B,K,D]
    const float* __restrict__ a_sum,     // [B,K]
    const float* __restrict__ clusters2, // [D,K]
    int B, int D, int K
) {
    int bk = (int)blockIdx.x;
    int b = bk / K;
    int k = bk - b * K;
    if (b >= B) return;

    float asum = a_sum[b * K + k];

    float sumsq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = asum * __ldg(&clusters2[d * K + k]);
        float v = vlad[((int64_t)b * K + k) * D + d] - a;
        vlad[((int64_t)b * K + k) * D + d] = v;
        sumsq += v * v;
    }
    sumsq = block_sum(sumsq);
    __shared__ float sh_inv;
    if (threadIdx.x == 0) sh_inv = rsqrtf(sumsq + 1e-12f);
    __syncthreads();
    float inv = sh_inv;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        vlad[((int64_t)b * K + k) * D + d] *= inv;
    }
}

// Kernel 5: flatten to out_tmp[B,DK] where i=d*K+k, and compute partial sumsq per chunk
__global__ void flatten_partials_f32(
    const float* __restrict__ vlad,      // [B,K,D]
    float* __restrict__ out_tmp,         // [B,DK]
    float* __restrict__ partial_sums,    // [B,numChunks]
    int B, int D, int K, int numChunks
) {
    int b = (int)blockIdx.x;
    int chunk = (int)blockIdx.y;
    if (b >= B) return;

    int DK = D * K;
    int chunkSize = (DK + numChunks - 1) / numChunks;
    int start = chunk * chunkSize;
    int end = min(DK, start + chunkSize);

    float psum = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int d = i / K;
        int k = i - d * K;
        float v = vlad[((int64_t)b * K + k) * D + d];
        out_tmp[(int64_t)b * DK + i] = v;
        psum += v * v;
    }
    psum = block_sum(psum);
    if (threadIdx.x == 0) partial_sums[(int64_t)b * numChunks + chunk] = psum;
}

// Kernel 6: finalize final L2 norm per b and scale out_tmp -> out
__global__ void finalize_f32(
    const float* __restrict__ out_tmp,      // [B,DK]
    const float* __restrict__ partial_sums, // [B,numChunks]
    float* __restrict__ out,                // [B,DK]
    int B, int DK, int numChunks
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < numChunks; i += blockDim.x) {
        sumsq += partial_sums[(int64_t)b * numChunks + i];
    }
    sumsq = block_sum(sumsq);
    __shared__ float sh_inv;
    if (threadIdx.x == 0) sh_inv = rsqrtf(sumsq + 1e-12f);
    __syncthreads();
    float inv = sh_inv;

    for (int i = threadIdx.x; i < DK; i += blockDim.x) {
        out[(int64_t)b * DK + i] = out_tmp[(int64_t)b * DK + i] * inv;
    }
}

torch::Tensor net_vlad_no_ghost_clusters_cuda(
    torch::Tensor x,          // [B,N,D]
    torch::Tensor bn_out,     // [B*N,K]
    torch::Tensor clusters2   // [1,D,K]
) {
    CHECK_CUDA(x);
    CHECK_CUDA(bn_out);
    CHECK_CUDA(clusters2);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(bn_out);
    CHECK_CONTIGUOUS(clusters2);
    CHECK_FLOAT(x);
    CHECK_FLOAT(bn_out);
    CHECK_FLOAT(clusters2);

    TORCH_CHECK(x.dim() == 3, "x must be [B,N,D]");
    TORCH_CHECK(bn_out.dim() == 2, "bn_out must be [B*N,K]");
    TORCH_CHECK(clusters2.dim() == 3, "clusters2 must be [1,D,K]");
    TORCH_CHECK(clusters2.size(0) == 1, "clusters2 must have first dim 1");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);
    int K = (int)bn_out.size(1);
    TORCH_CHECK((int64_t)B * N == bn_out.size(0), "bn_out first dim must be B*N");
    TORCH_CHECK((int)clusters2.size(1) == D && (int)clusters2.size(2) == K, "clusters2 must be [1,D,K]");

    auto opts = x.options();
    int64_t BN = (int64_t)B * N;

    torch::Tensor P = torch::empty({BN, K}, opts);
    torch::Tensor a_sum = torch::empty({B, K}, opts);
    torch::Tensor vlad = torch::empty({B, K, D}, opts);

    int DK = D * K;
    torch::Tensor out_tmp = torch::empty({B, DK}, opts);
    torch::Tensor out = torch::empty({B, DK}, opts);

    // Kernel 1: softmax
    {
        int threads = 128; // K small (e.g., 32)
        int blocks = (int)BN;
        softmax_probs_f32<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            bn_out.data_ptr<float>(), P.data_ptr<float>(), (int)BN, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 2: a_sum
    {
        int threads = 256;
        int blocks = B * K;
        asum_f32<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            P.data_ptr<float>(), a_sum.data_ptr<float>(), B, N, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 3: vlad accumulate
    bool can_vec4 = false;
    if ((D % 4) == 0) {
        uintptr_t ax = (uintptr_t)x.data_ptr<float>();
        uintptr_t av = (uintptr_t)vlad.data_ptr<float>();
        can_vec4 = ((ax & 0xF) == 0) && ((av & 0xF) == 0);
    }

    if (can_vec4) {
        // tuned for K=32, D=512: KTILE=4 -> 8 k-tiles
        constexpr int KTILE = 4;
        constexpr int THREADS = 128; // one thread per d4 element in tile
        int D4 = D / 4;
        int d4_tiles = (D4 + THREADS - 1) / THREADS;
        int k_tiles = (K + KTILE - 1) / KTILE;
        dim3 grid(B, d4_tiles, k_tiles);
        size_t shmem = THREADS * sizeof(float4);
        vlad_accum_k_tiled_vec4_f32<KTILE, THREADS><<<grid, THREADS, shmem, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(), P.data_ptr<float>(), vlad.data_ptr<float>(), B, N, D, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        int threads = 256;
        int tiles = (D + threads - 1) / threads;
        dim3 grid(B, K, tiles);
        vlad_accum_scalar_f32<<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(), P.data_ptr<float>(), vlad.data_ptr<float>(), B, N, D, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 4: subtract + intra norm
    {
        int threads = 256;
        int blocks = B * K;
        const float* c2 = clusters2.data_ptr<float>(); // treat as [D,K]
        subtract_intra_f32<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            vlad.data_ptr<float>(), a_sum.data_ptr<float>(), c2, B, D, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 5: flatten + partial sums
    int numChunks = 16;
    torch::Tensor partial = torch::empty({B, numChunks}, opts);
    {
        int threads = 256;
        dim3 grid(B, numChunks, 1);
        flatten_partials_f32<<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            vlad.data_ptr<float>(), out_tmp.data_ptr<float>(), partial.data_ptr<float>(),
            B, D, K, numChunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel 6: finalize
    {
        int threads = 256;
        int blocks = B;
        finalize_f32<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            out_tmp.data_ptr<float>(), partial.data_ptr<float>(), out.data_ptr<float>(),
            B, DK, numChunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

netvlad_cpp_source = r"""
torch::Tensor net_vlad_no_ghost_clusters_cuda(torch::Tensor x, torch::Tensor bn_out, torch::Tensor clusters2);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_netvlad_no_ghost_opt_inc3",
    cpp_sources=netvlad_cpp_source,
    cuda_sources=netvlad_cuda_source,
    functions=["net_vlad_no_ghost_clusters_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        if ghost_clusters != 0:
            raise ValueError("ModelNew custom kernel supports ghost_clusters==0 only.")
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        if mask is not None:
            pass

        if x.device != self.clusters.device:
            raise ValueError(f"x.device {x.device} != cluster.device {self.clusters.device}")

        if not x.is_contiguous():
            x = x.contiguous()
        clusters = self.clusters.contiguous() if not self.clusters.is_contiguous() else self.clusters
        clusters2 = self.clusters2.contiguous() if not self.clusters2.is_contiguous() else self.clusters2

        x2 = x.view(-1, self.feature_size)          # [B*N, D]
        assignment = torch.matmul(x2, clusters)     # [B*N, K]
        assignment = self.batch_norm(assignment)    # [B*N, K]

        out = custom_ops_lib.net_vlad_no_ghost_clusters_cuda(
            x, assignment.contiguous(), clusters2
        )
        return out