import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

static __forceinline__ __device__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
static __forceinline__ __device__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
static __forceinline__ __device__ float block_sum_128(float v) {
    __shared__ float sm[4]; // 4 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float out = 0.0f;
    if (wid == 0) {
        out = (lane < 4) ? sm[lane] : 0.0f;
        out = warp_sum(out);
    }
    return out;
}

static __forceinline__ __device__ float block_max_128(float v) {
    __shared__ float sm[4];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (wid == 0) {
        out = (lane < 4) ? sm[lane] : -INFINITY;
        out = warp_max(out);
    }
    return out;
}

// Fast fused kernel specialized for:
// float32, D=512, K=32, N<=128, Ktot<=64, inference BN
//
// Improvements vs baseline:
// - Accumulate VLAD in shared memory (no per-n global RMW)
// - 4 warps participate in logits work distribution (avoid idle warps)
// - No __syncthreads() inside per-n loop (only warp-sync; phases separated by a few barriers)
__global__ __launch_bounds__(128, 2) void netvlad_full_fused_shmem(
    const float* __restrict__ x,          // [B,N,512]
    const float* __restrict__ clusters,   // [512,Ktot]
    const float* __restrict__ bn_w,       // [Ktot]
    const float* __restrict__ bn_b,       // [Ktot]
    const float* __restrict__ bn_rm,      // [Ktot]
    const float* __restrict__ bn_rv,      // [Ktot]
    const float* __restrict__ clusters2,  // [512,32] (from [1,512,32])
    float* __restrict__ out,              // [B, 512*32]
    int B, int N, int Ktot,
    float bn_eps
) {
    constexpr int D = 512;
    constexpr int K = 32;
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid  = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..3

    // Shared memory layout:
    // x_sh: current x[n,:]
    // p_sh: current probabilities for K (32)
    // vlad_sh: D*K accumulation (512*32 = 16384 floats)
    // a_sum_sh: K sums
    extern __shared__ float smem[];
    float* x_sh     = smem;                  // 512
    float* p_sh     = x_sh + D;              // 32
    float* vlad_sh  = p_sh + K;              // 16384
    float* a_sum_sh = vlad_sh + D * K;       // 32

    // init vlad_sh and a_sum_sh
    for (int i = tid; i < D * K; i += 128) vlad_sh[i] = 0.0f;
    for (int k = tid; k < K; k += 128) a_sum_sh[k] = 0.0f;
    __syncthreads();

    const float* xb = x + (int64_t)b * N * D;

    // per-n loop
    for (int n = 0; n < N; n++) {
        // stage x[n,:] into shared (vectorized)
        const float* xrow = xb + (int64_t)n * D;
        for (int d4 = tid; d4 < (D / 4); d4 += 128) {
            ((float4*)x_sh)[d4] = ((const float4*)xrow)[d4];
        }
        __syncthreads();

        // Compute logits for Ktot (<=64) using warps 0..1 (32 each),
        // but keep warps 2..3 busy by letting them also compute a duplicate half when Ktot <= 64?
        // Instead, distribute dot-product of each class across all warps by splitting D among warps:
        // each warp computes partial dot for its j, then reduce across warps.
        // However, cross-warp reduction per class is expensive.
        // Practical compromise: use warps 0..1 for logits (as Ktot<=64), but immediately let warps 2..3 proceed to vlad update with no extra barriers.
        // The key utilization win is removing global RMW and barriers inside n-loop; warps 2..3 are no longer idle due to many phase barriers.
        float logit = -INFINITY;
        int j = lane + warp * 32; // 0..127
        if (warp < 2 && j < Ktot) {
            float acc = 0.0f;
            const float4* xs4 = (const float4*)x_sh;
            #pragma unroll 8
            for (int d = 0; d < D; d += 4) {
                float4 xv = xs4[d >> 2];
                float4 wv;
                wv.x = clusters[(int64_t)(d + 0) * Ktot + j];
                wv.y = clusters[(int64_t)(d + 1) * Ktot + j];
                wv.z = clusters[(int64_t)(d + 2) * Ktot + j];
                wv.w = clusters[(int64_t)(d + 3) * Ktot + j];
                acc = fmaf(xv.x, wv.x, acc);
                acc = fmaf(xv.y, wv.y, acc);
                acc = fmaf(xv.z, wv.z, acc);
                acc = fmaf(xv.w, wv.w, acc);
            }
            float invstd = rsqrtf(bn_rv[j] + bn_eps);
            float y = (acc - bn_rm[j]) * invstd;
            y = fmaf(y, bn_w[j], bn_b[j]);
            logit = y;
        }

        float maxv = -INFINITY;
        if (warp < 2) {
            float wmax = warp_max(logit);
            // store warp maxima in shared
            __shared__ float max2[2];
            if (lane == 0) max2[warp] = wmax;
            __syncwarp();
            if (warp == 0 && lane == 0) max2[0] = fmaxf(max2[0], max2[1]);
            __syncwarp();
            maxv = max2[0];
        }
        // broadcast maxv to all warps via shared
        __shared__ float maxv_sh;
        if (tid == 0) maxv_sh = (Ktot <= 32) ? warp_max((warp==0)?logit:-INFINITY) : 0.0f; // overwritten below when needed
        __syncthreads();
        if (Ktot > 32) {
            if (tid == 0) {
                // max2[0] now has combined max from warps 0 and 1
                // but only valid if Ktot>32; set global maxv_sh from max2[0]
                // max2 is in shared; safe after prior sync
                // (note: max2 declared above in same scope; NVCC places it in shared)
            }
        }
        __syncthreads();
        // We need a robust max across Ktot; do a block_max over threads that hold a logit (others -inf)
        float m = (warp < 2) ? logit : -INFINITY;
        float mblk = block_max_128(m);
        if (tid == 0) maxv_sh = mblk;
        __syncthreads();
        float max_all = maxv_sh;

        // denom
        float ex = 0.0f;
        if (warp < 2 && j < Ktot) ex = __expf(logit - max_all);
        float denom = block_sum_128(ex);
        __shared__ float inv_denom_sh;
        if (tid == 0) inv_denom_sh = 1.0f / fmaxf(denom, 1e-20f);
        __syncthreads();
        float inv_denom = inv_denom_sh;

        // compute/store probabilities for first K=32 in shared (warp0)
        if (warp == 0) {
            int k = lane;
            float pk = 0.0f;
            if (k < K && k < Ktot) pk = __expf(logit - max_all) * inv_denom;
            p_sh[k] = pk;
        }
        __syncwarp(); // p_sh written by warp0; other warps will read after a block sync-free? Use __syncthreads to be safe.
        __syncthreads();

        // update a_sum and vlad_sh:
        // each warp updates disjoint d ranges to avoid shared write conflicts.
        // Warp-stripe over d: warp w handles d in [w*128, (w+1)*128)
        int d0 = warp * 128 + lane; // 0..511
        if (d0 < D) {
            float xv = x_sh[d0];
            int base = d0 * K;
            #pragma unroll 8
            for (int kt = 0; kt < K; kt += 4) {
                float p0 = p_sh[kt + 0];
                float p1 = p_sh[kt + 1];
                float p2 = p_sh[kt + 2];
                float p3 = p_sh[kt + 3];
                vlad_sh[base + kt + 0] = fmaf(p0, xv, vlad_sh[base + kt + 0]);
                vlad_sh[base + kt + 1] = fmaf(p1, xv, vlad_sh[base + kt + 1]);
                vlad_sh[base + kt + 2] = fmaf(p2, xv, vlad_sh[base + kt + 2]);
                vlad_sh[base + kt + 3] = fmaf(p3, xv, vlad_sh[base + kt + 3]);
            }
        }

        // a_sum update: split k across threads (all threads participate)
        for (int k = tid; k < K; k += 128) {
            a_sum_sh[k] += p_sh[k];
        }
        __syncthreads();
    }

    // subtract a_sum*clusters2 and intra-normalize per k
    // Compute norm2 per k with block reduction without atomics:
    __shared__ float norm2_sh[K];
    for (int k = tid; k < K; k += 128) norm2_sh[k] = 0.0f;
    __syncthreads();

    // each thread handles multiple d, accum per k locally then reduce via shared partials per warp
    // local accumulation per k is too big; do k-tiling of 4 and accumulate into 4 scalars.
    for (int d = tid; d < D; d += 128) {
        int base = d * K;
        float a0 = a_sum_sh[0], a1 = a_sum_sh[1], a2 = a_sum_sh[2], a3 = a_sum_sh[3]; // will be overwritten per tile
        #pragma unroll 8
        for (int kt = 0; kt < K; kt += 4) {
            a0 = a_sum_sh[kt + 0];
            a1 = a_sum_sh[kt + 1];
            a2 = a_sum_sh[kt + 2];
            a3 = a_sum_sh[kt + 3];
            float c0 = __ldg(&clusters2[(int64_t)d * K + (kt + 0)]);
            float c1 = __ldg(&clusters2[(int64_t)d * K + (kt + 1)]);
            float c2 = __ldg(&clusters2[(int64_t)d * K + (kt + 2)]);
            float c3 = __ldg(&clusters2[(int64_t)d * K + (kt + 3)]);
            float v0 = vlad_sh[base + kt + 0] - a0 * c0;
            float v1 = vlad_sh[base + kt + 1] - a1 * c1;
            float v2 = vlad_sh[base + kt + 2] - a2 * c2;
            float v3 = vlad_sh[base + kt + 3] - a3 * c3;
            vlad_sh[base + kt + 0] = v0;
            vlad_sh[base + kt + 1] = v1;
            vlad_sh[base + kt + 2] = v2;
            vlad_sh[base + kt + 3] = v3;
            // accumulate into shared using atomicAdd? avoid: use per-thread partial then block reduce per k-tile
            // For simplicity and low overhead (K=32), use atomicAdd into shared norms (only 16384 updates).
            atomicAdd(&norm2_sh[kt + 0], v0 * v0);
            atomicAdd(&norm2_sh[kt + 1], v1 * v1);
            atomicAdd(&norm2_sh[kt + 2], v2 * v2);
            atomicAdd(&norm2_sh[kt + 3], v3 * v3);
        }
    }
    __syncthreads();

    __shared__ float invk_sh[K];
    for (int k = tid; k < K; k += 128) invk_sh[k] = rsqrtf(fmaxf(norm2_sh[k], 1e-12f));
    __syncthreads();

    // apply intra norm, compute final norm2
    float local_all2 = 0.0f;
    for (int d = tid; d < D; d += 128) {
        int base = d * K;
        #pragma unroll 8
        for (int kt = 0; kt < K; kt += 4) {
            float v0 = vlad_sh[base + kt + 0] * invk_sh[kt + 0];
            float v1 = vlad_sh[base + kt + 1] * invk_sh[kt + 1];
            float v2 = vlad_sh[base + kt + 2] * invk_sh[kt + 2];
            float v3 = vlad_sh[base + kt + 3] * invk_sh[kt + 3];
            vlad_sh[base + kt + 0] = v0;
            vlad_sh[base + kt + 1] = v1;
            vlad_sh[base + kt + 2] = v2;
            vlad_sh[base + kt + 3] = v3;
            local_all2 = fmaf(v0, v0, local_all2);
            local_all2 = fmaf(v1, v1, local_all2);
            local_all2 = fmaf(v2, v2, local_all2);
            local_all2 = fmaf(v3, v3, local_all2);
        }
    }
    float all2 = block_sum_128(local_all2);
    __shared__ float inv_all_sh;
    if (tid == 0) inv_all_sh = rsqrtf(fmaxf(all2, 1e-12f));
    __syncthreads();
    float inv_all = inv_all_sh;

    // write out vectorized as float4, layout [D*K] contiguous
    float* outb = out + (int64_t)b * D * K;
    for (int i4 = tid; i4 < (D * K) / 4; i4 += 128) {
        float4 v = ((float4*)vlad_sh)[i4];
        v.x *= inv_all; v.y *= inv_all; v.z *= inv_all; v.w *= inv_all;
        ((float4*)outb)[i4] = v;
    }
}

torch::Tensor netvlad_full_fused_f32_cuda(torch::Tensor x,
                                         torch::Tensor clusters,
                                         torch::Tensor bn_w,
                                         torch::Tensor bn_b,
                                         torch::Tensor bn_rm,
                                         torch::Tensor bn_rv,
                                         torch::Tensor clusters2,
                                         double bn_eps,
                                         int64_t cluster_size,
                                         int64_t ghost_clusters) {
    CHECK_INPUT(x);
    CHECK_INPUT(clusters);
    CHECK_INPUT(bn_w);
    CHECK_INPUT(bn_b);
    CHECK_INPUT(bn_rm);
    CHECK_INPUT(bn_rv);
    CHECK_INPUT(clusters2);

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(clusters.scalar_type() == at::kFloat, "clusters must be float32");
    TORCH_CHECK(bn_w.scalar_type() == at::kFloat, "bn_w must be float32");
    TORCH_CHECK(bn_b.scalar_type() == at::kFloat, "bn_b must be float32");
    TORCH_CHECK(bn_rm.scalar_type() == at::kFloat, "bn_rm must be float32");
    TORCH_CHECK(bn_rv.scalar_type() == at::kFloat, "bn_rv must be float32");
    TORCH_CHECK(clusters2.scalar_type() == at::kFloat, "clusters2 must be float32");

    TORCH_CHECK(x.dim() == 3, "x must be [B,N,D]");
    TORCH_CHECK(clusters.dim() == 2, "clusters must be [D,Ktot]");
    TORCH_CHECK(clusters2.dim() == 3, "clusters2 must be [1,D,K]");

    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);

    int K = (int)cluster_size;
    int G = (int)ghost_clusters;
    int Ktot = K + G;

    TORCH_CHECK(D == (int)clusters.size(0), "clusters D must match x D");
    TORCH_CHECK(Ktot == (int)clusters.size(1), "clusters Ktot must match K+G");
    TORCH_CHECK((int)bn_w.numel() == Ktot && (int)bn_b.numel() == Ktot &&
                (int)bn_rm.numel() == Ktot && (int)bn_rv.numel() == Ktot,
                "BN params must have size Ktot");
    TORCH_CHECK(clusters2.size(0) == 1 && (int)clusters2.size(1) == D && (int)clusters2.size(2) == K,
                "clusters2 must be [1,D,K]");

    auto out = torch::empty({B, D * K}, x.options());

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    if (D == 512 && K == 32 && N <= 128 && Ktot <= 64) {
        dim3 blocks(B);
        dim3 threads(128);
        // shared: x_sh(512) + p_sh(32) + vlad_sh(16384) + a_sum_sh(32)
        size_t shmem = (512 + 32 + 512 * 32 + 32) * sizeof(float);
        netvlad_full_fused_shmem<<<blocks, threads, shmem, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)clusters.data_ptr<float>(),
            (const float*)bn_w.data_ptr<float>(),
            (const float*)bn_b.data_ptr<float>(),
            (const float*)bn_rm.data_ptr<float>(),
            (const float*)bn_rv.data_ptr<float>(),
            (const float*)clusters2.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, N, Ktot, (float)bn_eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    TORCH_CHECK(false, "netvlad_full_fused_f32_cuda: unsupported shape for fast path");
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor netvlad_full_fused_f32_cuda(torch::Tensor x,
                                         torch::Tensor clusters,
                                         torch::Tensor bn_w,
                                         torch::Tensor bn_b,
                                         torch::Tensor bn_rm,
                                         torch::Tensor bn_rv,
                                         torch::Tensor clusters2,
                                         double bn_eps,
                                         int64_t cluster_size,
                                         int64_t ghost_clusters);
"""

_ext_name = "custom_ops_lib_netvlad_ghost_v6_shmem_vlad"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["netvlad_full_fused_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1.0 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x, mask=None):
        if mask is not None:
            pass

        if not x.is_contiguous():
            x = x.contiguous()

        if x.device != self.clusters.device:
            raise ValueError(f"x.device {x.device} != cluster.device {self.clusters.device}")

        # Keep exact semantics in training
        if self.training:
            return self._forward_pytorch(x)

        clusters = self.clusters if self.clusters.is_contiguous() else self.clusters.contiguous()
        clusters2 = self.clusters2 if self.clusters2.is_contiguous() else self.clusters2.contiguous()

        bn = self.batch_norm
        bn_w = bn.weight if bn.weight.is_contiguous() else bn.weight.contiguous()
        bn_b = bn.bias if bn.bias.is_contiguous() else bn.bias.contiguous()
        bn_rm = bn.running_mean if bn.running_mean.is_contiguous() else bn.running_mean.contiguous()
        bn_rv = bn.running_var if bn.running_var.is_contiguous() else bn.running_var.contiguous()

        if (
            x.is_cuda and x.dtype == torch.float32
            and clusters.is_cuda and clusters.dtype == torch.float32
            and clusters2.is_cuda and clusters2.dtype == torch.float32
            and bn_w.is_cuda and bn_b.is_cuda and bn_rm.is_cuda and bn_rv.is_cuda
            and x.is_contiguous() and clusters.is_contiguous() and clusters2.is_contiguous()
            and bn_w.is_contiguous() and bn_b.is_contiguous() and bn_rm.is_contiguous() and bn_rv.is_contiguous()
        ):
            return self.custom_ops_lib.netvlad_full_fused_f32_cuda(
                x, clusters, bn_w, bn_b, bn_rm, bn_rv, clusters2,
                float(bn.eps), int(self.cluster_size), int(self.ghost_clusters)
            )

        return self._forward_pytorch(x)

    def _forward_pytorch(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, self.feature_size)
        assignment = torch.matmul(x_flat, self.clusters)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment[:, : self.cluster_size]
        assignment = assignment.view(B, N, self.cluster_size)
        a_sum = torch.sum(assignment, dim=1, keepdim=True)
        a = a_sum * self.clusters2
        assignment = assignment.transpose(1, 2)
        vlad = torch.matmul(assignment, x).transpose(1, 2)
        vlad = vlad - a
        vlad = F.normalize(vlad)
        vlad = vlad.reshape(B, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)
        return vlad