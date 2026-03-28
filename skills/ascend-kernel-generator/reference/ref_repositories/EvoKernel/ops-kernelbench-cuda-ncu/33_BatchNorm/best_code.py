import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__inline__ __device__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    int nwarps = (blockDim.x + 31) >> 5;
    float out = (threadIdx.x < nwarps) ? warp_sums[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum(out);
    return out;
}

static __inline__ __device__ bool aligned16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// NHWC stats accumulation: x is [NHW, C] contiguous (channels_last)
__global__ __launch_bounds__(256, 2)
void bn2d_nhwc_stats_tiled(
    const float* __restrict__ x,  // [NHW, C]
    float* __restrict__ sum,      // [C]
    float* __restrict__ sumsq,    // [C]
    int NHW, int C, int tile_elems
) {
    int c0 = (int)blockIdx.x * 4; // 4 channels per vector
    int tile = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    if (c0 >= C) return;

    int start = tile * tile_elems;
    int end = start + tile_elems;
    if (start >= NHW) return;
    if (end > NHW) end = NHW;

    float4 acc_sum = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc_sumsq = make_float4(0.f, 0.f, 0.f, 0.f);

    const float* base = x + (int64_t)start * (int64_t)C + c0;
    bool vec_ok = (c0 + 3 < C) && aligned16(base) && ((C & 3) == 0);

    if (vec_ok) {
        // small ILP: 2 loads per loop when possible
        int i = start + tid;
        int stride = (int)blockDim.x;
        for (; i + stride < end; i += stride * 2) {
            const float4* p40 = reinterpret_cast<const float4*>(x + (int64_t)i * (int64_t)C + c0);
            const float4* p41 = reinterpret_cast<const float4*>(x + (int64_t)(i + stride) * (int64_t)C + c0);
            float4 v0 = *p40;
            float4 v1 = *p41;

            acc_sum.x += v0.x + v1.x;
            acc_sum.y += v0.y + v1.y;
            acc_sum.z += v0.z + v1.z;
            acc_sum.w += v0.w + v1.w;

            acc_sumsq.x = fmaf(v0.x, v0.x, acc_sumsq.x);
            acc_sumsq.y = fmaf(v0.y, v0.y, acc_sumsq.y);
            acc_sumsq.z = fmaf(v0.z, v0.z, acc_sumsq.z);
            acc_sumsq.w = fmaf(v0.w, v0.w, acc_sumsq.w);

            acc_sumsq.x = fmaf(v1.x, v1.x, acc_sumsq.x);
            acc_sumsq.y = fmaf(v1.y, v1.y, acc_sumsq.y);
            acc_sumsq.z = fmaf(v1.z, v1.z, acc_sumsq.z);
            acc_sumsq.w = fmaf(v1.w, v1.w, acc_sumsq.w);
        }
        for (; i < end; i += stride) {
            const float4* p4 = reinterpret_cast<const float4*>(x + (int64_t)i * (int64_t)C + c0);
            float4 v = *p4;
            acc_sum.x += v.x; acc_sum.y += v.y; acc_sum.z += v.z; acc_sum.w += v.w;
            acc_sumsq.x = fmaf(v.x, v.x, acc_sumsq.x);
            acc_sumsq.y = fmaf(v.y, v.y, acc_sumsq.y);
            acc_sumsq.z = fmaf(v.z, v.z, acc_sumsq.z);
            acc_sumsq.w = fmaf(v.w, v.w, acc_sumsq.w);
        }
    } else {
        for (int i = start + tid; i < end; i += (int)blockDim.x) {
            const float* p = x + (int64_t)i * (int64_t)C + c0;
            float v0 = (c0 + 0 < C) ? ldg_f32(p + 0) : 0.f;
            float v1 = (c0 + 1 < C) ? ldg_f32(p + 1) : 0.f;
            float v2 = (c0 + 2 < C) ? ldg_f32(p + 2) : 0.f;
            float v3 = (c0 + 3 < C) ? ldg_f32(p + 3) : 0.f;
            acc_sum.x += v0; acc_sum.y += v1; acc_sum.z += v2; acc_sum.w += v3;
            acc_sumsq.x = fmaf(v0, v0, acc_sumsq.x);
            acc_sumsq.y = fmaf(v1, v1, acc_sumsq.y);
            acc_sumsq.z = fmaf(v2, v2, acc_sumsq.z);
            acc_sumsq.w = fmaf(v3, v3, acc_sumsq.w);
        }
    }

    float s0 = block_reduce_sum(acc_sum.x);
    float s1 = block_reduce_sum(acc_sum.y);
    float s2 = block_reduce_sum(acc_sum.z);
    float s3 = block_reduce_sum(acc_sum.w);

    float q0 = block_reduce_sum(acc_sumsq.x);
    float q1 = block_reduce_sum(acc_sumsq.y);
    float q2 = block_reduce_sum(acc_sumsq.z);
    float q3 = block_reduce_sum(acc_sumsq.w);

    if (tid == 0) {
        if (c0 + 0 < C) { atomicAdd(sum + (c0 + 0), s0); atomicAdd(sumsq + (c0 + 0), q0); }
        if (c0 + 1 < C) { atomicAdd(sum + (c0 + 1), s1); atomicAdd(sumsq + (c0 + 1), q1); }
        if (c0 + 2 < C) { atomicAdd(sum + (c0 + 2), s2); atomicAdd(sumsq + (c0 + 2), q2); }
        if (c0 + 3 < C) { atomicAdd(sum + (c0 + 3), s3); atomicAdd(sumsq + (c0 + 3), q3); }
    }
}

__global__ __launch_bounds__(256, 4)
void bn2d_compute_ab(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ alpha, // [C]
    float* __restrict__ beta,  // [C]
    int NHW, int C, float eps
) {
    int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (c >= C) return;

    float invm = 1.0f / (float)NHW;
    float m = sum[c] * invm;
    float ex2 = sumsq[c] * invm;
    float var = ex2 - m * m;
    var = var > 0.0f ? var : 0.0f;
    float invstd = rsqrtf(var + eps);

    float g = ldg_f32(weight + c);
    float b = ldg_f32(bias + c);

    float a = invstd * g;
    alpha[c] = a;
    beta[c]  = b - m * a;
}

__global__ __launch_bounds__(256, 2)
void bn2d_nhwc_apply_ab(
    const float* __restrict__ x,      // [NHW, C]
    const float* __restrict__ alpha,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [NHW, C]
    int NHW, int C
) {
    // vectorized path for C%4==0 (common for BN)
    if ((C & 3) == 0) {
        int C4 = C >> 2;
        int total_vec = NHW * C4;

        int tid = (int)threadIdx.x;
        int idx0 = ((int)blockIdx.x * (int)blockDim.x + tid) * 2; // unroll x2

        if (idx0 >= total_vec) return;

        // process up to 2 vectors per thread
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx = idx0 + u;
            if (idx >= total_vec) break;

            int i = idx / C4;
            int c4 = idx - i * C4;
            int c0 = c4 * 4;

            const float4* x4 = reinterpret_cast<const float4*>(x + (int64_t)i * (int64_t)C);
            float4 v = x4[c4];

            float a0 = ldg_f32(alpha + c0 + 0), bt0 = ldg_f32(beta + c0 + 0);
            float a1 = ldg_f32(alpha + c0 + 1), bt1 = ldg_f32(beta + c0 + 1);
            float a2 = ldg_f32(alpha + c0 + 2), bt2 = ldg_f32(beta + c0 + 2);
            float a3 = ldg_f32(alpha + c0 + 3), bt3 = ldg_f32(beta + c0 + 3);

            float4 o;
            o.x = fmaf(v.x, a0, bt0);
            o.y = fmaf(v.y, a1, bt1);
            o.z = fmaf(v.z, a2, bt2);
            o.w = fmaf(v.w, a3, bt3);

            float4* y4 = reinterpret_cast<float4*>(y + (int64_t)i * (int64_t)C);
            y4[c4] = o;
        }
    } else {
        int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
        int total = NHW * C;
        if (t >= total) return;
        int i = t / C;
        int c = t - i * C;
        float v = x[(int64_t)i * (int64_t)C + c];
        float o = fmaf(v, ldg_f32(alpha + c), ldg_f32(beta + c));
        y[(int64_t)i * (int64_t)C + c] = o;
    }
}

torch::Tensor bn2d_train_fwd_cuda_nhwc(torch::Tensor x_nhwc, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x_nhwc.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x_nhwc.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x_nhwc.is_contiguous(at::MemoryFormat::ChannelsLast), "x must be contiguous channels_last (NHWC)");
    TORCH_CHECK(x_nhwc.dim() == 4, "x must be 4D");

    int N = (int)x_nhwc.size(0);
    int H = (int)x_nhwc.size(1);
    int W = (int)x_nhwc.size(2);
    int C = (int)x_nhwc.size(3);
    TORCH_CHECK(weight.numel() == C && bias.numel() == C, "weight/bias must be [C]");

    int NHW = N * H * W;

    auto y_nhwc = torch::empty_like(x_nhwc);
    auto sum = torch::zeros({C}, x_nhwc.options());
    auto sumsq = torch::zeros({C}, x_nhwc.options());

    // Compute alpha/beta instead of mean/invstd to reduce work/loads in apply
    auto alpha = torch::empty({C}, x_nhwc.options());
    auto beta_t = torch::empty({C}, x_nhwc.options());

    const int threads = 256;

    // Stats tiling
    int tile_elems = 8192; // slightly larger to reduce block count (and atomic pressure) at huge NHW
    int grid_x = (C + 3) / 4;
    int grid_y_full = (NHW + tile_elems - 1) / tile_elems;

    // Cap grid_y to avoid overlaunch; each block covers a disjoint tile range.
    // To keep correctness when capping, we increase tile size accordingly to cover full NHW with fewer tiles.
    int dev = x_nhwc.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    int max_total_blocks = sm * 32;
    long long total_blocks = (long long)grid_x * (long long)grid_y_full;
    if (total_blocks > max_total_blocks && grid_x > 0) {
        int capped_grid_y = max_total_blocks / grid_x;
        if (capped_grid_y < 1) capped_grid_y = 1;
        // recompute tile_elems so capped_grid_y tiles cover NHW
        tile_elems = (NHW + capped_grid_y - 1) / capped_grid_y;
        // round tile_elems up to a multiple of 256 for nicer scheduling
        tile_elems = ((tile_elems + 255) / 256) * 256;
    }
    int grid_y = (NHW + tile_elems - 1) / tile_elems;

    dim3 grid((unsigned)grid_x, (unsigned)grid_y, 1);
    bn2d_nhwc_stats_tiled<<<grid, threads>>>(
        (const float*)x_nhwc.data_ptr<float>(),
        (float*)sum.data_ptr<float>(),
        (float*)sumsq.data_ptr<float>(),
        NHW, C, tile_elems
    );

    int blocks_c = (C + threads - 1) / threads;
    bn2d_compute_ab<<<blocks_c, threads>>>(
        (const float*)sum.data_ptr<float>(),
        (const float*)sumsq.data_ptr<float>(),
        (const float*)weight.data_ptr<float>(),
        (const float*)bias.data_ptr<float>(),
        (float*)alpha.data_ptr<float>(),
        (float*)beta_t.data_ptr<float>(),
        NHW, C, (float)eps
    );

    // Apply: use unroll-by-2 on vector index space; adjust blocks accordingly.
    if ((C & 3) == 0) {
        int C4 = C >> 2;
        int total_vec = NHW * C4;
        int total_threads = (total_vec + 1) / 2; // because each thread handles 2 vecs
        int blocks_apply = (total_threads + threads - 1) / threads;
        bn2d_nhwc_apply_ab<<<blocks_apply, threads>>>(
            (const float*)x_nhwc.data_ptr<float>(),
            (const float*)alpha.data_ptr<float>(),
            (const float*)beta_t.data_ptr<float>(),
            (float*)y_nhwc.data_ptr<float>(),
            NHW, C
        );
    } else {
        int total = NHW * C;
        int blocks_apply = (total + threads - 1) / threads;
        bn2d_nhwc_apply_ab<<<blocks_apply, threads>>>(
            (const float*)x_nhwc.data_ptr<float>(),
            (const float*)alpha.data_ptr<float>(),
            (const float*)beta_t.data_ptr<float>(),
            (float*)y_nhwc.data_ptr<float>(),
            NHW, C
        );
    }

    return y_nhwc;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor bn2d_train_fwd_cuda_nhwc(torch::Tensor x_nhwc, torch::Tensor weight, torch::Tensor bias, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_bn2d_train_ops_nhwc_opt3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["bn2d_train_fwd_cuda_nhwc"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    nn.BatchNorm2d(num_features) optimized for training forward on CUDA when input is channels_last.
    Falls back to PyTorch BatchNorm otherwise.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=int(num_features),
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
            track_running_stats=bool(track_running_stats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not x.is_cuda) or (x.dtype != torch.float32) or (x.dim() != 4):
            return self.bn(x)

        if (not self.bn.affine) or (self.bn.weight is None) or (self.bn.bias is None):
            return self.bn(x)

        if not x.is_contiguous(memory_format=torch.channels_last):
            return self.bn(x)

        w = self.bn.weight
        b = self.bn.bias
        if (w.device != x.device) or (b.device != x.device):
            w = w.to(device=x.device)
            b = b.to(device=x.device)
        if (not w.is_contiguous()) or (not b.is_contiguous()):
            w = w.contiguous()
            b = b.contiguous()

        y = custom_ops_lib.bn2d_train_fwd_cuda_nhwc(x, w, b, float(self.bn.eps))

        if self.bn.track_running_stats:
            with torch.no_grad():
                torch.nn.functional.batch_norm(
                    x,
                    self.bn.running_mean,
                    self.bn.running_var,
                    weight=None,
                    bias=None,
                    training=True,
                    momentum=self.bn.momentum,
                    eps=self.bn.eps,
                )
        return y