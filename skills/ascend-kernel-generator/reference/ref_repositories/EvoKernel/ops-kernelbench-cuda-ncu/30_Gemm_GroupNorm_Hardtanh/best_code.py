import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_reduce.cuh>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    return v;
}

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Fast path: group_size == 512 (C=8192, G=16). One warp handles one (n,g).
// Each lane processes exactly 16 floats => perfectly coalesced.
__global__ __launch_bounds__(256, 2)
void gn_hardtanh_g512_warp_kernel(
    const float* __restrict__ x,     // [N, C]
    const float* __restrict__ gamma, // [C]
    const float* __restrict__ beta,  // [C]
    float* __restrict__ out,         // [N, C]
    int N, int C, int G,
    float eps, float ht_min, float ht_max
) {
    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;     // 0..7
    int lane    = tid & 31;     // 0..31
    int warps_per_block = (int)blockDim.x >> 5; // 8
    int global_warp = (int)blockIdx.x * warps_per_block + warp_id;

    int NG = N * G;
    if (global_warp >= NG) return;

    int n = global_warp / G;
    int g = global_warp - n * G;

    // group_size = 512, group offset
    int c0 = g * 512;
    int base = n * C + c0;

    // 16 elements per lane: indices lane + k*32, k=0..15
    float sum = 0.0f;
    float sumsq = 0.0f;

    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int i = lane + (k << 5);
        float v = x[base + i];
        sum += v;
        sumsq += v * v;
    }

    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 512.0f);
    float msq  = __shfl_sync(0xffffffffu, sumsq, 0) * (1.0f / 512.0f);
    float var = msq - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    float invstd = rsqrtf(var + eps);

    // Vectorized apply if aligned (likely true for contiguous float tensors).
    uintptr_t xa = (uintptr_t)(x + base);
    uintptr_t oa = (uintptr_t)(out + base);
    uintptr_t ga = (uintptr_t)(gamma + c0);
    uintptr_t ba = (uintptr_t)(beta + c0);
    bool vec_ok = ((xa | oa | ga | ba) & 15) == 0;

    if (vec_ok) {
        // 512 floats = 128 float4. Each lane handles 4 float4 (total 16 floats).
        const float4* __restrict__ x4 = (const float4*)(x + base);
        const float4* __restrict__ g4 = (const float4*)(gamma + c0);
        const float4* __restrict__ b4 = (const float4*)(beta + c0);
        float4* __restrict__ o4 = (float4*)(out + base);

        // lane owns float indices: lane + 32*k -> float4 index = (lane + 32*k)/4.
        // This implies lane must map to 4 consecutive lanes per float4; we instead
        // assign each lane contiguous float4 indices: lane + 32*m over float4 space.
        // Total float4 = 128. 128/32 = 4 per lane. Perfect.
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int i4 = lane + (m << 5); // lane + 32*m
            float4 xv = x4[i4];
            float4 gv = g4[i4];
            float4 bv = b4[i4];

            float y0 = (xv.x - mean) * invstd; y0 = y0 * gv.x + bv.x; y0 = clampf(y0, ht_min, ht_max);
            float y1 = (xv.y - mean) * invstd; y1 = y1 * gv.y + bv.y; y1 = clampf(y1, ht_min, ht_max);
            float y2 = (xv.z - mean) * invstd; y2 = y2 * gv.z + bv.z; y2 = clampf(y2, ht_min, ht_max);
            float y3 = (xv.w - mean) * invstd; y3 = y3 * gv.w + bv.w; y3 = clampf(y3, ht_min, ht_max);

            o4[i4] = make_float4(y0, y1, y2, y3);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            int i = lane + (k << 5);
            int c = c0 + i;
            float v = x[n * C + c];
            float y = (v - mean) * invstd;
            y = y * ldg_f32(gamma + c) + ldg_f32(beta + c);
            y = clampf(y, ht_min, ht_max);
            out[n * C + c] = y;
        }
    }
}

struct PairSum { float sum; float sumsq; };
struct PairReduce {
    __device__ __forceinline__ PairSum operator()(const PairSum& a, const PairSum& b) const {
        return PairSum{a.sum + b.sum, a.sumsq + b.sumsq};
    }
};

template<int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void gn_hardtanh_kernel_generic_cub(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, int G,
    float eps, float ht_min, float ht_max
) {
    int ng = (int)blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;

    int group_size = C / G;
    int c0 = g * group_size;
    int base = n * C + c0;

    int tid = (int)threadIdx.x;

    using BlockReduce = cub::BlockReduce<PairSum, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float sh_mean;
    __shared__ float sh_invstd;

    PairSum local{0.0f, 0.0f};
    for (int i = tid; i < group_size; i += BLOCK_THREADS) {
        float v = x[base + i];
        local.sum += v;
        local.sumsq += v * v;
    }

    PairSum total = BlockReduce(temp_storage).Reduce(local, PairReduce{});
    if (tid == 0) {
        float invN = 1.0f / (float)group_size;
        float mean = total.sum * invN;
        float var = total.sumsq * invN - mean * mean;
        var = var < 0.0f ? 0.0f : var;
        sh_mean = mean;
        sh_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = sh_mean;
    float invstd = sh_invstd;

    uintptr_t x_addr = (uintptr_t)(x + base);
    uintptr_t g_addr = (uintptr_t)(gamma + c0);
    uintptr_t b_addr = (uintptr_t)(beta + c0);
    uintptr_t o_addr = (uintptr_t)(out + base);
    bool vec_ok = ((group_size & 3) == 0) &&
                  (((x_addr | g_addr | b_addr | o_addr) & 15) == 0);

    if (vec_ok) {
        const float4* __restrict__ x4 = (const float4*)(x + base);
        const float4* __restrict__ g4 = (const float4*)(gamma + c0);
        const float4* __restrict__ b4 = (const float4*)(beta + c0);
        float4* __restrict__ o4 = (float4*)(out + base);
        int n4 = group_size >> 2;

        for (int i4 = tid; i4 < n4; i4 += BLOCK_THREADS) {
            float4 xv = x4[i4];
            float4 gv = g4[i4];
            float4 bv = b4[i4];

            float y0 = (xv.x - mean) * invstd; y0 = y0 * gv.x + bv.x; y0 = clampf(y0, ht_min, ht_max);
            float y1 = (xv.y - mean) * invstd; y1 = y1 * gv.y + bv.y; y1 = clampf(y1, ht_min, ht_max);
            float y2 = (xv.z - mean) * invstd; y2 = y2 * gv.z + bv.z; y2 = clampf(y2, ht_min, ht_max);
            float y3 = (xv.w - mean) * invstd; y3 = y3 * gv.w + bv.w; y3 = clampf(y3, ht_min, ht_max);

            o4[i4] = make_float4(y0, y1, y2, y3);
        }
    } else {
        for (int i = tid; i < group_size; i += BLOCK_THREADS) {
            int c = c0 + i;
            float v = x[n * C + c];
            float y = (v - mean) * invstd;
            y = y * ldg_f32(gamma + c) + ldg_f32(beta + c);
            y = clampf(y, ht_min, ht_max);
            out[n * C + c] = y;
        }
    }
}

torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor x,          // [N, C]
    torch::Tensor gamma,      // [C]
    torch::Tensor beta,       // [C]
    int64_t num_groups,
    double eps,
    double ht_min,
    double ht_max
) {
    TORCH_CHECK(x.is_cuda(), "group_norm_hardtanh_cuda: x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "group_norm_hardtanh_cuda: gamma/beta must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "group_norm_hardtanh_cuda: x must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
                "group_norm_hardtanh_cuda: gamma/beta must be float32");
    TORCH_CHECK(x.is_contiguous(), "group_norm_hardtanh_cuda: x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "group_norm_hardtanh_cuda: gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() == 2, "group_norm_hardtanh_cuda: x must be 2D [N, C]");

    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    int64_t G64 = num_groups;

    TORCH_CHECK(G64 > 0, "group_norm_hardtanh_cuda: num_groups must be > 0");
    TORCH_CHECK(C64 % G64 == 0, "group_norm_hardtanh_cuda: C must be divisible by num_groups");
    TORCH_CHECK(gamma.numel() == C64 && beta.numel() == C64, "group_norm_hardtanh_cuda: gamma/beta must be [C]");

    auto out = torch::empty_like(x);

    int N = (int)N64;
    int C = (int)C64;
    int G = (int)G64;

    int group_size = C / G;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (group_size == 512) {
        // Warp-per-group mapping
        int NG = N * G;
        const int threads = 256; // 8 warps
        const int warps_per_block = threads / 32;
        int blocks = (NG + warps_per_block - 1) / warps_per_block;

        gn_hardtanh_g512_warp_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            N, C, G,
            (float)eps, (float)ht_min, (float)ht_max
        );
    } else {
        // Generic fallback
        int blocks = N * G;
        constexpr int THREADS = 256;
        gn_hardtanh_kernel_generic_cub<THREADS><<<blocks, THREADS, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            N, C, G,
            (float)eps, (float)ht_min, (float)ht_max
        );
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    double ht_min,
    double ht_max
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gnht_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["group_norm_hardtanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


class ModelNew(nn.Module):
    """
    GEMM (nn.Linear) + fused GroupNorm+HardTanh via optimized custom CUDA op.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(int(num_groups), out_features)
        self.num_groups = int(num_groups)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)

        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        gamma = self.group_norm.weight
        beta = self.group_norm.bias

        if not gamma.is_cuda:
            gamma = gamma.cuda()
        if not beta.is_cuda:
            beta = beta.cuda()
        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()

        eps = float(self.group_norm.eps)

        return self.custom_ops_lib.group_norm_hardtanh_cuda(
            x, gamma, beta, self.num_groups, eps, self.hardtanh_min, self.hardtanh_max
        )