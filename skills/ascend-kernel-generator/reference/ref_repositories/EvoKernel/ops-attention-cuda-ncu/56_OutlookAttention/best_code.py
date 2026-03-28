import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Outlook Attention CUDA custom ops
# Baseline-safe:
#   - attn_scale_softmax_cuda: scale + softmax over last dim (KK)
#   - attn_matmul_v_cuda: (attn @ v) -> (B, C*KK, HW)
# Optimized eval (dropout inactive only):
#   - fused_softmax_matmul_warp_cuda: generic KK<=32 (one warp per row)
#   - fused_softmax_matmul_kk9_smem8w_cuda: KK=9 specialization:
#       8 warps/block, per-warp shared-memory probs[9], float4 V path, __ldg
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

// ----------------------- warp reductions -----------------------
static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
static __forceinline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
static __forceinline__ __device__ float warp_broadcast(float v, int src_lane) {
    return __shfl_sync(0xffffffff, v, src_lane);
}

// ----------------------- baseline kernels -----------------------
__global__ void attn_scale_softmax_kernel(
    const float* __restrict__ logits, // [B,heads,HW,KK,KK]
    float* __restrict__ probs,         // same
    int B, int heads, int HW, int KK,
    float scale
) {
    int idx = (int)blockIdx.x; // 0..B*heads*HW*KK-1
    int t = idx;
    int row_i = t % KK; t /= KK;
    int hw    = t % HW; t /= HW;
    int h     = t % heads; t /= heads;
    int b     = t;

    int base = (((b * heads + h) * HW + hw) * KK + row_i) * KK;

    float local_max = -1e20f;
    for (int j = threadIdx.x; j < KK; j += blockDim.x) {
        float x = logits[base + j] * scale;
        local_max = fmaxf(local_max, x);
    }

    __shared__ float smem_max[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float wmax = warp_reduce_max(local_max);
    if (lane == 0) smem_max[warp] = wmax;
    __syncthreads();
    float maxv = -1e20f;
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? smem_max[lane] : -1e20f;
        maxv = warp_reduce_max(v);
    }
    maxv = warp_broadcast(maxv, 0);
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < KK; j += blockDim.x) {
        float e = __expf(logits[base + j] * scale - maxv);
        probs[base + j] = e;
        local_sum += e;
    }

    __shared__ float smem_sum[8];
    float wsum = warp_reduce_sum(local_sum);
    if (lane == 0) smem_sum[warp] = wsum;
    __syncthreads();
    float sumv = 0.f;
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float v = (lane < nw) ? smem_sum[lane] : 0.f;
        sumv = warp_reduce_sum(v);
    }
    sumv = warp_broadcast(sumv, 0);
    __syncthreads();

    float inv = 1.f / (sumv + 1e-12f);
    for (int j = threadIdx.x; j < KK; j += blockDim.x) {
        probs[base + j] = probs[base + j] * inv;
    }
}

__global__ void attn_matmul_v_kernel(
    const float* __restrict__ attn,   // [B,heads,HW,KK,KK]
    const float* __restrict__ v,      // [B,heads,HW,KK,head_dim]
    float* __restrict__ out_flat,     // [B, heads*head_dim*KK, HW]
    int B, int heads, int HW, int KK, int head_dim
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = (int64_t)B * heads * HW * KK * head_dim;
    if (idx >= N) return;

    int d = (int)(idx % head_dim);
    int64_t t = idx / head_dim;
    int i = (int)(t % KK); t /= KK;
    int hw = (int)(t % HW); t /= HW;
    int h = (int)(t % heads); t /= heads;
    int b = (int)t;

    int attn_base = (((b * heads + h) * HW + hw) * KK + i) * KK;
    int v_base = (((b * heads + h) * HW + hw) * KK) * head_dim;

    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        if (j >= KK) break;
        float a = attn[attn_base + j];
        float vv = v[v_base + j * head_dim + d];
        acc += a * vv;
    }

    int c = (h * head_dim + d) * KK + i;
    int out_idx = (b * (heads * head_dim * KK) + c) * HW + hw;
    out_flat[out_idx] = acc;
}

// ----------------------- generic fused eval kernel (one warp per row) -----------------------
__global__ void fused_softmax_matmul_warp_kernel(
    const float* __restrict__ logits, // [B,heads,HW,KK,KK]
    const float* __restrict__ v,      // [B,heads,HW,KK,head_dim]
    float* __restrict__ out_flat,     // [B, heads*head_dim*KK, HW]
    int B, int heads, int HW, int KK, int head_dim,
    float scale
) {
    int row = (int)blockIdx.x;
    int lane = threadIdx.x & 31;

    int rows = B * heads * HW * KK;
    if (row >= rows) return;

    int t = row;
    int i  = t % KK; t /= KK;
    int hw = t % HW; t /= HW;
    int h  = t % heads; t /= heads;
    int b  = t;

    int d_tile0 = (int)blockIdx.y * 32;
    int d = d_tile0 + lane;
    bool valid_d = (d < head_dim);

    int logits_hw_base = ((b * heads + h) * HW + hw) * (KK * KK);
    int v_hw_base      = (((b * heads + h) * HW + hw) * KK) * head_dim;

    float x = -1e20f;
    if (lane < KK) x = logits[logits_hw_base + i * KK + lane] * scale;
    float maxv = warp_reduce_max(x);
    maxv = warp_broadcast(maxv, 0);

    float e = 0.f;
    if (lane < KK) e = __expf(x - maxv);
    float sumv = warp_reduce_sum(e);
    sumv = warp_broadcast(sumv, 0);
    float inv_sum = 1.f / (sumv + 1e-12f);

    float acc = 0.f;
    if (valid_d) {
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            if (j >= KK) break;
            float e_j = 0.f;
            if (lane == j) {
                float xj = logits[logits_hw_base + i * KK + j] * scale;
                e_j = __expf(xj - maxv);
            }
            e_j = warp_broadcast(e_j, j);
            float a = e_j * inv_sum;
            float vv = v[v_hw_base + j * head_dim + d];
            acc += a * vv;
        }

        int c = (h * head_dim + d) * KK + i;
        int out_idx = (b * (heads * head_dim * KK) + c) * HW + hw;
        out_flat[out_idx] = acc;
    }
}

__global__ void fused_softmax_matmul_warp_kernel_vec4(
    const float* __restrict__ logits, // [B,heads,HW,KK,KK]
    const float* __restrict__ v,      // [B,heads,HW,KK,head_dim]
    float* __restrict__ out_flat,     // [B, heads*head_dim*KK, HW]
    int B, int heads, int HW, int KK, int head_dim,
    float scale
) {
    int row = (int)blockIdx.x;
    int lane = threadIdx.x & 31;

    int rows = B * heads * HW * KK;
    if (row >= rows) return;

    int t = row;
    int i  = t % KK; t /= KK;
    int hw = t % HW; t /= HW;
    int h  = t % heads; t /= heads;
    int b  = t;

    int hd4 = head_dim >> 2;
    int d4_tile0 = (int)blockIdx.y * 32;
    int d4 = d4_tile0 + lane;
    bool valid_d4 = (d4 < hd4);

    int logits_hw_base = ((b * heads + h) * HW + hw) * (KK * KK);
    int v_hw_base      = (((b * heads + h) * HW + hw) * KK) * head_dim;

    float x = -1e20f;
    if (lane < KK) x = logits[logits_hw_base + i * KK + lane] * scale;
    float maxv = warp_reduce_max(x);
    maxv = warp_broadcast(maxv, 0);

    float e = 0.f;
    if (lane < KK) e = __expf(x - maxv);
    float sumv = warp_reduce_sum(e);
    sumv = warp_broadcast(sumv, 0);
    float inv_sum = 1.f / (sumv + 1e-12f);

    float4 acc; acc.x = 0.f; acc.y = 0.f; acc.z = 0.f; acc.w = 0.f;
    if (valid_d4) {
        const float4* __restrict__ vp4 = reinterpret_cast<const float4*>(v + v_hw_base);
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            if (j >= KK) break;
            float e_j = 0.f;
            if (lane == j) {
                float xj = logits[logits_hw_base + i * KK + j] * scale;
                e_j = __expf(xj - maxv);
            }
            e_j = warp_broadcast(e_j, j);
            float a = e_j * inv_sum;
            float4 vv = vp4[j * hd4 + d4];
            acc.x += a * vv.x;
            acc.y += a * vv.y;
            acc.z += a * vv.z;
            acc.w += a * vv.w;
        }

        int d = d4 << 2;
        int base_b = b * (heads * head_dim * KK) * HW;
        int c0 = (h * head_dim + (d + 0)) * KK + i;
        int c1 = (h * head_dim + (d + 1)) * KK + i;
        int c2 = (h * head_dim + (d + 2)) * KK + i;
        int c3 = (h * head_dim + (d + 3)) * KK + i;
        out_flat[base_b + c0 * HW + hw] = acc.x;
        out_flat[base_b + c1 * HW + hw] = acc.y;
        out_flat[base_b + c2 * HW + hw] = acc.z;
        out_flat[base_b + c3 * HW + hw] = acc.w;
    }
}

// ----------------------- KK=9 specialization: 8 warps/block + per-warp smem probs -----------------------
template<bool VEC4>
__global__ __launch_bounds__(256, 2) void fused_softmax_matmul_kk9_smem8w_kernel(
    const float* __restrict__ logits, // [B,heads,HW,9,9]
    const float* __restrict__ v,      // [B,heads,HW,9,head_dim]
    float* __restrict__ out_flat,     // [B, heads*head_dim*9, HW]
    int B, int heads, int HW, int head_dim,
    float scale
) {
    int tid = (int)threadIdx.x;
    int warp_in_block = tid >> 5;   // 0..7
    int lane = tid & 31;

    int row = (int)blockIdx.x * 8 + warp_in_block; // row over (B*heads*HW*9)
    int rows = B * heads * HW * 9;
    if (row >= rows) return;

    int t = row;
    int i  = t % 9; t /= 9;
    int hw = t % HW; t /= HW;
    int h  = t % heads; t /= heads;
    int b  = t;

    // shared probs per warp: 9 floats
    extern __shared__ float smem[];
    float* w = smem + warp_in_block * 9;

    int logits_hw_base = ((b * heads + h) * HW + hw) * 81;      // 9*9
    int v_hw_base      = (((b * heads + h) * HW + hw) * 9) * head_dim;

    // load logits row i and compute softmax (lanes 0..8)
    float x = -1e20f;
    if (lane < 9) x = __ldg(logits + logits_hw_base + i * 9 + lane) * scale;
    float m = warp_reduce_max(x);
    m = warp_broadcast(m, 0);

    float e = 0.f;
    if (lane < 9) e = __expf(x - m);
    float s = warp_reduce_sum(e);
    s = warp_broadcast(s, 0);
    float inv = 1.f / (s + 1e-12f);

    if (lane < 9) w[lane] = e * inv;
    __syncwarp(); // ensure w visible within warp

    // output dimension tiling across lanes
    if constexpr (VEC4) {
        int hd4 = head_dim >> 2;
        int d4 = (int)blockIdx.y * 32 + lane;
        if (d4 >= hd4) return;

        const float4* __restrict__ vp4 = reinterpret_cast<const float4*>(v + v_hw_base);
        float4 acc; acc.x = 0.f; acc.y = 0.f; acc.z = 0.f; acc.w = 0.f;

        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            float a = w[j];
            float4 vv = __ldg(vp4 + j * hd4 + d4);
            acc.x = fmaf(a, vv.x, acc.x);
            acc.y = fmaf(a, vv.y, acc.y);
            acc.z = fmaf(a, vv.z, acc.z);
            acc.w = fmaf(a, vv.w, acc.w);
        }

        int d = d4 << 2;
        int base_b = b * (heads * head_dim * 9) * HW;

        int c0 = (h * head_dim + (d + 0)) * 9 + i;
        int c1 = (h * head_dim + (d + 1)) * 9 + i;
        int c2 = (h * head_dim + (d + 2)) * 9 + i;
        int c3 = (h * head_dim + (d + 3)) * 9 + i;

        // stores are contiguous in HW dimension; each lane stores 4 scalars
        out_flat[base_b + c0 * HW + hw] = acc.x;
        out_flat[base_b + c1 * HW + hw] = acc.y;
        out_flat[base_b + c2 * HW + hw] = acc.z;
        out_flat[base_b + c3 * HW + hw] = acc.w;
    } else {
        int d = (int)blockIdx.y * 32 + lane;
        if (d >= head_dim) return;

        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            float a = w[j];
            float vv = __ldg(v + v_hw_base + j * head_dim + d);
            acc = fmaf(a, vv, acc);
        }

        int base_b = b * (heads * head_dim * 9) * HW;
        int c = (h * head_dim + d) * 9 + i;
        out_flat[base_b + c * HW + hw] = acc;
    }
}

// ----------------------- host wrappers -----------------------
torch::Tensor attn_scale_softmax_cuda(torch::Tensor logits, double scale) {
    CHECK_CUDA(logits);
    CHECK_CONTIGUOUS(logits);
    CHECK_FLOAT(logits);
    TORCH_CHECK(logits.dim() == 5, "logits must be (B,heads,HW,KK,KK)");
    TORCH_CHECK(logits.size(3) == logits.size(4), "last two dims must be KK,KK");

    int B = (int)logits.size(0);
    int heads = (int)logits.size(1);
    int HW = (int)logits.size(2);
    int KK = (int)logits.size(3);

    auto probs = torch::empty_like(logits);

    int blocks = B * heads * HW * KK;
    int threads = 128;
    if (KK <= 16) threads = 64;
    if (KK <= 8)  threads = 32;
    if (threads > 256) threads = 256;

    attn_scale_softmax_kernel<<<blocks, threads>>>(
        (const float*)logits.data_ptr<float>(),
        (float*)probs.data_ptr<float>(),
        B, heads, HW, KK, (float)scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return probs;
}

torch::Tensor attn_matmul_v_cuda(torch::Tensor attn, torch::Tensor v) {
    CHECK_CUDA(attn);
    CHECK_CUDA(v);
    CHECK_CONTIGUOUS(attn);
    CHECK_CONTIGUOUS(v);
    CHECK_FLOAT(attn);
    CHECK_FLOAT(v);
    TORCH_CHECK(attn.dim() == 5, "attn must be (B,heads,HW,KK,KK)");
    TORCH_CHECK(v.dim() == 5, "v must be (B,heads,HW,KK,head_dim)");
    TORCH_CHECK(attn.size(0) == v.size(0) && attn.size(1) == v.size(1) && attn.size(2) == v.size(2), "B/heads/HW must match");
    TORCH_CHECK(attn.size(3) == attn.size(4), "attn last dims must be KK,KK");
    TORCH_CHECK(attn.size(3) == v.size(3), "KK must match");
    int KK = (int)attn.size(3);
    TORCH_CHECK(KK <= 64, "KK too large for this kernel (max 64).");

    int B = (int)attn.size(0);
    int heads = (int)attn.size(1);
    int HW = (int)attn.size(2);
    int head_dim = (int)v.size(4);

    int C = heads * head_dim;
    auto out = torch::empty({B, C * KK, HW}, attn.options());

    int64_t N = (int64_t)B * heads * HW * KK * head_dim;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);

    attn_matmul_v_kernel<<<blocks, threads>>>(
        (const float*)attn.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, heads, HW, KK, head_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_softmax_matmul_warp_cuda(torch::Tensor logits, torch::Tensor v, double scale) {
    CHECK_CUDA(logits);
    CHECK_CUDA(v);
    CHECK_CONTIGUOUS(logits);
    CHECK_CONTIGUOUS(v);
    CHECK_FLOAT(logits);
    CHECK_FLOAT(v);
    TORCH_CHECK(logits.dim() == 5, "logits must be (B,heads,HW,KK,KK)");
    TORCH_CHECK(v.dim() == 5, "v must be (B,heads,HW,KK,head_dim)");
    TORCH_CHECK(logits.size(0) == v.size(0) && logits.size(1) == v.size(1) && logits.size(2) == v.size(2), "B/heads/HW must match");
    TORCH_CHECK(logits.size(3) == logits.size(4), "logits last dims must be KK,KK");
    TORCH_CHECK(logits.size(3) == v.size(3), "KK must match");
    int KK = (int)logits.size(3);
    TORCH_CHECK(KK <= 32, "fused warp kernel supports KK<=32 (one warp softmax).");
    int B = (int)logits.size(0);
    int heads = (int)logits.size(1);
    int HW = (int)logits.size(2);
    int head_dim = (int)v.size(4);

    int C = heads * head_dim;
    auto out = torch::empty({B, C * KK, HW}, logits.options());

    int rows = B * heads * HW * KK;
    dim3 block(32, 1, 1);

    int tiles_y = (head_dim + 31) / 32;

    bool vec4_ok = ((head_dim & 3) == 0) &&
                   (((uintptr_t)v.data_ptr<float>() & 0xF) == 0) &&
                   (((uintptr_t)out.data_ptr<float>() & 0xF) == 0);
    if (vec4_ok) {
        int tiles_y4 = ((head_dim >> 2) + 31) / 32;
        dim3 grid(rows, tiles_y4, 1);
        fused_softmax_matmul_warp_kernel_vec4<<<grid, block>>>(
            (const float*)logits.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, heads, HW, KK, head_dim, (float)scale
        );
    } else {
        dim3 grid(rows, tiles_y, 1);
        fused_softmax_matmul_warp_kernel<<<grid, block>>>(
            (const float*)logits.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, heads, HW, KK, head_dim, (float)scale
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_softmax_matmul_kk9_smem8w_cuda(torch::Tensor logits, torch::Tensor v, double scale) {
    CHECK_CUDA(logits);
    CHECK_CUDA(v);
    CHECK_CONTIGUOUS(logits);
    CHECK_CONTIGUOUS(v);
    CHECK_FLOAT(logits);
    CHECK_FLOAT(v);
    TORCH_CHECK(logits.dim() == 5, "logits must be (B,heads,HW,9,9)");
    TORCH_CHECK(v.dim() == 5, "v must be (B,heads,HW,9,head_dim)");
    TORCH_CHECK(logits.size(3) == 9 && logits.size(4) == 9, "KK must be 9");
    TORCH_CHECK(v.size(3) == 9, "KK must be 9");
    TORCH_CHECK(logits.size(0) == v.size(0) && logits.size(1) == v.size(1) && logits.size(2) == v.size(2), "B/heads/HW must match");

    int B = (int)logits.size(0);
    int heads = (int)logits.size(1);
    int HW = (int)logits.size(2);
    int head_dim = (int)v.size(4);

    int C = heads * head_dim;
    auto out = torch::empty({B, C * 9, HW}, logits.options());

    // row dimension: B*heads*HW*9 handled by 8 warps per block (256 threads)
    int rows = B * heads * HW * 9;
    int grid_x = (rows + 8 - 1) / 8;

    dim3 block(256, 1, 1);

    bool vec4_ok = ((head_dim & 3) == 0) &&
                   (((uintptr_t)v.data_ptr<float>() & 0xF) == 0) &&
                   (((uintptr_t)out.data_ptr<float>() & 0xF) == 0);

    // per-warp smem: 9 floats; 8 warps => 72 floats
    size_t smem = (size_t)8 * 9 * sizeof(float);

    if (vec4_ok) {
        int tiles_y4 = ((head_dim >> 2) + 31) / 32;
        dim3 grid(grid_x, tiles_y4, 1);
        fused_softmax_matmul_kk9_smem8w_kernel<true><<<grid, block, smem>>>(
            (const float*)logits.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, heads, HW, head_dim, (float)scale
        );
    } else {
        int tiles_y = (head_dim + 31) / 32;
        dim3 grid(grid_x, tiles_y, 1);
        fused_softmax_matmul_kk9_smem8w_kernel<false><<<grid, block, smem>>>(
            (const float*)logits.data_ptr<float>(),
            (const float*)v.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, heads, HW, head_dim, (float)scale
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor attn_scale_softmax_cuda(torch::Tensor logits, double scale);
torch::Tensor attn_matmul_v_cuda(torch::Tensor attn, torch::Tensor v);
torch::Tensor fused_softmax_matmul_warp_cuda(torch::Tensor logits, torch::Tensor v, double scale);
torch::Tensor fused_softmax_matmul_kk9_smem8w_cuda(torch::Tensor logits, torch::Tensor v, double scale);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_outlook_attention_opt_kk9_smem8w",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "attn_scale_softmax_cuda",
        "attn_matmul_v_cuda",
        "fused_softmax_matmul_warp_cuda",
        "fused_softmax_matmul_kk9_smem8w_cuda",
    ],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Outlook Attention with custom CUDA:
      - Training / dropout-active: scale+softmax + dropout + matmul (baseline-safe)
      - Eval / dropout-inactive:
          * KK=9: specialized 8-warps/block + per-warp smem probs + float4 V path
          * else (KK<=32): generic fused warp-softmax+matmul
    """
    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size, padding, stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        B, H, W, C = x.shape
        ks = self.kernel_size
        KK = ks * ks

        # v path
        v = self.v_pj(x).permute(0, 3, 1, 2)  # (B,C,H,W)
        h = math.ceil(H / self.stride)
        w = math.ceil(W / self.stride)
        HW = h * w

        v = self.unfold(v)  # (B, C*KK, HW)
        v = v.reshape(B, self.num_heads, self.head_dim, KK, HW).permute(0, 1, 4, 3, 2)  # (B,heads,HW,KK,head_dim)

        # logits path
        attn_in = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (B,h,w,C)
        logits = self.attn(attn_in).reshape(B, HW, self.num_heads, KK, KK).permute(0, 2, 1, 3, 4)  # (B,heads,HW,KK,KK)

        # custom kernels require float32 contiguous
        logits_c = logits.contiguous()
        v_c = v.contiguous()
        if logits_c.dtype != torch.float32:
            logits_c = logits_c.float()
        if v_c.dtype != torch.float32:
            v_c = v_c.float()

        dropout_inactive = (not self.training) or (self.attn_drop.p == 0.0)

        if dropout_inactive:
            if KK == 9:
                out_cols = self.custom_ops_lib.fused_softmax_matmul_kk9_smem8w_cuda(
                    logits_c, v_c, float(self.scale)
                )
            elif KK <= 32:
                out_cols = self.custom_ops_lib.fused_softmax_matmul_warp_cuda(
                    logits_c, v_c, float(self.scale)
                )
            else:
                attn = self.custom_ops_lib.attn_scale_softmax_cuda(logits_c, float(self.scale))
                out_cols = self.custom_ops_lib.attn_matmul_v_cuda(attn.contiguous(), v_c)
        else:
            attn = self.custom_ops_lib.attn_scale_softmax_cuda(logits_c, float(self.scale))
            attn = self.attn_drop(attn)
            out_cols = self.custom_ops_lib.attn_matmul_v_cuda(attn.contiguous(), v_c)

        out = F.fold(
            out_cols,
            output_size=(H, W),
            kernel_size=ks,
            padding=self.padding,
            stride=self.stride,
        )  # (B,C,H,W)

        out = self.proj(out.permute(0, 2, 3, 1))  # (B,H,W,C)
        out = self.proj_drop(out)
        return out