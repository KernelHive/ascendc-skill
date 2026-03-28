import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# CUDA extension: ufo_fused_qkv_cuda(q, kv, gamma, eps) -> out
# q:     [B,H,NQ,DK] float32 contiguous
# kv:    [B,H,DK,DV] float32 contiguous
# gamma: [1,H,1,1] or any contiguous tensor with >=H elements when flattened
# out:   [B,H,NQ,DV] float32 contiguous
#
# Fast path DK=64, DV=64:
# - CTA owns one (b,h) and a tile of queries: WARPS_PER_BLOCK queries per block
# - KV is staged into shared once per block (vectorized float4 loads)
# - per-k row invnorm computed once per block and stored in shared
# - each warp computes one query row using shared KV+invnorm; predication avoids OOB
#
# General path:
# - baseline-like warp-per-(b,h,nq) kernel (no __syncthreads()).
# ------------------------------------------------------------

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static __device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ float load_gamma_head(const float* gamma, int h) {
#if __CUDA_ARCH__ >= 350
    return __ldg(gamma + h);
#else
    return gamma[h];
#endif
}

template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4) void ufo_cta_kvshared_d64_kernel(
    const float* __restrict__ q,       // [BH,NQ,64]
    const float* __restrict__ kv,      // [BH,64,64]
    const float* __restrict__ gamma_h, // [H]
    float* __restrict__ out,           // [BH,NQ,64]
    int H, int NQ,
    float eps)
{
    constexpr int DK = 64;
    constexpr int DV = 64;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..WARPS_PER_BLOCK-1

    int bh = (int)blockIdx.x;   // [0, B*H)
    int tile = (int)blockIdx.y; // tile over queries
    int nq = tile * WARPS_PER_BLOCK + warp;

    // predication (no early return before barriers)
    bool valid_q = (nq < NQ);

    // gamma per head
    int h = bh % H;
    float g = load_gamma_head(gamma_h, h);

    // Safe q pointer: redirect invalid warps to a valid row (0) to avoid OOB loads
    int safe_nq = valid_q ? nq : 0;

    const float* q_row = q + (((int64_t)bh * (int64_t)NQ + (int64_t)safe_nq) * DK);
    const float* kv_bh = kv + ((int64_t)bh * DK * DV);
    float* out_row = out + (((int64_t)bh * (int64_t)NQ + (int64_t)nq) * DV);

    extern __shared__ float smem[];
    float* sm_kv  = smem;                 // 4096 floats
    float* sm_inv = sm_kv + DK * DV;      // 64 floats

    // Stage KV into shared memory.
    // Use float4 loads/stores when aligned to reduce instruction count.
    int nthreads = WARPS_PER_BLOCK * 32;
    uintptr_t gptr = (uintptr_t)kv_bh;
    uintptr_t sptr = (uintptr_t)sm_kv;
    bool vec_ok = ((gptr & 0xF) == 0) && ((sptr & 0xF) == 0);

    if (vec_ok) {
        int n4 = (DK * DV) / 4; // 4096/4 = 1024
        const float4* __restrict__ kv4 = (const float4*)kv_bh;
        float4* __restrict__ sm4 = (float4*)sm_kv;
        for (int idx4 = tid; idx4 < n4; idx4 += nthreads) {
#if __CUDA_ARCH__ >= 350
            float4 v = __ldg(kv4 + idx4);
#else
            float4 v = kv4[idx4];
#endif
            sm4[idx4] = v;
        }
    } else {
        for (int idx = tid; idx < DK * DV; idx += nthreads) {
#if __CUDA_ARCH__ >= 350
            sm_kv[idx] = __ldg(kv_bh + idx);
#else
            sm_kv[idx] = kv_bh[idx];
#endif
        }
    }
    __syncthreads();

    // Compute invnorm per k once per CTA.
    // Use first 2 warps when available, else just warp 0.
    int compute_warps = (WARPS_PER_BLOCK >= 2) ? 2 : 1;
    if (warp < compute_warps) {
        for (int k = warp; k < DK; k += compute_warps) {
            const float* row = sm_kv + k * DV;
            float a = row[lane];
            float b = row[lane + 32];
            float ss = a*a + b*b;
            ss = warp_sum(ss);
            ss = __shfl_sync(0xffffffffu, ss, 0);
            if (lane == 0) sm_inv[k] = g * rsqrtf(ss + eps);
        }
    }
    __syncthreads();

    // Load q (2 elems per lane) and compute q_scale
#if __CUDA_ARCH__ >= 350
    float q0 = __ldg(q_row + lane);
    float q1 = __ldg(q_row + lane + 32);
#else
    float q0 = q_row[lane];
    float q1 = q_row[lane + 32];
#endif

    float q_ss = q0*q0 + q1*q1;
    q_ss = warp_sum(q_ss);
    q_ss = __shfl_sync(0xffffffffu, q_ss, 0);
    float q_scale = g * rsqrtf(q_ss + eps);

    float acc0 = 0.f, acc1 = 0.f;

#pragma unroll
    for (int k = 0; k < DK; ++k) {
        const float* row = sm_kv + k * DV;
        float invk = sm_inv[k];

        float vA = row[lane];
        float vB = row[lane + 32];

        float qk = (k < 32) ? __shfl_sync(0xffffffffu, q0, k)
                            : __shfl_sync(0xffffffffu, q1, k - 32);

        float s = qk * (q_scale * invk);
        acc0 = fmaf(s, vA, acc0);
        acc1 = fmaf(s, vB, acc1);
    }

    if (valid_q) {
        out_row[lane]      = acc0;
        out_row[lane + 32] = acc1;
    }
}

__global__ __launch_bounds__(32, 8) void ufo_warp_general_kernel(
    const float* __restrict__ q,      // [B,H,NQ,DK]
    const float* __restrict__ kv,     // [B,H,DK,DV]
    const float* __restrict__ gamma_h,// [H]
    float* __restrict__ out,          // [B,H,NQ,DV]
    int B, int H, int NQ, int DK, int DV,
    float eps)
{
    int lane = threadIdx.x & 31;

    int bh = (int)blockIdx.x;
    int nq = (int)blockIdx.y;
    int b = bh / H;
    int h = bh - b * H;
    (void)b;

    float g = load_gamma_head(gamma_h, h);

    const float* q_row = q + (((int64_t)bh * (int64_t)NQ + (int64_t)nq) * (int64_t)DK);
    const float* kv_bh = kv + ((int64_t)bh * (int64_t)DK * (int64_t)DV);
    float* out_row = out + (((int64_t)bh * (int64_t)NQ + (int64_t)nq) * (int64_t)DV);

    float q_ss = 0.0f;
    for (int k = lane; k < DK; k += 32) {
#if __CUDA_ARCH__ >= 350
        float v = __ldg(q_row + k);
#else
        float v = q_row[k];
#endif
        q_ss = fmaf(v, v, q_ss);
    }
    q_ss = warp_sum(q_ss);
    q_ss = __shfl_sync(0xffffffffu, q_ss, 0);
    float q_scale = g * rsqrtf(q_ss + eps);

    for (int dv = lane; dv < DV; dv += 32) {
        float acc = 0.0f;
        for (int k = 0; k < DK; ++k) {
            const float* kv_row = kv_bh + (int64_t)k * (int64_t)DV;

            float kv_ss = 0.0f;
            for (int j = lane; j < DV; j += 32) {
#if __CUDA_ARCH__ >= 350
                float vv = __ldg(kv_row + j);
#else
                float vv = kv_row[j];
#endif
                kv_ss = fmaf(vv, vv, kv_ss);
            }
            kv_ss = warp_sum(kv_ss);
            kv_ss = __shfl_sync(0xffffffffu, kv_ss, 0);
            float kv_scale = g * rsqrtf(kv_ss + eps);

#if __CUDA_ARCH__ >= 350
            float qk = __ldg(q_row + k);
            float vdv = __ldg(kv_row + dv);
#else
            float qk = q_row[k];
            float vdv = kv_row[dv];
#endif
            acc = fmaf(qk * (q_scale * kv_scale), vdv, acc);
        }
        out_row[dv] = acc;
    }
}

torch::Tensor ufo_fused_qkv_cuda(torch::Tensor q, torch::Tensor kv, torch::Tensor gamma, double eps_d) {
    CHECK_CUDA(q);
    CHECK_CUDA(kv);
    CHECK_CUDA(gamma);
    CHECK_FLOAT(q);
    CHECK_FLOAT(kv);
    CHECK_FLOAT(gamma);
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(kv);
    CHECK_CONTIGUOUS(gamma);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B,H,NQ,DK]");
    TORCH_CHECK(kv.dim() == 4, "kv must be 4D [B,H,DK,DV]");

    int64_t B  = q.size(0);
    int64_t H  = q.size(1);
    int64_t NQ = q.size(2);
    int64_t DK = q.size(3);
    TORCH_CHECK(kv.size(0) == B && kv.size(1) == H && kv.size(2) == DK, "kv shape mismatch");
    int64_t DV = kv.size(3);

    TORCH_CHECK(DK > 0 && DV > 0 && NQ > 0, "invalid sizes");
    TORCH_CHECK(gamma.numel() >= H, "gamma must have at least H elements");
    auto gamma_flat = gamma.view({-1});

    auto out = torch::empty({B, H, NQ, DV}, torch::TensorOptions().dtype(q.dtype()).device(q.device()));

    float eps = (float)eps_d;
    auto stream = at::cuda::getDefaultCUDAStream();
    int BH = (int)(B * H);

    if (DK == 64 && DV == 64) {
        // Use 8 warps to improve latency hiding while keeping shared small (16KB + 256B).
        constexpr int WARPS = 8;
        dim3 block(WARPS * 32, 1, 1);
        dim3 grid((unsigned int)BH, (unsigned int)((NQ + WARPS - 1) / WARPS), 1);
        size_t shmem = (64 * 64 + 64) * sizeof(float);
        ufo_cta_kvshared_d64_kernel<WARPS><<<grid, block, shmem, stream>>>(
            (const float*)q.data_ptr<float>(),
            (const float*)kv.data_ptr<float>(),
            (const float*)gamma_flat.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            (int)H, (int)NQ,
            eps
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    dim3 block(32, 1, 1);
    dim3 grid((unsigned int)BH, (unsigned int)NQ, 1);
    ufo_warp_general_kernel<<<grid, block, 0, stream>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)kv.data_ptr<float>(),
        (const float*)gamma_flat.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)B, (int)H, (int)NQ, (int)DK, (int)DV,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor ufo_fused_qkv_cuda(torch::Tensor q, torch::Tensor kv, torch::Tensor gamma, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ufo_attention_ops_cta_kvshared_v2",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["ufo_fused_qkv_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    UFO Attention with optimized CUDA kernel for:
      out = XNorm(q,gamma) @ XNorm(k@v,gamma)
    Linear layers and k@v matmul remain in PyTorch (cuBLAS).
    """

    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1), dtype=torch.float32))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.custom_ops_lib = custom_ops_lib
        self._eps = 1e-12

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()

        kv = torch.matmul(k, v).contiguous()  # [B,H,DK,DV]

        gamma = self.gamma.contiguous()
        out = self.custom_ops_lib.ufo_fused_qkv_cuda(q, kv, gamma, float(self._eps))  # [B,H,NQ,DV]

        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out