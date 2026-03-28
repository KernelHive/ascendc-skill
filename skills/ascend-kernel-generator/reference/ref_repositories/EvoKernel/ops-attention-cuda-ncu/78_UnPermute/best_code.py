import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Further-optimized fused CUDA op: MoE un_permute + weight + reduce
# Adds:
#  - topk==8 bf16x4 (16B) vector path when aligned and K%4==0
#  - optional inv_perm int32 fast path (selected by dtype)
# Keeps:
#  - bf16x2 pipelined path (K even, 4B aligned)
#  - generic scalar fallback
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_I64(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be int64")
#define CHECK_I32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)
#define CHECK_INPUT_I64(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_I64(x)
#define CHECK_INPUT_I32(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_I32(x)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ int64_t ldg_i64(const int64_t* p) { return __ldg(p); }
__device__ __forceinline__ int ldg_i32(const int* p) { return __ldg(p); }
__device__ __forceinline__ __nv_bfloat16 ldg_bf16(const __nv_bfloat16* p) { return __ldg(p); }
#else
__device__ __forceinline__ int64_t ldg_i64(const int64_t* p) { return *p; }
__device__ __forceinline__ int ldg_i32(const int* p) { return *p; }
__device__ __forceinline__ __nv_bfloat16 ldg_bf16(const __nv_bfloat16* p) { return *p; }
#endif

// --------------------------- Generic scalar kernel ---------------------------
__global__ void unpermute_weight_sum_bf16_scalar_kernel(
    const __nv_bfloat16* __restrict__ expert,     // [E, K]
    const __nv_bfloat16* __restrict__ topk_vals,  // [M, T]
    const int64_t* __restrict__ inv_perm,         // [M*T]
    __nv_bfloat16* __restrict__ out,              // [M, K]
    int M, int T, int K
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || k >= K) return;

    float acc = 0.0f;
    int base = m * T;
    #pragma unroll 4
    for (int t = 0; t < T; ++t) {
        int64_t idx = ldg_i64(inv_perm + base + t);
        __nv_bfloat16 v_bf = expert[(int64_t)idx * (int64_t)K + (int64_t)k];
        __nv_bfloat16 w_bf = ldg_bf16(topk_vals + (int64_t)m * (int64_t)T + (int64_t)t);
        acc = fmaf(__bfloat162float(v_bf), __bfloat162float(w_bf), acc);
    }
    out[(int64_t)m * (int64_t)K + (int64_t)k] = __float2bfloat16(acc);
}

// ---------------------- Fast path: topk=8, bf16x2, pipelined -----------------
template <typename IndexT>
__global__ __launch_bounds__(256, 2)
void unpermute_weight_sum_bf16x2_topk8_pipelined_kernel(
    const __nv_bfloat16* __restrict__ expert,     // [E, K]
    const __nv_bfloat16* __restrict__ topk_vals,  // [M, 8]
    const IndexT* __restrict__ inv_perm,          // [M*8]
    __nv_bfloat16* __restrict__ out,              // [M, K]
    int M, int K
) {
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;
    int warps_per_block = (int)blockDim.x >> 5;

    int m = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (m >= M) return;

    int idx_i32[8];
    float w_f[8];
    if (lane == 0) {
        int base = m * 8;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if constexpr (std::is_same<IndexT, int>::value) idx_i32[t] = ldg_i32(inv_perm + base + t);
            else idx_i32[t] = (int)ldg_i64((const int64_t*)(inv_perm + base + t));
            w_f[t] = __bfloat162float(ldg_bf16(topk_vals + (int64_t)m * 8 + t));
        }
    }
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        idx_i32[t] = __shfl_sync(mask, idx_i32[t], 0);
        w_f[t] = __shfl_sync(mask, w_f[t], 0);
    }

    const __nv_bfloat162* __restrict__ expert2 = (const __nv_bfloat162*)expert;
    __nv_bfloat162* __restrict__ out2 = (__nv_bfloat162*)out;

    int K2 = K >> 1;
    int out_row2 = m * K2;

    for (int k2 = lane; k2 < K2; k2 += 32) {
        float acc0 = 0.f, acc1 = 0.f;

        int idx0 = idx_i32[0];
        __nv_bfloat162 v_cur = expert2[(int64_t)idx0 * (int64_t)K2 + (int64_t)k2];

        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            int idx_next = idx_i32[t + 1];
            __nv_bfloat162 v_next = expert2[(int64_t)idx_next * (int64_t)K2 + (int64_t)k2];

            float2 vf = __bfloat1622float2(v_cur);
            float wt = w_f[t];
            acc0 = fmaf(vf.x, wt, acc0);
            acc1 = fmaf(vf.y, wt, acc1);

            v_cur = v_next;
        }
        {
            float2 vf = __bfloat1622float2(v_cur);
            float wt = w_f[7];
            acc0 = fmaf(vf.x, wt, acc0);
            acc1 = fmaf(vf.y, wt, acc1);
        }

        out2[(int64_t)out_row2 + (int64_t)k2] =
            __float22bfloat162_rn(make_float2(acc0, acc1));
    }
}

// ---------------------- Faster path: topk=8, bf16x4 (16B) --------------------
// Requirements: K % 4 == 0, expert/out pointers 16B aligned.
template <typename IndexT>
__global__ __launch_bounds__(256, 2)
void unpermute_weight_sum_bf16x4_topk8_kernel(
    const __nv_bfloat16* __restrict__ expert,     // [E, K]
    const __nv_bfloat16* __restrict__ topk_vals,  // [M, 8]
    const IndexT* __restrict__ inv_perm,          // [M*8]
    __nv_bfloat16* __restrict__ out,              // [M, K]
    int M, int K
) {
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;
    int warps_per_block = (int)blockDim.x >> 5;

    int m = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (m >= M) return;

    int idx_i32[8];
    float w_f[8];
    if (lane == 0) {
        int base = m * 8;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if constexpr (std::is_same<IndexT, int>::value) idx_i32[t] = ldg_i32(inv_perm + base + t);
            else idx_i32[t] = (int)ldg_i64((const int64_t*)(inv_perm + base + t));
            w_f[t] = __bfloat162float(ldg_bf16(topk_vals + (int64_t)m * 8 + t));
        }
    }
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        idx_i32[t] = __shfl_sync(mask, idx_i32[t], 0);
        w_f[t] = __shfl_sync(mask, w_f[t], 0);
    }

    // Treat bf16 as 16-bit lanes; we load 16 bytes = 8 bf16 = 4 bf16x2 = 4 columns of bf16x2
    // But we want 4 scalar bf16 outputs. We'll load/store as uint4 and unpack.
    const uint4* __restrict__ expert4 = reinterpret_cast<const uint4*>(expert);
    uint4* __restrict__ out4 = reinterpret_cast<uint4*>(out);

    int K4 = K >> 3; // each uint4 covers 8 bf16 elements (16 bytes). K must be multiple of 8 for this mapping.
    // However we only require K%4==0; to keep correctness, we additionally guard K%8==0 for this uint4 path.
    // Host dispatch enforces K%8==0 for x4 kernel; otherwise fall back to x2.
    int out_row4 = m * K4;

    for (int k4 = lane; k4 < K4; k4 += 32) {
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;

        // Pipeline first load
        int idx0 = idx_i32[0];
        uint4 v_cur = expert4[(int64_t)idx0 * (int64_t)K4 + (int64_t)k4];

        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            int idx_next = idx_i32[t + 1];
            uint4 v_next = expert4[(int64_t)idx_next * (int64_t)K4 + (int64_t)k4];

            // unpack 8 bf16 from 16B
            uint32_t p0 = v_cur.x, p1 = v_cur.y, p2 = v_cur.z, p3 = v_cur.w;
            __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&p0);
            __nv_bfloat16 b1 = *(reinterpret_cast<__nv_bfloat16*>(&p0) + 1);
            __nv_bfloat16 b2 = *reinterpret_cast<__nv_bfloat16*>(&p1);
            __nv_bfloat16 b3 = *(reinterpret_cast<__nv_bfloat16*>(&p1) + 1);
            __nv_bfloat16 b4 = *reinterpret_cast<__nv_bfloat16*>(&p2);
            __nv_bfloat16 b5 = *(reinterpret_cast<__nv_bfloat16*>(&p2) + 1);
            __nv_bfloat16 b6 = *reinterpret_cast<__nv_bfloat16*>(&p3);
            __nv_bfloat16 b7 = *(reinterpret_cast<__nv_bfloat16*>(&p3) + 1);

            float wt = w_f[t];
            acc[0] = fmaf(__bfloat162float(b0), wt, acc[0]);
            acc[1] = fmaf(__bfloat162float(b1), wt, acc[1]);
            acc[2] = fmaf(__bfloat162float(b2), wt, acc[2]);
            acc[3] = fmaf(__bfloat162float(b3), wt, acc[3]);
            acc[4] = fmaf(__bfloat162float(b4), wt, acc[4]);
            acc[5] = fmaf(__bfloat162float(b5), wt, acc[5]);
            acc[6] = fmaf(__bfloat162float(b6), wt, acc[6]);
            acc[7] = fmaf(__bfloat162float(b7), wt, acc[7]);

            v_cur = v_next;
        }
        // t=7
        {
            uint32_t p0 = v_cur.x, p1 = v_cur.y, p2 = v_cur.z, p3 = v_cur.w;
            __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&p0);
            __nv_bfloat16 b1 = *(reinterpret_cast<__nv_bfloat16*>(&p0) + 1);
            __nv_bfloat16 b2 = *reinterpret_cast<__nv_bfloat16*>(&p1);
            __nv_bfloat16 b3 = *(reinterpret_cast<__nv_bfloat16*>(&p1) + 1);
            __nv_bfloat16 b4 = *reinterpret_cast<__nv_bfloat16*>(&p2);
            __nv_bfloat16 b5 = *(reinterpret_cast<__nv_bfloat16*>(&p2) + 1);
            __nv_bfloat16 b6 = *reinterpret_cast<__nv_bfloat16*>(&p3);
            __nv_bfloat16 b7 = *(reinterpret_cast<__nv_bfloat16*>(&p3) + 1);

            float wt = w_f[7];
            acc[0] = fmaf(__bfloat162float(b0), wt, acc[0]);
            acc[1] = fmaf(__bfloat162float(b1), wt, acc[1]);
            acc[2] = fmaf(__bfloat162float(b2), wt, acc[2]);
            acc[3] = fmaf(__bfloat162float(b3), wt, acc[3]);
            acc[4] = fmaf(__bfloat162float(b4), wt, acc[4]);
            acc[5] = fmaf(__bfloat162float(b5), wt, acc[5]);
            acc[6] = fmaf(__bfloat162float(b6), wt, acc[6]);
            acc[7] = fmaf(__bfloat162float(b7), wt, acc[7]);
        }

        // pack 8 bf16 back into uint4
        __nv_bfloat16 o0 = __float2bfloat16(acc[0]);
        __nv_bfloat16 o1 = __float2bfloat16(acc[1]);
        __nv_bfloat16 o2 = __float2bfloat16(acc[2]);
        __nv_bfloat16 o3 = __float2bfloat16(acc[3]);
        __nv_bfloat16 o4 = __float2bfloat16(acc[4]);
        __nv_bfloat16 o5 = __float2bfloat16(acc[5]);
        __nv_bfloat16 o6 = __float2bfloat16(acc[6]);
        __nv_bfloat16 o7 = __float2bfloat16(acc[7]);

        uint32_t q0 = (uint32_t)(reinterpret_cast<uint16_t&>(o0)) | ((uint32_t)(reinterpret_cast<uint16_t&>(o1)) << 16);
        uint32_t q1 = (uint32_t)(reinterpret_cast<uint16_t&>(o2)) | ((uint32_t)(reinterpret_cast<uint16_t&>(o3)) << 16);
        uint32_t q2 = (uint32_t)(reinterpret_cast<uint16_t&>(o4)) | ((uint32_t)(reinterpret_cast<uint16_t&>(o5)) << 16);
        uint32_t q3 = (uint32_t)(reinterpret_cast<uint16_t&>(o6)) | ((uint32_t)(reinterpret_cast<uint16_t&>(o7)) << 16);

        uint4 outv;
        outv.x = q0; outv.y = q1; outv.z = q2; outv.w = q3;
        out4[(int64_t)out_row4 + (int64_t)k4] = outv;
    }
}

// -------------------------- Host dispatchers --------------------------------
torch::Tensor unpermute_weight_sum_bf16_cuda_i64(
    torch::Tensor expert_output,
    torch::Tensor topk_vals,
    torch::Tensor inv_perm
) {
    CHECK_INPUT_BF16(expert_output);
    CHECK_INPUT_BF16(topk_vals);
    CHECK_INPUT_I64(inv_perm);

    TORCH_CHECK(expert_output.dim() == 2, "expert_output must be [E, K]");
    TORCH_CHECK(topk_vals.dim() == 2, "topk_vals must be [M, T]");
    TORCH_CHECK(inv_perm.dim() == 1, "inv_perm must be [M*T]");

    int64_t K64 = expert_output.size(1);
    int64_t M64 = topk_vals.size(0);
    int64_t T64 = topk_vals.size(1);
    TORCH_CHECK(inv_perm.size(0) == M64 * T64, "inv_perm must have length M*T");
    TORCH_CHECK(K64 <= INT_MAX && M64 <= INT_MAX && T64 <= INT_MAX, "sizes too large");

    int K = (int)K64;
    int M = (int)M64;
    int T = (int)T64;

    auto out = torch::empty({M64, K64}, torch::TensorOptions().dtype(torch::kBFloat16).device(expert_output.device()));

    // Alignment checks
    uintptr_t expert_ptr = (uintptr_t)expert_output.data_ptr<at::BFloat16>();
    uintptr_t out_ptr = (uintptr_t)out.data_ptr<at::BFloat16>();
    bool aligned16 = ((expert_ptr | out_ptr) & 0xF) == 0;
    bool aligned4 = ((expert_ptr | out_ptr) & 0x3) == 0;

    if (T == 8) {
        int warps_per_block = 8;
        dim3 block(warps_per_block * 32, 1, 1);
        dim3 grid((M + warps_per_block - 1) / warps_per_block, 1, 1);

        // Prefer bf16x4 only when safe: K%8==0 for uint4 mapping + 16B alignment
        if ((K % 8) == 0 && aligned16) {
            unpermute_weight_sum_bf16x4_topk8_kernel<int64_t><<<grid, block>>>(
                (const __nv_bfloat16*)expert_output.data_ptr<at::BFloat16>(),
                (const __nv_bfloat16*)topk_vals.data_ptr<at::BFloat16>(),
                (const int64_t*)inv_perm.data_ptr<int64_t>(),
                (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
                M, K
            );
            return out;
        }

        // Else bf16x2 pipelined when K even and 4B aligned
        if (((K & 1) == 0) && aligned4) {
            unpermute_weight_sum_bf16x2_topk8_pipelined_kernel<int64_t><<<grid, block>>>(
                (const __nv_bfloat16*)expert_output.data_ptr<at::BFloat16>(),
                (const __nv_bfloat16*)topk_vals.data_ptr<at::BFloat16>(),
                (const int64_t*)inv_perm.data_ptr<int64_t>(),
                (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
                M, K
            );
            return out;
        }
    }

    // Fallback: scalar kernel
    dim3 block(32, 8, 1);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y, 1);
    unpermute_weight_sum_bf16_scalar_kernel<<<grid, block>>>(
        (const __nv_bfloat16*)expert_output.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)topk_vals.data_ptr<at::BFloat16>(),
        (const int64_t*)inv_perm.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        M, T, K
    );
    return out;
}

torch::Tensor unpermute_weight_sum_bf16_cuda_i32(
    torch::Tensor expert_output,
    torch::Tensor topk_vals,
    torch::Tensor inv_perm
) {
    CHECK_INPUT_BF16(expert_output);
    CHECK_INPUT_BF16(topk_vals);
    CHECK_INPUT_I32(inv_perm);

    TORCH_CHECK(expert_output.dim() == 2, "expert_output must be [E, K]");
    TORCH_CHECK(topk_vals.dim() == 2, "topk_vals must be [M, T]");
    TORCH_CHECK(inv_perm.dim() == 1, "inv_perm must be [M*T]");

    int64_t K64 = expert_output.size(1);
    int64_t M64 = topk_vals.size(0);
    int64_t T64 = topk_vals.size(1);
    TORCH_CHECK(inv_perm.size(0) == M64 * T64, "inv_perm must have length M*T");
    TORCH_CHECK(K64 <= INT_MAX && M64 <= INT_MAX && T64 <= INT_MAX, "sizes too large");

    int K = (int)K64;
    int M = (int)M64;
    int T = (int)T64;

    auto out = torch::empty({M64, K64}, torch::TensorOptions().dtype(torch::kBFloat16).device(expert_output.device()));

    uintptr_t expert_ptr = (uintptr_t)expert_output.data_ptr<at::BFloat16>();
    uintptr_t out_ptr = (uintptr_t)out.data_ptr<at::BFloat16>();
    bool aligned16 = ((expert_ptr | out_ptr) & 0xF) == 0;
    bool aligned4 = ((expert_ptr | out_ptr) & 0x3) == 0;

    if (T == 8) {
        int warps_per_block = 8;
        dim3 block(warps_per_block * 32, 1, 1);
        dim3 grid((M + warps_per_block - 1) / warps_per_block, 1, 1);

        if ((K % 8) == 0 && aligned16) {
            unpermute_weight_sum_bf16x4_topk8_kernel<int><<<grid, block>>>(
                (const __nv_bfloat16*)expert_output.data_ptr<at::BFloat16>(),
                (const __nv_bfloat16*)topk_vals.data_ptr<at::BFloat16>(),
                (const int*)inv_perm.data_ptr<int>(),
                (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
                M, K
            );
            return out;
        }

        if (((K & 1) == 0) && aligned4) {
            unpermute_weight_sum_bf16x2_topk8_pipelined_kernel<int><<<grid, block>>>(
                (const __nv_bfloat16*)expert_output.data_ptr<at::BFloat16>(),
                (const __nv_bfloat16*)topk_vals.data_ptr<at::BFloat16>(),
                (const int*)inv_perm.data_ptr<int>(),
                (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
                M, K
            );
            return out;
        }
    }

    // Generic fallback requires int64 indices; keep this op specialized for topk==8 fast paths.
    // If invoked for other T, fall back to PyTorch path in Python.
    TORCH_CHECK(false, "unpermute_weight_sum_bf16_cuda_i32 only supports optimized topk==8 paths");
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor unpermute_weight_sum_bf16_cuda_i64(torch::Tensor expert_output, torch::Tensor topk_vals, torch::Tensor inv_perm);
torch::Tensor unpermute_weight_sum_bf16_cuda_i32(torch::Tensor expert_output, torch::Tensor topk_vals, torch::Tensor inv_perm);
"""

custom_ops_lib = load_inline(
    name="custom_unpermute_ops_opt_x4_i32i64_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "unpermute_weight_sum_bf16_cuda_i64",
        "unpermute_weight_sum_bf16_cuda_i32",
    ],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    MoE UnPermute with a fused custom CUDA kernel.

    Fast paths:
      - topk==8, bf16, contiguous
      - inv_perm int64: uses bf16x4 (K%8 and 16B aligned) else bf16x2 pipelined (K even, 4B aligned)
      - inv_perm int32: same fast paths, lower index bandwidth

    Fallback: reference PyTorch implementation.
    """

    def __init__(self, topk: int):
        super().__init__()
        self.topk = int(topk)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, expert_output: torch.Tensor, topk_vals: torch.Tensor, inv_perm: torch.Tensor) -> torch.Tensor:
        if (
            expert_output.is_cuda
            and topk_vals.is_cuda
            and inv_perm.is_cuda
            and expert_output.dtype == torch.bfloat16
            and topk_vals.dtype == torch.bfloat16
            and expert_output.is_contiguous()
            and topk_vals.is_contiguous()
            and inv_perm.is_contiguous()
            and expert_output.dim() == 2
            and topk_vals.dim() == 2
            and inv_perm.dim() == 1
            and topk_vals.size(1) == self.topk
            and self.topk == 8
        ):
            if inv_perm.dtype == torch.int32:
                # Only optimized topk==8 supported in i32 op; safe due to check above.
                return self.custom_ops_lib.unpermute_weight_sum_bf16_cuda_i32(expert_output, topk_vals, inv_perm)
            if inv_perm.dtype == torch.int64:
                return self.custom_ops_lib.unpermute_weight_sum_bf16_cuda_i64(expert_output, topk_vals, inv_perm)

        # Reference fallback (preserve semantics)
        M, topk = topk_vals.shape
        K = expert_output.shape[1]
        reordered = expert_output[inv_perm.to(dtype=torch.int64)]
        reordered = reordered.view(M, topk, K)
        weighted = reordered * topk_vals.unsqueeze(-1)
        out = weighted.sum(dim=1)
        return out