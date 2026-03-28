import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Fused CUDA epilogue for MUSE conv branch:
#   out_new(B,N,D) = out(B,N,D) + permute( softmax(dy)[0]*t0 + ... )(B,N,D)
# where t0/t1/t2 are (B,D,N) contiguous.
#
# Key improvements vs baseline:
# - fuse weighted-sum+permute+residual add into ONE kernel (no out2 tensor)
# - 2D block mapping with coalesced stores to (B,N,D)
# - vectorized float4 stores/loads when aligned (FP32 path)
# - optional FP16 half2 fast path (mixed accumulation)
# - read-only cache loads (__ldg) for streaming reads
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float expf_fast(float x) { return __expf(x); }

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ half ldg_f16(const half* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ void softmax3_to_shared(const float* __restrict__ dy, float &w0, float &w1, float &w2) {
    float a0 = dy[0], a1 = dy[1], a2 = dy[2];
    float m = a0;
    if (a1 > m) m = a1;
    if (a2 > m) m = a2;
    float e0 = expf_fast(a0 - m);
    float e1 = expf_fast(a1 - m);
    float e2 = expf_fast(a2 - m);
    float invs = 1.0f / (e0 + e1 + e2);
    w0 = e0 * invs;
    w1 = e1 * invs;
    w2 = e2 * invs;
}

// -------------------------------- FP32 fused kernel --------------------------------
// Mapping: each block covers one (b, tile_n). Threads cover d (vectorized by 4 optionally).
// out layout is (B,N,D) contiguous -> coalesced when x spans D.
// t layout is (B,D,N) contiguous in N -> strided when x spans D, but cached; N is small.
template <int VEC4>
__global__ void muse_fused_epilogue_f32_kernel(
    const float* __restrict__ out_in,   // (B,N,D) contiguous
    const float* __restrict__ t0,       // (B,D,N) contiguous
    const float* __restrict__ t1,       // (B,D,N) contiguous
    const float* __restrict__ t2,       // (B,D,N) contiguous
    const float* __restrict__ dy,       // (3)
    float* __restrict__ out,            // (B,N,D) contiguous
    int B, int N, int D
) {
    __shared__ float w0, w1, w2;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float a,b,c;
        softmax3_to_shared(dy, a,b,c);
        w0 = a; w1 = b; w2 = c;
    }
    __syncthreads();

    const int b = (int)blockIdx.x;
    const int n = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (b >= B || n >= N) return;

    if constexpr (VEC4) {
        // x indexes float4 groups along D
        const int d4 = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x; // groups of 4
        const int d = d4 * 4;
        if (d + 3 >= D) return;

        // out index base (b,n,d)
        const int64_t out_base = ((int64_t)b * (int64_t)N + (int64_t)n) * (int64_t)D + (int64_t)d;

        // t base for each d lane: t[(b*D + d)*N + n]
        const int64_t t_base0 = ((int64_t)b * (int64_t)D + (int64_t)(d + 0)) * (int64_t)N + (int64_t)n;
        const int64_t t_base1 = ((int64_t)b * (int64_t)D + (int64_t)(d + 1)) * (int64_t)N + (int64_t)n;
        const int64_t t_base2 = ((int64_t)b * (int64_t)D + (int64_t)(d + 2)) * (int64_t)N + (int64_t)n;
        const int64_t t_base3 = ((int64_t)b * (int64_t)D + (int64_t)(d + 3)) * (int64_t)N + (int64_t)n;

        float4 o4 = *reinterpret_cast<const float4*>(out_in + out_base);

        float r0 = w0 * ldg_f32(t0 + t_base0) + w1 * ldg_f32(t1 + t_base0) + w2 * ldg_f32(t2 + t_base0);
        float r1 = w0 * ldg_f32(t0 + t_base1) + w1 * ldg_f32(t1 + t_base1) + w2 * ldg_f32(t2 + t_base1);
        float r2 = w0 * ldg_f32(t0 + t_base2) + w1 * ldg_f32(t1 + t_base2) + w2 * ldg_f32(t2 + t_base2);
        float r3 = w0 * ldg_f32(t0 + t_base3) + w1 * ldg_f32(t1 + t_base3) + w2 * ldg_f32(t2 + t_base3);

        float4 out4;
        out4.x = o4.x + r0;
        out4.y = o4.y + r1;
        out4.z = o4.z + r2;
        out4.w = o4.w + r3;

        *reinterpret_cast<float4*>(out + out_base) = out4;
    } else {
        const int d = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x;
        if (d >= D) return;

        const int64_t out_idx = ((int64_t)b * (int64_t)N + (int64_t)n) * (int64_t)D + (int64_t)d;
        const int64_t t_idx   = ((int64_t)b * (int64_t)D + (int64_t)d) * (int64_t)N + (int64_t)n;

        float o = ldg_f32(out_in + out_idx);
        float r = w0 * ldg_f32(t0 + t_idx) + w1 * ldg_f32(t1 + t_idx) + w2 * ldg_f32(t2 + t_idx);
        out[out_idx] = o + r;
    }
}

// -------------------------------- FP16 fused kernel --------------------------------
// out/t0/t1/t2 are half. Accumulate in float, store half.
// Vectorize along D with half2 where possible (D even, aligned).
template <int USE_HALF2>
__global__ void muse_fused_epilogue_f16_kernel(
    const half* __restrict__ out_in, // (B,N,D)
    const half* __restrict__ t0,     // (B,D,N)
    const half* __restrict__ t1,
    const half* __restrict__ t2,
    const float* __restrict__ dy,    // float(3)
    half* __restrict__ out,          // (B,N,D)
    int B, int N, int D
) {
    __shared__ float w0, w1, w2;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float a,b,c;
        softmax3_to_shared(dy, a,b,c);
        w0 = a; w1 = b; w2 = c;
    }
    __syncthreads();

    const int b = (int)blockIdx.x;
    const int n = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (b >= B || n >= N) return;

    if constexpr (USE_HALF2) {
        const int d2 = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x; // half2 groups
        const int d = d2 * 2;
        if (d + 1 >= D) return;

        const int64_t out_base = ((int64_t)b * (int64_t)N + (int64_t)n) * (int64_t)D + (int64_t)d;

        // out half2
        const half2 o2 = *reinterpret_cast<const half2*>(out_in + out_base);

        // t indices for (d,n) and (d+1,n)
        const int64_t t_base0 = ((int64_t)b * (int64_t)D + (int64_t)(d + 0)) * (int64_t)N + (int64_t)n;
        const int64_t t_base1 = ((int64_t)b * (int64_t)D + (int64_t)(d + 1)) * (int64_t)N + (int64_t)n;

        float o_lo = __half2float(__low2half(o2));
        float o_hi = __half2float(__high2half(o2));

        float r_lo = w0 * __half2float(ldg_f16(t0 + t_base0)) + w1 * __half2float(ldg_f16(t1 + t_base0)) + w2 * __half2float(ldg_f16(t2 + t_base0));
        float r_hi = w0 * __half2float(ldg_f16(t0 + t_base1)) + w1 * __half2float(ldg_f16(t1 + t_base1)) + w2 * __half2float(ldg_f16(t2 + t_base1));

        half2 out2 = __floats2half2_rn(o_lo + r_lo, o_hi + r_hi);
        *reinterpret_cast<half2*>(out + out_base) = out2;
    } else {
        const int d = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x;
        if (d >= D) return;

        const int64_t out_idx = ((int64_t)b * (int64_t)N + (int64_t)n) * (int64_t)D + (int64_t)d;
        const int64_t t_idx   = ((int64_t)b * (int64_t)D + (int64_t)d) * (int64_t)N + (int64_t)n;

        float o = __half2float(ldg_f16(out_in + out_idx));
        float r = w0 * __half2float(ldg_f16(t0 + t_idx)) + w1 * __half2float(ldg_f16(t1 + t_idx)) + w2 * __half2float(ldg_f16(t2 + t_idx));
        out[out_idx] = __float2half_rn(o + r);
    }
}

torch::Tensor muse_fused_conv_branch_add_cuda(torch::Tensor out_in, torch::Tensor t0, torch::Tensor t1, torch::Tensor t2, torch::Tensor dy_paras) {
    TORCH_CHECK(out_in.is_cuda() && t0.is_cuda() && t1.is_cuda() && t2.is_cuda() && dy_paras.is_cuda(), "all must be CUDA");
    TORCH_CHECK(out_in.is_contiguous(), "out_in must be contiguous (B,N,D)");
    TORCH_CHECK(t0.is_contiguous() && t1.is_contiguous() && t2.is_contiguous(), "t0/t1/t2 must be contiguous (B,D,N)");
    TORCH_CHECK(dy_paras.is_contiguous() && dy_paras.numel() == 3, "dy_paras must be contiguous and have 3 elements");

    TORCH_CHECK(out_in.dim() == 3, "out_in must be (B,N,D)");
    TORCH_CHECK(t0.dim() == 3, "t0 must be (B,D,N)");
    TORCH_CHECK(t1.sizes() == t0.sizes() && t2.sizes() == t0.sizes(), "t1/t2 must match t0 shape");
    TORCH_CHECK(out_in.size(0) == t0.size(0), "B mismatch");
    TORCH_CHECK(out_in.size(1) == t0.size(2), "N mismatch between out_in and t0");
    TORCH_CHECK(out_in.size(2) == t0.size(1), "D mismatch between out_in and t0");

    int B = (int)out_in.size(0);
    int N = (int)out_in.size(1);
    int D = (int)out_in.size(2);

    // dy_paras expected float32 (compute softmax in float)
    TORCH_CHECK(dy_paras.scalar_type() == torch::kFloat32, "dy_paras must be float32");

    auto out = torch::empty_like(out_in);

    // 2D over (n,d) with small N; choose blockDim.y to cover N tiles, blockDim.x for D coalescing.
    // Empirically good defaults for D=512,N~49: x=128 (or 32 for vec4 groups), y=4 or 8.
    const int ty = (N >= 64) ? 4 : 8;

    if (out_in.scalar_type() == torch::kFloat32) {
        TORCH_CHECK(t0.scalar_type() == torch::kFloat32 && t1.scalar_type() == torch::kFloat32 && t2.scalar_type() == torch::kFloat32,
                    "t0/t1/t2 must match out_in dtype (float32) in FP32 path");

        // Prefer vec4 when D multiple of 4 and out pointer aligned.
        bool vec4_ok = (D % 4) == 0;
        vec4_ok = vec4_ok && (((uintptr_t)out_in.data_ptr<float>() & 0xF) == 0) && (((uintptr_t)out.data_ptr<float>() & 0xF) == 0);

        if (vec4_ok) {
            const int vx = 32; // threads for float4 groups -> covers 32*4=128 D per block
            dim3 block(vx, ty, 1);
            dim3 grid(B, (N + ty - 1) / ty, (D/4 + vx - 1) / vx);
            muse_fused_epilogue_f32_kernel<1><<<grid, block>>>(
                (const float*)out_in.data_ptr<float>(),
                (const float*)t0.data_ptr<float>(),
                (const float*)t1.data_ptr<float>(),
                (const float*)t2.data_ptr<float>(),
                (const float*)dy_paras.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, N, D
            );
        } else {
            const int vx = 128;
            dim3 block(vx, ty, 1);
            dim3 grid(B, (N + ty - 1) / ty, (D + vx - 1) / vx);
            muse_fused_epilogue_f32_kernel<0><<<grid, block>>>(
                (const float*)out_in.data_ptr<float>(),
                (const float*)t0.data_ptr<float>(),
                (const float*)t1.data_ptr<float>(),
                (const float*)t2.data_ptr<float>(),
                (const float*)dy_paras.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, N, D
            );
        }
        return out;
    }

    if (out_in.scalar_type() == torch::kFloat16) {
        TORCH_CHECK(t0.scalar_type() == torch::kFloat16 && t1.scalar_type() == torch::kFloat16 && t2.scalar_type() == torch::kFloat16,
                    "t0/t1/t2 must match out_in dtype (float16) in FP16 path");

        bool half2_ok = (D % 2) == 0;
        half2_ok = half2_ok && (((uintptr_t)out_in.data_ptr<at::Half>() & 0x3) == 0) && (((uintptr_t)out.data_ptr<at::Half>() & 0x3) == 0);

        if (half2_ok) {
            const int vx = 128; // half2 groups per block; covers 256 D elements
            dim3 block(vx, ty, 1);
            dim3 grid(B, (N + ty - 1) / ty, ((D/2) + vx - 1) / vx);
            muse_fused_epilogue_f16_kernel<1><<<grid, block>>>(
                (const half*)out_in.data_ptr<at::Half>(),
                (const half*)t0.data_ptr<at::Half>(),
                (const half*)t1.data_ptr<at::Half>(),
                (const half*)t2.data_ptr<at::Half>(),
                (const float*)dy_paras.data_ptr<float>(),
                (half*)out.data_ptr<at::Half>(),
                B, N, D
            );
        } else {
            const int vx = 128;
            dim3 block(vx, ty, 1);
            dim3 grid(B, (N + ty - 1) / ty, (D + vx - 1) / vx);
            muse_fused_epilogue_f16_kernel<0><<<grid, block>>>(
                (const half*)out_in.data_ptr<at::Half>(),
                (const half*)t0.data_ptr<at::Half>(),
                (const half*)t1.data_ptr<at::Half>(),
                (const half*)t2.data_ptr<at::Half>(),
                (const float*)dy_paras.data_ptr<float>(),
                (half*)out.data_ptr<at::Half>(),
                B, N, D
            );
        }
        return out;
    }

    TORCH_CHECK(false, "Unsupported dtype for out_in: expected float32 or float16");
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor muse_fused_conv_branch_add_cuda(torch::Tensor out_in, torch::Tensor t0, torch::Tensor t1, torch::Tensor t2, torch::Tensor dy_paras);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_muse_attention_v4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["muse_fused_conv_branch_add_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2,
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1,
        )

    def forward(self, x):
        return self.pointwise_conv(self.depth_conv(x))


class ModelNew(nn.Module):
    """
    MUSE attention with a fused CUDA epilogue for the conv branch:
      out_new = out + permute( softmax(dy) * [conv1(v2), conv3(v2), conv5(v2)] )
    The fusion removes the intermediate out2 tensor and reduces memory traffic/launch overhead.
    Supports FP32 and FP16 (FP16 uses mixed accumulation).
    """
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)

        self.dy_paras = nn.Parameter(torch.ones(3))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.custom_ops_lib = custom_ops_lib

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (B,N,D), typically float32 unless autocast is used

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # (B, h*d_v, N)

        # Conv branch outputs are (B,D,N)
        t0 = self.conv1(v2)
        t1 = self.conv3(v2)
        t2 = self.conv5(v2)

        # Ensure contiguous layouts expected by CUDA kernel
        out = out.contiguous()
        t0 = t0.contiguous()
        t1 = t1.contiguous()
        t2 = t2.contiguous()

        # dy in float32 for stable softmax (very small tensor)
        dy = self.dy_paras
        if dy.dtype != torch.float32:
            dy = dy.float()
        dy = dy.contiguous()

        # Dtype handling: keep a fast FP16 path if the model is in FP16.
        # If out is FP16, require conv outputs FP16 too; otherwise keep FP32.
        if out.dtype == torch.float16:
            if t0.dtype != torch.float16:
                t0 = t0.to(dtype=torch.float16)
                t1 = t1.to(dtype=torch.float16)
                t2 = t2.to(dtype=torch.float16)
        else:
            # Default to FP32 path for correctness/perf stability
            if out.dtype != torch.float32:
                out = out.float()
            if t0.dtype != torch.float32:
                t0 = t0.float()
                t1 = t1.float()
                t2 = t2.float()

        out = out.contiguous()
        t0 = t0.contiguous()
        t1 = t1.contiguous()
        t2 = t2.contiguous()

        # Fused: weighted sum + permute + residual add
        out = self.custom_ops_lib.muse_fused_conv_branch_add_cuda(out, t0, t1, t2, dy)
        return out