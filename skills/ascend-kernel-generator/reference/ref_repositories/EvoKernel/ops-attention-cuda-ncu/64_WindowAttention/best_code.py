import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------
# CUDA extension: fused W-MSA forward (online softmax, warp-per-row)
# - Fast path: N=49, D=64 with constant-memory rpi + optional float4 K/V loads
# - Fallback: generic online-softmax
# -------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_INPUT_F(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)
#define CHECK_INPUT_I32(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x)

#ifndef __CUDA_ARCH__
#define __ldg(x) (*(x))
#endif

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return __shfl_sync(0xffffffff, v, 0);
}

// Constant memory for rpi in the hot path (49*49 = 2401 ints)
__device__ __constant__ int c_rpi_49[49 * 49];

static inline void set_rpi_49_const(torch::Tensor rpi_i32_49x49) {
    TORCH_CHECK(rpi_i32_49x49.is_cuda(), "rpi must be CUDA tensor");
    TORCH_CHECK(rpi_i32_49x49.is_contiguous(), "rpi must be contiguous");
    TORCH_CHECK(rpi_i32_49x49.scalar_type() == at::ScalarType::Int, "rpi must be int32");
    TORCH_CHECK(rpi_i32_49x49.numel() == 49 * 49, "rpi must be 49x49");

    auto stream = c10::cuda::getDefaultCUDAStream(rpi_i32_49x49.device().index());
    cudaError_t err = cudaMemcpyToSymbolAsync(
        c_rpi_49,
        rpi_i32_49x49.data_ptr<int>(),
        sizeof(int) * 49 * 49,
        0,
        cudaMemcpyDeviceToDevice,
        stream.stream()
    );
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbolAsync failed");
}

static __forceinline__ __device__ float4 ldg_f4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

template<bool VEC4>
__global__ void window_attn_online_fwd_dh64_n49_kernel(
    const float* __restrict__ Q,   // (B,H,49,64)
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ rpb_table, // (M,H)
    float* __restrict__ Out,             // (B,H,49,64)
    int B, int Hh, int M,
    float scale
) {
    constexpr int N = 49;
    constexpr int Dh = 64;

    int warps_per_block = (int)(blockDim.x >> 5);
    int warp_in_block   = (int)(threadIdx.x >> 5);
    int lane            = (int)(threadIdx.x & 31);

    int global_warp = (int)blockIdx.x * warps_per_block + warp_in_block;
    int total_warps = B * Hh * N;
    if (global_warp >= total_warps) return;

    int tmp = global_warp;
    int n = tmp % N; tmp /= N;
    int h = tmp % Hh; tmp /= Hh;
    int b = tmp;

    long base_bh = ((long)b * Hh + h) * (long)N * (long)Dh;
    const float* q_ptr = Q + base_bh + (long)n * Dh;

    // lane owns 2 dims (Dh=64)
    int dv = lane * 2;
    float q0 = __ldg(q_ptr + dv);
    float q1 = __ldg(q_ptr + dv + 1);

    float m_i = -INFINITY;
    float l_i = 0.f;
    float o0 = 0.f, o1 = 0.f;

    // running pointers reduce address arithmetic
    const float* k_ptr = K + base_bh;
    const float* v_ptr = V + base_bh;

    #pragma unroll
    for (int t = 0; t < N; ++t) {
        float partial;

        if constexpr (VEC4) {
            // dv in {0..62} step 2. Map to float4 group.
            int f4_idx = dv >> 2;          // 0..15
            bool hi = (lane & 1);          // even: x,y ; odd: z,w
            const float4 k4 = ldg_f4(k_ptr + (f4_idx << 2));
            float k0 = hi ? k4.z : k4.x;
            float k1 = hi ? k4.w : k4.y;
            partial = q0 * k0 + q1 * k1;
        } else {
            float k0 = __ldg(k_ptr + dv);
            float k1 = __ldg(k_ptr + dv + 1);
            partial = q0 * k0 + q1 * k1;
        }

        float dot = warp_reduce_sum(partial);

        float score = dot * scale;
        int ridx = c_rpi_49[n * N + t];
        score += __ldg(rpb_table + (long)ridx * Hh + h);

        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float beta  = __expf(score - m_new);

        o0 *= alpha;
        o1 *= alpha;

        if constexpr (VEC4) {
            int f4_idx = dv >> 2;
            bool hi = (lane & 1);
            const float4 v4 = ldg_f4(v_ptr + (f4_idx << 2));
            float vv0 = hi ? v4.z : v4.x;
            float vv1 = hi ? v4.w : v4.y;
            o0 += beta * vv0;
            o1 += beta * vv1;
        } else {
            float vv0 = __ldg(v_ptr + dv);
            float vv1 = __ldg(v_ptr + dv + 1);
            o0 += beta * vv0;
            o1 += beta * vv1;
        }

        l_i = l_i * alpha + beta;
        m_i = m_new;

        k_ptr += Dh;
        v_ptr += Dh;
    }

    float inv_l = 1.f / fmaxf(l_i, 1e-9f);
    o0 *= inv_l;
    o1 *= inv_l;

    float* out_ptr = Out + base_bh + (long)n * Dh + dv;
    out_ptr[0] = o0;
    out_ptr[1] = o1;
}

// Generic online-softmax kernel (any N, any D)
__global__ void window_attn_online_fwd_generic_kernel(
    const float* __restrict__ Q,   // (B,H,N,D)
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ rpb_table, // (M,H)
    const int*   __restrict__ rpi,       // (N,N)
    float* __restrict__ Out,             // (B,H,N,D)
    int B, int Hh, int N, int D, int M,
    float scale
) {
    int warps_per_block = (int)(blockDim.x >> 5);
    int warp_in_block = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);

    int global_warp = (int)blockIdx.x * warps_per_block + warp_in_block;
    int total_warps = B * Hh * N;
    if (global_warp >= total_warps) return;

    int tmp = global_warp;
    int n = tmp % N; tmp /= N;
    int h = tmp % Hh; tmp /= Hh;
    int b = tmp;

    long base_bh = ((long)b * Hh + h) * (long)N * (long)D;
    const float* q_ptr = Q + base_bh + (long)n * D;

    float m_i = -INFINITY;
    float l_i = 0.f;

    int dv0 = lane * 2;
    float acc0 = 0.f, acc1 = 0.f;
    bool active0 = (dv0 < D);
    bool active1 = (dv0 + 1 < D);

    float q0 = active0 ? __ldg(q_ptr + dv0) : 0.f;
    float q1 = active1 ? __ldg(q_ptr + dv0 + 1) : 0.f;

    for (int t = 0; t < N; ++t) {
        const float* k_ptr = K + base_bh + (long)t * D;
        const float* v_ptr = V + base_bh + (long)t * D;

        float partial = 0.f;
        for (int d = dv0; d < D; d += 64) {
            float qq0 = __ldg(q_ptr + d);
            float kk0 = __ldg(k_ptr + d);
            partial += qq0 * kk0;
            if (d + 1 < D) {
                float qq1 = __ldg(q_ptr + d + 1);
                float kk1 = __ldg(k_ptr + d + 1);
                partial += qq1 * kk1;
            }
        }
        float dot = warp_reduce_sum(partial);

        float score = dot * scale;
        int ridx = rpi[n * N + t];
        score += __ldg(rpb_table + (long)ridx * Hh + h);

        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float beta  = __expf(score - m_new);

        acc0 *= alpha;
        acc1 *= alpha;

        if (active0) acc0 += beta * __ldg(v_ptr + dv0);
        if (active1) acc1 += beta * __ldg(v_ptr + dv0 + 1);

        l_i = l_i * alpha + beta;
        m_i = m_new;
    }

    float inv_l = 1.f / fmaxf(l_i, 1e-9f);

    if (active0) Out[base_bh + (long)n * D + dv0] = acc0 * inv_l;
    if (active1) Out[base_bh + (long)n * D + dv0 + 1] = acc1 * inv_l;
}

torch::Tensor window_attention_fused_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor relative_position_bias_table,
    torch::Tensor relative_position_index_i32,
    double scale_double
) {
    CHECK_INPUT_F(Q);
    CHECK_INPUT_F(K);
    CHECK_INPUT_F(V);
    CHECK_INPUT_F(relative_position_bias_table);
    CHECK_INPUT_I32(relative_position_index_i32);

    TORCH_CHECK(Q.dim() == 4, "Q must be (B,H,N,D)");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(), "K,V must match Q shape");
    TORCH_CHECK(relative_position_bias_table.dim() == 2, "relative_position_bias_table must be (M,H)");
    TORCH_CHECK(relative_position_index_i32.dim() == 2, "relative_position_index_i32 must be (N,N)");

    int B  = (int)Q.size(0);
    int Hh = (int)Q.size(1);
    int N  = (int)Q.size(2);
    int D  = (int)Q.size(3);

    TORCH_CHECK((int)relative_position_index_i32.size(0) == N && (int)relative_position_index_i32.size(1) == N,
                "relative_position_index_i32 must be (N,N) matching Q N");
    TORCH_CHECK((int)relative_position_bias_table.size(1) == Hh,
                "relative_position_bias_table second dim must match num_heads");

    int M = (int)relative_position_bias_table.size(0);

    auto Out = torch::empty_like(Q);
    float scale = (float)scale_double;

    // Warp-per-row launch.
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int total_warps = B * Hh * N;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    if (N == 49 && D == 64) {
        // Populate constant memory (small, once per call; uses default stream for the device)
        set_rpi_49_const(relative_position_index_i32);

        uintptr_t k_addr = (uintptr_t)K.data_ptr<float>();
        uintptr_t v_addr = (uintptr_t)V.data_ptr<float>();
        bool vec4_ok = ((k_addr & 0xF) == 0) && ((v_addr & 0xF) == 0);

        if (vec4_ok) {
            window_attn_online_fwd_dh64_n49_kernel<true><<<blocks, threads, 0>>>(
                (const float*)Q.data_ptr<float>(),
                (const float*)K.data_ptr<float>(),
                (const float*)V.data_ptr<float>(),
                (const float*)relative_position_bias_table.data_ptr<float>(),
                (float*)Out.data_ptr<float>(),
                B, Hh, M, scale
            );
        } else {
            window_attn_online_fwd_dh64_n49_kernel<false><<<blocks, threads, 0>>>(
                (const float*)Q.data_ptr<float>(),
                (const float*)K.data_ptr<float>(),
                (const float*)V.data_ptr<float>(),
                (const float*)relative_position_bias_table.data_ptr<float>(),
                (float*)Out.data_ptr<float>(),
                B, Hh, M, scale
            );
        }
    } else {
        window_attn_online_fwd_generic_kernel<<<blocks, threads, 0>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (const float*)relative_position_bias_table.data_ptr<float>(),
            (const int*)relative_position_index_i32.data_ptr<int>(),
            (float*)Out.data_ptr<float>(),
            B, Hh, N, D, M, scale
        );
    }

    return Out;
}
"""

cpp_src = r"""
torch::Tensor window_attention_fused_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor relative_position_bias_table,
    torch::Tensor relative_position_index_i32,
    double scale_double
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_window_attn_fused_v5_const_rpi_vec4_streamfix",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["window_attention_fused_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    verbose=False,
)


# -------------------------
# Init helpers (same behavior)
# -------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# -------------------------
# Optimized model
# -------------------------
class ModelNew(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) with relative position bias.
    Uses a fused CUDA kernel for: softmax((q) @ k^T * scale + rpb) @ v
    when running on CUDA and attn_drop==0 and proj_drop==0 and not training.
    Otherwise falls back to PyTorch reference path.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size
        self.query_size = self.window_size
        self.key_size = self.window_size[0] * 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(qk_scale or head_dim ** -0.5)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.register_buffer("relative_position_index_i32", torch.empty(0, dtype=torch.int32), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.custom_ops_lib = custom_ops_lib
        self._attn_drop_p = float(attn_drop)
        self._proj_drop_p = float(proj_drop)

    def _get_rpi_i32(self, device):
        rpi = self.relative_position_index
        if (self.relative_position_index_i32.numel() != rpi.numel() or
            self.relative_position_index_i32.device != device):
            self.relative_position_index_i32 = rpi.to(device=device, dtype=torch.int32).contiguous()
        return self.relative_position_index_i32

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_,H,N,D)

        use_fused = (
            x.is_cuda and
            (not self.training) and
            self._attn_drop_p == 0.0 and
            self._proj_drop_p == 0.0
        )

        if use_fused:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            rpb = self.relative_position_bias_table
            if (not rpb.is_cuda) or (rpb.device != x.device):
                rpb = rpb.to(device=x.device)
            rpb = rpb.contiguous()

            rpi_i32 = self._get_rpi_i32(x.device)

            # fused expects float32
            if q.dtype != torch.float32:
                qf = q.float()
                kf = k.float()
                vf = v.float()
            else:
                qf, kf, vf = q, k, v

            out = self.custom_ops_lib.window_attention_fused_forward_cuda(
                qf, kf, vf, rpb.float(), rpi_i32, float(self.scale)
            )

            if out.dtype != q.dtype:
                out = out.to(dtype=q.dtype)

            x = out.transpose(1, 2).contiguous().reshape(B_, N, C)
            x = self.proj(x)
            return x

        # Reference PyTorch path
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x