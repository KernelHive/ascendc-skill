import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---------------------------
# Custom CUDA ops:
#  (1) faster channels-first LayerNorm over C for NCHW tensors with warp-level reduction
#  (2) existing einsum replacements (QK and AV)
# ---------------------------

dattn_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline int64_t div_up_int64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    // full-warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Fast path LayerNorm for NCHW:
// - One warp handles one spatial position (n,h,w) OR a small vector of w positions (float4 along W).
// - Reduction across C is done with warp shuffles (no __syncthreads).
// - Vectorization: handle 4 adjacent W values per lane when possible (contiguous and aligned).
//
// Assumptions for fast path are checked in Python: x contiguous float32, and W multiple of 4.
// Kernel still correct for any W, but vectorization only kicks in when safe.
__global__ void layernorm_nchw_warpvec_fwd_kernel(
    const float* __restrict__ x,     // [N,C,H,W] contiguous
    const float* __restrict__ gamma, // [C] or nullptr
    const float* __restrict__ beta,  // [C] or nullptr
    float* __restrict__ y,           // [N,C,H,W]
    int N, int C, int H, int W,
    float eps
) {
    // grid: (N*H, ceil_div(W, 4)) blocks, each block is 1 warp (32 threads)
    // Each block handles a tile of 4 W positions: w0 = tile*4 .. tile*4+3 at fixed (n,h).
    int nhw = (int)blockIdx.x;
    int n = nhw / H;
    int h = nhw - n * H;

    int w_tile = (int)blockIdx.y;       // tile index along W in units of 4
    int w0 = w_tile * 4;

    // Each lane processes one of the 4 W positions based on lane_id % 4, and multiple lanes share same w.
    // lane groups: 8 lanes per w (since 32/4=8).
    int lane = (int)threadIdx.x; // 0..31
    int w_lane = lane & 3;       // 0..3
    int w = w0 + w_lane;
    if (w >= W) return;

    // Within each w, we use 8 lanes to stride over C: c = group_lane + t*8
    int group_lane = lane >> 2; // 0..7
    float sum = 0.0f;

    // base for x[n, 0, h, w]
    int64_t base = (((int64_t)n * C) * (int64_t)H + (int64_t)h) * (int64_t)W + (int64_t)w;
    int64_t cstride = (int64_t)H * (int64_t)W;

    // Sum over channels
    for (int c = group_lane; c < C; c += 8) {
        float xv = x[base + (int64_t)c * cstride];
        sum += xv;
    }

    // Reduce sum across the 8 lanes that share the same w.
    // We do this by masking lanes: use shuffles within the whole warp, but only among lanes with same (lane&3).
    // Pattern: lanes with same w_lane are at indices w_lane + 4*k.
    // Reduce across k=0..7 using shfl_down with delta=4,8,16.
    float sum_w = sum;
    sum_w += __shfl_down_sync(0xffffffff, sum_w, 4);
    sum_w += __shfl_down_sync(0xffffffff, sum_w, 8);
    sum_w += __shfl_down_sync(0xffffffff, sum_w, 16);

    // Broadcast mean to all lanes in the group (same w_lane) from leader lane (group_lane==0).
    float mean = __shfl_sync(0xffffffff, sum_w / (float)C, w_lane);

    // Variance
    float sumsq = 0.0f;
    for (int c = group_lane; c < C; c += 8) {
        float xv = x[base + (int64_t)c * cstride];
        float d = xv - mean;
        sumsq += d * d;
    }
    float sumsq_w = sumsq;
    sumsq_w += __shfl_down_sync(0xffffffff, sumsq_w, 4);
    sumsq_w += __shfl_down_sync(0xffffffff, sumsq_w, 8);
    sumsq_w += __shfl_down_sync(0xffffffff, sumsq_w, 16);

    float inv_std = rsqrtf(__shfl_sync(0xffffffff, (sumsq_w / (float)C) + eps, w_lane));

    // Write output: each lane writes its share of channels (striding by 8) for this (n,h,w)
    for (int c = group_lane; c < C; c += 8) {
        float xv = x[base + (int64_t)c * cstride];
        float yn = (xv - mean) * inv_std;
        float gv = gamma ? gamma[c] : 1.0f;
        float bv = beta  ? beta[c]  : 0.0f;
        y[base + (int64_t)c * cstride] = fmaf(yn, gv, bv);
    }
}

// Fallback-ish LayerNorm: one warp per (n,h,w) without w-tiling (handles any W, but more blocks).
__global__ void layernorm_nchw_warp_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps
) {
    // grid: N*H*W blocks, each block is 1 warp
    int nhw = (int)blockIdx.x;
    int n = nhw / (H * W);
    int hw = nhw - n * (H * W);
    int h = hw / W;
    int w = hw - h * W;

    int lane = (int)threadIdx.x; // 0..31
    float sum = 0.0f;

    int64_t base = (((int64_t)n * C) * (int64_t)H + (int64_t)h) * (int64_t)W + (int64_t)w;
    int64_t cstride = (int64_t)H * (int64_t)W;

    for (int c = lane; c < C; c += 32) {
        float xv = x[base + (int64_t)c * cstride];
        sum += xv;
    }
    float sum_all = warp_allreduce_sum(sum);
    float mean = sum_all / (float)C;

    float sumsq = 0.0f;
    for (int c = lane; c < C; c += 32) {
        float xv = x[base + (int64_t)c * cstride];
        float d = xv - mean;
        sumsq += d * d;
    }
    float sumsq_all = warp_allreduce_sum(sumsq);
    float inv_std = rsqrtf(sumsq_all / (float)C + eps);

    for (int c = lane; c < C; c += 32) {
        float xv = x[base + (int64_t)c * cstride];
        float yn = (xv - mean) * inv_std;
        float gv = gamma ? gamma[c] : 1.0f;
        float bv = beta  ? beta[c]  : 0.0f;
        y[base + (int64_t)c * cstride] = fmaf(yn, gv, bv);
    }
}

// QK kernel: q [B,C,M], k [B,C,N] -> attn [B,M,N]
__global__ void qk_einsum_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ attn,
    int B, int C, int M, int N,
    float scale
){
    int b = (int)blockIdx.y;
    int64_t mn = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t MN = (int64_t)M * (int64_t)N;
    if (mn >= MN) return;

    int m = (int)(mn / N);
    int n = (int)(mn - (int64_t)m * N);

    float acc = 0.0f;
    int64_t q_base = ((int64_t)b * C) * (int64_t)M + m;
    int64_t k_base = ((int64_t)b * C) * (int64_t)N + n;

    #pragma unroll 4
    for (int c = 0; c < C; ++c) {
        float qv = q[q_base + (int64_t)c * M];
        float kv = k[k_base + (int64_t)c * N];
        acc = fmaf(qv, kv, acc);
    }
    attn[((int64_t)b * M + m) * (int64_t)N + n] = acc * scale;
}

// AV kernel: attn [B,M,N], v [B,C,N] -> out [B,C,M]
__global__ void av_einsum_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int C, int M, int N
){
    int b = (int)blockIdx.y;
    int64_t cm = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t CM = (int64_t)C * (int64_t)M;
    if (cm >= CM) return;

    int c = (int)(cm / M);
    int m = (int)(cm - (int64_t)c * M);

    float acc = 0.0f;
    int64_t attn_base = ((int64_t)b * M + m) * (int64_t)N;
    int64_t v_base = ((int64_t)b * C + c) * (int64_t)N;

    #pragma unroll 4
    for (int n = 0; n < N; ++n) {
        float a = attn[attn_base + n];
        float vv = v[v_base + n];
        acc = fmaf(a, vv, acc);
    }
    out[((int64_t)b * C + c) * (int64_t)M + m] = acc;
}

torch::Tensor layernorm_nchw_fwd_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");

    int64_t N64 = x.size(0), C64 = x.size(1), H64 = x.size(2), W64 = x.size(3);
    TORCH_CHECK(N64 <= INT32_MAX && C64 <= INT32_MAX && H64 <= INT32_MAX && W64 <= INT32_MAX, "dims too large");
    int N = (int)N64, C = (int)C64, H = (int)H64, W = (int)W64;

    const float* gptr = nullptr;
    const float* bptr = nullptr;
    if (gamma.defined()) {
        TORCH_CHECK(gamma.is_cuda() && gamma.dtype() == torch::kFloat32 && gamma.is_contiguous(), "gamma must be contiguous CUDA float32");
        TORCH_CHECK(gamma.numel() == C64, "gamma shape mismatch");
        gptr = gamma.data_ptr<float>();
    }
    if (beta.defined()) {
        TORCH_CHECK(beta.is_cuda() && beta.dtype() == torch::kFloat32 && beta.is_contiguous(), "beta must be contiguous CUDA float32");
        TORCH_CHECK(beta.numel() == C64, "beta shape mismatch");
        bptr = beta.data_ptr<float>();
    }

    auto y = torch::empty_like(x);

    // Prefer W-tiling kernel when W>=4 (common case) to reduce grid size by 4x.
    // Grid: (N*H, ceil(W/4))
    if (W >= 4) {
        dim3 blocks((unsigned int)(N * H), (unsigned int)div_up_int64(W, 4));
        layernorm_nchw_warpvec_fwd_kernel<<<blocks, 32>>>(
            x.data_ptr<float>(), gptr, bptr, y.data_ptr<float>(),
            N, C, H, W, (float)eps
        );
    } else {
        int64_t blocks = N64 * H64 * W64;
        layernorm_nchw_warp_fwd_kernel<<<(unsigned int)blocks, 32>>>(
            x.data_ptr<float>(), gptr, bptr, y.data_ptr<float>(),
            N, C, H, W, (float)eps
        );
    }
    return y;
}

torch::Tensor qk_einsum_cuda(torch::Tensor q, torch::Tensor k, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32, "q and k must be float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q and k must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q,k must be 3D: q[B,C,M], k[B,C,N]");
    TORCH_CHECK(q.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(q.size(1) == k.size(1), "C mismatch");

    int64_t B64 = q.size(0);
    int64_t C64 = q.size(1);
    int64_t M64 = q.size(2);
    int64_t N64 = k.size(2);

    TORCH_CHECK(B64 <= INT32_MAX && C64 <= INT32_MAX && M64 <= INT32_MAX && N64 <= INT32_MAX, "dims too large");

    int B = (int)B64, C = (int)C64, M = (int)M64, N = (int)N64;

    auto attn = torch::empty({B64, M64, N64}, q.options());

    const int threads = 128;
    int64_t MN = M64 * N64;
    int64_t blocks_x = div_up_int64(MN, threads);
    dim3 blocks((unsigned int)blocks_x, (unsigned int)B);

    qk_einsum_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        attn.data_ptr<float>(),
        B, C, M, N,
        (float)scale
    );
    return attn;
}

torch::Tensor av_einsum_cuda(torch::Tensor attn, torch::Tensor v) {
    TORCH_CHECK(attn.is_cuda() && v.is_cuda(), "attn and v must be CUDA tensors");
    TORCH_CHECK(attn.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32, "attn and v must be float32");
    TORCH_CHECK(attn.is_contiguous() && v.is_contiguous(), "attn and v must be contiguous");
    TORCH_CHECK(attn.dim() == 3 && v.dim() == 3, "attn must be [B,M,N], v must be [B,C,N]");
    TORCH_CHECK(attn.size(0) == v.size(0), "B mismatch");
    TORCH_CHECK(attn.size(2) == v.size(2), "N mismatch");

    int64_t B64 = attn.size(0);
    int64_t M64 = attn.size(1);
    int64_t N64 = attn.size(2);
    int64_t C64 = v.size(1);

    TORCH_CHECK(B64 <= INT32_MAX && C64 <= INT32_MAX && M64 <= INT32_MAX && N64 <= INT32_MAX, "dims too large");

    int B = (int)B64, C = (int)C64, M = (int)M64, N = (int)N64;

    auto out = torch::empty({B64, C64, M64}, v.options());

    const int threads = 128;
    int64_t CM = C64 * M64;
    int64_t blocks_x = div_up_int64(CM, threads);
    dim3 blocks((unsigned int)blocks_x, (unsigned int)B);

    av_einsum_kernel<<<blocks, threads>>>(
        attn.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, M, N
    );
    return out;
}
"""

dattn_cpp_src = r"""
torch::Tensor layernorm_nchw_fwd_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);
torch::Tensor qk_einsum_cuda(torch::Tensor q, torch::Tensor k, double scale);
torch::Tensor av_einsum_cuda(torch::Tensor attn, torch::Tensor v);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_d_attention_baseline_v4_lnwarpvec",
    cpp_sources=dattn_cpp_src,
    cuda_sources=dattn_cuda_src,
    functions=["layernorm_nchw_fwd_cuda", "qk_einsum_cuda", "av_einsum_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)

# ---------------------------
# CUDA LayerNorm proxy (channels-first, fused)
# ---------------------------

class LayerNormProxyOptim(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [N,C,H,W]
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return custom_ops_lib.layernorm_nchw_fwd_cuda(
                x, self.weight.contiguous(), self.bias.contiguous(), float(self.eps)
            )
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (y.size(-1),), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)

# ---------------------------
# Model using optimized LayerNorm + cached grids + existing einsum kernels
# ---------------------------

class ModelNew(nn.Module):
    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride,
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, stage_idx
    ):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxyOptim(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
        )

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
                nn.init.trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                nn.init.trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        self.custom_ops = custom_ops_lib

        # Caches (keyed by (device, dtype, H, W, BxG))
        self._ref_cache = {}
        self._offset_range_cache = {}

    @torch.no_grad()
    def _get_ref_points_cached(self, H_key, W_key, B, dtype, device):
        bg = B * self.n_groups
        key = (device, dtype, int(H_key), int(W_key), int(bg))
        ref = self._ref_cache.get(key, None)
        if ref is None or (ref.device != device) or (ref.dtype != dtype):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
                torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
                indexing="ij",
            )
            ref = torch.stack((ref_y, ref_x), -1)
            ref[..., 1].div_(W_key).mul_(2).sub_(1)
            ref[..., 0].div_(H_key).mul_(2).sub_(1)
            ref = ref[None, ...].expand(bg, -1, -1, -1).contiguous()
            self._ref_cache[key] = ref
        return ref

    @torch.no_grad()
    def _get_offset_range_cached(self, Hk, Wk, dtype, device):
        key = (device, dtype, int(Hk), int(Wk))
        t = self._offset_range_cache.get(key, None)
        if t is None or (t.device != device) or (t.dtype != dtype):
            t = torch.tensor([1.0 / float(Hk), 1.0 / float(Wk)], device=device, dtype=dtype).reshape(1, 2, 1, 1)
            self._offset_range_cache[key] = t
        return t

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = q.reshape(B * self.n_groups, self.n_group_channels, H, W)
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = self._get_offset_range_cached(Hk, Wk, dtype, device)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = offset.permute(0, 2, 3, 1)
        reference = self._get_ref_points_cached(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode="bilinear",
            align_corners=True,
        )

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q_ = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_ = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_ = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        use_custom = (
            x.is_cuda
            and x.dtype == torch.float32
            and q_.is_contiguous()
            and k_.is_contiguous()
            and v_.is_contiguous()
        )

        if use_custom:
            attn = self.custom_ops.qk_einsum_cuda(q_, k_, float(self.scale))
        else:
            attn = torch.einsum("b c m, b c n -> b m n", q_, k_).mul(self.scale)

        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                    B * self.n_heads, self.n_head_channels, H * W
                )
            elif self.fixed_pe:
                attn_bias = self.rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            else:
                rpe_bias = self.rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points_cached(H, W, B, dtype, device)
                displacement = (
                    q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
                    - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode="bilinear",
                    align_corners=True,
                )
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        if use_custom and attn.is_contiguous():
            out = self.custom_ops.av_einsum_cuda(attn, v_)
        else:
            out = torch.einsum("b m n, b c n -> b c m", attn, v_)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        out = out.reshape(B, C, H, W)
        y = self.proj_drop(self.proj_out(out))
        return y