import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA/C++ extension: fused GroupNorm(tiny Cg=4) + tanh + hardswish + residual + logsumexp(channel) ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline __device__ float hardswish(float x) {
    float t = x + 3.0f;
    t = fminf(fmaxf(t, 0.0f), 6.0f);
    return x * (t * (1.0f / 6.0f));
}

static inline __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static inline __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__global__ __launch_bounds__(128, 3)
void fused_gn_tanh_hswish_res_lse_c64g16_vec4(
    const float* __restrict__ xconv,   // [N,C,H,W] contiguous
    const float* __restrict__ gamma,   // [C]
    const float* __restrict__ beta,    // [C]
    float* __restrict__ out,           // [N,1,H,W]
    int N, int H, int W, float eps)
{
    // One block per (n, hw)
    int idx = (int)blockIdx.x;
    int HW = H * W;
    int n = idx / HW;
    int hw = idx - n * HW;

    // Only warp0 computes; other warps idle (keeps register allocation bounded with launch_bounds)
    int t = (int)threadIdx.x;
    int lane = t & 31;
    if (t >= 32) return;

    // shape specialization: C=64, G=16 => Cg=4
    // GN stats are per (n,g,hw) over 4 channels at that spatial position
    // We'll load our two channels (c0,c1), compute group mean/rstd for our groups, then residual values,
    // then LSE over all 64 channels using warp reductions.

    // Compute spatial indices
    int h = hw / W;
    int w = hw - h * W;

    // Base pointers for this spatial location for channel c:
    // off(c) = ((n*C + c)*HW + hw)
    // We want to reduce index arithmetic: precompute base_n = n*HW, then channel base = (n*C + c)*HW
    int64_t HW64 = (int64_t)HW;
    int64_t base_n = (int64_t)n * 64 * HW64 + (int64_t)hw;

    // Each lane handles two channels
    int c0 = lane;
    int c1 = lane + 32;

    float xv0, xv1;

    // Attempt vectorized load along W when possible:
    // Load xconv for channel c at (h, w) via float4 on the HW dimension (W contiguous).
    // Works best when W % 4 == 0 and w aligned to 4.
    bool can_vec4 = ((W & 3) == 0) && ((w & 3) == 0);
    if (can_vec4) {
        int w4 = w >> 2; // index of float4 within row
        // Pointer to start of row for channel c: ((n*64 + c)*HW + h*W)
        int64_t row0 = (int64_t)n * 64 * HW64 + (int64_t)c0 * HW64 + (int64_t)h * W;
        int64_t row1 = (int64_t)n * 64 * HW64 + (int64_t)c1 * HW64 + (int64_t)h * W;
        const float4* p0 = reinterpret_cast<const float4*>(xconv + row0);
        const float4* p1 = reinterpret_cast<const float4*>(xconv + row1);
        float4 v0 = p0[w4];
        float4 v1 = p1[w4];
        // select lane within float4
        int k = w & 3;
        xv0 = (k == 0) ? v0.x : (k == 1) ? v0.y : (k == 2) ? v0.z : v0.w;
        xv1 = (k == 0) ? v1.x : (k == 1) ? v1.y : (k == 2) ? v1.z : v1.w;
    } else {
        xv0 = xconv[base_n + (int64_t)c0 * HW64];
        xv1 = xconv[base_n + (int64_t)c1 * HW64];
    }

    // Group id for each channel: g = c / 4 (since Cg=4)
    int g0 = c0 >> 2;
    int g1 = c1 >> 2;

    // Fetch the other two channels in each group to compute mean/var.
    // For g0: channels are (g0*4 + 0..3). c0 is one of them.
    // For g1 similarly.
    // We'll load all four values for both groups, but only for the two groups our lane belongs to.
    // Note: many lanes share groups; redundant loads are acceptable because Cg is tiny and we avoid extra kernels/buffers.
    auto load_x = [&](int c)->float {
        if (can_vec4) {
            int w4 = w >> 2;
            int64_t row = (int64_t)n * 64 * HW64 + (int64_t)c * HW64 + (int64_t)h * W;
            const float4* p = reinterpret_cast<const float4*>(xconv + row);
            float4 v = p[w4];
            int k = w & 3;
            return (k == 0) ? v.x : (k == 1) ? v.y : (k == 2) ? v.z : v.w;
        } else {
            return xconv[base_n + (int64_t)c * HW64];
        }
    };

    int base0 = g0 << 2;
    int base1 = g1 << 2;

    float a0 = load_x(base0 + 0);
    float a1 = load_x(base0 + 1);
    float a2 = load_x(base0 + 2);
    float a3 = load_x(base0 + 3);

    float b0 = load_x(base1 + 0);
    float b1 = load_x(base1 + 1);
    float b2 = load_x(base1 + 2);
    float b3 = load_x(base1 + 3);

    float m0 = 0.25f * (a0 + a1 + a2 + a3);
    float v0s = 0.25f * (a0*a0 + a1*a1 + a2*a2 + a3*a3) - m0*m0;
    float r0 = rsqrtf(v0s + eps);

    float m1 = 0.25f * (b0 + b1 + b2 + b3);
    float v1s = 0.25f * (b0*b0 + b1*b1 + b2*b2 + b3*b3) - m1*m1;
    float r1 = rsqrtf(v1s + eps);

    // Apply GN affine + tanh + hswish + residual for our two channels
    float gn0 = (xv0 - m0) * r0;
    float gn1 = (xv1 - m1) * r1;

    float y0 = gn0 * gamma[c0] + beta[c0];
    float y1 = gn1 * gamma[c1] + beta[c1];

    // fast-math tanh/exp enabled via --use_fast_math; tanhf maps to approx where allowed
    float t0 = tanhf(y0);
    float t1 = tanhf(y1);

    float hs0 = hardswish(t0);
    float hs1 = hardswish(t1);

    float xr0 = xv0 + hs0;
    float xr1 = xv1 + hs1;

    // LSE over 64 channels: each lane has two values -> reduce max then sumexp
    float vmax = fmaxf(xr0, xr1);
    vmax = warp_reduce_max(vmax);
    vmax = __shfl_sync(0xffffffff, vmax, 0);

    float s = __expf(xr0 - vmax) + __expf(xr1 - vmax);
    s = warp_reduce_sum(s);
    s = __shfl_sync(0xffffffff, s, 0);

    if (lane == 0) {
        float lse = logf(s) + vmax;
        out[(int64_t)n * HW64 + (int64_t)hw] = lse;
    }
}

__global__ void fused_residual_lse_generic(
    const float* __restrict__ xconv,   // [N,C,H,W]
    const float* __restrict__ mean,    // [N*G*H*W]
    const float* __restrict__ rstd,    // [N*G*H*W]
    const float* __restrict__ gn_gamma,// [C]
    const float* __restrict__ gn_beta, // [C]
    float* __restrict__ out,           // [N,1,H,W]
    int N, int C, int H, int W, int G)
{
    int idx = (int)blockIdx.x;
    int HW = H * W;
    int n = idx / HW;
    int hw = idx - n * HW;
    int h = hw / W;
    int w = hw - h * W;

    int Cg = C / G;
    int t = (int)threadIdx.x;

    float vmax = -INFINITY;
    for (int c = t; c < C; c += (int)blockDim.x) {
        int g = c / Cg;
        int64_t off = ((int64_t)n * C + c) * (int64_t)HW + (int64_t)h * W + w;

        float xv = xconv[off];
        int mg_idx = (n * G + g) * HW + hw;
        float m = mean[mg_idx];
        float inv = rstd[mg_idx];

        float gn = (xv - m) * inv;
        float yn = gn * gn_gamma[c] + gn_beta[c];
        float yt = tanhf(yn);
        float yh = hardswish(yt);
        float xr = xv + yh;
        vmax = fmaxf(vmax, xr);
    }

    __shared__ float sh_max[256];
    sh_max[t] = vmax;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) sh_max[t] = fmaxf(sh_max[t], sh_max[t + stride]);
        __syncthreads();
    }
    float maxv = sh_max[0];

    float s = 0.0f;
    for (int c = t; c < C; c += (int)blockDim.x) {
        int g = c / Cg;
        int64_t off = ((int64_t)n * C + c) * (int64_t)HW + (int64_t)h * W + w;

        float xv = xconv[off];
        int mg_idx = (n * G + g) * HW + hw;
        float m = mean[mg_idx];
        float inv = rstd[mg_idx];

        float gn = (xv - m) * inv;
        float yn = gn * gn_gamma[c] + gn_beta[c];
        float yt = tanhf(yn);
        float yh = hardswish(yt);
        float xr = xv + yh;
        s += __expf(xr - maxv);
    }

    __shared__ float sh_sum[256];
    sh_sum[t] = s;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) sh_sum[t] += sh_sum[t + stride];
        __syncthreads();
    }

    if (t == 0) {
        float lse = logf(sh_sum[0]) + maxv;
        out[(int64_t)n * HW + hw] = lse;
    }
}

__global__ void gn_stats_hw_kernel_generic(
    const float* __restrict__ x,   // [N,C,H,W] contiguous
    float* __restrict__ mean,      // [N*G*H*W]
    float* __restrict__ rstd,      // [N*G*H*W]
    int N, int C, int H, int W, int G, float eps)
{
    int idx = (int)blockIdx.x;
    int HW = H * W;
    int ng = idx / HW;
    int hw = idx - ng * HW;
    int n = ng / G;
    int g = ng - n * G;

    int Cg = C / G;
    int h = hw / W;
    int w = hw - h * W;

    float sum = 0.0f;
    float sumsq = 0.0f;

    int t = (int)threadIdx.x;
    for (int ci = t; ci < Cg; ci += (int)blockDim.x) {
        int c = g * Cg + ci;
        int64_t off = ((int64_t)n * C + c) * (int64_t)HW + (int64_t)h * W + w;
        float v = x[off];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float sh_sum[256];
    __shared__ float sh_sumsq[256];
    sh_sum[t] = sum;
    sh_sumsq[t] = sumsq;
    __syncthreads();

    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_sum[t] += sh_sum[t + stride];
            sh_sumsq[t] += sh_sumsq[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        float m = sh_sum[0] / (float)Cg;
        float var = sh_sumsq[0] / (float)Cg - m * m;
        float inv = rsqrtf(var + eps);
        mean[idx] = m;
        rstd[idx] = inv;
    }
}

torch::Tensor gn_tanh_hswish_residual_lse_forward_cuda(
    torch::Tensor xconv,
    torch::Tensor gn_gamma,
    torch::Tensor gn_beta,
    int64_t num_groups,
    double gn_eps)
{
    TORCH_CHECK(xconv.is_cuda(), "xconv must be CUDA");
    TORCH_CHECK(xconv.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(xconv.is_contiguous(), "xconv must be contiguous NCHW");
    TORCH_CHECK(xconv.dim() == 4, "xconv must be NCHW");

    TORCH_CHECK(gn_gamma.is_cuda() && gn_beta.is_cuda(), "gn params must be CUDA");
    TORCH_CHECK(gn_gamma.dtype() == torch::kFloat32 && gn_beta.dtype() == torch::kFloat32, "gn params must be float32");
    TORCH_CHECK(gn_gamma.is_contiguous() && gn_beta.is_contiguous(), "gn params must be contiguous");
    TORCH_CHECK(gn_gamma.dim() == 1 && gn_beta.dim() == 1, "gn params must be 1D");

    int64_t N = xconv.size(0);
    int64_t C = xconv.size(1);
    int64_t H = xconv.size(2);
    int64_t W = xconv.size(3);
    int64_t G64 = num_groups;

    TORCH_CHECK((int64_t)gn_gamma.size(0) == C && (int64_t)gn_beta.size(0) == C, "gn params must match channels");
    TORCH_CHECK(C % G64 == 0, "C must be divisible by num_groups");

    auto out = torch::empty({N, 1, H, W}, xconv.options());

    // Fast path: exact target shape (C=64,G=16), forward-only, computes GN stats on-the-fly per pixel.
    if (C == 64 && G64 == 16) {
        int64_t HW = H * W;
        int64_t blocks64 = N * HW;
        TORCH_CHECK(blocks64 <= (int64_t)2147483647, "too many blocks");
        int blocks = (int)blocks64;

        // output is [N,1,H,W] contiguous; write as [N*HW]
        fused_gn_tanh_hswish_res_lse_c64g16_vec4<<<blocks, 128>>>(
            (const float*)xconv.data_ptr<float>(),
            (const float*)gn_gamma.data_ptr<float>(),
            (const float*)gn_beta.data_ptr<float>(),
            (float*)out.view({N * HW}).data_ptr<float>(),
            (int)N, (int)H, (int)W, (float)gn_eps
        );
        return out;
    }

    // Generic fallback: original two-kernel path
    int G = (int)G64;
    int threads = 256;

    int64_t HW = H * W;
    auto stats_opts = xconv.options();
    auto mean = torch::empty({N * G64, H, W}, stats_opts).view({N * G64 * H * W});
    auto rstd = torch::empty({N * G64, H, W}, stats_opts).view({N * G64 * H * W});

    {
        int64_t blocks64 = N * G64 * H * W;
        TORCH_CHECK(blocks64 <= (int64_t)2147483647, "too many blocks");
        int blocks = (int)blocks64;
        gn_stats_hw_kernel_generic<<<blocks, threads>>>(
            (const float*)xconv.data_ptr<float>(),
            (float*)mean.data_ptr<float>(),
            (float*)rstd.data_ptr<float>(),
            (int)N, (int)C, (int)H, (int)W, (int)G, (float)gn_eps
        );
    }

    {
        int64_t blocks64 = N * H * W;
        TORCH_CHECK(blocks64 <= (int64_t)2147483647, "too many blocks");
        int blocks = (int)blocks64;
        fused_residual_lse_generic<<<blocks, threads>>>(
            (const float*)xconv.data_ptr<float>(),
            (const float*)mean.data_ptr<float>(),
            (const float*)rstd.data_ptr<float>(),
            (const float*)gn_gamma.data_ptr<float>(),
            (const float*)gn_beta.data_ptr<float>(),
            (float*)out.view({N * HW}).data_ptr<float>(),
            (int)N, (int)C, (int)H, (int)W, (int)G
        );
    }

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor gn_tanh_hswish_residual_lse_forward_cuda(
    torch::Tensor xconv,
    torch::Tensor gn_gamma,
    torch::Tensor gn_beta,
    int64_t num_groups,
    double gn_eps);
"""

custom_ops_lib = load_inline(
    name="custom_conv_gn_tanh_hswish_residual_lse_ops_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gn_tanh_hswish_residual_lse_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Conv2d + fused (GroupNorm + tanh + hardswish + residual add + logsumexp over channels) via custom CUDA.
    Fast path: inference-only specialized for C=64,G=16.
    Training: falls back to PyTorch ops (no custom backward).
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(int(groups), out_channels, eps=float(eps))
        self.groups = int(groups)
        self.gn_eps = float(eps)
        self.custom_ops_lib = custom_ops_lib

        # for state_dict compatibility
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x)

        if self.training:
            x_norm = self.group_norm(x_conv)
            x_tanh = torch.tanh(x_norm)
            x_hs = torch.nn.functional.hardswish(x_tanh)
            x_res = x_conv + x_hs
            return torch.logsumexp(x_res, dim=1, keepdim=True)

        if x_conv.dtype != torch.float32:
            x_conv = x_conv.float()
        if not x_conv.is_contiguous():
            x_conv = x_conv.contiguous()

        gn_w = self.group_norm.weight
        gn_b = self.group_norm.bias
        dev = x_conv.device
        if gn_w.device != dev:
            gn_w = gn_w.to(dev)
            gn_b = gn_b.to(dev)

        return self.custom_ops_lib.gn_tanh_hswish_residual_lse_forward_cuda(
            x_conv,
            gn_w.contiguous(),
            gn_b.contiguous(),
            int(self.groups),
            float(self.gn_eps),
        )