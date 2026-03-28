import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA/C++ extension: fused GELU (exact default, optional fast tanh) + GroupNorm ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__)
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

// -------- GELU implementations --------
static __device__ __forceinline__ float gelu_exact(float x) {
    // Exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

static __device__ __forceinline__ float gelu_tanh_fast(float x) {
    // Common tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    // Using fast math intrinsics (tanhf under --use_fast_math maps to fast approx).
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

// -------- Welford helpers --------
struct WelfordData {
    float mean;
    float m2;
    float count;
};

static __device__ __forceinline__ WelfordData welford_init() {
    WelfordData w;
    w.mean = 0.0f;
    w.m2 = 0.0f;
    w.count = 0.0f;
    return w;
}

static __device__ __forceinline__ void welford_update(WelfordData &w, float x) {
    w.count += 1.0f;
    float delta = x - w.mean;
    w.mean += delta / w.count;
    float delta2 = x - w.mean;
    w.m2 += delta * delta2;
}

static __device__ __forceinline__ WelfordData welford_combine(WelfordData a, WelfordData b) {
    if (b.count == 0.0f) return a;
    if (a.count == 0.0f) return b;
    float delta = b.mean - a.mean;
    float count = a.count + b.count;
    WelfordData out;
    out.mean = a.mean + delta * (b.count / count);
    out.m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / count);
    out.count = count;
    return out;
}

static __device__ __forceinline__ WelfordData welford_warp_reduce(WelfordData v) {
    unsigned mask = 0xffffffffu;
    // Reduce within warp using shuffles
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordData b;
        b.mean  = __shfl_down_sync(mask, v.mean,  offset);
        b.m2    = __shfl_down_sync(mask, v.m2,    offset);
        b.count = __shfl_down_sync(mask, v.count, offset);
        v = welford_combine(v, b);
    }
    return v;
}

template<int THREADS, bool FAST_GELU>
__global__ __launch_bounds__(THREADS, 2)
void gelu_groupnorm_fused_kernel_v4(
    const float* __restrict__ x,    // [N,C,H,W] contiguous
    float* __restrict__ y,          // [N,C,H,W] contiguous
    const float* __restrict__ gamma,// [C]
    const float* __restrict__ beta, // [C]
    int C, int HxW, int G, float eps)
{
    // One block per (n,g)
    int idx = (int)blockIdx.x;
    int n = idx / G;
    int g = idx - n * G;

    int Cg = C / G;
    int D  = Cg * HxW;

    int64_t base = (int64_t)(n * C + g * Cg) * (int64_t)HxW;
    const float* __restrict__ x_ptr = x + base;
    float* __restrict__ y_ptr = y + base;

    // ---- Pass1: per-thread Welford over GELU(x) ----
    WelfordData w = welford_init();

    // Iterate channel-major to avoid i/HxW div in the hot loop:
    // For each channel in group, each thread walks spatial with stride THREADS.
    for (int c = 0; c < Cg; ++c) {
        const float* __restrict__ xc = x_ptr + (int64_t)c * (int64_t)HxW;
        for (int hw = threadIdx.x; hw < HxW; hw += THREADS) {
            float xv = LDG(xc + hw);
            float a = FAST_GELU ? gelu_tanh_fast(xv) : gelu_exact(xv);
            welford_update(w, a);
        }
    }

    // Warp reduction
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int WARPS = (THREADS + 31) / 32;

    w = welford_warp_reduce(w);

    // Shared only for warp leaders
    __shared__ float sh_mean[WARPS];
    __shared__ float sh_m2[WARPS];
    __shared__ float sh_count[WARPS];

    if (lane == 0) {
        sh_mean[warp]  = w.mean;
        sh_m2[warp]    = w.m2;
        sh_count[warp] = w.count;
    }
    __syncthreads();

    // Final reduction by first warp
    WelfordData wb = welford_init();
    if (warp == 0) {
        if (lane < WARPS) {
            wb.mean  = sh_mean[lane];
            wb.m2    = sh_m2[lane];
            wb.count = sh_count[lane];
        }
        wb = welford_warp_reduce(wb);
        if (lane == 0) {
            sh_mean[0]  = wb.mean;
            sh_m2[0]    = wb.m2;
            sh_count[0] = wb.count;
        }
    }
    __syncthreads();

    float mean = sh_mean[0];
    float var  = (sh_count[0] > 0.0f) ? (sh_m2[0] / sh_count[0]) : 0.0f;
    float inv  = rsqrtf(var + eps);

    // ---- Pass2: write y = ((gelu(x)-mean)*inv)*gamma + beta ----
    // Also channel-major to avoid div/mod.
    for (int c = 0; c < Cg; ++c) {
        int ch = g * Cg + c;
        float gg = LDG(gamma + ch);
        float bb = LDG(beta + ch);

        const float* __restrict__ xc = x_ptr + (int64_t)c * (int64_t)HxW;
        float* __restrict__ yc = y_ptr + (int64_t)c * (int64_t)HxW;

        for (int hw = threadIdx.x; hw < HxW; hw += THREADS) {
            float xv = LDG(xc + hw);
            float a  = FAST_GELU ? gelu_tanh_fast(xv) : gelu_exact(xv);
            float norm = (a - mean) * inv;
            yc[hw] = norm * gg + bb;
        }
    }
}

torch::Tensor gelu_groupnorm_forward_cuda(
    torch::Tensor x,        // [N,C,H,W]
    torch::Tensor gamma,    // [C]
    torch::Tensor beta,     // [C]
    int64_t num_groups,
    double eps,
    bool fast_gelu)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");

    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32, "gamma/beta must be float32");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma/beta must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(gamma.numel() == C && beta.numel() == C, "gamma/beta must have C elements");
    TORCH_CHECK(num_groups > 0, "num_groups must be > 0");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    auto y = torch::empty_like(x);

    int HxW = (int)(H * W);
    int G = (int)num_groups;

    int blocks = (int)(N * num_groups);
    // 256 threads works well for large HxW; warp-reduction reduces sync cost vs baseline.
    constexpr int threads = 256;

    if (fast_gelu) {
        gelu_groupnorm_fused_kernel_v4<threads, true><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (int)C, (int)HxW, (int)G, (float)eps
        );
    } else {
        gelu_groupnorm_fused_kernel_v4<threads, false><<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            (int)C, (int)HxW, (int)G, (float)eps
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor gelu_groupnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    bool fast_gelu);
"""

custom_ops_lib = load_inline(
    name="custom_gelu_gn_ops_v4_warpwelford_chmajor",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gelu_groupnorm_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    ConvTranspose2d -> fused (GELU + GroupNorm) via custom CUDA in eval/inference.

    - Default GELU is exact (erf) for correctness parity.
    - Optional fast GELU (tanh approx) can be enabled via env var:
        CUSTOM_FUSED_FAST_GELU=1
    Training falls back to PyTorch ops (autograd correctness).
    Fused path supports CUDA float32 contiguous NCHW.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=int(num_groups), num_channels=out_channels, affine=True)

        self.num_groups = int(num_groups)
        self.gn_eps = float(self.group_norm.eps)
        self.custom_ops_lib = custom_ops_lib
        self._unused_groups_arg = int(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        if self.training or (not x.is_cuda):
            x = F.gelu(x)  # exact
            x = self.group_norm(x)
            return x

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        dev = x.device
        gn_w = self.group_norm.weight
        gn_b = self.group_norm.bias
        if gn_w.device != dev:
            gn_w = gn_w.to(dev)
            gn_b = gn_b.to(dev)

        fast_gelu = os.environ.get("CUSTOM_FUSED_FAST_GELU", "0").strip() not in ("0", "", "false", "False")

        return self.custom_ops_lib.gelu_groupnorm_forward_cuda(
            x,
            gn_w.contiguous(),
            gn_b.contiguous(),
            int(self.num_groups),
            float(self.gn_eps),
            bool(fast_gelu),
        )