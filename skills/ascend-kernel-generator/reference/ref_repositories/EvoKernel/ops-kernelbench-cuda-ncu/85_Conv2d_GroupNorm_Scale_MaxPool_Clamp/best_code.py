import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# CUDA/C++ extension
# Fused tail: GroupNorm stats (Welford) + (normalize -> affine(a/b)) + MaxPool2d + clamp
#
# Optimizations vs baseline:
#  - Specialized fast-path for k=s=4, p=0, ceil_mode=0 with fully unrolled pooling.
#  - Safe per-row vectorized loads (int4) only when that row pointer is 16B-aligned.
#  - Python-side caching of fused per-channel affine params a=gamma*scale, b=beta*scale
#    to reduce parameter traffic WITHOUT extra CUDA precompute kernel/launch.
#  - Add C10_CUDA_KERNEL_LAUNCH_CHECK to avoid async error masking.
#
# Constraints:
#  - CUDA only
#  - float32 only
#  - contiguous NCHW only
# ------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

// ------------------------
// Welford helpers
// ------------------------
static __device__ __forceinline__ void welford_combine(float& mean, float& m2, float& count,
                                                      float mean_b, float m2_b, float count_b) {
    if (count_b == 0.0f) return;
    if (count == 0.0f) {
        mean = mean_b;
        m2 = m2_b;
        count = count_b;
        return;
    }
    float delta = mean_b - mean;
    float new_count = count + count_b;
    mean = mean + delta * (count_b / new_count);
    m2 = m2 + m2_b + delta * delta * (count * count_b / new_count);
    count = new_count;
}

static __device__ __forceinline__ void welford_update(float x, float& mean, float& m2, float& count) {
    count += 1.0f;
    float delta = x - mean;
    mean += delta / count;
    float delta2 = x - mean;
    m2 += delta * delta2;
}

static __device__ __forceinline__ void warp_welford_reduce(float& mean, float& m2, float& count) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mean_b = __shfl_down_sync(mask, mean, offset);
        float m2_b   = __shfl_down_sync(mask, m2, offset);
        float cnt_b  = __shfl_down_sync(mask, count, offset);
        welford_combine(mean, m2, count, mean_b, m2_b, cnt_b);
    }
}

static __host__ __device__ __forceinline__ int out_dim_maxpool(int in, int k, int s, int p, int ceil_mode) {
    int out;
    if (ceil_mode) out = (in + 2*p - k + s - 1) / s + 1;
    else          out = (in + 2*p - k) / s + 1;
    if (out < 0) out = 0;
    if (out > 0) {
        if ((out - 1) * s >= in + p) out -= 1;
    }
    return out;
}

// ------------------------
// Kernel: compute GN stats per (n,g): mean/rstd over Cg*H*W
// ------------------------
__global__ void gn2d_stats_welford_kernel(
    const float* __restrict__ x, // [N,C,H,W]
    float* __restrict__ mean,    // [N*G]
    float* __restrict__ rstd,    // [N*G]
    int N, int C, int H, int W, int G, float eps)
{
    int idx = (int)blockIdx.x;
    int n = idx / G;
    int g = idx - n * G;
    int Cg = C / G;
    int HW = H * W;
    int D = Cg * HW;

    const float* x_ptr = x + ((n * C + g * Cg) * HW);

    float m = 0.0f, m2 = 0.0f, cnt = 0.0f;
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) {
        float v = x_ptr[i];
        welford_update(v, m, m2, cnt);
    }

    warp_welford_reduce(m, m2, cnt);

    __shared__ float sh_mean[8];
    __shared__ float sh_m2[8];
    __shared__ float sh_cnt[8];

    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;

    if (lane == 0) {
        sh_mean[warp] = m;
        sh_m2[warp] = m2;
        sh_cnt[warp] = cnt;
    }
    __syncthreads();

    if (warp == 0) {
        int nwarps = (int)(blockDim.x >> 5);
        float m0 = (lane < nwarps) ? sh_mean[lane] : 0.0f;
        float s0 = (lane < nwarps) ? sh_m2[lane]   : 0.0f;
        float c0 = (lane < nwarps) ? sh_cnt[lane]  : 0.0f;
        float mm = m0, ss = s0, cc = c0;
        warp_welford_reduce(mm, ss, cc);
        if (lane == 0) {
            float var = ss / (float)D;
            mean[idx] = mm;
            rstd[idx] = rsqrtf(var + eps);
        }
    }
}

// ------------------------
// Generic kernel: normalize + affine(a/b) + maxpool + clamp
// ------------------------
__global__ void gn_affineab_maxpool_clamp_generic_kernel(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ mean,   // [N*G]
    const float* __restrict__ rstd,   // [N*G]
    const float* __restrict__ a,      // [C]
    const float* __restrict__ b,      // [C]
    float* __restrict__ y,            // [N,C,Ho,Wo]
    int N, int C, int H, int W, int G,
    int Ho, int Wo,
    int kH, int kW, int sH, int sW, int pH, int pW,
    float clamp_min, float clamp_max)
{
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)Ho * (int64_t)Wo;
    if (idx >= total) return;

    int ow = (int)(idx % Wo);
    int oh = (int)((idx / Wo) % Ho);
    int c  = (int)((idx / (int64_t)(Wo * Ho)) % C);
    int n  = (int)(idx / (int64_t)(Wo * Ho * C));

    int Cg = C / G;
    int g = c / Cg;
    float m = mean[n * G + g];
    float inv = rstd[n * G + g];

    int ih0 = oh * sH - pH;
    int iw0 = ow * sW - pW;

    float maxv = -FLT_MAX;
    const float* x_ptr = x + ((n * C + c) * H * W);

    float aa = a[c];
    float bb = b[c];

    #pragma unroll 1
    for (int kh = 0; kh < kH; ++kh) {
        int ih = ih0 + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        int row = ih * W;
        #pragma unroll 1
        for (int kw = 0; kw < kW; ++kw) {
            int iw = iw0 + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            float v = x_ptr[row + iw];
            v = (v - m) * inv;
            v = v * aa + bb;
            maxv = fmaxf(maxv, v);
        }
    }

    float outv = fminf(fmaxf(maxv, clamp_min), clamp_max);
    y[idx] = outv;
}

// ------------------------
// Fast-path kernel: assumes kH=kW=sH=sW=4, pH=pW=0, ceil_mode=0
// Uses per-row alignment check to safely vectorize via int4 when possible.
// ------------------------
static __device__ __forceinline__ void load4_safe(const float* ptr, float& v0, float& v1, float& v2, float& v3) {
    uintptr_t addr = (uintptr_t)ptr;
    if ((addr & 0xF) == 0) {
        // safe 16B load
        int4 raw = *reinterpret_cast<const int4*>(ptr);
        v0 = __int_as_float(raw.x);
        v1 = __int_as_float(raw.y);
        v2 = __int_as_float(raw.z);
        v3 = __int_as_float(raw.w);
    } else {
        v0 = __ldg(ptr + 0);
        v1 = __ldg(ptr + 1);
        v2 = __ldg(ptr + 2);
        v3 = __ldg(ptr + 3);
    }
}

__global__ void gn_affineab_maxpool4_clamp_fast_kernel(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ mean,   // [N*G]
    const float* __restrict__ rstd,   // [N*G]
    const float* __restrict__ a,      // [C]
    const float* __restrict__ b,      // [C]
    float* __restrict__ y,            // [N,C,Ho,Wo]
    int N, int C, int H, int W, int G,
    int Ho, int Wo,
    float clamp_min, float clamp_max)
{
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)Ho * (int64_t)Wo;
    if (idx >= total) return;

    int ow = (int)(idx % Wo);
    int oh = (int)((idx / Wo) % Ho);
    int c  = (int)((idx / (int64_t)(Wo * Ho)) % C);
    int n  = (int)(idx / (int64_t)(Wo * Ho * C));

    int Cg = C / G;
    int g = c / Cg;
    float m = mean[n * G + g];
    float inv = rstd[n * G + g];

    // top-left corner of 4x4 window
    int ih0 = oh * 4;
    int iw0 = ow * 4;

    const float* base = x + ((n * C + c) * H * W) + ih0 * W + iw0;

    float aa = __ldg(a + c);
    float bb = __ldg(b + c);

    float maxv = -FLT_MAX;

    #pragma unroll
    for (int kh = 0; kh < 4; ++kh) {
        const float* row = base + kh * W;
        float r0, r1, r2, r3;
        load4_safe(row, r0, r1, r2, r3);

        float v0 = (r0 - m) * inv; v0 = v0 * aa + bb; maxv = fmaxf(maxv, v0);
        float v1 = (r1 - m) * inv; v1 = v1 * aa + bb; maxv = fmaxf(maxv, v1);
        float v2 = (r2 - m) * inv; v2 = v2 * aa + bb; maxv = fmaxf(maxv, v2);
        float v3 = (r3 - m) * inv; v3 = v3 * aa + bb; maxv = fmaxf(maxv, v3);
    }

    float outv = fminf(fmaxf(maxv, clamp_min), clamp_max);
    y[idx] = outv;
}

torch::Tensor gn2d_scale_maxpool_clamp_cuda_opt2(
    torch::Tensor x,
    torch::Tensor mean,   // optional prealloc? (ignored, kept simple) - not exposed
    torch::Tensor rstd,   // optional prealloc? (ignored, kept simple) - not exposed
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_groups,
    double eps,
    int64_t kH, int64_t kW,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    bool ceil_mode,
    double clamp_min,
    double clamp_max)
{
    (void)mean; (void)rstd; // not used; kept for signature stability if needed

    CHECK_CUDA(x); CHECK_CUDA(a); CHECK_CUDA(b);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    CHECK_FLOAT(x); CHECK_FLOAT(a); CHECK_FLOAT(b);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(a.dim() == 1 && b.dim() == 1, "a/b must be [C]");
    TORCH_CHECK(num_groups > 0, "num_groups must be > 0");
    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be > 0");
    TORCH_CHECK(sH > 0 && sW > 0, "stride must be > 0");
    TORCH_CHECK(pH >= 0 && pW >= 0, "padding must be >= 0");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int G = (int)num_groups;

    TORCH_CHECK((int)a.numel() == C && (int)b.numel() == C, "a/b must have C elements");
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    const int Ho = out_dim_maxpool(H, (int)kH, (int)sH, (int)pH, (int)ceil_mode);
    const int Wo = out_dim_maxpool(W, (int)kW, (int)sW, (int)pW, (int)ceil_mode);

    auto y = torch::empty({N, C, Ho, Wo}, x.options());

    // mean/rstd buffers
    auto mean_buf = torch::empty({N * G}, x.options());
    auto rstd_buf = torch::empty({N * G}, x.options());

    const int blocks_stats = N * G;
    const int threads_stats = 256;
    gn2d_stats_welford_kernel<<<blocks_stats, threads_stats, 0, at::cuda::getDefaultCUDAStream()>>>(
        (const float*)x.data_ptr<float>(),
        (float*)mean_buf.data_ptr<float>(),
        (float*)rstd_buf.data_ptr<float>(),
        N, C, H, W, G, (float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t total = (int64_t)N * (int64_t)C * (int64_t)Ho * (int64_t)Wo;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    const bool fast_path = (!ceil_mode) &&
                           (kH == 4 && kW == 4 && sH == 4 && sW == 4 && pH == 0 && pW == 0) &&
                           (Ho == H / 4) && (Wo == W / 4); // ensure exact tiling

    if (fast_path) {
        gn_affineab_maxpool4_clamp_fast_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)mean_buf.data_ptr<float>(),
            (const float*)rstd_buf.data_ptr<float>(),
            (const float*)a.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W, G, Ho, Wo,
            (float)clamp_min, (float)clamp_max
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        gn_affineab_maxpool_clamp_generic_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)mean_buf.data_ptr<float>(),
            (const float*)rstd_buf.data_ptr<float>(),
            (const float*)a.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, C, H, W, G, Ho, Wo,
            (int)kH, (int)kW, (int)sH, (int)sW, (int)pH, (int)pW,
            (float)clamp_min, (float)clamp_max
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor gn2d_scale_maxpool_clamp_cuda_opt2(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor rstd,
    torch::Tensor a,
    torch::Tensor b,
    int64_t num_groups,
    double eps,
    int64_t kH, int64_t kW,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    bool ceil_mode,
    double clamp_min,
    double clamp_max);
"""

custom_ops_lib = load_inline(
    name="custom_gn2d_scale_maxpool_clamp_ops_opt2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gn2d_scale_maxpool_clamp_cuda_opt2"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Conv2d (cuDNN) -> (custom) GroupNorm + scale + MaxPool2d + clamp

    Custom op constraints:
      - CUDA only
      - float32 only
      - contiguous NCHW only
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups,
                 scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels, eps=1e-5, affine=True)
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=torch.float32))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)  # stride defaults to kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        self.num_groups = int(num_groups)
        self.eps = float(self.group_norm.eps)

        k = self.maxpool.kernel_size
        s = self.maxpool.stride if self.maxpool.stride is not None else self.maxpool.kernel_size
        p = self.maxpool.padding
        self.kH, self.kW = (int(k), int(k)) if isinstance(k, int) else (int(k[0]), int(k[1]))
        self.sH, self.sW = (int(s), int(s)) if isinstance(s, int) else (int(s[0]), int(s[1]))
        self.pH, self.pW = (int(p), int(p)) if isinstance(p, int) else (int(p[0]), int(p[1]))
        self.ceil_mode = bool(self.maxpool.ceil_mode)

        self.custom_ops_lib = custom_ops_lib

        # cache for fused affine params
        self._cache_dev = None
        self._cache_C = None
        self._cache_gamma_ptr = None
        self._cache_beta_ptr = None
        self._cache_scale_ptr = None
        self._cache_a = None
        self._cache_b = None

    def _get_fused_affine_ab(self, x: torch.Tensor):
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        scale_c = self.scale.view(-1)

        # move to device (avoid repeated .to if already correct)
        if gamma.device != x.device:
            gamma = gamma.to(device=x.device)
            beta = beta.to(device=x.device)
            scale_c = scale_c.to(device=x.device)

        gamma = gamma.contiguous().to(dtype=torch.float32)
        beta = beta.contiguous().to(dtype=torch.float32)
        scale_c = scale_c.contiguous().to(dtype=torch.float32)

        C = x.size(1)
        if gamma.numel() != C or beta.numel() != C or scale_c.numel() != C:
            raise RuntimeError(
                f"Expected gamma/beta/scale to have C={C} elements; "
                f"got gamma={gamma.numel()}, beta={beta.numel()}, scale={scale_c.numel()}"
            )

        # cache keyed by (device, C, storage pointers)
        gptr = gamma.untyped_storage().data_ptr()
        bptr = beta.untyped_storage().data_ptr()
        sptr = scale_c.untyped_storage().data_ptr()

        if (self._cache_dev == x.device and self._cache_C == C and
            self._cache_gamma_ptr == gptr and self._cache_beta_ptr == bptr and self._cache_scale_ptr == sptr and
            self._cache_a is not None and self._cache_b is not None):
            return self._cache_a, self._cache_b

        # compute fused a,b (keep as tensors on GPU)
        a = (gamma * scale_c).contiguous()
        b = (beta * scale_c).contiguous()

        self._cache_dev = x.device
        self._cache_C = C
        self._cache_gamma_ptr = gptr
        self._cache_beta_ptr = bptr
        self._cache_scale_ptr = sptr
        self._cache_a = a
        self._cache_b = b
        return a, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if not x.is_cuda:
            raise RuntimeError("ModelNew only supports CUDA tensors")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        a, b = self._get_fused_affine_ab(x)

        # dummy mean/rstd args (not used by C++ wrapper; kept for ABI flexibility)
        dummy = torch.empty(0, device=x.device, dtype=torch.float32)

        y = self.custom_ops_lib.gn2d_scale_maxpool_clamp_cuda_opt2(
            x, dummy, dummy, a, b,
            int(self.num_groups),
            float(self.eps),
            int(self.kH), int(self.kW),
            int(self.sH), int(self.sW),
            int(self.pH), int(self.pW),
            bool(self.ceil_mode),
            float(self.clamp_min), float(self.clamp_max),
        )
        return y