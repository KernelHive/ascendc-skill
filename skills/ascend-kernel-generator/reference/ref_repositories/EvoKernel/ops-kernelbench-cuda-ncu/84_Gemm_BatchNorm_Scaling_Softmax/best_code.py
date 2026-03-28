import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FP32
#define CHECK_FP32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

// -------------------- reductions --------------------
static __forceinline__ __device__ float warp_sum(float v) {
    unsigned m = 0xffffffffu;
    v += __shfl_down_sync(m, v, 16);
    v += __shfl_down_sync(m, v, 8);
    v += __shfl_down_sync(m, v, 4);
    v += __shfl_down_sync(m, v, 2);
    v += __shfl_down_sync(m, v, 1);
    return v;
}

static __forceinline__ __device__ float warp_max(float v) {
    unsigned m = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(m, v, 16));
    v = fmaxf(v, __shfl_down_sync(m, v, 8));
    v = fmaxf(v, __shfl_down_sync(m, v, 4));
    v = fmaxf(v, __shfl_down_sync(m, v, 2));
    v = fmaxf(v, __shfl_down_sync(m, v, 1));
    return v;
}

template<int THREADS>
static __forceinline__ __device__ float block_sum_broadcast(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float smem[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        out = (tid < WARPS) ? smem[lane] : 0.0f;
        out = warp_sum(out);
    }
    if (tid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

template<int THREADS>
static __forceinline__ __device__ float block_max_broadcast(float v) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float smem[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    v = warp_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    float out = -INFINITY;
    if (wid == 0) {
        out = (tid < WARPS) ? smem[lane] : -INFINITY;
        out = warp_max(out);
    }
    if (tid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

// -------------------- loads + exp --------------------
static __forceinline__ __device__ float4 ldg4(const float* p) {
#if __CUDA_ARCH__ >= 350
    float4 o;
    o.x = __ldg(p + 0);
    o.y = __ldg(p + 1);
    o.z = __ldg(p + 2);
    o.w = __ldg(p + 3);
    return o;
#else
    return *reinterpret_cast<const float4*>(p);
#endif
}

static __forceinline__ __device__ float fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f);
}

// -------------------- precompute a,b (BN+scale affine) --------------------
// a[c] = scale*gamma[c]*rsqrt(var[c]+eps)
// b[c] = scale*(beta[c] - mean[c]*gamma[c]*rsqrt(var[c]+eps))
__global__ void precompute_ab_vec4(
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ a,
    float* __restrict__ b,
    int C,
    float eps,
    float scale_scalar
) {
    int idx4 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int C4 = C >> 2;
    if (idx4 >= C4) return;
    int base = idx4 << 2;

    float4 g  = ldg4(gamma + base);
    float4 bt = ldg4(beta + base);
    float4 m  = ldg4(mean + base);
    float4 v  = ldg4(var + base);

    float inv0 = rsqrtf(v.x + eps);
    float inv1 = rsqrtf(v.y + eps);
    float inv2 = rsqrtf(v.z + eps);
    float inv3 = rsqrtf(v.w + eps);

    float a0 = scale_scalar * (g.x * inv0);
    float a1 = scale_scalar * (g.y * inv1);
    float a2 = scale_scalar * (g.z * inv2);
    float a3 = scale_scalar * (g.w * inv3);

    float b0 = scale_scalar * (bt.x - m.x * g.x * inv0);
    float b1 = scale_scalar * (bt.y - m.y * g.y * inv1);
    float b2 = scale_scalar * (bt.z - m.z * g.z * inv2);
    float b3 = scale_scalar * (bt.w - m.w * g.w * inv3);

    *reinterpret_cast<float4*>(a + base) = make_float4(a0, a1, a2, a3);
    *reinterpret_cast<float4*>(b + base) = make_float4(b0, b1, b2, b3);
}

// -------------------- fused BN(scale)+softmax streaming (no logits staging) --------------------
template<int THREADS, int C_FIXED>  // C_FIXED either 8192 or -1
__global__ __launch_bounds__(THREADS, 4)
void bn_scale_softmax_streaming_vec4(
    const float* __restrict__ X,   // [B,C]
    const float* __restrict__ A,   // [C]
    const float* __restrict__ Bv,  // [C]
    float* __restrict__ Out,       // [B,C]
    int B, int C
) {
    int row = (int)blockIdx.x;
    if (row >= B) return;
    int tid = (int)threadIdx.x;

    const float* xrow = X + (size_t)row * (size_t)C;
    float* orow = Out + (size_t)row * (size_t)C;

    // Assumes C%4==0 and all pointers 16B aligned, checked on host.
    const int Ceff = (C_FIXED > 0) ? C_FIXED : C;
    int C4 = Ceff >> 2;

    // Pass 1: max
    float tmax = -INFINITY;

    if (C_FIXED == 8192) {
        // C4=2048; for THREADS=256 => 8 iters; THREADS=128 => 16 iters
        constexpr int C4_FIXED = 8192 / 4;
        constexpr int ITERS = (C4_FIXED + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int j4 = it * THREADS + tid;
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);
            float lm = fmaxf(fmaxf(y0, y1), fmaxf(y2, y3));
            tmax = fmaxf(tmax, lm);
        }
    } else {
        #pragma unroll 1
        for (int j4 = tid; j4 < C4; j4 += THREADS) {
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);
            float lm = fmaxf(fmaxf(y0, y1), fmaxf(y2, y3));
            tmax = fmaxf(tmax, lm);
        }
    }

    float rmax = block_max_broadcast<THREADS>(tmax);

    // Pass 2: sum exp
    float tsum = 0.0f;
    if (C_FIXED == 8192) {
        constexpr int C4_FIXED = 8192 / 4;
        constexpr int ITERS = (C4_FIXED + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int j4 = it * THREADS + tid;
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);
            tsum += fast_exp(y0 - rmax) + fast_exp(y1 - rmax) + fast_exp(y2 - rmax) + fast_exp(y3 - rmax);
        }
    } else {
        #pragma unroll 1
        for (int j4 = tid; j4 < C4; j4 += THREADS) {
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);
            tsum += fast_exp(y0 - rmax) + fast_exp(y1 - rmax) + fast_exp(y2 - rmax) + fast_exp(y3 - rmax);
        }
    }
    float rsum = block_sum_broadcast<THREADS>(tsum);
    float inv = 1.0f / fmaxf(rsum, 1e-20f);

    // Pass 3: write
    if (C_FIXED == 8192) {
        constexpr int C4_FIXED = 8192 / 4;
        constexpr int ITERS = (C4_FIXED + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int j4 = it * THREADS + tid;
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);

            float4 o;
            o.x = fast_exp(y0 - rmax) * inv;
            o.y = fast_exp(y1 - rmax) * inv;
            o.z = fast_exp(y2 - rmax) * inv;
            o.w = fast_exp(y3 - rmax) * inv;
            *reinterpret_cast<float4*>(orow + base) = o;
        }
    } else {
        #pragma unroll 1
        for (int j4 = tid; j4 < C4; j4 += THREADS) {
            int base = j4 << 2;
            float4 xv = *reinterpret_cast<const float4*>(xrow + base);
            float4 av = ldg4(A + base);
            float4 bv = ldg4(Bv + base);
            float y0 = fmaf(av.x, xv.x, bv.x);
            float y1 = fmaf(av.y, xv.y, bv.y);
            float y2 = fmaf(av.z, xv.z, bv.z);
            float y3 = fmaf(av.w, xv.w, bv.w);

            float4 o;
            o.x = fast_exp(y0 - rmax) * inv;
            o.y = fast_exp(y1 - rmax) * inv;
            o.z = fast_exp(y2 - rmax) * inv;
            o.w = fast_exp(y3 - rmax) * inv;
            *reinterpret_cast<float4*>(orow + base) = o;
        }
    }
}

// Host entrypoint: expects precomputed A/Bv passed in.
torch::Tensor bn_scale_softmax_from_ab_cuda(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor Bv,
    bool inplace_out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(A);
    CHECK_CUDA(Bv);
    CHECK_FP32(x);
    CHECK_FP32(A);
    CHECK_FP32(Bv);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(Bv);

    TORCH_CHECK(x.dim() == 2, "x must be 2D (B,C)");
    TORCH_CHECK(A.dim() == 1 && Bv.dim() == 1, "A/Bv must be 1D (C)");
    TORCH_CHECK(x.size(1) == A.size(0) && A.size(0) == Bv.size(0), "C mismatch");

    int B = (int)x.size(0);
    int C = (int)x.size(1);

    TORCH_CHECK((C & 3) == 0, "C must be divisible by 4");
    TORCH_CHECK((((uintptr_t)x.data_ptr<float>()) & 0xF) == 0, "x must be 16B aligned");
    TORCH_CHECK((((uintptr_t)A.data_ptr<float>()) & 0xF) == 0, "A must be 16B aligned");
    TORCH_CHECK((((uintptr_t)Bv.data_ptr<float>()) & 0xF) == 0, "Bv must be 16B aligned");

    torch::Tensor out;
    if (inplace_out) {
        out = x; // overwrite GEMM output
    } else {
        out = torch::empty({B, C}, x.options());
        TORCH_CHECK((((uintptr_t)out.data_ptr<float>()) & 0xF) == 0, "out must be 16B aligned");
    }

    // Tune threads: 256 tends to help latency hiding; 128 may reduce regs on some GPUs.
    // We'll decide in C++ based on SM count and B, but keep it simple: try 256 for large C==8192.
    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    // 1 CTA per row; cap grid to B
    dim3 grid((unsigned)B);

    if (C == 8192) {
        // heuristic: use 256 threads unless B is tiny; our B is typically 1024.
        if (B >= sm * 2) {
            constexpr int THREADS = 256;
            bn_scale_softmax_streaming_vec4<THREADS, 8192><<<grid, THREADS>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)A.data_ptr<float>(),
                (const float*)Bv.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, C
            );
        } else {
            constexpr int THREADS = 128;
            bn_scale_softmax_streaming_vec4<THREADS, 8192><<<grid, THREADS>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)A.data_ptr<float>(),
                (const float*)Bv.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, C
            );
        }
    } else {
        // generic vec4 path
        if (B >= sm * 2) {
            constexpr int THREADS = 256;
            bn_scale_softmax_streaming_vec4<THREADS, -1><<<grid, THREADS>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)A.data_ptr<float>(),
                (const float*)Bv.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, C
            );
        } else {
            constexpr int THREADS = 128;
            bn_scale_softmax_streaming_vec4<THREADS, -1><<<grid, THREADS>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)A.data_ptr<float>(),
                (const float*)Bv.data_ptr<float>(),
                (float*)out.data_ptr<float>(),
                B, C
            );
        }
    }

    return out;
}

torch::Tensor precompute_ab_cuda(
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    double eps
) {
    CHECK_CUDA(gamma);
    CHECK_CUDA(beta);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);
    CHECK_CUDA(scale);

    CHECK_FP32(gamma);
    CHECK_FP32(beta);
    CHECK_FP32(running_mean);
    CHECK_FP32(running_var);
    CHECK_FP32(scale);

    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);
    CHECK_CONTIGUOUS(scale);

    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");
    TORCH_CHECK(running_mean.dim() == 1 && running_var.dim() == 1, "running stats must be 1D");
    TORCH_CHECK(gamma.size(0) == beta.size(0), "gamma/beta mismatch");
    TORCH_CHECK(gamma.size(0) == running_mean.size(0), "mean mismatch");
    TORCH_CHECK(gamma.size(0) == running_var.size(0), "var mismatch");
    TORCH_CHECK(scale.numel() == 1, "scale must be scalar");

    int C = (int)gamma.size(0);
    TORCH_CHECK((C & 3) == 0, "C must be divisible by 4");
    TORCH_CHECK((((uintptr_t)gamma.data_ptr<float>()) & 0xF) == 0, "gamma must be 16B aligned");
    TORCH_CHECK((((uintptr_t)beta.data_ptr<float>()) & 0xF) == 0, "beta must be 16B aligned");
    TORCH_CHECK((((uintptr_t)running_mean.data_ptr<float>()) & 0xF) == 0, "mean must be 16B aligned");
    TORCH_CHECK((((uintptr_t)running_var.data_ptr<float>()) & 0xF) == 0, "var must be 16B aligned");

    auto A = torch::empty({C}, gamma.options());
    auto Bv = torch::empty({C}, gamma.options());

    TORCH_CHECK((((uintptr_t)A.data_ptr<float>()) & 0xF) == 0, "A must be 16B aligned");
    TORCH_CHECK((((uintptr_t)Bv.data_ptr<float>()) & 0xF) == 0, "Bv must be 16B aligned");

    float scale_scalar = scale.item<float>();

    int C4 = C >> 2;
    int threads = 256;
    int blocks = (C4 + threads - 1) / threads;
    precompute_ab_vec4<<<blocks, threads>>>(
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (const float*)running_mean.data_ptr<float>(),
        (const float*)running_var.data_ptr<float>(),
        (float*)A.data_ptr<float>(),
        (float*)Bv.data_ptr<float>(),
        C, (float)eps, scale_scalar
    );
    return torch::stack({A, Bv}, 0); // shape (2,C)
}
"""

cpp_source = r"""
torch::Tensor precompute_ab_cuda(
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor scale,
    double eps
);

torch::Tensor bn_scale_softmax_from_ab_cuda(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor Bv,
    bool inplace_out
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_bn_scale_softmax_v6",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["precompute_ab_cuda", "bn_scale_softmax_from_ab_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    GEMM (Linear) stays in cuBLAS.
    In eval/inference on CUDA+fp32, fuse:
      BatchNorm1d (running stats) + scale(scalar) + Softmax(dim=1)
    using a cached BN+scale affine precompute (A,Bv) + streaming softmax CUDA kernel.

    Key improvements vs baseline:
      - no logits staging to global memory
      - cached A/Bv buffers (no per-forward allocation; recompute only on changes)
      - optional in-place overwrite of GEMM output when safe to remove output allocation
    """

    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        self.custom_ops_lib = custom_ops_lib
        self._bn_eps = float(bn_eps)

        # Cached affine params
        self.register_buffer("_ab_cache", torch.empty(0), persistent=False)  # shape (2,C)
        self._ab_meta = {
            "device": None,
            "dtype": None,
            "C": None,
            "eps": None,
            "scale": None,
            "gamma_ver": None,
            "beta_ver": None,
            "mean_ver": None,
            "var_ver": None,
        }

    def _maybe_refresh_ab(self, gamma, beta, mean, var, scale, eps: float):
        # Use PyTorch internal version counters when available; fall back to data_ptr identity.
        def _ver(t):
            try:
                return int(t._version)
            except Exception:
                return int(t.data_ptr())

        C = gamma.numel()
        dev = gamma.device
        dt = gamma.dtype
        sc = float(scale.item())

        meta = self._ab_meta
        need = (
            self._ab_cache.numel() == 0
            or meta["device"] != dev
            or meta["dtype"] != dt
            or meta["C"] != C
            or meta["eps"] != float(eps)
            or meta["scale"] != sc
            or meta["gamma_ver"] != _ver(gamma)
            or meta["beta_ver"] != _ver(beta)
            or meta["mean_ver"] != _ver(mean)
            or meta["var_ver"] != _ver(var)
        )
        if need:
            ab = self.custom_ops_lib.precompute_ab_cuda(gamma, beta, mean, var, scale, float(eps))
            self._ab_cache = ab
            meta["device"] = dev
            meta["dtype"] = dt
            meta["C"] = C
            meta["eps"] = float(eps)
            meta["scale"] = sc
            meta["gamma_ver"] = _ver(gamma)
            meta["beta_ver"] = _ver(beta)
            meta["mean_ver"] = _ver(mean)
            meta["var_ver"] = _ver(var)

        return self._ab_cache[0], self._ab_cache[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gemm(x)

        if (not y.is_cuda) or self.training or (y.dtype != torch.float32):
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        if not y.is_contiguous():
            y = y.contiguous()

        gamma = self.bn.weight
        beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        scale = self.scale

        # Ensure fp32 contiguous params
        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        if running_mean.dtype != torch.float32:
            running_mean = running_mean.float()
        if running_var.dtype != torch.float32:
            running_var = running_var.float()
        if scale.dtype != torch.float32:
            scale = scale.float()

        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()
        if not running_mean.is_contiguous():
            running_mean = running_mean.contiguous()
        if not running_var.is_contiguous():
            running_var = running_var.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()

        C = y.size(1)
        aligned = (
            (C % 4) == 0
            and (y.data_ptr() % 16) == 0
            and (gamma.data_ptr() % 16) == 0
            and (beta.data_ptr() % 16) == 0
            and (running_mean.data_ptr() % 16) == 0
            and (running_var.data_ptr() % 16) == 0
            and (scale.numel() == 1)
        )
        if not aligned:
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        A, Bv = self._maybe_refresh_ab(gamma, beta, running_mean, running_var, scale, self._bn_eps)

        # In-place overwrite can remove output allocation; only do it when tensor is uniquely owned.
        inplace_ok = bool(y.is_contiguous() and (y.data_ptr() % 16 == 0))
        try:
            # _is_view is not public; use storage refcount heuristic where available.
            if hasattr(y, "_base") and y._base is not None:
                inplace_ok = False
        except Exception:
            pass

        out = self.custom_ops_lib.bn_scale_softmax_from_ab_cuda(y, A, Bv, bool(inplace_ok))
        return out