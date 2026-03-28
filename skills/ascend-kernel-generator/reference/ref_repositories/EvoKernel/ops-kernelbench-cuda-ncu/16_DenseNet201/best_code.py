import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA extension:
# 1) bn_relu_inference_cuda: fused BN(inference using running stats) + ReLU (NCHW)
# 2) copy_base_and_append_nchw_cuda: fused "optional base copy + append new" into a preallocated buffer
# 3) append_into_nchw_cuda: fast append-only into a preallocated buffer
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_DTYPE
#define CHECK_DTYPE(x) TORCH_CHECK((x.scalar_type() == at::ScalarType::Float) || (x.scalar_type() == at::ScalarType::Half), #x " must be float16 or float32")
#endif

static inline __host__ __device__ bool is_aligned_16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xFULL) == 0ULL);
}
static inline __host__ __device__ bool is_aligned_8(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0x7ULL) == 0ULL);
}
static inline __host__ __device__ bool is_aligned_4(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0x3ULL) == 0ULL);
}

template <typename T>
__device__ __forceinline__ T relu_scalar(T v) { return v > (T)0 ? v : (T)0; }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
__device__ __forceinline__ half2 relu_half2(half2 v) {
    const half2 z = __float2half2_rn(0.f);
    return __hmax2(v, z);
}
#endif

// ------------------------------------
// Fused BN(inference) + ReLU, NCHW contiguous.
// Grid: (N, C). Each block streams HW.
// ------------------------------------
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void bn_relu_infer_f32_nchw(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int HW,
    float eps
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (n >= N || c >= C) return;

    float m = mean[c];
    float v = var[c];
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta  ? beta[c]  : 0.0f;
    float invstd = rsqrtf(v + eps);

    int64_t base = ((int64_t)n * C + c) * (int64_t)HW;
    const float* __restrict__ xp = x + base;
    float* __restrict__ yp = y + base;

    int tid = (int)threadIdx.x;

    if (((HW & 3) == 0) && is_aligned_16(xp) && is_aligned_16(yp)) {
        int HW4 = HW >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xp);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(yp);
        for (int i = tid; i < HW4; i += THREADS) {
            float4 xv = x4[i];
            xv.x = relu_scalar((xv.x - m) * invstd * g + b);
            xv.y = relu_scalar((xv.y - m) * invstd * g + b);
            xv.z = relu_scalar((xv.z - m) * invstd * g + b);
            xv.w = relu_scalar((xv.w - m) * invstd * g + b);
            y4[i] = xv;
        }
        return;
    }

    if (((HW & 1) == 0) && is_aligned_8(xp) && is_aligned_8(yp)) {
        int HW2 = HW >> 1;
        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(xp);
        float2* __restrict__ y2 = reinterpret_cast<float2*>(yp);
        for (int i = tid; i < HW2; i += THREADS) {
            float2 xv = x2[i];
            xv.x = relu_scalar((xv.x - m) * invstd * g + b);
            xv.y = relu_scalar((xv.y - m) * invstd * g + b);
            y2[i] = xv;
        }
        return;
    }

    for (int i = tid; i < HW; i += THREADS) {
        float xv = xp[i];
        yp[i] = relu_scalar((xv - m) * invstd * g + b);
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void bn_relu_infer_f16_nchw(
    const half* __restrict__ x,
    half* __restrict__ y,
    const half* __restrict__ mean,
    const half* __restrict__ var,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    int N, int C, int HW,
    float eps
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (n >= N || c >= C) return;

    float m = __half2float(mean[c]);
    float v = __half2float(var[c]);
    float g = gamma ? __half2float(gamma[c]) : 1.0f;
    float b = beta  ? __half2float(beta[c])  : 0.0f;
    float invstd = rsqrtf(v + eps);

    int64_t base = ((int64_t)n * C + c) * (int64_t)HW;
    const half* __restrict__ xp = x + base;
    half* __restrict__ yp = y + base;

    int tid = (int)threadIdx.x;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    if (((HW & 1) == 0) && is_aligned_4(xp) && is_aligned_4(yp)) {
        int HW2 = HW >> 1;
        const half2* __restrict__ x2 = reinterpret_cast<const half2*>(xp);
        half2* __restrict__ y2 = reinterpret_cast<half2*>(yp);

        half2 hm = __float2half2_rn(m);
        half2 hg = __float2half2_rn(g);
        half2 hb = __float2half2_rn(b);
        half2 hinv = __float2half2_rn(invstd);

        for (int i = tid; i < HW2; i += THREADS) {
            half2 xv = x2[i];
            half2 t = __hadd2(__hmul2(__hmul2(__hsub2(xv, hm), hinv), hg), hb);
            y2[i] = relu_half2(t);
        }
        return;
    }
#endif

    for (int i = tid; i < HW; i += THREADS) {
        float xv = __half2float(xp[i]);
        float tv = (xv - m) * invstd * g + b;
        tv = tv > 0.f ? tv : 0.f;
        yp[i] = __float2half(tv);
    }
}

torch::Tensor bn_relu_inference_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_DTYPE(x);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);
    CHECK_DTYPE(running_mean);
    CHECK_DTYPE(running_var);
    TORCH_CHECK(running_mean.numel() == x.size(1), "mean size mismatch");
    TORCH_CHECK(running_var.numel() == x.size(1), "var size mismatch");
    TORCH_CHECK(running_mean.scalar_type() == x.scalar_type(), "mean dtype mismatch");
    TORCH_CHECK(running_var.scalar_type() == x.scalar_type(), "var dtype mismatch");

    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_b = bias.defined() && bias.numel() > 0;

    if (has_w) {
        CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight); CHECK_DTYPE(weight);
        TORCH_CHECK(weight.numel() == x.size(1), "weight size mismatch");
        TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype mismatch");
    }
    if (has_b) {
        CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_DTYPE(bias);
        TORCH_CHECK(bias.numel() == x.size(1), "bias size mismatch");
        TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype mismatch");
    }

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;

    auto y = torch::empty_like(x);

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int threads = 128;
    dim3 block(threads);
    dim3 grid((unsigned)N, (unsigned)C, 1);
    float feps = (float)eps;

    if (x.scalar_type() == at::ScalarType::Float) {
        bn_relu_infer_f32_nchw<threads><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            has_w ? weight.data_ptr<float>() : nullptr,
            has_b ? bias.data_ptr<float>() : nullptr,
            N, C, HW, feps
        );
    } else {
        bn_relu_infer_f16_nchw<threads><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(running_mean.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(running_var.data_ptr<at::Half>()),
            has_w ? reinterpret_cast<const half*>(weight.data_ptr<at::Half>()) : nullptr,
            has_b ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
            N, C, HW, feps
        );
    }
    return y;
}

// ------------------------------------
// Append-only into preallocated dst buffer (NCHW).
// Grid: (N, Cnew). Each block copies one plane (HW).
// ------------------------------------
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void append_only_f32(
    const float* __restrict__ new_feat,
    float* __restrict__ dst,
    int N, int Cnew, int Cdst, int HW,
    int c_offset
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (n >= N || c >= Cnew) return;

    int64_t src_base = ((int64_t)n * Cnew + c) * (int64_t)HW;
    int64_t dst_base = ((int64_t)n * Cdst + (c + c_offset)) * (int64_t)HW;
    const float* sp = new_feat + src_base;
    float* dp = dst + dst_base;

    int tid = (int)threadIdx.x;

    if (((HW & 3) == 0) && is_aligned_16(sp) && is_aligned_16(dp)) {
        int HW4 = HW >> 2;
        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(sp);
        float4* __restrict__ d4 = reinterpret_cast<float4*>(dp);
        for (int i = tid; i < HW4; i += THREADS) d4[i] = s4[i];
        return;
    }
    if (((HW & 1) == 0) && is_aligned_8(sp) && is_aligned_8(dp)) {
        int HW2 = HW >> 1;
        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(sp);
        float2* __restrict__ d2 = reinterpret_cast<float2*>(dp);
        for (int i = tid; i < HW2; i += THREADS) d2[i] = s2[i];
        return;
    }
    for (int i = tid; i < HW; i += THREADS) dp[i] = sp[i];
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void append_only_f16(
    const half* __restrict__ new_feat,
    half* __restrict__ dst,
    int N, int Cnew, int Cdst, int HW,
    int c_offset
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (n >= N || c >= Cnew) return;

    int64_t src_base = ((int64_t)n * Cnew + c) * (int64_t)HW;
    int64_t dst_base = ((int64_t)n * Cdst + (c + c_offset)) * (int64_t)HW;
    const half* sp = new_feat + src_base;
    half* dp = dst + dst_base;

    int tid = (int)threadIdx.x;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    if (((HW & 1) == 0) && is_aligned_4(sp) && is_aligned_4(dp)) {
        int HW2 = HW >> 1;
        const half2* __restrict__ s2 = reinterpret_cast<const half2*>(sp);
        half2* __restrict__ d2 = reinterpret_cast<half2*>(dp);
        for (int i = tid; i < HW2; i += THREADS) d2[i] = s2[i];
        return;
    }
#endif
    for (int i = tid; i < HW; i += THREADS) dp[i] = sp[i];
}

void append_into_nchw_cuda(
    torch::Tensor new_feat,  // [N, Cnew, H, W]
    torch::Tensor dst,       // [N, Cdst, H, W]
    int64_t c_offset
) {
    CHECK_CUDA(new_feat); CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(new_feat); CHECK_CONTIGUOUS(dst);
    CHECK_DTYPE(new_feat); CHECK_DTYPE(dst);
    TORCH_CHECK(new_feat.scalar_type() == dst.scalar_type(), "dtype mismatch");
    TORCH_CHECK(new_feat.dim() == 4 && dst.dim() == 4, "new_feat/dst must be NCHW");
    TORCH_CHECK(new_feat.size(0) == dst.size(0), "N mismatch");
    TORCH_CHECK(new_feat.size(2) == dst.size(2) && new_feat.size(3) == dst.size(3), "H/W mismatch");

    int N = (int)new_feat.size(0);
    int Cnew = (int)new_feat.size(1);
    int Cdst = (int)dst.size(1);
    int HW = (int)(dst.size(2) * dst.size(3));
    int coff = (int)c_offset;

    TORCH_CHECK(coff >= 0 && coff + Cnew <= Cdst, "append range out of bounds");

    c10::cuda::CUDAGuard device_guard(dst.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((unsigned)N, (unsigned)Cnew, 1);

    if (dst.scalar_type() == at::ScalarType::Float) {
        append_only_f32<threads><<<grid, block, 0, stream>>>(
            new_feat.data_ptr<float>(),
            dst.data_ptr<float>(),
            N, Cnew, Cdst, HW, coff
        );
    } else {
        append_only_f16<threads><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(new_feat.data_ptr<at::Half>()),
            reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
            N, Cnew, Cdst, HW, coff
        );
    }
}

// ------------------------------------
// Fused: optional base copy (x -> dst[:, :Cin]) + append new_feat -> dst[:, coff:coff+Cnew].
// Implemented as a single elementwise kernel over dst range [0, Cin + Cnew) planes,
// selecting source by channel. This avoids launching two separate copy kernels on first layer.
// ------------------------------------
template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void copy_base_and_append_f32(
    const float* __restrict__ x,        // [N, Cin, HW]
    const float* __restrict__ new_feat, // [N, Cnew, HW]
    float* __restrict__ dst,            // [N, Cdst, HW]
    int N, int Cin, int Cnew, int Cdst, int HW,
    int coff,
    int do_copy_base
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y; // 0..(Cin+Cnew-1)
    if (n >= N) return;

    int Cwork = do_copy_base ? (Cin + Cnew) : Cnew;
    if (c >= Cwork) return;

    const float* sp;
    int dst_c;

    if (do_copy_base) {
        if (c < Cin) { sp = x + ((int64_t)n * Cin + c) * (int64_t)HW; dst_c = c; }
        else         { int cc = c - Cin; sp = new_feat + ((int64_t)n * Cnew + cc) * (int64_t)HW; dst_c = coff + cc; }
    } else {
        sp = new_feat + ((int64_t)n * Cnew + c) * (int64_t)HW;
        dst_c = coff + c;
    }

    float* dp = dst + ((int64_t)n * Cdst + dst_c) * (int64_t)HW;

    int tid = (int)threadIdx.x;

    if (((HW & 3) == 0) && is_aligned_16(sp) && is_aligned_16(dp)) {
        int HW4 = HW >> 2;
        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(sp);
        float4* __restrict__ d4 = reinterpret_cast<float4*>(dp);
        for (int i = tid; i < HW4; i += THREADS) d4[i] = s4[i];
        return;
    }
    if (((HW & 1) == 0) && is_aligned_8(sp) && is_aligned_8(dp)) {
        int HW2 = HW >> 1;
        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(sp);
        float2* __restrict__ d2 = reinterpret_cast<float2*>(dp);
        for (int i = tid; i < HW2; i += THREADS) d2[i] = s2[i];
        return;
    }
    for (int i = tid; i < HW; i += THREADS) dp[i] = sp[i];
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void copy_base_and_append_f16(
    const half* __restrict__ x,
    const half* __restrict__ new_feat,
    half* __restrict__ dst,
    int N, int Cin, int Cnew, int Cdst, int HW,
    int coff,
    int do_copy_base
) {
    int n = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (n >= N) return;

    int Cwork = do_copy_base ? (Cin + Cnew) : Cnew;
    if (c >= Cwork) return;

    const half* sp;
    int dst_c;

    if (do_copy_base) {
        if (c < Cin) { sp = x + ((int64_t)n * Cin + c) * (int64_t)HW; dst_c = c; }
        else         { int cc = c - Cin; sp = new_feat + ((int64_t)n * Cnew + cc) * (int64_t)HW; dst_c = coff + cc; }
    } else {
        sp = new_feat + ((int64_t)n * Cnew + c) * (int64_t)HW;
        dst_c = coff + c;
    }

    half* dp = dst + ((int64_t)n * Cdst + dst_c) * (int64_t)HW;

    int tid = (int)threadIdx.x;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    if (((HW & 1) == 0) && is_aligned_4(sp) && is_aligned_4(dp)) {
        int HW2 = HW >> 1;
        const half2* __restrict__ s2 = reinterpret_cast<const half2*>(sp);
        half2* __restrict__ d2 = reinterpret_cast<half2*>(dp);
        for (int i = tid; i < HW2; i += THREADS) d2[i] = s2[i];
        return;
    }
#endif
    for (int i = tid; i < HW; i += THREADS) dp[i] = sp[i];
}

void copy_base_and_append_nchw_cuda(
    torch::Tensor x,         // [N, Cin, H, W]
    torch::Tensor new_feat,  // [N, Cnew, H, W]
    torch::Tensor dst,       // [N, Cdst, H, W]
    int64_t c_offset,
    bool copy_base
) {
    CHECK_CUDA(x); CHECK_CUDA(new_feat); CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(new_feat); CHECK_CONTIGUOUS(dst);
    CHECK_DTYPE(x); CHECK_DTYPE(new_feat); CHECK_DTYPE(dst);
    TORCH_CHECK(x.scalar_type() == new_feat.scalar_type() && x.scalar_type() == dst.scalar_type(), "dtype mismatch");
    TORCH_CHECK(x.dim() == 4 && new_feat.dim() == 4 && dst.dim() == 4, "must be NCHW");
    TORCH_CHECK(x.size(0) == new_feat.size(0) && x.size(0) == dst.size(0), "N mismatch");
    TORCH_CHECK(x.size(2) == new_feat.size(2) && x.size(3) == new_feat.size(3), "H/W mismatch");
    TORCH_CHECK(dst.size(2) == x.size(2) && dst.size(3) == x.size(3), "dst H/W mismatch");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Cnew = (int)new_feat.size(1);
    int Cdst = (int)dst.size(1);
    int HW = (int)(x.size(2) * x.size(3));
    int coff = (int)c_offset;

    TORCH_CHECK(coff >= 0 && coff + Cnew <= Cdst, "append range out of bounds");
    if (copy_base) TORCH_CHECK(Cin <= Cdst, "Cin > Cdst");

    c10::cuda::CUDAGuard device_guard(dst.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int threads = 256;
    dim3 block(threads);
    int Cwork = copy_base ? (Cin + Cnew) : Cnew;
    dim3 grid((unsigned)N, (unsigned)Cwork, 1);

    if (dst.scalar_type() == at::ScalarType::Float) {
        copy_base_and_append_f32<threads><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            new_feat.data_ptr<float>(),
            dst.data_ptr<float>(),
            N, Cin, Cnew, Cdst, HW, coff, copy_base ? 1 : 0
        );
    } else {
        copy_base_and_append_f16<threads><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(new_feat.data_ptr<at::Half>()),
            reinterpret_cast<half*>(dst.data_ptr<at::Half>()),
            N, Cin, Cnew, Cdst, HW, coff, copy_base ? 1 : 0
        );
    }
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor bn_relu_inference_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);

void copy_base_and_append_nchw_cuda(
    torch::Tensor x,
    torch::Tensor new_feat,
    torch::Tensor dst,
    int64_t c_offset,
    bool copy_base
);

void append_into_nchw_cuda(
    torch::Tensor new_feat,
    torch::Tensor dst,
    int64_t c_offset
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dense_net201_blockbuf_v10",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "bn_relu_inference_cuda",
        "copy_base_and_append_nchw_cuda",
        "append_into_nchw_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)

# -----------------------------------------------------------------------------
# Model rewrite:
# - DenseBlock: single preallocated accumulation buffer per block; in-place append slices
# - BN+ReLU: fused custom kernel for inference; training uses PyTorch BN + inplace ReLU
# - Convs/pools/linear remain PyTorch (highly tuned)
# -----------------------------------------------------------------------------

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.growth_rate = int(growth_rate)
        self.num_input_features = int(num_input_features)

        layers = []
        for i in range(self.num_layers):
            in_features = self.num_input_features + i * self.growth_rate
            layers.append(
                nn.ModuleDict(
                    {
                        "bn": nn.BatchNorm2d(in_features),
                        "conv": nn.Conv2d(in_features, self.growth_rate, kernel_size=3, padding=1, bias=False),
                        "drop": nn.Dropout(0.0),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)

    def _bn_relu(self, bn: nn.BatchNorm2d, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and (not bn.training):
            x = x.contiguous()
            w = bn.weight if bn.affine else torch.Tensor()
            b = bn.bias if bn.affine else torch.Tensor()
            return custom_ops_lib.bn_relu_inference_cuda(
                x,
                bn.running_mean.contiguous(),
                bn.running_var.contiguous(),
                w.contiguous() if (bn.affine and w is not None) else torch.Tensor(),
                b.contiguous() if (bn.affine and b is not None) else torch.Tensor(),
                float(bn.eps),
            )
        y = F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            bn.training,
            bn.momentum,
            bn.eps,
        )
        return F.relu(y, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        N, C0, H, W = x.shape
        total_out = C0 + self.num_layers * self.growth_rate

        # Preallocate full output buffer once per block.
        buf = torch.empty((N, total_out, H, W), device=x.device, dtype=x.dtype)

        cur_channels = C0
        cur_view = x  # first conv reads from original input (no need to have it in buf yet)

        for i, layer in enumerate(self.layers):
            y = self._bn_relu(layer["bn"], cur_view)
            y = layer["conv"](y)
            y = layer["drop"](y)
            y = y.contiguous()

            if i == 0:
                # One fused kernel: copy base x into buf and append first new slice.
                custom_ops_lib.copy_base_and_append_nchw_cuda(
                    x, y, buf, int(cur_channels), True
                )
            else:
                # Append-only (buf already contains previous channels).
                custom_ops_lib.append_into_nchw_cuda(
                    y, buf, int(cur_channels)
                )

            cur_channels += self.growth_rate
            cur_view = buf[:, :cur_channels, :, :]

        return buf[:, :cur_channels, :, :]


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(int(num_input_features))
        self.conv = nn.Conv2d(int(num_input_features), int(num_output_features), kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def _bn_relu(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and (not self.bn.training):
            x = x.contiguous()
            return custom_ops_lib.bn_relu_inference_cuda(
                x,
                self.bn.running_mean.contiguous(),
                self.bn.running_var.contiguous(),
                self.bn.weight.contiguous() if self.bn.affine else torch.Tensor(),
                self.bn.bias.contiguous() if self.bn.affine else torch.Tensor(),
                float(self.bn.eps),
            )
        y = F.batch_norm(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.training,
            self.bn.momentum,
            self.bn.eps,
        )
        return F.relu(y, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        x = self._bn_relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()
        self.growth_rate = int(growth_rate)

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, nl in enumerate(block_layers):
            nl = int(nl)
            block = DenseBlockNew(num_layers=nl, num_input_features=int(num_features), growth_rate=self.growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + nl * self.growth_rate

            if i != len(block_layers) - 1:
                trans = TransitionLayerNew(num_input_features=int(num_features), num_output_features=int(num_features // 2))
                self.transition_layers.append(trans)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(int(num_features))
        self.classifier = nn.Linear(int(num_features), int(num_classes))

    def _bn_relu0(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and (not self.bn0.training):
            x = x.contiguous()
            return custom_ops_lib.bn_relu_inference_cuda(
                x,
                self.bn0.running_mean.contiguous(),
                self.bn0.running_var.contiguous(),
                self.bn0.weight.contiguous() if self.bn0.affine else torch.Tensor(),
                self.bn0.bias.contiguous() if self.bn0.affine else torch.Tensor(),
                float(self.bn0.eps),
            )
        y = F.batch_norm(
            x,
            self.bn0.running_mean,
            self.bn0.running_var,
            self.bn0.weight,
            self.bn0.bias,
            self.bn0.training,
            self.bn0.momentum,
            self.bn0.eps,
        )
        return F.relu(y, inplace=True)

    def _bn_relu_final(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and (not self.final_bn.training):
            x = x.contiguous()
            return custom_ops_lib.bn_relu_inference_cuda(
                x,
                self.final_bn.running_mean.contiguous(),
                self.final_bn.running_var.contiguous(),
                self.final_bn.weight.contiguous() if self.final_bn.affine else torch.Tensor(),
                self.final_bn.bias.contiguous() if self.final_bn.affine else torch.Tensor(),
                float(self.final_bn.eps),
            )
        y = F.batch_norm(
            x,
            self.final_bn.running_mean,
            self.final_bn.running_var,
            self.final_bn.weight,
            self.final_bn.bias,
            self.final_bn.training,
            self.final_bn.momentum,
            self.final_bn.eps,
        )
        return F.relu(y, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = x.contiguous()
        x = self._bn_relu0(x)
        x = self.pool0(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self._bn_relu_final(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x