import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------
# Custom CUDA op: HardTanh (clamp) optimized
# y = clamp(x, lo, hi) for float32 CUDA tensors
# ---------------------------------------------------------

hardtanh_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

static __forceinline__ __device__ float clamp_branchless(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

static __forceinline__ __device__ float ld_ro(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__global__ __launch_bounds__(256, 2)
void hardtanh_scalar_ilp_gs(const float* __restrict__ x,
                           float* __restrict__ out,
                           int64_t n,
                           float lo,
                           float hi) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // ILP: process 4 scalars per iteration when possible (contiguous per-thread)
    for (int64_t base = tid; base < n; base += stride * 4) {
        int64_t i0 = base;
        int64_t i1 = base + stride;
        int64_t i2 = base + 2 * stride;
        int64_t i3 = base + 3 * stride;

        if (i0 < n) { float v = ld_ro(x + i0); out[i0] = clamp_branchless(v, lo, hi); }
        if (i1 < n) { float v = ld_ro(x + i1); out[i1] = clamp_branchless(v, lo, hi); }
        if (i2 < n) { float v = ld_ro(x + i2); out[i2] = clamp_branchless(v, lo, hi); }
        if (i3 < n) { float v = ld_ro(x + i3); out[i3] = clamp_branchless(v, lo, hi); }
    }
}

__global__ __launch_bounds__(256, 2)
void hardtanh_float2_ilp_gs(const float2* __restrict__ x2,
                           float2* __restrict__ out2,
                           int64_t n2,
                           float lo,
                           float hi) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // ILP: 2x float2 per loop
    for (int64_t i = tid; i < n2; i += stride * 2) {
        int64_t i0 = i;
        int64_t i1 = i + stride;

        if (i0 < n2) {
            float2 v = x2[i0];
            float2 o;
            o.x = clamp_branchless(v.x, lo, hi);
            o.y = clamp_branchless(v.y, lo, hi);
            out2[i0] = o;
        }
        if (i1 < n2) {
            float2 v = x2[i1];
            float2 o;
            o.x = clamp_branchless(v.x, lo, hi);
            o.y = clamp_branchless(v.y, lo, hi);
            out2[i1] = o;
        }
    }
}

__global__ __launch_bounds__(256, 2)
void hardtanh_float4_ilp_gs(const float4* __restrict__ x4,
                           float4* __restrict__ out4,
                           int64_t n4,
                           float lo,
                           float hi) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    // ILP: 2x float4 per loop
    for (int64_t i = tid; i < n4; i += stride * 2) {
        int64_t i0 = i;
        int64_t i1 = i + stride;

        if (i0 < n4) {
            float4 v = x4[i0];
            float4 o;
            o.x = clamp_branchless(v.x, lo, hi);
            o.y = clamp_branchless(v.y, lo, hi);
            o.z = clamp_branchless(v.z, lo, hi);
            o.w = clamp_branchless(v.w, lo, hi);
            out4[i0] = o;
        }
        if (i1 < n4) {
            float4 v = x4[i1];
            float4 o;
            o.x = clamp_branchless(v.x, lo, hi);
            o.y = clamp_branchless(v.y, lo, hi);
            o.z = clamp_branchless(v.z, lo, hi);
            o.w = clamp_branchless(v.w, lo, hi);
            out4[i1] = o;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "hardtanh_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "hardtanh_cuda: only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "hardtanh_cuda: input must be contiguous");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;

    const float lo = (float)min_val;
    const float hi = (float)max_val;

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

    const int threads = 256;
    int64_t blocks64 = (n + threads - 1) / threads;
    // Reasonable grid cap: enough to fill GPU and provide MLP without huge launch
    int blocks = (int)((blocks64 > 65535) ? 65535 : blocks64);
    if (blocks < 1) blocks = 1;

    const uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    const uintptr_t yp = (uintptr_t)out.data_ptr<float>();
    const uintptr_t ap = (xp | yp);

    // Prefer float4 when possible, else float2, else scalar.
    if ((ap & 0xF) == 0 && (n % 4 == 0)) {
        int64_t n4 = n / 4;
        hardtanh_float4_ilp_gs<<<blocks, threads, 0, stream>>>(
            (const float4*)x.data_ptr<float>(),
            (float4*)out.data_ptr<float>(),
            n4, lo, hi
        );
    } else if ((ap & 0x7) == 0 && (n % 2 == 0)) {
        int64_t n2 = n / 2;
        hardtanh_float2_ilp_gs<<<blocks, threads, 0, stream>>>(
            (const float2*)x.data_ptr<float>(),
            (float2*)out.data_ptr<float>(),
            n2, lo, hi
        );
    } else {
        hardtanh_scalar_ilp_gs<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            n, lo, hi
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

hardtanh_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor hardtanh_cuda(torch::Tensor x, double min_val, double max_val);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_hardtanh_opt2",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_cuda_source,
    functions=["hardtanh_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
)


class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA kernel for HardTanh.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return F.hardtanh(x, min_val=-1.0, max_val=1.0)
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.hardtanh_cuda(x, -1.0, 1.0)