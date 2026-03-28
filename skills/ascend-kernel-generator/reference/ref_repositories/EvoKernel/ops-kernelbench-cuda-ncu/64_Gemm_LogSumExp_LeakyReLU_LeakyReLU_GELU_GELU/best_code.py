import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: fused logsumexp(dim=1, keepdim=True) + leaky_relu + leaky_relu + gelu + gelu ----

gemm_lse_act_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float leaky_relu_f(float x, float negative_slope) {
    return (x >= 0.0f) ? x : x * negative_slope;
}

// Tanh-based GELU approximation
__device__ __forceinline__ float gelu_f(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    float t = kAlpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template<int THREADS>
__device__ __forceinline__ float block_reduce_max(float v) {
    static_assert(THREADS % 32 == 0, "THREADS must be multiple of warp");
    __shared__ float smem[32]; // max 32 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int WARPS = THREADS / 32;

    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = -INFINITY;
    if (warp == 0) {
        float x = (lane < WARPS) ? smem[lane] : -INFINITY;
        out = warp_reduce_max(x);
    }
    return out;
}

template<int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    static_assert(THREADS % 32 == 0, "THREADS must be multiple of warp");
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    constexpr int WARPS = THREADS / 32;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        float x = (lane < WARPS) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(x);
    }
    return out;
}

__device__ __forceinline__ float ro_load_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

template<int THREADS, int UNROLL4>
__global__ __launch_bounds__(THREADS, 4)
void logsumexp_act_persistent_kernel(
    const float* __restrict__ x,   // [B, N]
    float* __restrict__ out,       // [B]
    int B,
    int N,
    float negative_slope
) {
    __shared__ float s_scalar; // reuse for max then sum
    const int tid = threadIdx.x;
    const int row_stride = (int)gridDim.x;

    for (int row = (int)blockIdx.x; row < B; row += row_stride) {
        const float* row_ptr = x + (int64_t)row * (int64_t)N;

        // -------- Pass 1: max --------
        float local_max = -INFINITY;

        // Vectorized path if 16-byte aligned and N multiple-of-4 friendly.
        uintptr_t addr = (uintptr_t)row_ptr;
        bool aligned16 = ((addr & 0xF) == 0);

        if (aligned16) {
            const float4* p4 = reinterpret_cast<const float4*>(row_ptr);
            int N4 = N >> 2;
            int idx4 = tid;

            if (UNROLL4) {
                int step = THREADS;
                for (; idx4 + 3 * step < N4; idx4 += 4 * step) {
                    float4 v0 = p4[idx4];
                    float4 v1 = p4[idx4 + step];
                    float4 v2 = p4[idx4 + 2 * step];
                    float4 v3 = p4[idx4 + 3 * step];
                    local_max = fmaxf(local_max, v0.x); local_max = fmaxf(local_max, v0.y);
                    local_max = fmaxf(local_max, v0.z); local_max = fmaxf(local_max, v0.w);
                    local_max = fmaxf(local_max, v1.x); local_max = fmaxf(local_max, v1.y);
                    local_max = fmaxf(local_max, v1.z); local_max = fmaxf(local_max, v1.w);
                    local_max = fmaxf(local_max, v2.x); local_max = fmaxf(local_max, v2.y);
                    local_max = fmaxf(local_max, v2.z); local_max = fmaxf(local_max, v2.w);
                    local_max = fmaxf(local_max, v3.x); local_max = fmaxf(local_max, v3.y);
                    local_max = fmaxf(local_max, v3.z); local_max = fmaxf(local_max, v3.w);
                }
            }
            for (; idx4 < N4; idx4 += THREADS) {
                float4 v = p4[idx4];
                local_max = fmaxf(local_max, v.x);
                local_max = fmaxf(local_max, v.y);
                local_max = fmaxf(local_max, v.z);
                local_max = fmaxf(local_max, v.w);
            }

            // tail (N % 4)
            int tail = N & 3;
            if (tail) {
                int base = (N4 << 2);
                for (int j = base + tid; j < N; j += THREADS) {
                    local_max = fmaxf(local_max, ro_load_f32(row_ptr + j));
                }
            }
        } else {
            int j = tid;
            int step = THREADS;

            if (UNROLL4) {
                for (; j + 3 * step < N; j += 4 * step) {
                    float v0 = ro_load_f32(row_ptr + j);
                    float v1 = ro_load_f32(row_ptr + j + step);
                    float v2 = ro_load_f32(row_ptr + j + 2 * step);
                    float v3 = ro_load_f32(row_ptr + j + 3 * step);
                    local_max = fmaxf(local_max, v0);
                    local_max = fmaxf(local_max, v1);
                    local_max = fmaxf(local_max, v2);
                    local_max = fmaxf(local_max, v3);
                }
            }
            for (; j < N; j += step) {
                local_max = fmaxf(local_max, ro_load_f32(row_ptr + j));
            }
        }

        float max_val = block_reduce_max<THREADS>(local_max);
        if (tid == 0) s_scalar = max_val;
        __syncthreads();
        max_val = s_scalar;

        // -------- Pass 2: sum exp(x - max) --------
        float local_sum = 0.0f;

        if (aligned16) {
            const float4* p4 = reinterpret_cast<const float4*>(row_ptr);
            int N4 = N >> 2;
            int idx4 = tid;
            int step = THREADS;

            if (UNROLL4) {
                for (; idx4 + 3 * step < N4; idx4 += 4 * step) {
                    float4 v0 = p4[idx4];
                    float4 v1 = p4[idx4 + step];
                    float4 v2 = p4[idx4 + 2 * step];
                    float4 v3 = p4[idx4 + 3 * step];
                    local_sum += expf(v0.x - max_val); local_sum += expf(v0.y - max_val);
                    local_sum += expf(v0.z - max_val); local_sum += expf(v0.w - max_val);
                    local_sum += expf(v1.x - max_val); local_sum += expf(v1.y - max_val);
                    local_sum += expf(v1.z - max_val); local_sum += expf(v1.w - max_val);
                    local_sum += expf(v2.x - max_val); local_sum += expf(v2.y - max_val);
                    local_sum += expf(v2.z - max_val); local_sum += expf(v2.w - max_val);
                    local_sum += expf(v3.x - max_val); local_sum += expf(v3.y - max_val);
                    local_sum += expf(v3.z - max_val); local_sum += expf(v3.w - max_val);
                }
            }
            for (; idx4 < N4; idx4 += THREADS) {
                float4 v = p4[idx4];
                local_sum += expf(v.x - max_val);
                local_sum += expf(v.y - max_val);
                local_sum += expf(v.z - max_val);
                local_sum += expf(v.w - max_val);
            }

            int tail = N & 3;
            if (tail) {
                int base = (N4 << 2);
                for (int j = base + tid; j < N; j += THREADS) {
                    local_sum += expf(ro_load_f32(row_ptr + j) - max_val);
                }
            }
        } else {
            int j = tid;
            int step = THREADS;

            if (UNROLL4) {
                for (; j + 3 * step < N; j += 4 * step) {
                    float v0 = ro_load_f32(row_ptr + j) - max_val;
                    float v1 = ro_load_f32(row_ptr + j + step) - max_val;
                    float v2 = ro_load_f32(row_ptr + j + 2 * step) - max_val;
                    float v3 = ro_load_f32(row_ptr + j + 3 * step) - max_val;
                    local_sum += expf(v0);
                    local_sum += expf(v1);
                    local_sum += expf(v2);
                    local_sum += expf(v3);
                }
            }
            for (; j < N; j += step) {
                local_sum += expf(ro_load_f32(row_ptr + j) - max_val);
            }
        }

        float sum_val = block_reduce_sum<THREADS>(local_sum);
        if (tid == 0) s_scalar = sum_val;
        __syncthreads();
        sum_val = s_scalar;

        if (tid == 0) {
            float y = logf(sum_val) + max_val;       // logsumexp
            y = leaky_relu_f(y, negative_slope);
            y = leaky_relu_f(y, negative_slope);
            y = gelu_f(y);
            y = gelu_f(y);
            out[row] = y;
        }
        __syncthreads(); // ensure s_scalar not raced in next iteration
    }
}

torch::Tensor logsumexp_leakyrelu_leakyrelu_gelu_gelu_forward_cuda(
    torch::Tensor x,
    double negative_slope
) {
    TORCH_CHECK(x.is_cuda(), "logsumexp...forward_cuda: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "logsumexp...forward_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "logsumexp...forward_cuda: x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "logsumexp...forward_cuda: x must be 2D [B, N]");

    const int64_t B64 = x.size(0);
    const int64_t N64 = x.size(1);
    TORCH_CHECK(B64 > 0 && N64 > 0, "logsumexp...forward_cuda: invalid shape");
    TORCH_CHECK(B64 <= INT32_MAX && N64 <= INT32_MAX, "logsumexp...forward_cuda: shape too large");

    const int B = (int)B64;
    const int N = (int)N64;

    // Output as [B, 1] to match keepdim=True
    auto out = torch::empty({B, 1}, x.options());

    constexpr int THREADS = 128;

    const at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Persistent-ish launch: blocks = min(B, sms * k)
    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sms = prop.multiProcessorCount;
    int blocks = sms * 8;
    if (blocks > B) blocks = B;
    if (blocks < 1) blocks = 1;

    // Write to out[:,0]
    float* out_ptr = (float*)out.data_ptr<float>();

    logsumexp_act_persistent_kernel<THREADS, 1><<<blocks, THREADS, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        out_ptr,
        B, N,
        (float)negative_slope
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

gemm_lse_act_cpp_source = r"""
torch::Tensor logsumexp_leakyrelu_leakyrelu_gelu_gelu_forward_cuda(torch::Tensor x, double negative_slope);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_v7",
    cpp_sources=gemm_lse_act_cpp_source,
    cuda_sources=gemm_lse_act_cuda_source,
    functions=["logsumexp_leakyrelu_leakyrelu_gelu_gelu_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ---- Model wrapper using the custom op ----

class ModelNew(nn.Module):
    """
    GEMM (nn.Linear) followed by fused:
      logsumexp(dim=1, keepdim=True) -> leaky_relu -> leaky_relu -> gelu -> gelu
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.negative_slope = 0.01
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)  # cuBLAS GEMM
        if not x.is_contiguous():
            x = x.contiguous()
        return self.custom_ops_lib.logsumexp_leakyrelu_leakyrelu_gelu_gelu_forward_cuda(
            x, float(self.negative_slope)
        )

# Keep the same input helpers for compatibility with the original harness.
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]