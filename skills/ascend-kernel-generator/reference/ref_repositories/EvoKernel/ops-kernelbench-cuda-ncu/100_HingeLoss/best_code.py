import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

// NOTE: avoid "static inline __device__ __forceinline__" to dodge toolchain duplicate-specifier issues.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// block reduce using warp shuffles + shared for warp sums
template<int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    constexpr int WARP = 32;
    constexpr int WARPS = (BLOCK_THREADS + WARP - 1) / WARP;
    __shared__ float warp_sums[WARPS];

    int tid = threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp_id == 0) {
        out = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__global__ void write_nan_f32_kernel(float* __restrict__ out) {
    out[0] = NAN;
}

// Baseline fallback: one atomicAdd per block.
__global__ __launch_bounds__(256, 4) void hinge_loss_mean_atomic_block_f32_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ out,
    int64_t n,
    float inv_n
) {
    float thread_sum = 0.0f;

    const int64_t idx0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const uintptr_t p_addr = (uintptr_t)preds;
    const uintptr_t t_addr = (uintptr_t)targets;
    const bool aligned16 = ((p_addr | t_addr) & 0xF) == 0;

    if (aligned16) {
        const int64_t n4 = n >> 2;
        const float4* __restrict__ p4 = reinterpret_cast<const float4*>(preds);
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(targets);

        for (int64_t j4 = idx0; j4 < n4; j4 += stride) {
            float4 a = __ldg(&p4[j4]);
            float4 b = __ldg(&t4[j4]);

            float m0 = 1.0f - a.x * b.x; thread_sum += (m0 > 0.0f) ? m0 : 0.0f;
            float m1 = 1.0f - a.y * b.y; thread_sum += (m1 > 0.0f) ? m1 : 0.0f;
            float m2 = 1.0f - a.z * b.z; thread_sum += (m2 > 0.0f) ? m2 : 0.0f;
            float m3 = 1.0f - a.w * b.w; thread_sum += (m3 > 0.0f) ? m3 : 0.0f;
        }

        const int64_t tail_start = (n4 << 2);
        for (int64_t k = tail_start + idx0; k < n; k += stride) {
            float a = __ldg(&preds[k]);
            float b = __ldg(&targets[k]);
            float m = 1.0f - a * b;
            thread_sum += (m > 0.0f) ? m : 0.0f;
        }
    } else {
        for (int64_t i = idx0; i < n; i += stride * 4) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int64_t j = i + (int64_t)u * stride;
                if (j < n) {
                    float a = __ldg(&preds[j]);
                    float b = __ldg(&targets[j]);
                    float m = 1.0f - a * b;
                    thread_sum += (m > 0.0f) ? m : 0.0f;
                }
            }
        }
    }

    thread_sum *= inv_n;

    float bsum = block_reduce_sum<256>(thread_sum);
    if (threadIdx.x == 0) atomicAdd(out, bsum);
}

// Cooperative-groups single-kernel reduction: blocks write partials, grid sync, block0 reduces partials.
__global__ __launch_bounds__(256, 2) void hinge_loss_mean_cg_f32_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ out,
    float* __restrict__ partials, // length >= gridDim.x
    int64_t n,
    float inv_n
) {
    cg::grid_group grid = cg::this_grid();

    float thread_sum = 0.0f;

    const int64_t idx0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const uintptr_t p_addr = (uintptr_t)preds;
    const uintptr_t t_addr = (uintptr_t)targets;
    const bool aligned16 = ((p_addr | t_addr) & 0xF) == 0;

    if (aligned16) {
        const int64_t n4 = n >> 2;
        const float4* __restrict__ p4 = reinterpret_cast<const float4*>(preds);
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(targets);

        for (int64_t j4 = idx0; j4 < n4; j4 += stride) {
            float4 a = __ldg(&p4[j4]);
            float4 b = __ldg(&t4[j4]);
            float m0 = 1.0f - a.x * b.x; thread_sum += (m0 > 0.0f) ? m0 : 0.0f;
            float m1 = 1.0f - a.y * b.y; thread_sum += (m1 > 0.0f) ? m1 : 0.0f;
            float m2 = 1.0f - a.z * b.z; thread_sum += (m2 > 0.0f) ? m2 : 0.0f;
            float m3 = 1.0f - a.w * b.w; thread_sum += (m3 > 0.0f) ? m3 : 0.0f;
        }

        const int64_t tail_start = (n4 << 2);
        for (int64_t k = tail_start + idx0; k < n; k += stride) {
            float a = __ldg(&preds[k]);
            float b = __ldg(&targets[k]);
            float m = 1.0f - a * b;
            thread_sum += (m > 0.0f) ? m : 0.0f;
        }
    } else {
        for (int64_t i = idx0; i < n; i += stride * 4) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int64_t j = i + (int64_t)u * stride;
                if (j < n) {
                    float a = __ldg(&preds[j]);
                    float b = __ldg(&targets[j]);
                    float m = 1.0f - a * b;
                    thread_sum += (m > 0.0f) ? m : 0.0f;
                }
            }
        }
    }

    thread_sum *= inv_n;
    float bsum = block_reduce_sum<256>(thread_sum);

    if (threadIdx.x == 0) partials[blockIdx.x] = bsum;

    grid.sync();

    if (blockIdx.x == 0) {
        float v = 0.0f;
        // reduce partials using block0; partials length is gridDim.x (<= SM count in our launch)
        for (int64_t i = (int64_t)threadIdx.x; i < (int64_t)gridDim.x; i += (int64_t)blockDim.x) {
            v += __ldg(&partials[i]);
        }
        float total = block_reduce_sum<256>(v);
        if (threadIdx.x == 0) out[0] = total;
    }
}

static inline int get_sm_count() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.multiProcessorCount;
}

static inline bool cooperative_supported() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int supported = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
    return supported != 0;
}

torch::Tensor hinge_loss_fwd_mean_f32_cuda(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kFloat, "targets must be float32");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have same numel");

    const int64_t n = predictions.numel();
    auto out = torch::zeros({}, predictions.options());

    c10::cuda::CUDAGuard device_guard(predictions.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    if (n == 0) {
        write_nan_f32_kernel<<<1, 1, 0, stream>>>((float*)out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    constexpr int threads = 256;
    float inv_n = 1.0f / (float)n;

    const int sms = get_sm_count();

    // Try cooperative path with blocks <= sms (to guarantee residency).
    if (cooperative_supported()) {
        int blocks = sms; // cooperative-resident
        // Don't oversubscribe tiny problems with too many CTAs.
        int blocks_by_n = (int)((n + threads - 1) / threads);
        if (blocks > blocks_by_n) blocks = blocks_by_n;
        if (blocks < 1) blocks = 1;

        // partials buffer (small): allocate per-call to keep python wrapper simple.
        // Size <= sms, overhead is small; avoids adding extra API parameters.
        auto partials = torch::empty({blocks}, predictions.options());

        void* args[] = {
            (void*)predictions.data_ptr<float>(),
            (void*)targets.data_ptr<float>(),
            (void*)out.data_ptr<float>(),
            (void*)partials.data_ptr<float>(),
            (void*)&n,
            (void*)&inv_n
        };

        cudaError_t st = cudaLaunchCooperativeKernel(
            (void*)hinge_loss_mean_cg_f32_kernel,
            dim3(blocks), dim3(threads),
            args, 0, stream
        );

        if (st == cudaSuccess) {
            return out;
        }
        // If cooperative launch fails for any reason, fall back.
        C10_CUDA_CHECK(cudaGetLastError());
    }

    // Fallback: atomic per block, slightly higher blocks for more MLP.
    int blocks_by_sms = sms * 6;
    int64_t blocks_by_n64 = (n + threads - 1) / threads;
    int blocks = (int)blocks_by_n64;
    if (blocks < blocks_by_sms) blocks = blocks_by_sms;
    if (blocks > sms * 20) blocks = sms * 20;
    if (blocks < 1) blocks = 1;

    hinge_loss_mean_atomic_block_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)predictions.data_ptr<float>(),
        (const float*)targets.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        n,
        inv_n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor hinge_loss_fwd_mean_f32_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

_ext_name = "custom_ops_lib_hinge_loss_opt6_cg"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["hinge_loss_fwd_mean_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Hinge Loss model using an optimized custom CUDA implementation (forward only):
      - computes mean(clamp(1 - predictions * targets, min=0))
      - predictions: float32 CUDA contiguous
      - targets: float32 CUDA contiguous (same shape/numel), values typically in {-1, +1}
    Falls back to torch implementation otherwise.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if (
            predictions.is_cuda and targets.is_cuda
            and predictions.dtype == torch.float32
            and targets.dtype == torch.float32
            and predictions.is_contiguous()
            and targets.is_contiguous()
            and predictions.numel() == targets.numel()
        ):
            return self.custom_ops_lib.hinge_loss_fwd_mean_f32_cuda(predictions, targets)

        return torch.mean(torch.clamp(1 - predictions * targets, min=0))