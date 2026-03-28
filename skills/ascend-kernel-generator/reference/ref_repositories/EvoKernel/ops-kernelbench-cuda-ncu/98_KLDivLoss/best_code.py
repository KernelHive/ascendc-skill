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

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

static __forceinline__ __device__ float warp_reduce_sum_f32(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static __forceinline__ __device__ float block_reduce_sum_f32(float v) {
    __shared__ float shared[32]; // up to 1024 threads
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_sum_f32(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        out = (lane < nw) ? shared[lane] : 0.0f;
        out = warp_reduce_sum_f32(out);
    }
    return out;
}

// PyTorch-like xlogy(t, t): if t == 0 => 0 else t * log(t)
static __forceinline__ __device__ float xlogx_f32(float t) {
    return (t == 0.0f) ? 0.0f : t * logf(t);
}

// Correct KLDivLoss for input = log(pred):
// sum_i [ xlogx(t_i) - t_i * log(pred_i) ] , with t_i==0 => 0
__global__ void kldiv_partial_sums_f32_kernel(const float* __restrict__ p,
                                             const float* __restrict__ t,
                                             float* __restrict__ partial,
                                             int64_t N) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int64_t i = tid; i < N; i += stride) {
        float tv = __ldg(t + i);
        if (tv != 0.0f) {
            float pv = __ldg(p + i);
            // p are probabilities from softmax; pv>0 and finite.
            // Use logf(pv) directly (no clamp) to match semantics as closely as possible.
            acc += xlogx_f32(tv) - tv * logf(pv);
        }
    }

    float sum = block_reduce_sum_f32(acc);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

__global__ void kldiv_partial_sums_f32_vec4_kernel(const float4* __restrict__ p4,
                                                  const float4* __restrict__ t4,
                                                  float* __restrict__ partial,
                                                  int64_t N4) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int64_t i4 = tid; i4 < N4; i4 += stride) {
        float4 pv = __ldg(p4 + i4);
        float4 tv = __ldg(t4 + i4);

        if (tv.x != 0.0f) acc += xlogx_f32(tv.x) - tv.x * logf(pv.x);
        if (tv.y != 0.0f) acc += xlogx_f32(tv.y) - tv.y * logf(pv.y);
        if (tv.z != 0.0f) acc += xlogx_f32(tv.z) - tv.z * logf(pv.z);
        if (tv.w != 0.0f) acc += xlogx_f32(tv.w) - tv.w * logf(pv.w);
    }

    float sum = block_reduce_sum_f32(acc);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

__global__ void reduce_partials_f32_kernel(const float* __restrict__ partial,
                                          float* __restrict__ out,
                                          int n) {
    // single-block reduction is enough: n is at most a few thousand
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += partial[i];
    float sum = block_reduce_sum_f32(acc);
    if (threadIdx.x == 0) out[0] = sum;
}

torch::Tensor kldiv_batchmean_prob_f32_cuda(torch::Tensor predictions,
                                           torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kFloat, "targets must be float32");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D [B, D]");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D [B, D]");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have same shape");

    const int64_t B = predictions.size(0);
    const int64_t D = predictions.size(1);
    const int64_t N = B * D;

    auto out = torch::zeros({}, torch::TensorOptions().dtype(at::kFloat).device(predictions.device()));

    c10::cuda::CUDAGuard device_guard(predictions.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    int device = predictions.get_device();
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int sm_count = prop.multiProcessorCount;

    // Enough CTAs to cover DRAM latency without creating atomic hotspots (we don't use atomics here).
    const int threads = 256;
    int blocks = sm_count * 8;
    if (blocks < 1) blocks = 1;
    if (blocks > 8192) blocks = 8192;

    auto partial = torch::empty({blocks}, torch::TensorOptions().dtype(at::kFloat).device(predictions.device()));

    uintptr_t p_addr = (uintptr_t)predictions.data_ptr<float>();
    uintptr_t t_addr = (uintptr_t)targets.data_ptr<float>();
    bool aligned16 = ((p_addr & 15) == 0) && ((t_addr & 15) == 0);
    bool n_div4 = ((N & 3) == 0);

    if (aligned16 && n_div4) {
        int64_t N4 = N >> 2;
        kldiv_partial_sums_f32_vec4_kernel<<<blocks, threads, 0, stream>>>(
            (const float4*)predictions.data_ptr<float>(),
            (const float4*)targets.data_ptr<float>(),
            (float*)partial.data_ptr<float>(),
            N4
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        kldiv_partial_sums_f32_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)predictions.data_ptr<float>(),
            (const float*)targets.data_ptr<float>(),
            (float*)partial.data_ptr<float>(),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Final reduction (single block)
    reduce_partials_f32_kernel<<<1, 256, 0, stream>>>(
        (const float*)partial.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // batchmean: divide by B
    out = out / (float)B;
    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor kldiv_batchmean_prob_f32_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

_ext_name = "custom_ops_lib_kldiv_opt_streaming"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["kldiv_batchmean_prob_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Model using the custom op
# ----------------------------

class ModelNew(nn.Module):
    """
    KLDiv model using a custom CUDA kernel computing:
      F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    Fast path:
      - CUDA, float32, 2D contiguous, same shapes.
    Otherwise falls back to PyTorch for correctness.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if (
            predictions.is_cuda and targets.is_cuda and
            predictions.dtype == torch.float32 and targets.dtype == torch.float32 and
            predictions.dim() == 2 and targets.dim() == 2 and
            predictions.is_contiguous() and targets.is_contiguous() and
            predictions.shape == targets.shape
        ):
            return self.custom_ops_lib.kldiv_batchmean_prob_f32_cuda(predictions, targets)

        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")