import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float warp_inclusive_scan(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(mask, v, off);
        if ((int)(threadIdx.x & 31) >= off) v += n;
    }
    return v;
}
__device__ __forceinline__ float warp_broadcast(float v, int src_lane) {
    return __shfl_sync(0xffffffffu, v, src_lane);
}

static inline __host__ __device__ bool is_aligned_16_ptr(const void* p) {
    return (((uintptr_t)p) & 0xFULL) == 0;
}

__global__ __launch_bounds__(256, 2)
void cumsum_warp_rows_f32_scalar(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int B, int N) {
    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int warps_per_block = (int)blockDim.x >> 5;
    int row = (int)blockIdx.x * warps_per_block + warp_id;
    if (row >= B) return;

    int base = row * N;
    float carry = 0.0f;

    for (int chunk_start = 0; chunk_start < N; chunk_start += 32) {
        int col = chunk_start + lane;

        float v = 0.0f;
        if (col < N) {
#if __CUDA_ARCH__ >= 350
            v = __ldg(x + base + col);
#else
            v = x[base + col];
#endif
        }

        float ps = warp_inclusive_scan(v) + carry;

        if (col < N) out[base + col] = ps;

        int remaining = N - chunk_start;
        int last_lane = (remaining >= 32) ? 31 : (remaining - 1);
        carry = warp_broadcast(ps, last_lane);
    }
}

// Double-buffered/prefetch int4 path.
// Each lane loads/stores one int4 per iter (32 int4 = 128 floats).
__global__ __launch_bounds__(256, 2)
void cumsum_warp_rows_f32_i4_db(const int4* __restrict__ xI4,
                                int4* __restrict__ oI4,
                                int B, int N4) {
    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int warps_per_block = (int)blockDim.x >> 5;
    int row = (int)blockIdx.x * warps_per_block + warp_id;
    if (row >= B) return;

    const int base4 = row * N4;
    float carry = 0.0f;

    const int full_chunks = (N4 >> 5);  // /32
    const int rem = (N4 & 31);

    // Prefetch chunk 0
    int4 t_next;
    if (full_chunks > 0) {
        int idx4 = base4 + lane;
        t_next = xI4[idx4];
    }

    // Process full chunks with 1-iter lookahead
    #pragma unroll 1
    for (int ck = 0; ck < full_chunks; ++ck) {
        // current is prefetched
        int4 t_cur = t_next;

        // prefetch next chunk early (if any)
        if (ck + 1 < full_chunks) {
            int idx4n = base4 + ((ck + 1) << 5) + lane;
            t_next = xI4[idx4n];
        }

        float4 v4 = *reinterpret_cast<float4*>(&t_cur);

        float s0 = v4.x;
        float s1 = s0 + v4.y;
        float s2 = s1 + v4.z;
        float s3 = s2 + v4.w;

        float lane_total = s3;
        float lane_scan = warp_inclusive_scan(lane_total);
        float lane_excl = lane_scan - lane_total;
        float add = carry + lane_excl;

        float4 o4;
        o4.x = s0 + add;
        o4.y = s1 + add;
        o4.z = s2 + add;
        o4.w = s3 + add;

        int idx4 = base4 + (ck << 5) + lane;
        int4 ot = *reinterpret_cast<int4*>(&o4);
        oI4[idx4] = ot;

        carry = warp_broadcast(o4.w, 31);
    }

    // Tail (partial warp for remaining int4s)
    if (rem) {
        int idx4 = base4 + (full_chunks << 5) + lane;

        float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (lane < rem) {
            int4 t = xI4[idx4];
            v4 = *reinterpret_cast<float4*>(&t);
        }

        float s0 = v4.x;
        float s1 = s0 + v4.y;
        float s2 = s1 + v4.z;
        float s3 = s2 + v4.w;

        float lane_total = s3;
        float lane_scan = warp_inclusive_scan(lane_total);
        float lane_excl = lane_scan - lane_total;
        float add = carry + lane_excl;

        float4 o4;
        o4.x = s0 + add;
        o4.y = s1 + add;
        o4.z = s2 + add;
        o4.w = s3 + add;

        if (lane < rem) {
            int4 ot = *reinterpret_cast<int4*>(&o4);
            oI4[idx4] = ot;
        }
    }
}

static inline void set_l2_persisting_window(cudaStream_t stream,
                                           const void* base_ptr,
                                           size_t num_bytes) {
#if CUDART_VERSION >= 11020
    // Cap window size to something reasonable to avoid starving other workloads.
    // Works best when this kernel dominates.
    const size_t max_window = (size_t)64 * 1024 * 1024; // 64MB
    size_t win = num_bytes < max_window ? num_bytes : max_window;

    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
    attr.accessPolicyWindow.num_bytes = win;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
#else
    (void)stream; (void)base_ptr; (void)num_bytes;
#endif
}

static inline void clear_l2_persisting_window(cudaStream_t stream) {
#if CUDART_VERSION >= 11020
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
#else
    (void)stream;
#endif
}

torch::Tensor cumsum_dim1_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [B, N]");

    const auto B64 = x.size(0);
    const auto N64 = x.size(1);
    TORCH_CHECK(B64 > 0 && N64 > 0, "Invalid tensor sizes");
    TORCH_CHECK(B64 <= INT32_MAX && N64 <= INT32_MAX, "Tensor too large for int32 indexing");

    const int B = (int)B64;
    const int N = (int)N64;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty_like(x);

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    constexpr int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = (B + warps_per_block - 1) / warps_per_block;

    const bool can_vec4 = (N % 4 == 0) && is_aligned_16_ptr(xp) && is_aligned_16_ptr(op);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // L2 persisting hint: treat input/output as a persisting window.
    // This can reduce average latency (higher L2 hit rate) in some regimes.
    // Window set on input; output tends to follow same lines, so input hint helps.
    const size_t total_bytes = (size_t)B * (size_t)N * sizeof(float);
    set_l2_persisting_window(stream, xp, total_bytes);

    if (can_vec4) {
        const int N4 = N / 4;
        const int4* xI4 = reinterpret_cast<const int4*>(xp);
        int4* oI4 = reinterpret_cast<int4*>(op);
        cumsum_warp_rows_f32_i4_db<<<blocks, threads, 0, stream>>>(xI4, oI4, B, N4);
    } else {
        cumsum_warp_rows_f32_scalar<<<blocks, threads, 0, stream>>>(xp, op, B, N);
    }

    // Clear stream attribute to avoid impacting subsequent kernels on same stream.
    clear_l2_persisting_window(stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor cumsum_dim1_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_cumsum_dim1_ext_opt6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["cumsum_dim1_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replacement model using an optimized custom CUDA kernel for cumsum along dim=1.
    Specialized for input shape [B, N] float32 CUDA contiguous tensors (2D).
    Falls back to torch.cumsum otherwise.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.cumsum(x, dim=self.dim)
        if (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 2 or (not x.is_contiguous()):
            return torch.cumsum(x, dim=self.dim)
        return self.custom_ops_lib.cumsum_dim1_cuda(x)