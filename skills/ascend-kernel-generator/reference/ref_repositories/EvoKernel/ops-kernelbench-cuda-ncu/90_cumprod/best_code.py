import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

static __device__ __forceinline__ float ro_load(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float4 ro_load4(const float* p) {
#if __CUDA_ARCH__ >= 350
    // __ldg supports scalar; rely on normal vector load (still read-only cached on modern arch).
    return *reinterpret_cast<const float4*>(p);
#else
    return *reinterpret_cast<const float4*>(p);
#endif
}
static __device__ __forceinline__ void st_store4(float* p, const float4 &v) {
    *reinterpret_cast<float4*>(p) = v;
}

// Strict inclusive product scan within a warp (lane order).
static __device__ __forceinline__ float warp_inclusive_prod(float x, unsigned mask=0xFFFFFFFFu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(mask, x, offset);
        if ((int)(threadIdx.x & 31) >= offset) x *= y;
    }
    return x;
}

// Baseline general kernel retained (vec4 when possible, scalar otherwise).
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void cumprod_dim1_warp_strict_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int N
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_in_block = WARPS_PER_BLOCK;

    int row = (int)blockIdx.x * warps_in_block + warp;
    int row_stride = (int)gridDim.x * warps_in_block;

    for (; row < B; row += row_stride) {
        const float* row_in  = x   + (int64_t)row * (int64_t)N;
        float*       row_out = out + (int64_t)row * (int64_t)N;

        bool can_vec4 = ((N & 3) == 0) &&
                        ((((uintptr_t)row_in  & 0xF) == 0) && (((uintptr_t)row_out & 0xF) == 0));

        if (can_vec4) {
            const int N4 = N >> 2;
            float carry = 1.0f;

            for (int i4 = lane; i4 < N4; i4 += 32) {
                float4 v4 = ro_load4(row_in + (i4 << 2));

                float p0 = v4.x;
                float p1 = p0 * v4.y;
                float p2 = p1 * v4.z;
                float p3 = p2 * v4.w;
                float lane_prod = p3;

                float inc = warp_inclusive_prod(lane_prod);
                float exc = __shfl_up_sync(0xFFFFFFFFu, inc, 1);
                if (lane == 0) exc = 1.0f;
                float base = carry * exc;

                float o0 = base * p0;
                float o1 = base * p1;
                float o2 = base * p2;
                float o3 = base * p3;

                st_store4(row_out + (i4 << 2), make_float4(o0, o1, o2, o3));

                float step_total = __shfl_sync(0xFFFFFFFFu, inc, 31);
                carry *= step_total;
            }
        } else {
            float carry = 1.0f;
            for (int i = lane; i < N; i += 32) {
                float v = ro_load(row_in + i);
                float inc = warp_inclusive_prod(v);
                float outv = carry * inc;
                row_out[i] = outv;
                float step_total = __shfl_sync(0xFFFFFFFFu, inc, 31);
                carry *= step_total;
            }
        }
    }
}

// Persistent, always-float4 kernel for hot path.
// Grid of blocks; each warp grabs rows via atomic counter.
// Requirements: x/out base pointers 16B aligned, N % 4 == 0, contiguous [B,N].
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void cumprod_dim1_persistent_vec4_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int N4,
    int* __restrict__ row_counter
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // One warp handles one row at a time
    while (true) {
        int row;
        if (lane == 0) row = atomicAdd(row_counter, 1);
        row = __shfl_sync(0xFFFFFFFFu, row, 0);
        if (row >= B) break;

        const float* row_in  = x   + (int64_t)row * (int64_t)(N4 << 2);
        float*       row_out = out + (int64_t)row * (int64_t)(N4 << 2);

        float carry = 1.0f;

        // Iterate over float4 chunks; each iteration is a 32-wide "step" across the warp.
        // Strictness: carry is updated only after completing each step (in increasing i4).
        for (int i4 = lane; i4 < N4; i4 += 32) {
            float4 v4 = ro_load4(row_in + (i4 << 2));

            float p0 = v4.x;
            float p1 = p0 * v4.y;
            float p2 = p1 * v4.z;
            float p3 = p2 * v4.w;
            float lane_prod = p3;

            float inc = warp_inclusive_prod(lane_prod);
            float exc = __shfl_up_sync(0xFFFFFFFFu, inc, 1);
            if (lane == 0) exc = 1.0f;

            float base = carry * exc;
            float o0 = base * p0;
            float o1 = base * p1;
            float o2 = base * p2;
            float o3 = base * p3;

            st_store4(row_out + (i4 << 2), make_float4(o0, o1, o2, o3));

            float step_total = __shfl_sync(0xFFFFFFFFu, inc, 31);
            carry *= step_total;
        }
    }
}

static void set_l2_persisting_hint_if_supported(const void* ptr, size_t bytes) {
#if CUDART_VERSION >= 11000
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) return;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return;

    // Access policy window is supported on Volta+ generally, but size limits vary.
    // Best-effort: if persisting cache is tiny or disabled, skip.
    if (prop.persistingL2CacheMaxSize <= 0) return;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = (bytes > (size_t)prop.accessPolicyMaxWindowSize)
                                            ? (size_t)prop.accessPolicyMaxWindowSize
                                            : bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    // Ignore failure; hint only.
    (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
#else
    (void)ptr; (void)bytes;
#endif
}

torch::Tensor cumprod_dim1_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [B, N]");

    auto out = torch::empty_like(x);
    const int64_t B64 = x.size(0);
    const int64_t N64 = x.size(1);
    if (B64 == 0 || N64 == 0) return out;
    TORCH_CHECK(B64 <= INT_MAX && N64 <= INT_MAX, "Tensor too large for this kernel");

    int B = (int)B64;
    int N = (int)N64;

    const float* xp = (const float*)x.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();

    auto stream = at::cuda::getDefaultCUDAStream();
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // Hot-path eligibility: N%4==0 and base pointers 16B aligned.
    bool hot_vec4 = ((N & 3) == 0) && ((((uintptr_t)xp & 0xF) == 0) && (((uintptr_t)op & 0xF) == 0));

    // Apply an L2 persisting-cache hint for the input window (best-effort).
    // For very large B*N this won't fit, but it can still help locality/priority in L2.
    if (hot_vec4) {
        size_t bytes = (size_t)B * (size_t)N * sizeof(float);
        set_l2_persisting_hint_if_supported((const void*)xp, bytes);
    }

    if (hot_vec4 && N >= 1024) {
        // Persistent scheduling: cap blocks to a few waves over SMs.
        constexpr int WPB = 8; // 256 threads
        int threads = WPB * 32;
        int blocks = num_sms * 8;
        if (blocks < 1) blocks = 1;

        auto counter = torch::empty({1}, torch::TensorOptions().device(x.device()).dtype(torch::kInt32));
        counter.zero_();

        int N4 = N >> 2;
        cumprod_dim1_persistent_vec4_f32<WPB><<<blocks, threads, 0, stream>>>(
            xp, op, B, N4, (int*)counter.data_ptr<int>()
        );
        return out;
    }

    // Fallback: previous warp-per-row kernel.
    if (N <= 256) {
        constexpr int WPB = 4;
        int threads = WPB * 32;
        int blocks = (B + WPB - 1) / WPB;
        int max_blocks = num_sms * 8;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;
        cumprod_dim1_warp_strict_f32<WPB><<<blocks, threads, 0, stream>>>(xp, op, B, N);
    } else {
        constexpr int WPB = 8;
        int threads = WPB * 32;
        int blocks = (B + WPB - 1) / WPB;
        int max_blocks = num_sms * 12;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;
        cumprod_dim1_warp_strict_f32<WPB><<<blocks, threads, 0, stream>>>(xp, op, B, N);
    }
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor cumprod_dim1_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_cumprod_dim1_ext_persistent_vec4_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["cumprod_dim1_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Replacement model using a custom CUDA kernel for cumprod along dim=1.
    Specialized for input shape [B, N] float32 CUDA contiguous tensors.
    Falls back to torch.cumprod for unsupported cases.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.cumprod(x, dim=self.dim)
        if (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 2 or (not x.is_contiguous()):
            return torch.cumprod(x, dim=self.dim)
        return self.custom_ops_lib.cumprod_dim1_cuda(x)