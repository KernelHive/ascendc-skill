import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static inline void cuda_check_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, " CUDA error: ", cudaGetErrorString(err));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // up to 1024 threads => 32 warps
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        float out = (lane < nwarps) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) shared[0] = out;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ void unpack_bf16x8_from_int4(
    const int4 &v,
    uint16_t &a0, uint16_t &a1, uint16_t &a2, uint16_t &a3,
    uint16_t &a4, uint16_t &a5, uint16_t &a6, uint16_t &a7
) {
    a0 = (uint16_t)(v.x & 0xFFFF);
    a1 = (uint16_t)((uint32_t)v.x >> 16);
    a2 = (uint16_t)(v.y & 0xFFFF);
    a3 = (uint16_t)((uint32_t)v.y >> 16);
    a4 = (uint16_t)(v.z & 0xFFFF);
    a5 = (uint16_t)((uint32_t)v.z >> 16);
    a6 = (uint16_t)(v.w & 0xFFFF);
    a7 = (uint16_t)((uint32_t)v.w >> 16);
}

__device__ __forceinline__ int pack_bf16x2(uint16_t lo, uint16_t hi) {
    return (int)((uint32_t)lo | ((uint32_t)hi << 16));
}

__global__ void rmsnorm_fwd_bf16_vec8_shmem(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ w,
    at::BFloat16* __restrict__ y,
    int N, int H,
    float eps
) {
    int n = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    const at::BFloat16* x_row = x + (int64_t)n * H;
    at::BFloat16* y_row = y + (int64_t)n * H;

    // dynamic shared memory stores x_row as bf16
    extern __shared__ at::BFloat16 s_x[];

    // vectorized bf16x8
    int vec_elems = H >> 3;
    const int4* __restrict__ x4 = reinterpret_cast<const int4*>(x_row);
    const int4* __restrict__ w4 = reinterpret_cast<const int4*>(w);
    int4* __restrict__ y4 = reinterpret_cast<int4*>(y_row);
    int4* __restrict__ s4 = reinterpret_cast<int4*>(s_x);

    float local = 0.0f;

    // Pass 1: load x once, accumulate sumsq, store bf16 to shared (vectorized)
    for (int i = tid; i < vec_elems; i += (int)blockDim.x) {
        int4 xv = x4[i];
        s4[i] = xv;

        uint16_t a0,a1,a2,a3,a4,a5,a6,a7;
        unpack_bf16x8_from_int4(xv, a0,a1,a2,a3,a4,a5,a6,a7);

        __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&a0);
        __nv_bfloat16 b1 = *reinterpret_cast<__nv_bfloat16*>(&a1);
        __nv_bfloat16 b2 = *reinterpret_cast<__nv_bfloat16*>(&a2);
        __nv_bfloat16 b3 = *reinterpret_cast<__nv_bfloat16*>(&a3);
        __nv_bfloat16 b4 = *reinterpret_cast<__nv_bfloat16*>(&a4);
        __nv_bfloat16 b5 = *reinterpret_cast<__nv_bfloat16*>(&a5);
        __nv_bfloat16 b6 = *reinterpret_cast<__nv_bfloat16*>(&a6);
        __nv_bfloat16 b7 = *reinterpret_cast<__nv_bfloat16*>(&a7);

        float f0 = __bfloat162float(b0); local = fmaf(f0, f0, local);
        float f1 = __bfloat162float(b1); local = fmaf(f1, f1, local);
        float f2 = __bfloat162float(b2); local = fmaf(f2, f2, local);
        float f3 = __bfloat162float(b3); local = fmaf(f3, f3, local);
        float f4 = __bfloat162float(b4); local = fmaf(f4, f4, local);
        float f5 = __bfloat162float(b5); local = fmaf(f5, f5, local);
        float f6 = __bfloat162float(b6); local = fmaf(f6, f6, local);
        float f7 = __bfloat162float(b7); local = fmaf(f7, f7, local);
    }

    float sumsq = block_reduce_sum(local);
    float inv_rms = rsqrtf(sumsq * (1.0f / (float)H) + eps);

    // Ensure s_x fully written before reading in pass2
    __syncthreads();

    // Pass 2: read x from shared, stream weight (RO cache), write y (vectorized)
    const int4* __restrict__ sx4 = reinterpret_cast<const int4*>(s_x);

    for (int i = tid; i < vec_elems; i += (int)blockDim.x) {
        int4 xv = sx4[i];
        int4 wv = __ldg(w4 + i);

        uint16_t x0,x1,x2,x3,x4u,x5u,x6,x7;
        uint16_t w0,w1,w2,w3,w4v,w5v,w6v,w7v;
        unpack_bf16x8_from_int4(xv, x0,x1,x2,x3,x4u,x5u,x6,x7);
        unpack_bf16x8_from_int4(wv, w0,w1,w2,w3,w4v,w5v,w6v,w7v);

        __nv_bfloat16 bx0 = *reinterpret_cast<__nv_bfloat16*>(&x0);
        __nv_bfloat16 bx1 = *reinterpret_cast<__nv_bfloat16*>(&x1);
        __nv_bfloat16 bx2 = *reinterpret_cast<__nv_bfloat16*>(&x2);
        __nv_bfloat16 bx3 = *reinterpret_cast<__nv_bfloat16*>(&x3);
        __nv_bfloat16 bx4 = *reinterpret_cast<__nv_bfloat16*>(&x4u);
        __nv_bfloat16 bx5 = *reinterpret_cast<__nv_bfloat16*>(&x5u);
        __nv_bfloat16 bx6 = *reinterpret_cast<__nv_bfloat16*>(&x6);
        __nv_bfloat16 bx7 = *reinterpret_cast<__nv_bfloat16*>(&x7);

        __nv_bfloat16 bw0 = *reinterpret_cast<__nv_bfloat16*>(&w0);
        __nv_bfloat16 bw1 = *reinterpret_cast<__nv_bfloat16*>(&w1);
        __nv_bfloat16 bw2 = *reinterpret_cast<__nv_bfloat16*>(&w2);
        __nv_bfloat16 bw3 = *reinterpret_cast<__nv_bfloat16*>(&w3);
        __nv_bfloat16 bw4 = *reinterpret_cast<__nv_bfloat16*>(&w4v);
        __nv_bfloat16 bw5 = *reinterpret_cast<__nv_bfloat16*>(&w5v);
        __nv_bfloat16 bw6 = *reinterpret_cast<__nv_bfloat16*>(&w6v);
        __nv_bfloat16 bw7 = *reinterpret_cast<__nv_bfloat16*>(&w7v);

        float fx0 = __bfloat162float(bx0), fw0 = __bfloat162float(bw0);
        float fx1 = __bfloat162float(bx1), fw1 = __bfloat162float(bw1);
        float fx2 = __bfloat162float(bx2), fw2 = __bfloat162float(bw2);
        float fx3 = __bfloat162float(bx3), fw3 = __bfloat162float(bw3);
        float fx4 = __bfloat162float(bx4), fw4 = __bfloat162float(bw4);
        float fx5 = __bfloat162float(bx5), fw5 = __bfloat162float(bw5);
        float fx6 = __bfloat162float(bx6), fw6 = __bfloat162float(bw6);
        float fx7 = __bfloat162float(bx7), fw7 = __bfloat162float(bw7);

        __nv_bfloat16 oy0 = __float2bfloat16_rn((fx0 * inv_rms) * fw0);
        __nv_bfloat16 oy1 = __float2bfloat16_rn((fx1 * inv_rms) * fw1);
        __nv_bfloat16 oy2 = __float2bfloat16_rn((fx2 * inv_rms) * fw2);
        __nv_bfloat16 oy3 = __float2bfloat16_rn((fx3 * inv_rms) * fw3);
        __nv_bfloat16 oy4 = __float2bfloat16_rn((fx4 * inv_rms) * fw4);
        __nv_bfloat16 oy5 = __float2bfloat16_rn((fx5 * inv_rms) * fw5);
        __nv_bfloat16 oy6 = __float2bfloat16_rn((fx6 * inv_rms) * fw6);
        __nv_bfloat16 oy7 = __float2bfloat16_rn((fx7 * inv_rms) * fw7);

        uint16_t o0 = *reinterpret_cast<uint16_t*>(&oy0);
        uint16_t o1 = *reinterpret_cast<uint16_t*>(&oy1);
        uint16_t o2 = *reinterpret_cast<uint16_t*>(&oy2);
        uint16_t o3 = *reinterpret_cast<uint16_t*>(&oy3);
        uint16_t o4 = *reinterpret_cast<uint16_t*>(&oy4);
        uint16_t o5 = *reinterpret_cast<uint16_t*>(&oy5);
        uint16_t o6 = *reinterpret_cast<uint16_t*>(&oy6);
        uint16_t o7 = *reinterpret_cast<uint16_t*>(&oy7);

        int4 outv;
        outv.x = pack_bf16x2(o0, o1);
        outv.y = pack_bf16x2(o2, o3);
        outv.z = pack_bf16x2(o4, o5);
        outv.w = pack_bf16x2(o6, o7);
        y4[i] = outv;
    }
}

// ---------------- generic scalar conversions for fallback ----------------
template <typename T>
__device__ __forceinline__ float to_f32(T v) { return (float)v; }

template <>
__device__ __forceinline__ float to_f32<at::Half>(at::Half v) { return __half2float((__half)v); }

template <>
__device__ __forceinline__ float to_f32<at::BFloat16>(at::BFloat16 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __bfloat162float((__nv_bfloat16)v);
#else
    uint16_t x = *((const uint16_t*)&v);
    uint32_t bits = ((uint32_t)x) << 16;
    return __uint_as_float(bits);
#endif
}

template <typename T>
__device__ __forceinline__ T from_f32(float v) { return (T)v; }

template <>
__device__ __forceinline__ at::Half from_f32<at::Half>(float v) { return (at::Half)__float2half(v); }

template <>
__device__ __forceinline__ at::BFloat16 from_f32<at::BFloat16>(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return (at::BFloat16)__float2bfloat16_rn(v);
#else
    uint32_t bits = __float_as_uint(v);
    uint16_t hi = (uint16_t)(bits >> 16);
    at::BFloat16 out;
    *((uint16_t*)&out) = hi;
    return out;
#endif
}

template <typename x_t, typename w_t, typename out_t>
__global__ void rmsnorm_fwd_generic(
    const x_t* __restrict__ x,
    const w_t* __restrict__ w,
    out_t* __restrict__ y,
    int N, int H,
    float eps
) {
    int n = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    const x_t* x_row = x + (int64_t)n * H;

    float local = 0.0f;
    for (int h = tid; h < H; h += (int)blockDim.x) {
        float xf = to_f32<x_t>(x_row[h]);
        local = fmaf(xf, xf, local);
    }
    float sumsq = block_reduce_sum(local);
    float inv_rms = rsqrtf(sumsq * (1.0f / (float)H) + eps);

    for (int h = tid; h < H; h += (int)blockDim.x) {
        float xf = to_f32<x_t>(x_row[h]);
        float wf = to_f32<w_t>(w[h]);
        y[(int64_t)n * H + h] = from_f32<out_t>((xf * inv_rms) * wf);
    }
}

torch::Tensor rmsnorm_fwd_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    TORCH_CHECK(x.dim() == 2, "x must be 2D [N,H]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [H]");
    TORCH_CHECK(x.size(1) == weight.size(0), "hidden size mismatch");

    int64_t N64 = x.size(0);
    int64_t H64 = x.size(1);
    TORCH_CHECK(N64 <= INT32_MAX && H64 <= INT32_MAX, "N and H must fit in int32");
    int N = (int)N64;
    int H = (int)H64;

    auto y = torch::empty_like(x);

    auto x_type = x.scalar_type();
    auto w_type = weight.scalar_type();
    auto y_type = y.scalar_type();

    // Fast path: bf16, H divisible by 8, and 16B aligned pointers, and shmem fits.
    if (x_type == at::ScalarType::BFloat16 &&
        w_type == at::ScalarType::BFloat16 &&
        y_type == at::ScalarType::BFloat16 &&
        (H % 8 == 0)) {

        uintptr_t xp = (uintptr_t)x.data_ptr();
        uintptr_t wp = (uintptr_t)weight.data_ptr();
        uintptr_t yp = (uintptr_t)y.data_ptr();
        bool aligned16 = ((xp | wp | yp) & 0xF) == 0;

        size_t shmem_bytes = (size_t)H * sizeof(at::BFloat16);
        bool shmem_ok = (shmem_bytes <= (48 * 1024)); // conservative

        if (aligned16 && shmem_ok) {
            int threads = 256;
            if (H >= 8192) threads = 512;
            if (threads > 1024) threads = 1024;

            dim3 grid(N);
            dim3 block(threads);

            rmsnorm_fwd_bf16_vec8_shmem<<<grid, block, shmem_bytes>>>(
                (const at::BFloat16*)x.data_ptr<at::BFloat16>(),
                (const at::BFloat16*)weight.data_ptr<at::BFloat16>(),
                (at::BFloat16*)y.data_ptr<at::BFloat16>(),
                N, H, (float)eps
            );
            cuda_check_last_error("rmsnorm_fwd_bf16_vec8_shmem launch failed");
            return y;
        }
    }

    // Generic fallback
    int threads = 256;
    dim3 block(threads);
    dim3 grid(N);

    if (x_type == at::ScalarType::BFloat16) {
        if (w_type == at::ScalarType::BFloat16 && y_type == at::ScalarType::BFloat16) {
            rmsnorm_fwd_generic<at::BFloat16, at::BFloat16, at::BFloat16><<<grid, block>>>(
                (const at::BFloat16*)x.data_ptr<at::BFloat16>(),
                (const at::BFloat16*)weight.data_ptr<at::BFloat16>(),
                (at::BFloat16*)y.data_ptr<at::BFloat16>(),
                N, H, (float)eps
            );
        } else if (w_type == at::ScalarType::Float && y_type == at::ScalarType::BFloat16) {
            rmsnorm_fwd_generic<at::BFloat16, float, at::BFloat16><<<grid, block>>>(
                (const at::BFloat16*)x.data_ptr<at::BFloat16>(),
                (const float*)weight.data_ptr<float>(),
                (at::BFloat16*)y.data_ptr<at::BFloat16>(),
                N, H, (float)eps
            );
        } else if (w_type == at::ScalarType::Half && y_type == at::ScalarType::BFloat16) {
            rmsnorm_fwd_generic<at::BFloat16, at::Half, at::BFloat16><<<grid, block>>>(
                (const at::BFloat16*)x.data_ptr<at::BFloat16>(),
                (const at::Half*)weight.data_ptr<at::Half>(),
                (at::BFloat16*)y.data_ptr<at::BFloat16>(),
                N, H, (float)eps
            );
        } else {
            TORCH_CHECK(false, "Unsupported weight/output dtype combination for bf16 x");
        }
    } else if (x_type == at::ScalarType::Half) {
        if (w_type == at::ScalarType::Half && y_type == at::ScalarType::Half) {
            rmsnorm_fwd_generic<at::Half, at::Half, at::Half><<<grid, block>>>(
                (const at::Half*)x.data_ptr<at::Half>(),
                (const at::Half*)weight.data_ptr<at::Half>(),
                (at::Half*)y.data_ptr<at::Half>(),
                N, H, (float)eps
            );
        } else if (w_type == at::ScalarType::Float && y_type == at::ScalarType::Half) {
            rmsnorm_fwd_generic<at::Half, float, at::Half><<<grid, block>>>(
                (const at::Half*)x.data_ptr<at::Half>(),
                (const float*)weight.data_ptr<float>(),
                (at::Half*)y.data_ptr<at::Half>(),
                N, H, (float)eps
            );
        } else {
            TORCH_CHECK(false, "Unsupported weight/output dtype combination for fp16 x");
        }
    } else if (x_type == at::ScalarType::Float) {
        TORCH_CHECK(w_type == at::ScalarType::Float && y_type == at::ScalarType::Float,
                    "Unsupported weight/output dtype combination for fp32 x");
        rmsnorm_fwd_generic<float, float, float><<<grid, block>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)weight.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, H, (float)eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported x dtype; expected bf16/fp16/fp32");
    }

    cuda_check_last_error("rmsnorm_fwd_generic launch failed");
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor rmsnorm_fwd_cuda(torch::Tensor x, torch::Tensor weight, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_rmsnorm_ops_shmem_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["rmsnorm_fwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    RMSNorm using an optimized fused custom CUDA kernel for the forward pass.
    Fast path targets bf16 tensors with H divisible by 8, 16B alignment,
    and shared-memory staging to avoid reloading x from global memory.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.custom_ops_lib.rmsnorm_fwd_cuda(
                x.contiguous(),
                self.weight.contiguous(),
                self.eps,
            )

        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(x.dtype)