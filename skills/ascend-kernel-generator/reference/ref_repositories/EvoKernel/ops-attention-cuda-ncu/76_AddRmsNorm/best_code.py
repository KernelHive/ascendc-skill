import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------
# Custom CUDA: fused add + RMSNorm (optimized)
# - Fast path: bf16, H % 8 == 0, 16B aligned pointers, and enough SMEM
#   * Vectorized 16B loads for x/residual/weight and 16B stores for out
#   * Stage combined (x+residual) in shared memory to avoid re-reading x/residual
# - Fallback: baseline-like scalar kernel (still correct for all bf16 contiguous)
# -----------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::kBFloat16, #x " must be bfloat16")
#define CHECK_INPUT2D(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x); TORCH_CHECK(x.dim() == 2, #x " must be 2D")
#define CHECK_INPUT1D(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x); TORCH_CHECK(x.dim() == 1, #x " must be 1D")

static inline void cuda_check_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, " CUDA error: ", cudaGetErrorString(err));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // up to 1024 threads -> 32 warps
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        float v = (lane < nwarps) ? shared[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float bf16_to_f32(at::BFloat16 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __bfloat162float((__nv_bfloat16)v);
#else
    uint16_t x = *((const uint16_t*)&v);
    uint32_t bits = ((uint32_t)x) << 16;
    return __uint_as_float(bits);
#endif
}

__device__ __forceinline__ at::BFloat16 f32_to_bf16_rn(float v) {
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

// -----------------------------
// Fast path kernel: stage combined into shared and use vec8 (int4) loads/stores
// Requirements checked on host:
// - H % 8 == 0
// - x/residual/weight/out 16B aligned
// - dynamic shared bytes = H*sizeof(bf16) fits device limit
// -----------------------------
__global__ void add_rms_norm_bf16_stage_vec8(
    const at::BFloat16* __restrict__ x,        // [T,H]
    const at::BFloat16* __restrict__ residual, // [T,H]
    const at::BFloat16* __restrict__ weight,   // [H]
    at::BFloat16* __restrict__ out,            // [T,H]
    int T, int H,
    float eps
) {
    int t = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    const int64_t row_off = (int64_t)t * (int64_t)H;

    // dynamic shared: H bf16
    extern __shared__ at::BFloat16 sh_combined[];

    // process 8 bf16 at a time (16 bytes)
    int vec_elems = H >> 3;
    const int4* __restrict__ x4 = reinterpret_cast<const int4*>(x + row_off);
    const int4* __restrict__ r4 = reinterpret_cast<const int4*>(residual + row_off);
    int4* __restrict__ sh4 = reinterpret_cast<int4*>(sh_combined);

    float local = 0.0f;

    for (int i = tid; i < vec_elems; i += (int)blockDim.x) {
        int4 xv = x4[i];
        int4 rv = r4[i];

        // unpack 8 bf16 from each, add, store combined into shared, accumulate sumsq
        uint16_t xa[8], ra[8];
        xa[0] = (uint16_t)(xv.x & 0xFFFF); xa[1] = (uint16_t)((uint32_t)xv.x >> 16);
        xa[2] = (uint16_t)(xv.y & 0xFFFF); xa[3] = (uint16_t)((uint32_t)xv.y >> 16);
        xa[4] = (uint16_t)(xv.z & 0xFFFF); xa[5] = (uint16_t)((uint32_t)xv.z >> 16);
        xa[6] = (uint16_t)(xv.w & 0xFFFF); xa[7] = (uint16_t)((uint32_t)xv.w >> 16);

        ra[0] = (uint16_t)(rv.x & 0xFFFF); ra[1] = (uint16_t)((uint32_t)rv.x >> 16);
        ra[2] = (uint16_t)(rv.y & 0xFFFF); ra[3] = (uint16_t)((uint32_t)rv.y >> 16);
        ra[4] = (uint16_t)(rv.z & 0xFFFF); ra[5] = (uint16_t)((uint32_t)rv.z >> 16);
        ra[6] = (uint16_t)(rv.w & 0xFFFF); ra[7] = (uint16_t)((uint32_t)rv.w >> 16);

        at::BFloat16 bc[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            at::BFloat16 bx, br;
            *((uint16_t*)&bx) = xa[k];
            *((uint16_t*)&br) = ra[k];
            float c = bf16_to_f32(bx) + bf16_to_f32(br);
            local = fmaf(c, c, local);
            bc[k] = f32_to_bf16_rn(c);
        }

        // pack combined back into int4 and store to shared
        int4 cv;
        cv.x = (int)((uint32_t)(*((uint16_t*)&bc[0])) | ((uint32_t)(*((uint16_t*)&bc[1])) << 16));
        cv.y = (int)((uint32_t)(*((uint16_t*)&bc[2])) | ((uint32_t)(*((uint16_t*)&bc[3])) << 16));
        cv.z = (int)((uint32_t)(*((uint16_t*)&bc[4])) | ((uint32_t)(*((uint16_t*)&bc[5])) << 16));
        cv.w = (int)((uint32_t)(*((uint16_t*)&bc[6])) | ((uint32_t)(*((uint16_t*)&bc[7])) << 16));
        sh4[i] = cv;
    }

    float sumsq = block_reduce_sum(local);
    float inv_rms = rsqrtf(sumsq * (1.0f / (float)H) + eps);

    // write output: (combined * inv_rms) * weight
    const int4* __restrict__ w4 = reinterpret_cast<const int4*>(weight);
    int4* __restrict__ out4 = reinterpret_cast<int4*>(out + row_off);

    for (int i = tid; i < vec_elems; i += (int)blockDim.x) {
        int4 cv = sh4[i];
        int4 wv = w4[i];

        uint16_t ca[8], wa[8];
        ca[0] = (uint16_t)(cv.x & 0xFFFF); ca[1] = (uint16_t)((uint32_t)cv.x >> 16);
        ca[2] = (uint16_t)(cv.y & 0xFFFF); ca[3] = (uint16_t)((uint32_t)cv.y >> 16);
        ca[4] = (uint16_t)(cv.z & 0xFFFF); ca[5] = (uint16_t)((uint32_t)cv.z >> 16);
        ca[6] = (uint16_t)(cv.w & 0xFFFF); ca[7] = (uint16_t)((uint32_t)cv.w >> 16);

        wa[0] = (uint16_t)(wv.x & 0xFFFF); wa[1] = (uint16_t)((uint32_t)wv.x >> 16);
        wa[2] = (uint16_t)(wv.y & 0xFFFF); wa[3] = (uint16_t)((uint32_t)wv.y >> 16);
        wa[4] = (uint16_t)(wv.z & 0xFFFF); wa[5] = (uint16_t)((uint32_t)wv.z >> 16);
        wa[6] = (uint16_t)(wv.w & 0xFFFF); wa[7] = (uint16_t)((uint32_t)wv.w >> 16);

        at::BFloat16 oy[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            at::BFloat16 bc, bw;
            *((uint16_t*)&bc) = ca[k];
            *((uint16_t*)&bw) = wa[k];
            float y = (bf16_to_f32(bc) * inv_rms) * bf16_to_f32(bw);
            oy[k] = f32_to_bf16_rn(y);
        }

        int4 ov;
        ov.x = (int)((uint32_t)(*((uint16_t*)&oy[0])) | ((uint32_t)(*((uint16_t*)&oy[1])) << 16));
        ov.y = (int)((uint32_t)(*((uint16_t*)&oy[2])) | ((uint32_t)(*((uint16_t*)&oy[3])) << 16));
        ov.z = (int)((uint32_t)(*((uint16_t*)&oy[4])) | ((uint32_t)(*((uint16_t*)&oy[5])) << 16));
        ov.w = (int)((uint32_t)(*((uint16_t*)&oy[6])) | ((uint32_t)(*((uint16_t*)&oy[7])) << 16));
        out4[i] = ov;
    }
}

// -----------------------------
// Fallback kernel: scalar baseline-like (still bf16 only, contiguous)
// -----------------------------
__global__ void add_rms_norm_fwd_scalar(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int T, int H,
    float eps
) {
    int t = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int row_base = t * H;

    float local_sum = 0.0f;
    for (int j = tid; j < H; j += (int)blockDim.x) {
        float xv = __bfloat162float(x[row_base + j]);
        float rv = __bfloat162float(residual[row_base + j]);
        float c = xv + rv;
        local_sum += c * c;
    }

    float sumsq = block_reduce_sum(local_sum);
    float inv_rms = rsqrtf(sumsq * (1.0f / (float)H) + eps);

    for (int j = tid; j < H; j += (int)blockDim.x) {
        float xv = __bfloat162float(x[row_base + j]);
        float rv = __bfloat162float(residual[row_base + j]);
        float wv = __bfloat162float(weight[j]);
        float y = (xv + rv) * inv_rms * wv;
        out[row_base + j] = __float2bfloat16_rn(y);
    }
}

torch::Tensor add_rms_norm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    CHECK_INPUT2D(x);
    CHECK_INPUT2D(residual);
    CHECK_INPUT1D(weight);

    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    TORCH_CHECK(x.size(1) == weight.size(0), "weight must have shape [hidden_size]");
    TORCH_CHECK(x.device() == residual.device() && x.device() == weight.device(), "all inputs must be on same CUDA device");

    int64_t T64 = x.size(0);
    int64_t H64 = x.size(1);
    TORCH_CHECK(T64 > 0 && H64 > 0, "invalid shapes");
    TORCH_CHECK(T64 <= INT_MAX && H64 <= INT_MAX, "shapes too large");
    int T = (int)T64;
    int H = (int)H64;

    auto out = torch::empty_like(x);

    // Prefer 256 threads; good balance for H=4096.
    int threads = 256;
    dim3 block(threads);
    dim3 grid(T);

    // Fast path guards
    bool vec_ok = (H % 8 == 0);
    uintptr_t xp = (uintptr_t)x.data_ptr();
    uintptr_t rp = (uintptr_t)residual.data_ptr();
    uintptr_t wp = (uintptr_t)weight.data_ptr();
    uintptr_t op = (uintptr_t)out.data_ptr();
    bool aligned16 = ((xp | rp | wp | op) & 0xF) == 0;

    size_t shmem_bytes = (size_t)H * sizeof(at::BFloat16);

    int device = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // be conservative: require <= sharedMemPerBlockOptin if available else sharedMemPerBlock
    size_t shmem_limit = (size_t)prop.sharedMemPerBlock;
#if defined(cudaDevAttrMaxSharedMemoryPerBlockOptin)
    int optin = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess) {
        if (optin > 0) shmem_limit = (size_t)optin;
    }
#endif

    if (vec_ok && aligned16 && shmem_bytes <= shmem_limit) {
        // If opt-in is needed for larger SMEM, request it.
        cudaFuncSetAttribute(add_rms_norm_bf16_stage_vec8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

        add_rms_norm_bf16_stage_vec8<<<grid, block, shmem_bytes>>>(
            (const at::BFloat16*)x.data_ptr<at::BFloat16>(),
            (const at::BFloat16*)residual.data_ptr<at::BFloat16>(),
            (const at::BFloat16*)weight.data_ptr<at::BFloat16>(),
            (at::BFloat16*)out.data_ptr<at::BFloat16>(),
            T, H, (float)eps
        );
        cuda_check_last_error("add_rms_norm_bf16_stage_vec8 launch failed");
        return out;
    }

    // Fallback
    add_rms_norm_fwd_scalar<<<grid, block>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)residual.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        T, H, (float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor add_rms_norm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_add_rms_norm_opt2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_rms_norm_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Uses an optimized fused CUDA kernel for (x + residual) + RMSNorm + weight.
    Inputs/outputs: bfloat16 CUDA tensors, shape (num_tokens, hidden_size).
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x, residual):
        if (
            x.is_cuda
            and residual.is_cuda
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            return custom_ops_lib.add_rms_norm_cuda(
                x.contiguous(),
                residual.contiguous(),
                self.weight.contiguous(),
                self.eps,
            )

        combined = (x + residual).float()
        variance = combined.pow(2).mean(-1, keepdim=True)
        normed = combined * torch.rsqrt(variance + self.eps)
        return (self.weight.to(normed.dtype) * normed).to(x.dtype)