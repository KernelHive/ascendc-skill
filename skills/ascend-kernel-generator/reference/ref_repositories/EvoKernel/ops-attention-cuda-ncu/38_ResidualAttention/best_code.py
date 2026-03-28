import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v9: Improve dominant fused path (FP16, channels_last, C=512, S=49)
# by computing KPACK=2 outputs per warp and reducing KTILE to 4 to lower SMEM/regs
# while reusing x loads across two ks. Keep shared-memory staging for weights to
# avoid weight rereads across S. Only one __syncthreads() remains (after weight load).

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

static __device__ __forceinline__ half ldg_f16(const half* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// ---------------------------------------------
// Dominant fast-path: FP16, channels_last, C=512, S=49.
// Block: 4 warps (128 threads). Each warp computes KPACK=2 outputs (two ks)
// while sharing x loads. We stage weights for [KTILE*KPACK, C] in shared memory.
// Only one sync after weight load.
// ---------------------------------------------
template<int KTILE, int KPACK>
__global__ __launch_bounds__(128, 4) void fused_warpmulti_nhwc_f16_c512_s49(
    const half* __restrict__ x,   // [B,S,C] flattened NHWC-packed (channels_last NCHW)
    const half* __restrict__ w,   // [K,C] contiguous
    float* __restrict__ out,      // [B,K] float32
    int B, int K,
    float la)
{
    constexpr int C = 512;
    constexpr int S = 49;
    constexpr float invS = 1.0f / 49.0f;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..3

    int b = (int)blockIdx.y;
    int k0 = (int)blockIdx.x * (KTILE * KPACK);

    // shared weights: (KTILE*KPACK)*C half
    extern __shared__ half shw[];

    int total_k = KTILE * KPACK;
    int total_w = total_k * C;

    for (int i = tid; i < total_w; i += blockDim.x) {
        int kt = i / C;        // 0..total_k-1
        int c  = i - kt * C;
        int k  = k0 + kt;
        shw[i] = (k < K) ? ldg_f16(w + (int64_t)k * C + c) : __float2half(0.f);
    }
    __syncthreads();

    // warp maps to a tile-index (0..KTILE-1), producing KPACK outputs
    int tile = warp;
    if (tile >= KTILE) return;

    int k_base = k0 + tile * KPACK;
    int k1 = k_base;
    int k2 = k_base + 1;

    if (k1 >= K) return;

    const half* x_base = x + (int64_t)b * (int64_t)S * (int64_t)C;

    const half* w1_sh = shw + (int64_t)(tile * KPACK + 0) * C;
    const half* w2_sh = shw + (int64_t)(tile * KPACK + 1) * C;

    float sum1 = 0.f, max1 = -INFINITY;
    float sum2 = 0.f, max2 = -INFINITY;

    #pragma unroll
    for (int sidx = 0; sidx < S; ++sidx) {
        const half* x_ptr = x_base + (int64_t)sidx * C;

        const half2* x2  = reinterpret_cast<const half2*>(x_ptr);
        const half2* w12 = reinterpret_cast<const half2*>(w1_sh);
        const half2* w22 = reinterpret_cast<const half2*>(w2_sh);

        float acc1 = 0.f;
        float acc2 = 0.f;

        #pragma unroll
        for (int c2 = lane; c2 < (C / 2); c2 += 32) {
            half2 xv  = __ldg(x2 + c2);
            half2 wv1 = __ldg(w12 + c2);
            half2 wv2 = __ldg(w22 + c2);

            // fp16->fp32 multiply-accumulate for both outputs
            acc1 = fmaf(__half2float(xv.x), __half2float(wv1.x), acc1);
            acc1 = fmaf(__half2float(xv.y), __half2float(wv1.y), acc1);

            acc2 = fmaf(__half2float(xv.x), __half2float(wv2.x), acc2);
            acc2 = fmaf(__half2float(xv.y), __half2float(wv2.y), acc2);
        }

        float dot1 = warp_reduce_sum(acc1);
        float dot2 = warp_reduce_sum(acc2);

        if (lane == 0) {
            sum1 += dot1; max1 = fmaxf(max1, dot1);
            if (k2 < K) { sum2 += dot2; max2 = fmaxf(max2, dot2); }
        }
    }

    if (lane == 0) {
        out[(int64_t)b * (int64_t)K + (int64_t)k1] = sum1 * invS + la * max1;
        if (k2 < K) out[(int64_t)b * (int64_t)K + (int64_t)k2] = sum2 * invS + la * max2;
    }
}

// ---------------------------------------------
// Generic fused kernels (tile weights in shared memory)
// ---------------------------------------------
template<int KTILE, int CCONST, int SCONST>
__global__ __launch_bounds__(256, 2) void fused_tilew_nhwc_f16_sc_kernel(
    const half* __restrict__ x,   // [B,S,C] NHWC-packed
    const half* __restrict__ w,   // [K,C]
    float* __restrict__ out,      // [B,K]
    int B, int S, int C, int K,
    float la)
{
    int tid = threadIdx.x;
    int b = blockIdx.y;
    int k0 = blockIdx.x * KTILE;
    if (k0 >= K) return;

    extern __shared__ half shw[];
    int total_w = KTILE * C;
    for (int i = tid; i < total_w; i += blockDim.x) {
        int kt = i / C;
        int c = i - kt * C;
        int k = k0 + kt;
        half v = (k < K) ? ldg_f16(w + (int64_t)k * C + c) : __float2half(0.f);
        shw[i] = v;
    }
    __syncthreads();

    int kt = tid;
    if (kt >= KTILE) return;
    int k = k0 + kt;
    if (k >= K) return;

    const half* x_base = x + (int64_t)b * (int64_t)S * (int64_t)C;
    const half* w_sh = shw + kt * C;

    float sum_s = 0.f;
    float max_s = -INFINITY;

    int Ceff = (CCONST > 0) ? CCONST : C;
    int Seff = (SCONST > 0) ? SCONST : S;

    for (int sidx = 0; sidx < Seff; ++sidx) {
        const half* x_ptr = x_base + (int64_t)sidx * (int64_t)C;
        float acc = 0.f;

        for (int c = tid; c < Ceff; c += blockDim.x) {
            float xv = __half2float(ldg_f16(x_ptr + c));
            float wv = __half2float(w_sh[c]);
            acc = fmaf(xv, wv, acc);
        }

        float wsum = warp_reduce_sum(acc);

        __shared__ float warp_partials[8];
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) warp_partials[warp] = wsum;
        __syncthreads();

        float dot = 0.f;
        if (warp == 0) {
            float v = (lane < (blockDim.x >> 5)) ? warp_partials[lane] : 0.f;
            v = warp_reduce_sum(v);
            if (lane == 0) dot = v;
        }
        __shared__ float dot_shared;
        if (tid == 0) dot_shared = dot;
        __syncthreads();
        dot = dot_shared;

        sum_s += dot;
        max_s = fmaxf(max_s, dot);
    }

    float invS = 1.0f / (float)Seff;
    out[(int64_t)b * (int64_t)K + k] = sum_s * invS + la * max_s;
}

template<int KTILE, int CCONST, int SCONST>
__global__ __launch_bounds__(256, 2) void fused_tilew_nhwc_f32_sc_kernel(
    const float* __restrict__ x,  // [B,S,C]
    const float* __restrict__ w,  // [K,C]
    float* __restrict__ out,      // [B,K]
    int B, int S, int C, int K,
    float la)
{
    int tid = threadIdx.x;
    int b = blockIdx.y;
    int k0 = blockIdx.x * KTILE;
    if (k0 >= K) return;

    extern __shared__ float shw_f[];
    int total_w = KTILE * C;
    for (int i = tid; i < total_w; i += blockDim.x) {
        int kt = i / C;
        int c = i - kt * C;
        int k = k0 + kt;
        float v = (k < K) ? ldg_f32(w + (int64_t)k * C + c) : 0.f;
        shw_f[i] = v;
    }
    __syncthreads();

    int kt = tid;
    if (kt >= KTILE) return;
    int k = k0 + kt;
    if (k >= K) return;

    const float* x_base = x + (int64_t)b * (int64_t)S * (int64_t)C;
    const float* w_sh = shw_f + kt * C;

    float sum_s = 0.f;
    float max_s = -INFINITY;

    int Ceff = (CCONST > 0) ? CCONST : C;
    int Seff = (SCONST > 0) ? SCONST : S;

    for (int sidx = 0; sidx < Seff; ++sidx) {
        const float* x_ptr = x_base + (int64_t)sidx * (int64_t)C;
        float acc = 0.f;
        for (int c = tid; c < Ceff; c += blockDim.x) {
            float xv = ldg_f32(x_ptr + c);
            float wv = w_sh[c];
            acc = fmaf(xv, wv, acc);
        }

        float wsum = warp_reduce_sum(acc);

        __shared__ float warp_partials[8];
        int lane = tid & 31;
        int warp = tid >> 5;
        if (lane == 0) warp_partials[warp] = wsum;
        __syncthreads();

        float dot = 0.f;
        if (warp == 0) {
            float v = (lane < (blockDim.x >> 5)) ? warp_partials[lane] : 0.f;
            v = warp_reduce_sum(v);
            if (lane == 0) dot = v;
        }
        __shared__ float dot_shared;
        if (tid == 0) dot_shared = dot;
        __syncthreads();
        dot = dot_shared;

        sum_s += dot;
        max_s = fmaxf(max_s, dot);
    }

    float invS = 1.0f / (float)Seff;
    out[(int64_t)b * (int64_t)K + k] = sum_s * invS + la * max_s;
}

// ---------------------------------------------
// Standalone reduction kernels (unchanged)
// ---------------------------------------------
static __device__ __forceinline__ float warp_reduce_sum2(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static __device__ __forceinline__ float warp_reduce_max2(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

__global__ __launch_bounds__(256, 2) void residual_attention_hw49_kernel(
    const float* __restrict__ y,   // [B, K, 49]
    float* __restrict__ out,       // [B, K]
    int total,                     // B*K
    int K,
    float la)
{
    constexpr int HW = 49;
    constexpr float invHW = 1.0f / 49.0f;

    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    int stride_warps = gridDim.x * warps_per_block;

    for (int idx = warp_global; idx < total; idx += stride_warps) {
        int b = idx / K;
        int k = idx - b * K;

        const float* ptr = y + ((b * K + k) * HW);

        float sum = 0.0f;
        float mx = -INFINITY;

        int j0 = lane;
        int j1 = lane + 32;

        if (j0 < HW) { float v = ptr[j0]; sum += v; mx = fmaxf(mx, v); }
        if (j1 < HW) { float v = ptr[j1]; sum += v; mx = fmaxf(mx, v); }

        sum = warp_reduce_sum2(sum);
        mx  = warp_reduce_max2(mx);

        if (lane == 0) out[idx] = sum * invHW + la * mx;
    }
}

__global__ __launch_bounds__(256, 2) void residual_attention_generic_kernel(
    const float* __restrict__ y,   // [B, K, HW]
    float* __restrict__ out,       // [B, K]
    int total,                     // B*K
    int K,
    int HW,
    float invHW,
    float la)
{
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    int stride_warps = gridDim.x * warps_per_block;

    for (int idx = warp_global; idx < total; idx += stride_warps) {
        int b = idx / K;
        int k = idx - b * K;
        const float* ptr = y + ((b * K + k) * HW);

        float sum = 0.0f;
        float mx = -INFINITY;

        for (int j = lane; j < HW; j += 32) {
            float v = ptr[j];
            sum += v;
            mx = fmaxf(mx, v);
        }

        sum = warp_reduce_sum2(sum);
        mx  = warp_reduce_max2(mx);

        if (lane == 0) out[idx] = sum * invHW + la * mx;
    }
}

torch::Tensor residual_attention_cuda(torch::Tensor y, double la) {
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(y.scalar_type() == at::ScalarType::Float, "y must be float32");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(y.dim() == 4, "y must be 4D (B, K, H, W)");

    int64_t B = y.size(0);
    int64_t K = y.size(1);
    int64_t H = y.size(2);
    int64_t W = y.size(3);
    int64_t HW = H * W;
    int64_t total = B * K;

    auto out = torch::empty({B, K}, torch::TensorOptions().dtype(y.dtype()).device(y.device()));

    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;

    int blocks = (int)((total + warps_per_block - 1) / warps_per_block);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (HW == 49) {
        residual_attention_hw49_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)y.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            (int)total,
            (int)K,
            (float)la
        );
    } else {
        float invHW = 1.0f / (float)HW;
        residual_attention_generic_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)y.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            (int)total,
            (int)K,
            (int)HW,
            invHW,
            (float)la
        );
    }
    return out;
}

// ---------------------------------------------
// Fused entrypoint
// ---------------------------------------------
torch::Tensor fused_residual_attention_cuda(torch::Tensor x, torch::Tensor w, double la) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(w.dim() == 2, "w must be 2D [K,C]");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous [K,C]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (channels_last contiguous required)");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int K = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == C, "w.shape[1] must match x.channels");

    int S = H * W;

    auto xs = x.strides();
    bool is_channels_last = (xs[1] == 1);
    TORCH_CHECK(is_channels_last, "fused kernel requires channels_last contiguous x");

    auto out = torch::empty({B, K}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (x.scalar_type() == at::ScalarType::Half && w.scalar_type() == at::ScalarType::Half) {
        const half* xp = (const half*)x.data_ptr<at::Half>();
        const half* wp = (const half*)w.data_ptr<at::Half>();

        // Dominant fast path
        if (S == 49 && C == 512) {
            constexpr int KTILE = 4;   // warps/block
            constexpr int KPACK = 2;   // outputs/warp
            dim3 block(128, 1, 1);
            dim3 grid((K + (KTILE * KPACK) - 1) / (KTILE * KPACK), B, 1);
            size_t shmem = (size_t)(KTILE * KPACK) * (size_t)512 * sizeof(half);
            fused_warpmulti_nhwc_f16_c512_s49<KTILE, KPACK><<<grid, block, shmem, stream>>>(
                xp, wp, (float*)out.data_ptr<float>(), B, K, (float)la
            );
            return out;
        }

        // Generic fused
        constexpr int threads = 256;
        constexpr int KTILE = 32;
        dim3 grid((K + KTILE - 1) / KTILE, B, 1);
        size_t shmem = (size_t)KTILE * (size_t)C * sizeof(half);

        fused_tilew_nhwc_f16_sc_kernel<KTILE, -1, -1><<<grid, threads, shmem, stream>>>(
            xp, wp, (float*)out.data_ptr<float>(), B, S, C, K, (float)la
        );
        return out;
    }

    if (x.scalar_type() == at::ScalarType::Float && w.scalar_type() == at::ScalarType::Float) {
        constexpr int threads = 256;
        constexpr int KTILE = 16;
        dim3 grid((K + KTILE - 1) / KTILE, B, 1);
        size_t shmem = (size_t)KTILE * (size_t)C * sizeof(float);

        const float* xp = (const float*)x.data_ptr<float>();
        const float* wp = (const float*)w.data_ptr<float>();

        if (S == 49 && C == 512) {
            fused_tilew_nhwc_f32_sc_kernel<KTILE, 512, 49><<<grid, threads, shmem, stream>>>(
                xp, wp, (float*)out.data_ptr<float>(), B, S, C, K, (float)la
            );
        } else {
            fused_tilew_nhwc_f32_sc_kernel<KTILE, -1, -1><<<grid, threads, shmem, stream>>>(
                xp, wp, (float*)out.data_ptr<float>(), B, S, C, K, (float)la
            );
        }
        return out;
    }

    TORCH_CHECK(false, "fused kernel supports only float16 or float32 for both x and w");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residual_attention_cuda", &residual_attention_cuda, "Residual attention reduction (CUDA)");
    m.def("fused_residual_attention_cuda", &fused_residual_attention_cuda, "Fused 1x1 conv + residual attention (CUDA)");
}
"""

_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor residual_attention_cuda(torch::Tensor y, double la);
torch::Tensor fused_residual_attention_cuda(torch::Tensor x, torch::Tensor w, double la);
"""

custom_ops_lib = load_inline(
    name="custom_residual_attention_ops_v9_warpmulti",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Residual Attention Module with optimized fused CUDA kernel (channels_last fast path) + fallback."""
    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = float(la)
        self.fc = nn.Conv2d(
            in_channels=channel,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: CUDA + channels_last contiguous + contiguous weights + matching dtype
        if x.is_cuda and x.is_contiguous() and (x.stride(1) == 1) and self.fc.weight.is_contiguous():
            w = self.fc.weight.view(self.fc.out_channels, self.fc.in_channels)
            if (x.dtype == torch.float16 and w.dtype == torch.float16) or (x.dtype == torch.float32 and w.dtype == torch.float32):
                return self.custom_ops_lib.fused_residual_attention_cuda(x, w, self.la)

        # Fallback: PyTorch conv then custom reduction (expects contiguous float32 y)
        y = self.fc(x)
        if not y.is_contiguous():
            y = y.contiguous()
        if y.dtype != torch.float32:
            y = y.float()
        return self.custom_ops_lib.residual_attention_cuda(y, self.la)