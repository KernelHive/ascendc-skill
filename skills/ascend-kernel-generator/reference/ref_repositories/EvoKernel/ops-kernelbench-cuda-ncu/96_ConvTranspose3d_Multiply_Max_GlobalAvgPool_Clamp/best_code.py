import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

__device__ __forceinline__ float clamp01(float v){
    v = v < 0.0f ? 0.0f : v;
    v = v > 1.0f ? 1.0f : v;
    return v;
}

__device__ __forceinline__ float ldgf(const float* p){
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v){
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0){
        float t = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    out = smem[0];
    return out;
}

static __device__ __forceinline__ float max8(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7){
    float m0 = fmaxf(fmaxf(a0,a1), fmaxf(a2,a3));
    float m1 = fmaxf(fmaxf(a4,a5), fmaxf(a6,a7));
    return fmaxf(m0,m1);
}

// One block per nc. Persistent loop over pooled voxels; block-reduce sum; normalize+clamp+write.
// Specialize K=2 with vectorized loads when possible.
__global__ __launch_bounds__(128, 6)
void fused_mul_maxpool_gap_clamp_kernel(
    const float* __restrict__ x, // [NC, D, H, W] flattened
    float* __restrict__ y,       // [NC]
    int NC,
    int D, int H, int W,
    int K,
    int D2, int H2, int W2,
    float scale,
    float inv_S_pool
){
    int nc = (int)blockIdx.x;
    if (nc >= NC) return;

    const int64_t S_in = (int64_t)D * H * W;
    const int64_t base_nc = (int64_t)nc * S_in;
    const int64_t S_pool = (int64_t)D2 * H2 * W2;

    float thread_sum = 0.0f;

    // We want each block to traverse pooled voxels with stride = blockDim.x
    // to keep per-thread work balanced and enable latency hiding.
    if (K == 2){
        // For K=2, each pooled voxel reads a 2x2x2 cube = 8 floats.
        // Attempt to load each row as float2, and optionally as float4 when W alignment permits.
        for (int64_t p = (int64_t)threadIdx.x; p < S_pool; p += (int64_t)blockDim.x){
            int64_t t = p;
            int ow = (int)(t % W2); t /= W2;
            int oh = (int)(t % H2); t /= H2;
            int od = (int)t;

            int id0 = od << 1;
            int ih0 = oh << 1;
            int iw0 = ow << 1;

            int64_t d0 = base_nc + (int64_t)id0 * (int64_t)H * (int64_t)W;
            int64_t d1 = d0 + (int64_t)H * (int64_t)W;

            int64_t r00 = d0 + (int64_t)ih0 * (int64_t)W + (int64_t)iw0;
            int64_t r01 = r00 + (int64_t)W;
            int64_t r10 = d1 + (int64_t)ih0 * (int64_t)W + (int64_t)iw0;
            int64_t r11 = r10 + (int64_t)W;

            float v0,v1,v2,v3,v4,v5,v6,v7;

            const float* p00 = x + r00;
            const float* p01 = x + r01;
            const float* p10 = x + r10;
            const float* p11 = x + r11;

            // Use float2 loads if 8-byte aligned; scalar otherwise.
            if ((((uintptr_t)p00 | (uintptr_t)p01 | (uintptr_t)p10 | (uintptr_t)p11) & 0x7) == 0){
                float2 a = *reinterpret_cast<const float2*>(p00);
                float2 b = *reinterpret_cast<const float2*>(p01);
                float2 c = *reinterpret_cast<const float2*>(p10);
                float2 d = *reinterpret_cast<const float2*>(p11);
                v0=a.x; v1=a.y;
                v2=b.x; v3=b.y;
                v4=c.x; v5=c.y;
                v6=d.x; v7=d.y;
            } else {
                v0 = ldgf(p00 + 0); v1 = ldgf(p00 + 1);
                v2 = ldgf(p01 + 0); v3 = ldgf(p01 + 1);
                v4 = ldgf(p10 + 0); v5 = ldgf(p10 + 1);
                v6 = ldgf(p11 + 0); v7 = ldgf(p11 + 1);
            }

            float m = max8(v0,v1,v2,v3,v4,v5,v6,v7);
            thread_sum = fmaf(m, scale, thread_sum);
        }
    } else {
        for (int64_t p = (int64_t)threadIdx.x; p < S_pool; p += (int64_t)blockDim.x){
            int64_t t = p;
            int ow = (int)(t % W2); t /= W2;
            int oh = (int)(t % H2); t /= H2;
            int od = (int)t;

            int id0 = od * K;
            int ih0 = oh * K;
            int iw0 = ow * K;

            float m = -INFINITY;
            #pragma unroll 1
            for (int kd = 0; kd < K; ++kd){
                int id = id0 + kd;
                int64_t bd = base_nc + (int64_t)id * (int64_t)H * (int64_t)W;
                #pragma unroll 1
                for (int kh = 0; kh < K; ++kh){
                    int ih = ih0 + kh;
                    int64_t bdh = bd + (int64_t)ih * (int64_t)W;
                    #pragma unroll 1
                    for (int kw = 0; kw < K; ++kw){
                        int iw = iw0 + kw;
                        float v = ldgf(x + bdh + iw);
                        m = fmaxf(m, v);
                    }
                }
            }
            thread_sum = fmaf(m, scale, thread_sum);
        }
    }

    float sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0){
        float avg = sum * inv_S_pool;
        y[nc] = clamp01(avg);
    }
}

torch::Tensor mul_maxpool_gap_clamp_cuda(
    torch::Tensor x,      // [N,C,D,H,W]
    double scale_d,
    int64_t maxpool_k
){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be [N,C,D,H,W]");
    TORCH_CHECK(maxpool_k >= 1, "maxpool_k must be >= 1");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    auto xc = x.contiguous();

    int64_t N = xc.size(0), C = xc.size(1), D = xc.size(2), H = xc.size(3), W = xc.size(4);
    TORCH_CHECK(D >= maxpool_k && H >= maxpool_k && W >= maxpool_k, "maxpool_k larger than input dims");

    // MaxPool3d defaults: stride=K, padding=0, ceil_mode=False
    int64_t D2 = (D - maxpool_k) / maxpool_k + 1;
    int64_t H2 = (H - maxpool_k) / maxpool_k + 1;
    int64_t W2 = (W - maxpool_k) / maxpool_k + 1;
    TORCH_CHECK(D2 > 0 && H2 > 0 && W2 > 0, "pooled dims must be > 0");

    int64_t NC64 = N * C;
    TORCH_CHECK(NC64 <= (int64_t)2147483647, "N*C too large");
    int NC = (int)NC64;

    int64_t S_pool64 = D2 * H2 * W2;
    TORCH_CHECK(S_pool64 > 0 && S_pool64 <= (int64_t)2147483647, "S_pool invalid");

    float inv_S_pool = 1.0f / (float)S_pool64;
    float scale = (float)scale_d;

    auto y_nc = torch::empty({NC}, xc.options());

    // One block per nc avoids any atomic contention. NC is large enough (e.g. 2048).
    int threads = 128;
    int blocks = NC;

    fused_mul_maxpool_gap_clamp_kernel<<<blocks, threads, 0, stream>>>(
        xc.data_ptr<float>(),
        y_nc.data_ptr<float>(),
        NC,
        (int)D, (int)H, (int)W,
        (int)maxpool_k,
        (int)D2, (int)H2, (int)W2,
        scale,
        inv_S_pool
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y_nc.view({N, C, 1, 1, 1});
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor mul_maxpool_gap_clamp_cuda(torch::Tensor x, double scale_d, int64_t maxpool_k);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mul_maxpool_gap_clamp_v8_singlekernel",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["mul_maxpool_gap_clamp_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    conv_transpose3d (PyTorch) -> fused CUDA: (maxpool -> *scale -> global avg -> clamp(0,1))

    Assumptions for the fused CUDA:
      - x is CUDA float32 contiguous NCDHW
      - MaxPool3d: kernel_size == stride == maxpool_kernel_size, padding=0, ceil_mode=False
      - AdaptiveAvgPool3d((1,1,1)) == global average over pooled volume
      - clamp_min=0, clamp_max=1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=True,
        )
        self.scale = float(scale)
        self.maxpool_kernel_size = int(maxpool_kernel_size)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.custom_ops.mul_maxpool_gap_clamp_cuda(x, self.scale, self.maxpool_kernel_size)
        return x