import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: fused tanh + scaling + bias + maxpool(k, s=k) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

__device__ __forceinline__ float tanh_fast(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float ld_ro(const float* __restrict__ p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Supports maxpool with: dilation=1, padding=0, stride==kernel, ceil_mode=false
// Supports general kernel size k (small, e.g., 2..8), with an optimized path for k==4 using float4 row loads.
template<int K>
__global__ __launch_bounds__(256, 2) void tanh_scale_bias_maxpool_fwd_f32_kernel(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ bias,   // numel == 1 or C
    float* __restrict__ out,          // [N,C,Hp,Wp]
    int N, int C, int H, int W,
    int Hp, int Wp,
    int bias_numel,
    float scale
) {
    // 2D mapping:
    //  - threadIdx.x maps pw
    //  - blockIdx.y/threadIdx.y map (n,c,ph) flattened
    int pw = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (pw >= Wp) return;

    int ncp = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int NC = N * C;
    int total_ncp = NC * Hp;
    if (ncp >= total_ncp) return;

    int ph = ncp % Hp;
    int tmp = ncp / Hp;
    int c = tmp % C;
    int n = tmp / C;

    float b = (bias_numel == 1) ? ld_ro(bias) : ld_ro(bias + c);

    int ih0 = ph * K;
    int iw0 = pw * K;

    // base pointer at top-left of the pooling window
    const float* __restrict__ base = x + ((n * C + c) * H + ih0) * W + iw0;

    float m = -INFINITY;

    if constexpr (K == 4) {
        // Each row loads 4 contiguous floats. Always in-bounds in height by construction of Hp.
        // Width: iw0 + 3 < W guaranteed by construction of Wp.
        // Still keep a conservative fallback if something odd occurs.
        if (iw0 + 3 < W) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const float* __restrict__ rowp = base + r * W;

                // Use float4 load when aligned enough; otherwise scalar.
                // Alignment is not strictly required for correctness, but may help performance.
                uintptr_t addr = reinterpret_cast<uintptr_t>(rowp);
                float v0, v1, v2, v3;
                if ((addr & 0xF) == 0) {
                    float4 v = *reinterpret_cast<const float4*>(rowp);
                    v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
                } else {
                    v0 = rowp[0]; v1 = rowp[1]; v2 = rowp[2]; v3 = rowp[3];
                }

                v0 = tanh_fast(v0) * scale + b;
                v1 = tanh_fast(v1) * scale + b;
                v2 = tanh_fast(v2) * scale + b;
                v3 = tanh_fast(v3) * scale + b;

                m = fmaxf(m, v0);
                m = fmaxf(m, v1);
                m = fmaxf(m, v2);
                m = fmaxf(m, v3);
            }
        } else {
            // Safe fallback (should be unreachable with Hp/Wp computed as floor)
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const float* __restrict__ rowp = base + r * W;
                #pragma unroll
                for (int c0 = 0; c0 < 4; c0++) {
                    if (iw0 + c0 < W) {
                        float v = tanh_fast(rowp[c0]) * scale + b;
                        m = fmaxf(m, v);
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int r = 0; r < K; r++) {
            const float* __restrict__ rowp = base + r * W;
            #pragma unroll
            for (int c0 = 0; c0 < K; c0++) {
                float v = tanh_fast(rowp[c0]) * scale + b;
                m = fmaxf(m, v);
            }
        }
    }

    // out index: (((n*C + c)*Hp + ph)*Wp + pw)
    out[(((n * C + c) * Hp + ph) * Wp + pw)] = m;
}

torch::Tensor tanh_scale_bias_maxpool_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double scaling_factor,
    int64_t pool_kernel
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(pool_kernel >= 1 && pool_kernel <= 8, "supported pool_kernel in [1, 8]");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!bias.is_contiguous()) bias = bias.contiguous();

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    TORCH_CHECK(bias.numel() == 1 || bias.numel() == C,
                "bias must have numel 1 or C (supports shapes like (C), (C,1,1), (1,C,1,1))");

    const int K = (int)pool_kernel;
    const int S = K; // stride == kernel
    // ceil_mode=false, padding=0, dilation=1:
    // Hp = floor((H - K)/S) + 1 if H >= K else 0
    // Wp = floor((W - K)/S) + 1 if W >= K else 0
    int Hp = (H >= K) ? ((H - K) / S + 1) : 0;
    int Wp = (W >= K) ? ((W - K) / S + 1) : 0;

    auto out = torch::empty({N, C, Hp, Wp}, x.options());

    if (Hp == 0 || Wp == 0) return out;

    dim3 block;
    // block.x over Wp for coalesced writes, block.y for independent outputs
    // 32x8 = 256 threads
    block.x = 32;
    block.y = 8;
    block.z = 1;

    dim3 grid;
    grid.x = (Wp + block.x - 1) / block.x;
    int total_ncp = (N * C * Hp);
    grid.y = (total_ncp + block.y - 1) / block.y;
    grid.z = 1;

    const float* xp = (const float*)x.data_ptr<float>();
    const float* bp = (const float*)bias.data_ptr<float>();
    float* op = (float*)out.data_ptr<float>();
    float scale = (float)scaling_factor;
    int bias_numel = (int)bias.numel();

    switch (K) {
        case 1:
            tanh_scale_bias_maxpool_fwd_f32_kernel<1><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 2:
            tanh_scale_bias_maxpool_fwd_f32_kernel<2><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 3:
            tanh_scale_bias_maxpool_fwd_f32_kernel<3><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 4:
            tanh_scale_bias_maxpool_fwd_f32_kernel<4><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 5:
            tanh_scale_bias_maxpool_fwd_f32_kernel<5><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 6:
            tanh_scale_bias_maxpool_fwd_f32_kernel<6><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 7:
            tanh_scale_bias_maxpool_fwd_f32_kernel<7><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        case 8:
            tanh_scale_bias_maxpool_fwd_f32_kernel<8><<<grid, block>>>(xp, bp, op, N, C, H, W, Hp, Wp, bias_numel, scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported pool_kernel");
    }

    return out;
}
"""

cpp_src = r"""
torch::Tensor tanh_scale_bias_maxpool_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double scaling_factor,
    int64_t pool_kernel
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_tanh_scaling_bias_add_max_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["tanh_scale_bias_maxpool_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the fused op (safe, semantics-matched for this model) ---------

class ModelNew(nn.Module):
    """
    Convolution followed by fused CUDA op computing:
      y = maxpool_k( tanh(conv(x)) * scaling_factor + bias )
    for MaxPool2d configured as: kernel_size=k, stride=k, padding=0, dilation=1, ceil_mode=False.
    Falls back to PyTorch ops if configuration deviates.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.pool_kernel_size = int(pool_kernel_size)
        self.max_pool = nn.MaxPool2d(self.pool_kernel_size)  # default: stride=None->k, padding=0, dilation=1, ceil_mode=False
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        bias = self.bias.to(device=x.device, dtype=x.dtype)

        # Only fuse for the exact semantics we implement (matches nn.MaxPool2d(k) defaults)
        # and for float32 CUDA contiguous.
        can_fuse = (
            x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and
            isinstance(self.max_pool, nn.MaxPool2d) and
            self.max_pool.kernel_size == self.pool_kernel_size and
            (self.max_pool.stride is None or self.max_pool.stride == self.pool_kernel_size) and
            self.max_pool.padding == 0 and self.max_pool.dilation == 1 and
            self.max_pool.ceil_mode is False
        )

        if can_fuse:
            return self.custom_ops_lib.tanh_scale_bias_maxpool_cuda(x, bias, self.scaling_factor, self.pool_kernel_size)

        # Fallback (generic)
        x = torch.tanh(x) * self.scaling_factor
        x = x + bias
        x = self.max_pool(x)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]