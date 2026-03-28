import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
#include <math_constants.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ro_load(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float fmax2(float a, float b) { return a > b ? a : b; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, mask);
    }
    return v;
}

// Block reduction: warp shuffles + tiny shared array of warp sums.
// THREADS must be multiple of 32 and <= 1024.
template<int THREADS>
__device__ __forceinline__ float block_reduce_sum_warp(float v) {
    static_assert(THREADS % 32 == 0, "THREADS must be multiple of 32");
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[WARPS];
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        float x = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) warp_sums[0] = x;
    }
    __syncthreads();
    out = warp_sums[0];
    return out;
}

// Fused tail: (o/div) -> maxpool -> global avg -> +bias -> sum over C into out[n]
// Mapping: one block per (n,c), atomicAdd to out[n] (keeps high parallelism).
template<int THREADS>
__global__ __launch_bounds__(THREADS, 3)
void div_pool_gavg_bias_sum_nc_kernel_v5(
    const float* __restrict__ o,   // [N,C,D,H,W] contiguous NCDHW
    const float* __restrict__ b,   // [C]
    float* __restrict__ out,       // [N] (flattened)
    int N, int C, int D, int H, int W,
    float inv_div,
    int pd, int ph, int pw,
    int PD, int PH, int PW
) {
    int nc = (int)blockIdx.x;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    int pooled_count = PD * PH * PW;
    float inv_pooled = pooled_count > 0 ? (1.0f / (float)pooled_count) : 0.0f;

    // Decide whether we can use 32-bit indexing inside the per-(n,c) volume.
    // volume = D*H*W fits int32 for typical sizes here; use int64 if not.
    int64_t vol64 = (int64_t)D * (int64_t)H * (int64_t)W;
    bool use_i32 = (vol64 <= (int64_t)0x7fffffff);

    float partial = 0.0f;

    if (use_i32) {
        int vol = (int)vol64;
        int base = ((int)n * C + c) * vol;

        // Specialized fast path: pool = 2x2x2 and exact divisibility assumed.
        if (pd == 2 && ph == 2 && pw == 2) {
            // Iterate pooled cells using structured loops to avoid div/mod.
            // Flatten pooled index p = ((pz*PH + py)*PW + px).
            int tid = (int)threadIdx.x;
            for (int p = tid; p < pooled_count; p += THREADS) {
                int t = p;
                int px = t % PW; t /= PW;
                int py = t % PH;
                int pz = t / PH;

                int oz0 = pz << 1; // *2
                int oy0 = py << 1;
                int ox0 = px << 1;

                float m = -CUDART_INF_F;

                // Unrolled 2x2x2 max pool; keep address math simple (int).
#pragma unroll
                for (int kz = 0; kz < 2; ++kz) {
                    int z = oz0 + kz;
                    int zoff = z * (H * W);
#pragma unroll
                    for (int ky = 0; ky < 2; ++ky) {
                        int y = oy0 + ky;
                        int off = base + zoff + y * W + ox0;
                        const float* ptr = o + off;
                        float v0 = ptr[0] * inv_div;
                        float v1 = ptr[1] * inv_div;
                        m = fmax2(m, fmax2(v0, v1));
                    }
                }

                if (!isfinite(m)) m = 0.0f;
                partial += m;
            }
        } else {
            int tid = (int)threadIdx.x;
            for (int p = tid; p < pooled_count; p += THREADS) {
                // decode p -> (pz,py,px) (generic path retains div/mod)
                int t = p;
                int px = t % PW; t /= PW;
                int py = t % PH; t /= PH;
                int pz = t;

                int oz0 = pz * pd;
                int oy0 = py * ph;
                int ox0 = px * pw;

                float m = -CUDART_INF_F;

                for (int kz = 0; kz < pd; ++kz) {
                    int z = oz0 + kz;
                    int zoff = z * (H * W);
                    for (int ky = 0; ky < ph; ++ky) {
                        int y = oy0 + ky;
                        int off = base + zoff + y * W + ox0;
                        const float* ptr = o + off;
                        if (pw == 2) {
                            float v0 = ptr[0] * inv_div;
                            float v1 = ptr[1] * inv_div;
                            m = fmax2(m, fmax2(v0, v1));
                        } else {
                            for (int kx = 0; kx < pw; ++kx) {
                                float v = ptr[kx] * inv_div;
                                m = fmax2(m, v);
                            }
                        }
                    }
                }

                if (!isfinite(m)) m = 0.0f;
                partial += m;
            }
        }
    } else {
        // Safe int64 path (unlikely for provided shapes)
        int64_t strideC = vol64;
        int64_t base = ((int64_t)n * (int64_t)C + (int64_t)c) * strideC;

        int tid = (int)threadIdx.x;
        for (int p = tid; p < pooled_count; p += THREADS) {
            int t = p;
            int px = t % PW; t /= PW;
            int py = t % PH; t /= PH;
            int pz = t;

            int oz0 = pz * pd;
            int oy0 = py * ph;
            int ox0 = px * pw;

            float m = -CUDART_INF_F;

            for (int kz = 0; kz < pd; ++kz) {
                int z = oz0 + kz;
                int64_t zoff = (int64_t)z * (int64_t)H * (int64_t)W;
                for (int ky = 0; ky < ph; ++ky) {
                    int y = oy0 + ky;
                    int64_t off = base + zoff + (int64_t)y * (int64_t)W + (int64_t)ox0;
                    const float* ptr = o + off;
                    if (pw == 2) {
                        float v0 = ptr[0] * inv_div;
                        float v1 = ptr[1] * inv_div;
                        m = fmax2(m, fmax2(v0, v1));
                    } else {
                        for (int kx = 0; kx < pw; ++kx) {
                            float v = ptr[kx] * inv_div;
                            m = fmax2(m, v);
                        }
                    }
                }
            }

            if (!isfinite(m)) m = 0.0f;
            partial += m;
        }
    }

    float sum = block_reduce_sum_warp<THREADS>(partial);
    if (threadIdx.x == 0) {
        float mean_acc = sum * inv_pooled;
        float bv = ro_load(b + c);
        atomicAdd(out + n, mean_acc + bv);
    }
}

static torch::Tensor bias_to_C(torch::Tensor bias, int64_t C) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    auto b = bias.contiguous();
    if (b.dim() == 1) {
        TORCH_CHECK(b.size(0) == C, "bias [C] expected");
        return b;
    } else if (b.dim() == 4) {
        TORCH_CHECK(b.size(0) == C && b.size(1) == 1 && b.size(2) == 1 && b.size(3) == 1,
                    "bias [C,1,1,1] expected");
        return b.view({C});
    } else if (b.dim() == 5) {
        TORCH_CHECK(b.size(0) == 1 && b.size(1) == C && b.size(2) == 1 && b.size(3) == 1 && b.size(4) == 1,
                    "bias [1,C,1,1,1] expected");
        return b.view({C});
    } else {
        TORCH_CHECK(false, "Unsupported bias shape; expected [C], [C,1,1,1], or [1,C,1,1,1]");
    }
}

torch::Tensor div_pool_gavg_bias_sum_cuda(
    torch::Tensor o,        // [N,C,D,H,W]
    torch::Tensor bias,     // broadcastable to [N,C,1,1,1]
    double divisor,
    int64_t pool_d,
    int64_t pool_h,
    int64_t pool_w
) {
    TORCH_CHECK(o.is_cuda(), "o must be CUDA");
    TORCH_CHECK(o.dtype() == torch::kFloat32, "o must be float32");
    TORCH_CHECK(o.dim() == 5, "o must be [N,C,D,H,W]");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous (NCDHW)");
    TORCH_CHECK(divisor != 0.0, "divisor must be non-zero");

    int64_t N = o.size(0);
    int64_t C = o.size(1);
    int64_t D = o.size(2);
    int64_t H = o.size(3);
    int64_t W = o.size(4);

    int pd = (int)pool_d;
    int ph = (int)pool_h;
    int pw = (int)pool_w;
    TORCH_CHECK(pd > 0 && ph > 0 && pw > 0, "pool sizes must be > 0");

    TORCH_CHECK(D >= pd && H >= ph && W >= pw, "pool kernel must be <= input sizes");
    TORCH_CHECK((D % pd) == 0 && (H % ph) == 0 && (W % pw) == 0,
                "Fused kernel assumes exact divisibility: D%pd==0, H%ph==0, W%pw==0 (stride==kernel, padding==0, ceil_mode=False)");

    int PD = (int)(D / pd);
    int PH = (int)(H / ph);
    int PW = (int)(W / pw);
    TORCH_CHECK(PD > 0 && PH > 0 && PW > 0, "pooled sizes must be positive");

    auto b1 = bias_to_C(bias, C);

    auto out = torch::empty({N, 1, 1, 1}, o.options());
    auto out1d = out.view({N});
    out1d.zero_();

    float inv_div = (float)(1.0 / divisor);

    dim3 grid((unsigned)(N * C));
    constexpr int THREADS = 256;
    dim3 block(THREADS);

    div_pool_gavg_bias_sum_nc_kernel_v5<THREADS><<<grid, block>>>(
        (const float*)o.data_ptr<float>(),
        (const float*)b1.data_ptr<float>(),
        (float*)out1d.data_ptr<float>(),
        (int)N, (int)C, (int)D, (int)H, (int)W,
        inv_div,
        pd, ph, pw,
        PD, PH, PW
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor div_pool_gavg_bias_sum_cuda(
    torch::Tensor o,
    torch::Tensor bias,
    double divisor,
    int64_t pool_d,
    int64_t pool_h,
    int64_t pool_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_div_pool_gavg_bias_sum_v5",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["div_pool_gavg_bias_sum_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Conv3d in PyTorch/cuDNN + fused CUDA for:
      out = sum_dim1( adaptive_avg_pool3d(max_pool3d(conv(x)/divisor)) + bias )

    Output: [N,1,1,1]
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)

        if not (isinstance(pool_size, (tuple, list)) and len(pool_size) == 3):
            raise ValueError("pool_size must be a 3-tuple")
        self.pool_d, self.pool_h, self.pool_w = int(pool_size[0]), int(pool_size[1]), int(pool_size[2])

        self.bias = nn.Parameter(torch.randn(*bias_shape, dtype=torch.float32))
        self.sum_dim = int(sum_dim)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)

        if o.is_cuda and o.dtype == torch.float32 and self.sum_dim == 1:
            if not o.is_contiguous():
                o = o.contiguous()
            return self.custom_ops.div_pool_gavg_bias_sum_cuda(
                o, self.bias, float(self.divisor),
                int(self.pool_d), int(self.pool_h), int(self.pool_w)
            )

        y = o / self.divisor
        y = nn.functional.max_pool3d(
            y,
            kernel_size=(self.pool_d, self.pool_h, self.pool_w),
            stride=(self.pool_d, self.pool_h, self.pool_w),
            padding=0,
            dilation=1,
            ceil_mode=False,
        )
        y = nn.functional.adaptive_avg_pool3d(y, (1, 1, 1))
        y = y + self.bias
        y = torch.sum(y, dim=self.sum_dim)
        return y