import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension ----

matmul_gn_lrelu_sum_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
__device__ __forceinline__ float ro_load_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ro_load_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

__device__ __forceinline__ float lrelu_fwd(float x, float negative_slope) {
    return (x >= 0.0f) ? x : (x * negative_slope);
}

// Generic 256-thread kernel (baseline-like)
__device__ __forceinline__ float block_reduce_sum_256(float v) {
    __shared__ float smem[8]; // 8 warps
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < 8) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    out = __shfl_sync(0xffffffff, out, 0);
    return out;
}

__global__ __launch_bounds__(256, 2) void gn_fused_generic_kernel(
    const float* __restrict__ x,     // (N,C)
    float* __restrict__ out,         // (N,C)
    const float* __restrict__ gamma, // (C,)
    const float* __restrict__ beta,  // (C,)
    int N, int C, int G, int group_size,
    float eps, float negative_slope
) {
    int ng = (int)blockIdx.x; // 0..N*G-1
    int n = ng / G;
    int g = ng - n * G;

    int c0 = g * group_size;
    int base = n * C + c0;

    float sum = 0.0f;
    float sumsq = 0.0f;

    if ((group_size & 3) == 0) {
        int gs4 = group_size >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
        for (int i4 = (int)threadIdx.x; i4 < gs4; i4 += (int)blockDim.x) {
            float4 v = x4[i4];
            sum   += (v.x + v.y) + (v.z + v.w);
            sumsq += (v.x*v.x + v.y*v.y) + (v.z*v.z + v.w*v.w);
        }
    } else {
        for (int i = (int)threadIdx.x; i < group_size; i += (int)blockDim.x) {
            float v = x[base + i];
            sum   += v;
            sumsq += v * v;
        }
    }

    sum   = block_reduce_sum_256(sum);
    sumsq = block_reduce_sum_256(sumsq);

    float inv_gs = 1.0f / (float)group_size;
    float mean = sum * inv_gs;
    float ex2  = sumsq * inv_gs;
    float var  = ex2 - mean * mean;
    var = (var > 0.0f) ? var : 0.0f;
    float invstd = rsqrtf(var + eps);

    if ((group_size & 3) == 0) {
        int gs4 = group_size >> 2;
        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(out + base);

        for (int i4 = (int)threadIdx.x; i4 < gs4; i4 += (int)blockDim.x) {
            int c = c0 + (i4 << 2);
            float4 v = x4[i4];

            float g0 = ro_load_f32(gamma + c + 0);
            float g1 = ro_load_f32(gamma + c + 1);
            float g2 = ro_load_f32(gamma + c + 2);
            float g3 = ro_load_f32(gamma + c + 3);

            float b0 = ro_load_f32(beta + c + 0);
            float b1 = ro_load_f32(beta + c + 1);
            float b2 = ro_load_f32(beta + c + 2);
            float b3 = ro_load_f32(beta + c + 3);

            float y0 = (v.x - mean) * invstd; y0 = fmaf(y0, g0, b0); y0 = lrelu_fwd(y0, negative_slope); y0 = y0 + y0;
            float y1 = (v.y - mean) * invstd; y1 = fmaf(y1, g1, b1); y1 = lrelu_fwd(y1, negative_slope); y1 = y1 + y1;
            float y2 = (v.z - mean) * invstd; y2 = fmaf(y2, g2, b2); y2 = lrelu_fwd(y2, negative_slope); y2 = y2 + y2;
            float y3 = (v.w - mean) * invstd; y3 = fmaf(y3, g3, b3); y3 = lrelu_fwd(y3, negative_slope); y3 = y3 + y3;

            out4[i4] = make_float4(y0, y1, y2, y3);
        }
    } else {
        for (int i = (int)threadIdx.x; i < group_size; i += (int)blockDim.x) {
            int c = c0 + i;
            float v = x[base + i];
            float y = (v - mean) * invstd;
            y = fmaf(y, ro_load_f32(gamma + c), ro_load_f32(beta + c));
            y = lrelu_fwd(y, negative_slope);
            out[base + i] = y + y;
        }
    }
}

// Specialized fast path: group_size == 16, one warp per (n,g), persistent grid-stride over groups.
// No shared memory, no __syncthreads.
__global__ __launch_bounds__(32, 8) void gn_fused_gs16_warp_kernel(
    const float* __restrict__ x,     // (N,C)
    float* __restrict__ out,         // (N,C)
    const float* __restrict__ gamma, // (C,)
    const float* __restrict__ beta,  // (C,)
    int N, int C, int G,
    float eps, float negative_slope
) {
    int lane = (int)threadIdx.x & 31;
    int ng0 = (int)blockIdx.x;            // starting group index
    int stride = (int)gridDim.x;          // groups per grid-stride

    // Each block is exactly one warp (32 threads).
    for (int ng = ng0; ng < N * G; ng += stride) {
        int n = ng / G;
        int g = ng - n * G;

        int c0 = g << 4;         // g * 16
        int base = n * C + c0;

        const float* __restrict__ xp = x + base;
        float* __restrict__ op = out + base;
        const float* __restrict__ gp = gamma + c0;
        const float* __restrict__ bp = beta + c0;

        float v = (lane < 16) ? xp[lane] : 0.0f;
        float sum = warp_reduce_sum(v);
        float sumsq = warp_reduce_sum(v * v);

        float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / 16.0f);
        float ex2  = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / 16.0f);
        float var  = ex2 - mean * mean;
        var = (var > 0.0f) ? var : 0.0f;
        float invstd = rsqrtf(var + eps);

        if (lane < 16) {
            float y = (v - mean) * invstd;
            y = fmaf(y, ro_load_f32(gp + lane), ro_load_f32(bp + lane));
            y = lrelu_fwd(y, negative_slope);
            op[lane] = y + y;
        }
    }
}

torch::Tensor group_norm_leaky_relu_sum_forward_cuda(
    torch::Tensor x,              // (N, C)
    torch::Tensor gamma,          // (C,)
    torch::Tensor beta,           // (C,)
    int64_t num_groups,
    double eps,
    double negative_slope
) {
    TORCH_CHECK(x.is_cuda(), "group_norm_leaky_relu_sum_forward_cuda: x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "group_norm_leaky_relu_sum_forward_cuda: gamma/beta must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "group_norm_leaky_relu_sum_forward_cuda: x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32,
                "group_norm_leaky_relu_sum_forward_cuda: gamma/beta must be float32");
    TORCH_CHECK(x.is_contiguous(), "group_norm_leaky_relu_sum_forward_cuda: x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "group_norm_leaky_relu_sum_forward_cuda: gamma/beta must be contiguous");
    TORCH_CHECK(x.dim() == 2, "group_norm_leaky_relu_sum_forward_cuda: x must be 2D (N,C)");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "group_norm_leaky_relu_sum_forward_cuda: gamma/beta must be 1D");
    TORCH_CHECK(gamma.numel() == x.size(1) && beta.numel() == x.size(1), "group_norm_leaky_relu_sum_forward_cuda: gamma/beta must match C");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int G = (int)num_groups;
    TORCH_CHECK(G > 0 && (C % G) == 0, "group_norm_leaky_relu_sum_forward_cuda: num_groups must divide channels");
    int group_size = C / G;

    auto out = torch::empty_like(x);

    // Launch
    if (group_size == 16) {
        int total_groups = N * G;

        // Choose a grid that is large enough to fill the GPU but not absurdly large.
        // Persistent looping will cover all groups.
        int blocks = total_groups;
        // Clamp blocks to avoid excessive launch overhead on very large N*G.
        // 65535 is max gridDim.x for 1D grid on many platforms; stay within.
        if (blocks > 65535) blocks = 65535;

        int threads = 32;
        gn_fused_gs16_warp_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            N, C, G,
            (float)eps, (float)negative_slope
        );
    } else {
        int blocks = N * G;
        int threads = 256;
        gn_fused_generic_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            (const float*)gamma.data_ptr<float>(),
            (const float*)beta.data_ptr<float>(),
            N, C, G, group_size,
            (float)eps, (float)negative_slope
        );
    }

    return out;
}
"""

matmul_gn_lrelu_sum_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor group_norm_leaky_relu_sum_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    double negative_slope
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_group_norm_leaky_relu_sum_v7_gs16warp_persist",
    cpp_sources=matmul_gn_lrelu_sum_cpp_source,
    cuda_sources=matmul_gn_lrelu_sum_cuda_source,
    functions=["group_norm_leaky_relu_sum_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ---- Model wrapper ----

class ModelNew(nn.Module):
    """
    GEMM (nn.Linear) followed by a fused CUDA kernel:
    GroupNorm + affine + LeakyReLU + (x+x).
    Includes a fast path for group_size==16 (C/G==16) using a single-warp persistent kernel.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps, affine=True)
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.negative_slope = float(negative_slope)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if not x.is_contiguous():
            x = x.contiguous()
        w = self.gn.weight
        b = self.gn.bias
        if not w.is_contiguous():
            w = w.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        return self.custom_ops_lib.group_norm_leaky_relu_sum_forward_cuda(
            x, w, b, self.num_groups, self.eps, self.negative_slope
        )

# Keep the same input helpers for compatibility with the original harness.
batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]