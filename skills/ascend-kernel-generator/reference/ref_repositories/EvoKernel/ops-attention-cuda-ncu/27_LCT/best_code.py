import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // fast math enabled via --use_fast_math (expf maps to __expf)
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Kernel 1: compute gate[n, c] for all channels.
// Grid: (N, G). Block: e.g., 128 threads.
// For each (n,g), iterate channels in group. For each channel, compute avg over HW,
// then reduce across channels to get mean/var, then compute per-channel gate.
__global__ __launch_bounds__(128, 4)
void lct_gate_kernel(
    const float* __restrict__ x,    // [N,C,H,W] contiguous
    const float* __restrict__ w,    // [C]
    const float* __restrict__ b,    // [C]
    float* __restrict__ gate,       // [N,C]
    int N, int C, int H, int W, int G,
    float eps
) {
    int n = (int)blockIdx.x;
    int g = (int)blockIdx.y;

    int c_per_g = C / G;
    int hw = H * W;

    // warp-level partial reductions for sum(avg) and sum(avg^2)
    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;

    // We will recompute avg per channel twice (once for stats, once for gate) to avoid
    // storing avg array in shared memory. This cuts shared memory and registers.
    // Cost is extra reads; with high cache hit rates and reduced stalls, this can be faster overall.

    // First pass: compute sums across channels in group
    for (int ci = threadIdx.x; ci < c_per_g; ci += blockDim.x) {
        int c = g * c_per_g + ci;
        int base = ((n * C + c) * H) * W;

        float s0 = 0.0f;
        // Unroll by 4 for ILP (hw=49 typically; tail handled)
        int i = 0;
        for (; i + 3 < hw; i += 4) {
            float v0 = x[base + i + 0];
            float v1 = x[base + i + 1];
            float v2 = x[base + i + 2];
            float v3 = x[base + i + 3];
            s0 += v0 + v1 + v2 + v3;
        }
        for (; i < hw; i++) s0 += x[base + i];

        float a = s0 * (1.0f / (float)hw);
        thread_sum += a;
        thread_sum2 += a * a;
    }

    // Reduce across block using warp reductions + shared for warp sums
    float sum_w = warp_sum(thread_sum);
    float sum2_w = warp_sum(thread_sum2);

    __shared__ float sh_sum[32];   // up to 32 warps
    __shared__ float sh_sum2[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        sh_sum[warp_id] = sum_w;
        sh_sum2[warp_id] = sum2_w;
    }
    __syncthreads();

    float sum = 0.0f;
    float sum2 = 0.0f;
    if (warp_id == 0) {
        float v1 = (lane < num_warps) ? sh_sum[lane] : 0.0f;
        float v2 = (lane < num_warps) ? sh_sum2[lane] : 0.0f;
        v1 = warp_sum(v1);
        v2 = warp_sum(v2);
        if (lane == 0) {
            sh_sum[0] = v1;
            sh_sum2[0] = v2;
        }
    }
    __syncthreads();
    sum = sh_sum[0];
    sum2 = sh_sum2[0];

    float inv_c = 1.0f / (float)c_per_g;
    float mean = sum * inv_c;
    float mean_x2 = sum2 * inv_c;
    float var = mean_x2 - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = rsqrtf(var + eps);

    // Second pass: compute gate per channel and store gate[n,c]
    for (int ci = threadIdx.x; ci < c_per_g; ci += blockDim.x) {
        int c = g * c_per_g + ci;
        int base = ((n * C + c) * H) * W;

        float s0 = 0.0f;
        int i = 0;
        for (; i + 3 < hw; i += 4) {
            float v0 = x[base + i + 0];
            float v1 = x[base + i + 1];
            float v2 = x[base + i + 2];
            float v3 = x[base + i + 3];
            s0 += v0 + v1 + v2 + v3;
        }
        for (; i < hw; i++) s0 += x[base + i];

        float a = s0 * (1.0f / (float)hw);
        float yn = (a - mean) * inv_std;
        float z = fmaf(w[c], yn, b[c]);
        float gval = sigmoidf_fast(z);
        gate[n * C + c] = gval;
    }
}

// Kernel 2: apply gate to x
// out[idx] = x[idx] * gate[n*C + c], where idx maps to n,c,h,w
__global__ __launch_bounds__(256, 4)
void lct_apply_kernel(
    const float* __restrict__ x,     // [N,C,H,W]
    const float* __restrict__ gate,  // [N,C]
    float* __restrict__ out,         // [N,C,H,W]
    int N, int C, int HW
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * HW;
    if (idx >= total) return;

    int t = idx / HW;
    int n = t / C;
    int c = t - n * C;
    float g = gate[n * C + c];
    out[idx] = x[idx] * g;
}

torch::Tensor lct_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t groups, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda() && b.is_cuda(), "w and b must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "w and b must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW contiguous)");
    TORCH_CHECK(w.is_contiguous() && b.is_contiguous(), "w and b must be contiguous");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int G = (int)groups;
    TORCH_CHECK(C % G == 0, "channels must be divisible by groups");

    auto out = torch::empty_like(x);
    auto gate = torch::empty({N, C}, x.options()); // float32 on CUDA

    // Kernel 1 launch
    dim3 grid1(N, G, 1);

    int c_per_g = C / G;
    int block1 = 128;
    // For small groups, avoid too many idle threads
    if (c_per_g <= 32) block1 = 32;
    else if (c_per_g <= 64) block1 = 64;
    else block1 = 128;

    lct_gate_kernel<<<grid1, block1, 0>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        gate.data_ptr<float>(),
        N, C, H, W, G,
        (float)eps
    );

    // Kernel 2 launch
    int HW = H * W;
    int total = N * C * HW;
    int block2 = 256;
    int grid2 = (total + block2 - 1) / block2;

    lct_apply_kernel<<<grid2, block2, 0>>>(
        x.data_ptr<float>(),
        gate.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, HW
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor lct_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t groups, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lct_opt2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["lct_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    verbose=False,
)


class ModelNew(nn.Module):
    """LCT block using optimized CUDA kernels for forward (CUDA float32 contiguous)."""
    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.w = nn.Parameter(torch.ones(channels, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or (x.dtype != torch.float32) or (not x.is_contiguous()):
            batch_size = x.shape[0]
            y = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, self.groups, -1)
            mean = y.mean(dim=-1, keepdim=True)
            mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
            var = mean_x2 - mean ** 2
            y_norm = (y - mean) / torch.sqrt(var + self.eps)
            y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
            y_norm = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
            y_norm = torch.sigmoid(y_norm)
            return x * y_norm.expand_as(x)

        return custom_ops_lib.lct_forward_cuda(
            x,
            self.w.contiguous(),
            self.b.contiguous(),
            int(self.groups),
            float(self.eps),
        )