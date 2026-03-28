import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Notes on layout:
# We allocate a contiguous buffer of shape [N, C] (row-major) for coalesced stores.
# Then we return a view with shape [1, C, N, 1] using as_strided without copying:
# out_view[0, c, n, 0] maps to out_nc[n, c].
# This preserves the required output semantics and keeps the write path coalesced.

fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits>

__device__ __forceinline__ float fmin2(float a, float b) { return a < b ? a : b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}
__device__ __forceinline__ float warp_min(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmin2(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void gn_rowmin_bias_fused_coalesced_kernel(
    const float* __restrict__ x,      // [N,C] contiguous row-major
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    const float* __restrict__ bias,   // [C]
    float* __restrict__ out_nc,       // [N,C] contiguous row-major
    int N, int C, int G,
    float eps
) {
    int n = (int)blockIdx.x;
    if (n >= N) return;

    constexpr int WARP = 32;
    int tid  = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp = tid >> 5;
    int num_warps = THREADS / WARP;

    int group_size = C / G;
    const float* xrow = x + (int64_t)n * C;

    // One float per group for group minima
    extern __shared__ float shmem[];
    float* sh_gmin = shmem; // size >= G

    // Each warp processes groups g = warp, warp+num_warps, ...
    // For each group: compute mean/var and group min using only that warp.
    for (int g = warp; g < G; g += num_warps) {
        int c0 = g * group_size;

        // group_size expected small; handle generic anyway.
        float sum = 0.0f, sumsq = 0.0f;

        // Map lanes to group channels: each lane may do 0/1 elements when group_size<=32.
        // For group_size=16, lanes [0..15] participate.
        for (int i = lane; i < group_size; i += WARP) {
            float v = xrow[c0 + i];
            sum += v;
            sumsq += v * v;
        }

        sum = warp_sum(sum);
        sumsq = warp_sum(sumsq);

        float mean = __shfl_sync(0xffffffff, sum / (float)group_size, 0);
        float var  = __shfl_sync(0xffffffff, sumsq / (float)group_size, 0) - mean * mean;
        var = var < 0.0f ? 0.0f : var;
        float inv_std = rsqrtf(var + eps);

        float local_min = INFINITY;
        for (int i = lane; i < group_size; i += WARP) {
            int c = c0 + i;
            float v = xrow[c];
            float y = (v - mean) * inv_std;
            y = y * ldg_f32(gamma + c) + ldg_f32(beta + c);
            local_min = fmin2(local_min, y);
        }
        float gmin = warp_min(local_min);
        if (lane == 0) sh_gmin[g] = gmin;
    }

    __syncthreads();

    // Reduce sh_gmin[0..G-1] to a single row_min.
    // Parallel reduction over G floats using all THREADS threads.
    float vmin = INFINITY;
    for (int i = tid; i < G; i += THREADS) vmin = fmin2(vmin, sh_gmin[i]);

    // Block-wide min reduction
    // First reduce within warp, then across warps through shared.
    vmin = warp_min(vmin);

    __shared__ float sh_wmin[8]; // THREADS assumed <=256 => <=8 warps
    if (lane == 0) sh_wmin[warp] = vmin;
    __syncthreads();

    float row_min = INFINITY;
    if (warp == 0) {
        float t = (lane < num_warps) ? sh_wmin[lane] : INFINITY;
        t = warp_min(t);
        if (lane == 0) sh_wmin[0] = t;
    }
    __syncthreads();
    row_min = sh_wmin[0];

    // Coalesced writeout: out_nc[n, c] = row_min + bias[c]
    // Use float4 vectorization when possible. out_nc is contiguous row-major so c dimension is contiguous.
    float* outrow = out_nc + (int64_t)n * C;

    int vecC = C >> 2; // number of float4
    int tid4 = tid;
    for (int i4 = tid4; i4 < vecC; i4 += THREADS) {
        int c = i4 << 2;

        float4 b4;
        b4.x = ldg_f32(bias + c + 0);
        b4.y = ldg_f32(bias + c + 1);
        b4.z = ldg_f32(bias + c + 2);
        b4.w = ldg_f32(bias + c + 3);

        float4 o4;
        o4.x = row_min + b4.x;
        o4.y = row_min + b4.y;
        o4.z = row_min + b4.z;
        o4.w = row_min + b4.w;

        *reinterpret_cast<float4*>(outrow + c) = o4;
    }

    // Tail (if C not divisible by 4)
    for (int c = (vecC << 2) + tid; c < C; c += THREADS) {
        outrow[c] = row_min + ldg_f32(bias + c);
    }
}

torch::Tensor gemm_group_norm_min_bias_add_cuda(
    torch::Tensor x,          // [N, C]
    torch::Tensor gamma,      // [C]
    torch::Tensor beta,       // [C]
    int64_t num_groups,
    double eps,
    torch::Tensor bias_4d     // [1, C, 1, 1]
) {
    TORCH_CHECK(x.is_cuda(), "op: x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda() && bias_4d.is_cuda(), "op: gamma/beta/bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "op: x must be float32");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && bias_4d.scalar_type() == at::kFloat,
                "op: gamma/beta/bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "op: x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous() && bias_4d.is_contiguous(), "op: gamma/beta/bias must be contiguous");
    TORCH_CHECK(x.dim() == 2, "op: x must be 2D [N, C]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "op: gamma/beta must be 1D [C]");
    TORCH_CHECK(bias_4d.dim() == 4, "op: bias must be 4D [1, C, 1, 1]");
    TORCH_CHECK(bias_4d.size(0) == 1 && bias_4d.size(2) == 1 && bias_4d.size(3) == 1, "op: bias must be [1, C, 1, 1]");

    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    int64_t G64 = num_groups;

    TORCH_CHECK(G64 > 0, "op: num_groups must be > 0");
    TORCH_CHECK(C64 % G64 == 0, "op: C must be divisible by num_groups");
    TORCH_CHECK(gamma.numel() == C64 && beta.numel() == C64, "op: gamma/beta must be [C]");
    TORCH_CHECK(bias_4d.size(1) == C64, "op: bias second dim must be C");

    int N = (int)N64;
    int C = (int)C64;
    int G = (int)G64;

    auto bias_flat = bias_4d.view({C});

    // Write into [N,C] contiguous for coalescing, then return a strided view [1,C,N,1].
    auto out_nc = torch::empty({N, C}, x.options());

    constexpr int THREADS = 256; // 8 warps: good balance for occupancy vs per-row work
    dim3 block(THREADS, 1, 1);
    dim3 grid(N, 1, 1);

    size_t shmem_bytes = (size_t)G * sizeof(float);

    gn_rowmin_bias_fused_coalesced_kernel<THREADS><<<grid, block, shmem_bytes>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)gamma.data_ptr<float>(),
        (const float*)beta.data_ptr<float>(),
        (const float*)bias_flat.data_ptr<float>(),
        (float*)out_nc.data_ptr<float>(),
        N, C, G,
        (float)eps
    );

    // View as [1, C, N, 1] without copy:
    // out_view[0, c, n, 0] = out_nc[n, c]
    // Strides for out_nc [N,C]: strideN=C, strideC=1
    // Desired [1,C,N,1] strides: [C*N, 1, C, 1] but to map (c,n) -> n*C + c:
    // index = c*1 + n*C, so strides should be: s0=0 (or any), s1=1, s2=C, s3=0.
    // Use as_strided with sizes [1,C,N,1] and strides [C*N, 1, C, 1] would map to n with stride N? No.
    // Correct is: offset = c*1 + n*C; thus strides = [0, 1, C, 0] (last dim size 1).
    // PyTorch requires non-negative strides; 0 is allowed for broadcast-like dims.
    auto out_view = out_nc.as_strided({1, C, N, 1}, {0, 1, (int64_t)C, 0});

    return out_view;
}
"""

fused_cpp_source = r"""
torch::Tensor gemm_group_norm_min_bias_add_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double eps,
    torch::Tensor bias_4d
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gnmin_v6",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["gemm_group_norm_min_bias_add_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Linear (GEMM) + fused GroupNorm + Min(dim=1) + BiasAdd (broadcast) via custom CUDA op.
    Output is a view with shape [1, C, N, 1] (no extra copy), matching bias broadcast [1,C,1,1].
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = int(num_groups)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)

        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        if not gamma.is_cuda:
            gamma = gamma.cuda()
        if not beta.is_cuda:
            beta = beta.cuda()
        if gamma.dtype != torch.float32:
            gamma = gamma.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()

        bias = self.bias
        if not bias.is_cuda:
            bias = bias.cuda()
        if bias.dtype != torch.float32:
            bias = bias.float()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        C = x.size(1)
        if bias.dim() != 4 or bias.size(0) != 1 or bias.size(1) != C or bias.size(2) != 1 or bias.size(3) != 1:
            bias = bias.view(1, C, 1, 1).contiguous()

        eps = float(self.group_norm.eps)

        return self.custom_ops_lib.gemm_group_norm_min_bias_add_cuda(
            x, gamma, beta, self.num_groups, eps, bias
        )