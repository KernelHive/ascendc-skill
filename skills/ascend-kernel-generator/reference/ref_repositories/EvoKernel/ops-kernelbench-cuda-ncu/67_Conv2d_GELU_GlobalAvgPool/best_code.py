import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float fast_gelu(float x) {
    // x * sigmoid(1.702x) (fast approximation)
    float y = 1.702f * x;
    float s = 1.0f / (1.0f + __expf(-y));
    return x * s;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // full mask
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// One block computes one (n,c) output.
// blockDim.x should be a multiple of 32.
__global__ void gelu_global_avg_pool_fused_kernel(
    const float* __restrict__ x,  // [N,C,HW] flattened
    float* __restrict__ out,       // [N,C]
    int N, int C, int HW,
    float inv_hw
) {
    int c = (int)blockIdx.x;
    int n = (int)blockIdx.y;
    if (c >= C || n >= N) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (int)blockDim.x >> 5;

    const float* base = x + ((n * C + c) * HW);

    float acc = 0.0f;

    // Vectorized float4 loads over HW
    int vecN = HW >> 2; // /4
    const float4* base4 = reinterpret_cast<const float4*>(base);

    for (int v4 = tid; v4 < vecN; v4 += (int)blockDim.x) {
        float4 v = __ldg(base4 + v4);
        acc += fast_gelu(v.x);
        acc += fast_gelu(v.y);
        acc += fast_gelu(v.z);
        acc += fast_gelu(v.w);
    }

    // Tail
    int start = vecN << 2;
    for (int i = start + tid; i < HW; i += (int)blockDim.x) {
        acc += fast_gelu(__ldg(base + i));
    }

    // Reduce within warp
    acc = warp_reduce_sum(acc);

    // Shared memory for warp sums (max 32 warps -> 32 floats)
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    // Final reduce by warp 0
    float sum = 0.0f;
    if (warp == 0) {
        sum = (tid < num_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            out[n * C + c] = sum * inv_hw;
        }
    }
}

torch::Tensor gelu_global_avg_pool_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_global_avg_pool_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "gelu_global_avg_pool_cuda: only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "gelu_global_avg_pool_cuda: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "gelu_global_avg_pool_cuda: expected 4D NCHW input");

    int64_t N64 = x.size(0), C64 = x.size(1), H64 = x.size(2), W64 = x.size(3);
    TORCH_CHECK(N64 <= INT32_MAX && C64 <= INT32_MAX && H64 <= INT32_MAX && W64 <= INT32_MAX,
                "gelu_global_avg_pool_cuda: dims too large");
    int N = (int)N64, C = (int)C64;
    int HW = (int)(H64 * W64);

    auto out = torch::empty({N, C}, x.options());

    float inv_hw = 1.0f / (float)HW;

    // 256 threads (8 warps): good balance for HW-sized reductions; no atomics now.
    constexpr int THREADS = 256;
    dim3 block(THREADS, 1, 1);
    dim3 grid(C, N, 1);

    gelu_global_avg_pool_fused_kernel<<<grid, block>>>(
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        N, C, HW, inv_hw
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor gelu_global_avg_pool_cuda(torch::Tensor x);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv2d_gelu_gap_opt4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gelu_global_avg_pool_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Convolution -> fused (GELU + global average pooling) using a custom CUDA kernel.
    Output shape: (batch_size, out_channels)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        y = self.conv(x)
        if y.dtype != torch.float32:
            y = y.float()
        if not y.is_contiguous():
            y = y.contiguous()

        return self.custom_ops.gelu_global_avg_pool_cuda(y)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]