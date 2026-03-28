import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: optimized two-stage (reduce -> broadcast) ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    // tanh approximation
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float t = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 2) void reduce_mean_gelu_kernel_opt(
    const float* __restrict__ gemm_out,   // [B, F]
    const float* __restrict__ subtract,   // [F]
    float* __restrict__ scalar_out,       // [B]
    int B, int F
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    const float* row = gemm_out + (int64_t)b * (int64_t)F;

    float acc = 0.0f;

    // Prefer float4 vector path when aligned and F divisible by 4
    uintptr_t row_addr = (uintptr_t)row;
    uintptr_t sub_addr = (uintptr_t)subtract;
    bool vec_ok = ((F & 3) == 0) && ((row_addr & 0xF) == 0) && ((sub_addr & 0xF) == 0);

    if (vec_ok) {
        const float4* __restrict__ row4 = reinterpret_cast<const float4*>(row);
        const float4* __restrict__ sub4 = reinterpret_cast<const float4*>(subtract);
        int F4 = F >> 2;

        // Each thread processes float4s in a coalesced stride
        for (int j4 = tid; j4 < F4; j4 += THREADS) {
            float4 a = __ldg((const float4*)(row4 + j4));
            float4 s = __ldg((const float4*)(sub4 + j4));
            // sum(a - s)
            acc += (a.x - s.x) + (a.y - s.y) + (a.z - s.z) + (a.w - s.w);
        }
    } else {
        for (int j = tid; j < F; j += THREADS) {
            acc += __ldg(row + j) - __ldg(subtract + j);
        }
    }

    // Reduce within warp
    acc = warp_sum(acc);

    // Reduce across warps using shared memory
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[WARPS];
    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();

    // Warp 0 finalizes
    if (wid == 0) {
        float v = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            float mean = v * (1.0f / (float)F);
            scalar_out[b] = gelu_approx(mean);
        }
    }
}

// 2D grid-stride broadcast: vectorized float4 path + safe scalar fallback.
// grid.x = B (rows), grid.y = small factor to increase CTAs in-flight and hide mem latency.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4) void broadcast_add_kernel_2d_opt(
    const float* __restrict__ residual, // [B, R]
    const float* __restrict__ scalar,   // [B]
    float* __restrict__ out,            // [B, R]
    int B, int R
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    int tid = (int)threadIdx.x;
    float s = __ldg(scalar + b);

    const float* rptr = residual + (int64_t)b * (int64_t)R;
    float* optr = out + (int64_t)b * (int64_t)R;

    uintptr_t raddr = (uintptr_t)rptr;
    uintptr_t oaddr = (uintptr_t)optr;
    bool vec_ok = ((R & 3) == 0) && ((raddr & 0xF) == 0) && ((oaddr & 0xF) == 0);

    int y = (int)blockIdx.y;
    int gy = (int)gridDim.y;

    if (vec_ok) {
        const float4* __restrict__ r4 = reinterpret_cast<const float4*>(rptr);
        float4* __restrict__ o4 = reinterpret_cast<float4*>(optr);
        int R4 = R >> 2;

        // 2D grid-stride over vector columns
        int start = y * THREADS + tid;
        int step  = gy * THREADS;

        for (int i4 = start; i4 < R4; i4 += step) {
            float4 v = r4[i4];
            v.x += s; v.y += s; v.z += s; v.w += s;
            o4[i4] = v;
        }
    } else {
        int start = y * THREADS + tid;
        int step  = gy * THREADS;

        for (int i = start; i < R; i += step) {
            optr[i] = rptr[i] + s;
        }
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor gemm_out, torch::Tensor subtract, torch::Tensor residual) {
    TORCH_CHECK(gemm_out.is_cuda(), "fused_forward_cuda: gemm_out must be CUDA");
    TORCH_CHECK(subtract.is_cuda(), "fused_forward_cuda: subtract must be CUDA");
    TORCH_CHECK(residual.is_cuda(), "fused_forward_cuda: residual must be CUDA");

    TORCH_CHECK(gemm_out.scalar_type() == torch::kFloat32, "gemm_out must be float32");
    TORCH_CHECK(subtract.scalar_type() == torch::kFloat32, "subtract must be float32");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");

    TORCH_CHECK(gemm_out.is_contiguous(), "gemm_out must be contiguous");
    TORCH_CHECK(subtract.is_contiguous(), "subtract must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");

    TORCH_CHECK(gemm_out.dim() == 2, "gemm_out must be [B,F]");
    TORCH_CHECK(subtract.dim() == 1, "subtract must be [F]");
    TORCH_CHECK(residual.dim() == 2, "residual must be [B,R]");

    int64_t B64 = gemm_out.size(0);
    int64_t F64 = gemm_out.size(1);
    int64_t R64 = residual.size(1);
    TORCH_CHECK(residual.size(0) == B64, "residual batch must match gemm_out");
    TORCH_CHECK(subtract.size(0) == F64, "subtract length must match F");

    TORCH_CHECK(B64 <= INT_MAX && F64 <= INT_MAX && R64 <= INT_MAX, "sizes too large");
    int B = (int)B64, F = (int)F64, R = (int)R64;

    auto out = torch::empty_like(residual);
    auto scalar = torch::empty({B}, residual.options());

    // Choose reduction threads (power-of-two): 256 works well for F=8192; allow 128 for smaller F.
    int threads_red = (F >= 4096) ? 256 : 128;

    // Broadcast threads fixed; 2D grid.y chosen modestly to avoid overhead while increasing concurrency.
    constexpr int threads_brd = 256;

    // grid.y heuristic: enough to provide extra CTAs, but not so many that scheduling overhead dominates.
    int device = gemm_out.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm = prop.multiProcessorCount;

    int grid_y = 1;
    // Aim for ~4-8 CTAs per SM during broadcast when B is not huge; cap to 8.
    // Since grid.x=B, total CTAs = B*grid_y.
    if (B < sm * 4) {
        grid_y = 4;
    }
    if (B < sm * 2) {
        grid_y = 8;
    }
    if (grid_y > 8) grid_y = 8;
    if (grid_y < 1) grid_y = 1;

    // Kernel 1: one block per row (avoid persistent multi-row loop that previously regressed)
    if (threads_red == 256) {
        reduce_mean_gelu_kernel_opt<256><<<(unsigned int)B, 256>>>(
            (const float*)gemm_out.data_ptr<float>(),
            (const float*)subtract.data_ptr<float>(),
            (float*)scalar.data_ptr<float>(),
            B, F
        );
    } else {
        reduce_mean_gelu_kernel_opt<128><<<(unsigned int)B, 128>>>(
            (const float*)gemm_out.data_ptr<float>(),
            (const float*)subtract.data_ptr<float>(),
            (float*)scalar.data_ptr<float>(),
            B, F
        );
    }

    // Kernel 2: 2D grid-stride over columns to improve latency hiding (modest grid_y)
    dim3 grid((unsigned int)B, (unsigned int)grid_y, 1);
    broadcast_add_kernel_2d_opt<threads_brd><<<grid, threads_brd>>>(
        (const float*)residual.data_ptr<float>(),
        (const float*)scalar.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, R
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_forward_cuda(torch::Tensor gemm_out, torch::Tensor subtract, torch::Tensor residual);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    GEMM (nn.Linear) followed by custom CUDA:
      subtract -> global avg pool -> logsumexp (len-1 identity) -> GELU -> residual add (broadcast scalar)
    Implemented as two kernels (avoid fusion regression):
      1) vectorized row reduction to scalar
      2) 2D grid-stride vectorized broadcast add for better latency hiding
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.detach()
        if not residual.is_contiguous():
            residual = residual.contiguous()

        y = self.gemm(x)
        if not y.is_contiguous():
            y = y.contiguous()

        sub = self.subtract
        if not sub.is_contiguous():
            sub = sub.contiguous()

        return self.custom_ops_lib.fused_forward_cuda(y, sub, residual)


batch_size = 2048
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features]