import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA implementation (post conv_transpose):
#   Input:  x [N,C,H,W] float32 CUDA contiguous (NCHW)
#   Bias:   bias [B] flattened (typical B==C from [C,1,1])
#   Output: y [N,1] float32
#
# Pipeline:
#   1) pool_bias_kernel: pooled[n,c] = mean_{hw}(x[n,c,hw]) + bias[c%B]
#      - grid: (C, N) to preserve large parallelism
#      - float4 vectorized reads on HW when safe
#   2) lse_mul10_fused_kernel: for each n compute logsumexp over C and multiply by 10
#      - fuses "max over C" + "sumexp over C" + final write
#      - optional float4 vectorized reads on pooled when safe
#
# Additionally:
#   - pooled buffer can be provided by caller to avoid per-call allocation.
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIG
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_DTYPE_FLOAT
#define CHECK_DTYPE_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {                                   \
  cudaError_t err = cudaGetLastError();                                       \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err)); \
} while(0)
#endif

static __device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}
static __device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

template<int BLOCK_THREADS>
static __device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float shared[32]; // up to 1024 threads
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  float out = 0.0f;
  if (wid == 0) {
    int nw = (BLOCK_THREADS + 31) >> 5;
    out = (lane < nw) ? shared[lane] : 0.0f;
    out = warp_reduce_sum(out);
  }
  out = __shfl_sync(0xffffffff, out, 0);
  return out;
}

template<int BLOCK_THREADS>
static __device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  float out = -INFINITY;
  if (wid == 0) {
    int nw = (BLOCK_THREADS + 31) >> 5;
    out = (lane < nw) ? shared[lane] : -INFINITY;
    out = warp_reduce_max(out);
  }
  out = __shfl_sync(0xffffffff, out, 0);
  return out;
}

// Kernel1: pooled[n,c] = mean(x[n,c,:,:]) + bias[c%B]
// grid: (C, N)
__global__ void pool_bias_kernel(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ bias,   // [B] or nullptr
    float* __restrict__ pooled,       // [N,C]
    int N, int C, int H, int W, int B,
    int use_vec4_hw
) {
  int c = (int)blockIdx.x;
  int n = (int)blockIdx.y;
  if (n >= N || c >= C) return;

  const int HW = H * W;
  const int64_t stride_n = (int64_t)C * (int64_t)HW;
  const int64_t stride_c = (int64_t)HW;
  const float* x_nc = x + (int64_t)n * stride_n + (int64_t)c * stride_c;

  float sum = 0.0f;

  if (use_vec4_hw) {
    const float4* x4 = reinterpret_cast<const float4*>(x_nc);
    int HW4 = HW >> 2;
    // slight unroll to reduce loop overhead
    for (int i = (int)threadIdx.x; i < HW4; i += (int)blockDim.x) {
      float4 v = x4[i];
      sum += (v.x + v.y) + (v.z + v.w);
    }
  } else {
    // unroll by 4 on scalar path (safe for any HW)
    int i = (int)threadIdx.x;
    int stride = (int)blockDim.x;
    for (; i + 3 * stride < HW; i += 4 * stride) {
#if __CUDA_ARCH__ >= 350
      float v0 = __ldg(x_nc + i);
      float v1 = __ldg(x_nc + i + stride);
      float v2 = __ldg(x_nc + i + 2 * stride);
      float v3 = __ldg(x_nc + i + 3 * stride);
#else
      float v0 = x_nc[i];
      float v1 = x_nc[i + stride];
      float v2 = x_nc[i + 2 * stride];
      float v3 = x_nc[i + 3 * stride];
#endif
      sum += (v0 + v1) + (v2 + v3);
    }
    for (; i < HW; i += stride) {
#if __CUDA_ARCH__ >= 350
      sum += __ldg(x_nc + i);
#else
      sum += x_nc[i];
#endif
    }
  }

  float total = block_reduce_sum<256>(sum);
  if (threadIdx.x == 0) {
    float mean = total * (1.0f / (float)HW);
    float bv = 0.0f;
    if (bias != nullptr && B > 0) {
#if __CUDA_ARCH__ >= 350
      bv = __ldg(bias + (c % B));
#else
      bv = bias[c % B];
#endif
    }
    pooled[(int64_t)n * (int64_t)C + c] = mean + bv;
  }
}

// Kernel2: fused max+sumexp -> logsumexp -> *10
// grid: (N)
template<int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 4)
void lse_mul10_fused_kernel(
    const float* __restrict__ pooled, // [N,C]
    float* __restrict__ y,            // [N]
    int N, int C,
    int use_vec4_c
) {
  int n = (int)blockIdx.x;
  if (n >= N) return;

  const float* row = pooled + (int64_t)n * (int64_t)C;

  // Pass 1: max
  float tmax = -INFINITY;
  if (use_vec4_c) {
    const float4* r4 = reinterpret_cast<const float4*>(row);
    int C4 = C >> 2;
    for (int i = (int)threadIdx.x; i < C4; i += BLOCK_THREADS) {
      float4 v = r4[i];
      tmax = fmaxf(tmax, v.x);
      tmax = fmaxf(tmax, v.y);
      tmax = fmaxf(tmax, v.z);
      tmax = fmaxf(tmax, v.w);
    }
  } else {
    for (int c = (int)threadIdx.x; c < C; c += BLOCK_THREADS) {
#if __CUDA_ARCH__ >= 350
      float v = __ldg(row + c);
#else
      float v = row[c];
#endif
      tmax = fmaxf(tmax, v);
    }
  }
  float m = block_reduce_max<BLOCK_THREADS>(tmax);
  __syncthreads();

  // Pass 2: sumexp
  float tsum = 0.0f;
  if (use_vec4_c) {
    const float4* r4 = reinterpret_cast<const float4*>(row);
    int C4 = C >> 2;
    for (int i = (int)threadIdx.x; i < C4; i += BLOCK_THREADS) {
      float4 v = r4[i];
      tsum += __expf(v.x - m);
      tsum += __expf(v.y - m);
      tsum += __expf(v.z - m);
      tsum += __expf(v.w - m);
    }
  } else {
    for (int c = (int)threadIdx.x; c < C; c += BLOCK_THREADS) {
#if __CUDA_ARCH__ >= 350
      float v = __ldg(row + c);
#else
      float v = row[c];
#endif
      tsum += __expf(v - m);
    }
  }
  float s = block_reduce_sum<BLOCK_THREADS>(tsum);

  if (threadIdx.x == 0) {
    float lse = m + logf(s);
    y[n] = lse * 10.0f;
  }
}

// Entry point with optional pooled buffer (to allow reuse)
torch::Tensor gap_bias_logsumexp_mul10_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor pooled_opt) {
  CHECK_CUDA(x);
  CHECK_DTYPE_FLOAT(x);
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  auto x_c = x.contiguous();

  const int64_t N64 = x_c.size(0);
  const int64_t C64 = x_c.size(1);
  const int64_t H64 = x_c.size(2);
  const int64_t W64 = x_c.size(3);
  TORCH_CHECK(N64 > 0 && C64 > 0 && H64 > 0 && W64 > 0, "invalid x shape");
  TORCH_CHECK(N64 <= INT_MAX && C64 <= INT_MAX && H64 <= INT_MAX && W64 <= INT_MAX, "shape too large");

  torch::Tensor bias_c;
  const float* bias_ptr = nullptr;
  int B = 0;
  if (bias.defined() && bias.numel() > 0) {
    CHECK_CUDA(bias);
    CHECK_DTYPE_FLOAT(bias);
    bias_c = bias.contiguous().view({-1});
    TORCH_CHECK(bias_c.numel() <= INT_MAX, "bias too large");
    B = (int)bias_c.numel();
    bias_ptr = bias_c.data_ptr<float>();
  }

  torch::Tensor pooled;
  if (pooled_opt.defined() && pooled_opt.numel() > 0) {
    CHECK_CUDA(pooled_opt);
    CHECK_DTYPE_FLOAT(pooled_opt);
    CHECK_CONTIG(pooled_opt);
    TORCH_CHECK(pooled_opt.dim() == 2, "pooled_opt must be [N,C]");
    TORCH_CHECK(pooled_opt.size(0) >= N64 && pooled_opt.size(1) >= C64, "pooled_opt too small");
    pooled = pooled_opt.narrow(0, 0, N64).narrow(1, 0, C64);
  } else {
    pooled = torch::empty({N64, C64}, x_c.options());
  }

  auto y = torch::empty({N64, 1}, x_c.options());
  auto y1 = y.view({N64});

  const int N = (int)N64;
  const int C = (int)C64;
  const int H = (int)H64;
  const int W = (int)W64;
  const int HW = H * W;

  // vec4 eligibility for HW reads
  uintptr_t addr_x = (uintptr_t)(x_c.data_ptr<float>());
  int use_vec4_hw = ((addr_x % 16u) == 0u) && ((HW & 3) == 0);

  // launch pool kernel: fixed 256 threads for good reduction behavior
  dim3 block1(256, 1, 1);
  dim3 grid1((unsigned)C, (unsigned)N, 1);
  pool_bias_kernel<<<grid1, block1>>>(
      x_c.data_ptr<float>(), bias_ptr, pooled.data_ptr<float>(),
      N, C, H, W, B, use_vec4_hw
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // vec4 eligibility for C reads from pooled row
  uintptr_t addr_p = (uintptr_t)(pooled.data_ptr<float>());
  // row base alignment depends on C; require base pointer aligned and C divisible by 4
  int use_vec4_c = ((addr_p % 16u) == 0u) && ((C & 3) == 0);

  constexpr int THREADS2 = 128;
  dim3 block2(THREADS2, 1, 1);
  dim3 grid2((unsigned)N, 1, 1);
  lse_mul10_fused_kernel<THREADS2><<<grid2, block2>>>(
      pooled.data_ptr<float>(),
      y1.data_ptr<float>(),
      N, C,
      use_vec4_c
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor gap_bias_logsumexp_mul10_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor pooled_opt);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_t_gap_bias_lse_mul10_v5_fused_cachedpooled",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gap_bias_logsumexp_mul10_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Keeps ConvTranspose2d on cuDNN, fuses post-op chain:
      mean over (2,3) keepdim -> +bias -> logsumexp over C keepdim ->
      sum over (2,3) -> *10
    Output: [N,1]
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.custom_ops = custom_ops_lib
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.register_buffer("_pooled_buf", torch.empty(0, device="cuda", dtype=torch.float32), persistent=False)

    def _get_pooled(self, n: int, c: int, like: torch.Tensor) -> torch.Tensor:
        if (not self._pooled_buf.is_cuda) or (self._pooled_buf.dtype != torch.float32) or (self._pooled_buf.numel() == 0):
            self._pooled_buf = torch.empty((n, c), device=like.device, dtype=torch.float32)
        else:
            if self._pooled_buf.device != like.device or self._pooled_buf.size(0) < n or self._pooled_buf.size(1) < c:
                self._pooled_buf = torch.empty((n, c), device=like.device, dtype=torch.float32)
        return self._pooled_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        pooled = self._get_pooled(x.shape[0], x.shape[1], x)
        return self.custom_ops.gap_bias_logsumexp_mul10_cuda(x, self.bias, pooled)