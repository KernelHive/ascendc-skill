import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA op: softmax_fp32_from_bf16_cuda
# - Fast path specialized for E==64:
#     * warp-per-row, warp shuffle reductions (no shared mem, no __syncthreads)
#     * 4 warps/block => 4 rows/block
#     * vectorized bf16x2 loads
# - Fallback for general E:
#     * prior block-per-row shared-memory reduction kernel
# - Output: probs [T,E] float32 to feed torch.topk for exact topk semantics.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cfloat>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(p) __ldg(p)
#else
  #define LDG(p) (*(p))
#endif

__device__ __forceinline__ float bf16_to_f32(at::BFloat16 x) { return (float)x; }

// Warp reduce max/sum
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Specialized kernel for E == 64
// blockDim.x must be 128 (4 warps) or 256 (8 warps) etc; we use 128.
__global__ void softmax_fp32_from_bf16_e64_warp_kernel(
    const at::BFloat16* __restrict__ gate, // [T,64]
    float* __restrict__ probs,             // [T,64]
    int T
) {
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;           // warp id in block
    int lane = tid & 31;

    int row = (int)blockIdx.x * 4 + warp;
    if (row >= T) return;

    const int E = 64;
    int64_t base = (int64_t)row * (int64_t)E;

    // Vectorized bf16x2 loads: each lane loads two bf16 pairs => 4 elements total per lane across 2 pairs?
    // For E=64, we need 64 elements. If each lane loads 2 elements, we cover 64 with 32 lanes.
    // We'll do exactly 2 elems per lane using one uint32 load.
    const uint32_t* gate_u32 = reinterpret_cast<const uint32_t*>(gate + base);
    uint32_t pack = LDG(gate_u32 + lane); // loads bf16[2*lane], bf16[2*lane+1]

    // Unpack bf16x2
    uint16_t lo = (uint16_t)(pack & 0xFFFFu);
    uint16_t hi = (uint16_t)(pack >> 16);
    __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&lo);
    __nv_bfloat16 b1 = *reinterpret_cast<__nv_bfloat16*>(&hi);
    float x0 = __bfloat162float(b0);
    float x1 = __bfloat162float(b1);

    float local_max = fmaxf(x0, x1);
    float mx = warp_reduce_max(local_max);
    mx = __shfl_sync(0xffffffff, mx, 0);

    float e0 = expf(x0 - mx);
    float e1 = expf(x1 - mx);
    float local_sum = e0 + e1;
    float denom = warp_reduce_sum(local_sum);
    denom = __shfl_sync(0xffffffff, denom, 0);

    float inv = 1.0f / denom;

    // Store to probs (fp32)
    float* out = probs + base;
    out[2 * lane + 0] = e0 * inv;
    out[2 * lane + 1] = e1 * inv;
}

// Generic fallback kernel (block per row, shared-memory reduction)
__global__ void softmax_fp32_from_bf16_generic_kernel(
    const at::BFloat16* __restrict__ gate, // [T, E]
    float* __restrict__ probs,             // [T, E]
    int T, int E
) {
    int t = (int)blockIdx.x;
    if (t >= T) return;

    extern __shared__ float shmem[];
    float* sh_max = shmem;
    float* sh_sum = shmem + blockDim.x;

    int tid = (int)threadIdx.x;
    int64_t row = (int64_t)t * (int64_t)E;

    float local_max = -FLT_MAX;
    for (int e = tid; e < E; e += (int)blockDim.x) {
        float x = bf16_to_f32(gate[row + e]);
        local_max = fmaxf(local_max, x);
    }
    sh_max[tid] = local_max;
    __syncthreads();

    for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sh_max[tid] = fmaxf(sh_max[tid], sh_max[tid + offset]);
        __syncthreads();
    }
    float mx = sh_max[0];

    float local_sum = 0.0f;
    for (int e = tid; e < E; e += (int)blockDim.x) {
        float x = bf16_to_f32(gate[row + e]);
        local_sum += expf(x - mx);
    }
    sh_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sh_sum[tid] += sh_sum[tid + offset];
        __syncthreads();
    }
    float denom = sh_sum[0];

    float inv_denom = 1.0f / denom;
    for (int e = tid; e < E; e += (int)blockDim.x) {
        float x = bf16_to_f32(gate[row + e]);
        float p = expf(x - mx) * inv_denom;
        probs[row + e] = p;
    }
}

torch::Tensor softmax_fp32_from_bf16_cuda(torch::Tensor gate_logits) {
    CHECK_INPUT_BF16(gate_logits);
    TORCH_CHECK(gate_logits.dim() == 2, "gate_logits must be [T, E]");

    int64_t T64 = gate_logits.size(0);
    int64_t E64 = gate_logits.size(1);
    TORCH_CHECK(T64 > 0 && E64 > 0, "invalid sizes");
    TORCH_CHECK(T64 <= (int64_t)INT_MAX && E64 <= (int64_t)INT_MAX, "sizes too large");

    int T = (int)T64;
    int E = (int)E64;

    auto probs = torch::empty({T64, E64},
        torch::TensorOptions().device(gate_logits.device()).dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (E == 64) {
        // 4 warps/block => 4 rows/block, warp-per-row.
        dim3 block(128);
        dim3 grid(div_up_int(T, 4));
        softmax_fp32_from_bf16_e64_warp_kernel<<<grid, block, 0, stream>>>(
            (const at::BFloat16*)gate_logits.data_ptr<at::BFloat16>(),
            (float*)probs.data_ptr<float>(),
            T
        );
        return probs;
    }

    // Generic fallback: one block per row.
    int threads = 1;
    while (threads < E) threads <<= 1;
    if (threads > 256) threads = 256;

    dim3 grid(T);
    dim3 block(threads);
    size_t shmem_bytes = (size_t)threads * sizeof(float) * 2;

    softmax_fp32_from_bf16_generic_kernel<<<grid, block, shmem_bytes, stream>>>(
        (const at::BFloat16*)gate_logits.data_ptr<at::BFloat16>(),
        (float*)probs.data_ptr<float>(),
        T, E
    );
    return probs;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor softmax_fp32_from_bf16_cuda(torch::Tensor gate_logits);
"""

custom_ops_lib = load_inline(
    name="custom_router_softmax_fp32_v2_e64warp",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["softmax_fp32_from_bf16_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    MoE Router optimized:
      - Custom CUDA softmax: bf16 logits -> fp32 probs (warp-specialized for E=64)
      - torch.topk on fp32 probs for exact tie/ID semantics
      - cast values to bf16

    Fast path requires:
      gate_logits: CUDA, bfloat16, contiguous, [T,E]
    """

    def __init__(self, topk: int):
        super().__init__()
        self.topk = int(topk)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, gate_logits: torch.Tensor):
        if (
            gate_logits.is_cuda
            and gate_logits.dtype == torch.bfloat16
            and gate_logits.dim() == 2
            and gate_logits.is_contiguous()
            and 0 < self.topk <= gate_logits.size(1)
        ):
            probs = self.custom_ops_lib.softmax_fp32_from_bf16_cuda(gate_logits)
            topk_vals, topk_ids = torch.topk(probs, k=self.topk, dim=-1)
            return topk_ids, topk_vals.to(dtype=gate_logits.dtype)

        probs = torch.softmax(gate_logits.float(), dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=self.topk, dim=-1)
        return topk_ids, topk_vals.to(gate_logits.dtype)