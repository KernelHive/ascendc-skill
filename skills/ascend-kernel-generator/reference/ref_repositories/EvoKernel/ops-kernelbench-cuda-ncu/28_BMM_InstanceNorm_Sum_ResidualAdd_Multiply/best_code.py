import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# opt6: latency-bound streaming rewrite
# - Remove shared-memory y tiling (eliminates per-tile barriers and smem traffic)
# - Compute mean/var via sum/sumsq (single-pass reduction) + warp shuffles
# - Tiny shared memory for warp partials only
# - More warps/block (256 threads) to increase MLP and hide memory latency
# - float4 vectorization in both reduction and epilogue when aligned
# - __launch_bounds__ to cap registers and improve occupancy
# -----------------------------------------------------------------------------

fused_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int THREADS, int ROWS_PER_BLOCK>
__global__ __launch_bounds__(THREADS, 2)
void bmm_instance_norm_sum_residual_add_multiply_kernel_opt6(
    const float* __restrict__ x,   // [B, C]
    const float* __restrict__ y,   // [B, C]
    float* __restrict__ out,       // [B, C]
    int B, int C,
    float eps,
    int vec_ok
) {
    static_assert((THREADS % 32) == 0, "THREADS must be multiple of warp size");
    constexpr int NWARPS = THREADS / 32;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ float s_sum[NWARPS];
    __shared__ float s_sumsq[NWARPS];
    __shared__ float s_mean[ROWS_PER_BLOCK];
    __shared__ float s_invstd[ROWS_PER_BLOCK];

    // Grid-stride over rows in chunks of ROWS_PER_BLOCK
    for (int b0 = (int)blockIdx.x * ROWS_PER_BLOCK; b0 < B; b0 += (int)gridDim.x * ROWS_PER_BLOCK) {
        #pragma unroll
        for (int rb = 0; rb < ROWS_PER_BLOCK; ++rb) {
            const int b = b0 + rb;
            if (b >= B) break;

            const int base = b * C;

            // ------------------ Pass 1: sum/sumsq for mean/var ------------------
            float sum = 0.0f;
            float sumsq = 0.0f;

            if (vec_ok) {
                const int C4 = C >> 2;
                const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
                for (int i = tid; i < C4; i += THREADS) {
                    float4 v = __ldg(x4 + i);
                    sum   += (v.x + v.y) + (v.z + v.w);
                    sumsq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
                }
            } else {
                for (int c = tid; c < C; c += THREADS) {
                    float v = __ldg(x + base + c);
                    sum   += v;
                    sumsq += v * v;
                }
            }

            sum = warp_sum(sum);
            sumsq = warp_sum(sumsq);

            if (lane == 0) {
                s_sum[warp] = sum;
                s_sumsq[warp] = sumsq;
            }
            __syncthreads();

            if (warp == 0) {
                float block_sum = (lane < NWARPS) ? s_sum[lane] : 0.0f;
                float block_sumsq = (lane < NWARPS) ? s_sumsq[lane] : 0.0f;
                block_sum = warp_sum(block_sum);
                block_sumsq = warp_sum(block_sumsq);
                if (lane == 0) {
                    float mean = block_sum / (float)C;
                    float ex2  = block_sumsq / (float)C;
                    float var  = ex2 - mean * mean;           // population variance
                    var = fmaxf(var, 0.0f);                   // guard small negative due to fp error
                    s_mean[rb] = mean;
                    s_invstd[rb] = rsqrtf(var + eps);
                }
            }
            __syncthreads();

            const float mean = s_mean[rb];
            const float invstd = s_invstd[rb];

            // ------------------ Pass 2: normalize + fused add/mul ------------------
            if (vec_ok) {
                const int C4 = C >> 2;
                const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x + base);
                const float4* __restrict__ y4 = reinterpret_cast<const float4*>(y + base);
                float4* __restrict__ o4 = reinterpret_cast<float4*>(out + base);
                for (int i = tid; i < C4; i += THREADS) {
                    float4 xv = __ldg(x4 + i);
                    float4 yv = __ldg(y4 + i);

                    float xn0 = (xv.x - mean) * invstd;
                    float xn1 = (xv.y - mean) * invstd;
                    float xn2 = (xv.z - mean) * invstd;
                    float xn3 = (xv.w - mean) * invstd;

                    float4 oo;
                    oo.x = (xn0 + yv.x) * yv.x;
                    oo.y = (xn1 + yv.y) * yv.y;
                    oo.z = (xn2 + yv.z) * yv.z;
                    oo.w = (xn3 + yv.w) * yv.w;
                    o4[i] = oo;
                }
            } else {
                for (int c = tid; c < C; c += THREADS) {
                    float xv = __ldg(x + base + c);
                    float yv = __ldg(y + base + c);
                    float xn = (xv - mean) * invstd;
                    out[base + c] = (xn + yv) * yv;
                }
            }

            // no sync needed between rows because next row reuses shared buffers but overwrites after __syncthreads()
            __syncthreads();
        }
    }
}

torch::Tensor bmm_instance_norm_sum_residual_add_multiply_cuda(torch::Tensor x, torch::Tensor y, double eps) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32, "x and y must be float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y must be 2D [B, C]");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "x and y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    const int B = (int)x.size(0);
    const int C = (int)x.size(1);

    auto out = torch::empty_like(x);

    uintptr_t xp = (uintptr_t)x.data_ptr<float>();
    uintptr_t yp = (uintptr_t)y.data_ptr<float>();
    uintptr_t op = (uintptr_t)out.data_ptr<float>();
    int vec_ok = ((C & 3) == 0) && ((xp & 15) == 0) && ((yp & 15) == 0) && ((op & 15) == 0);

    int dev = x.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm = prop.multiProcessorCount;

    constexpr int THREADS = 256;
    constexpr int ROWS_PER_BLOCK = 1;

    int blocks = (B + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    // cap blocks to avoid excessive launch overhead while keeping SMs busy
    int max_blocks = sm * 16;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    bmm_instance_norm_sum_residual_add_multiply_kernel_opt6<THREADS, ROWS_PER_BLOCK>
        <<<blocks, THREADS>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)y.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C,
            (float)eps,
            vec_ok
        );

    return out;
}
"""

fused_cpp_decl = r"""
torch::Tensor bmm_instance_norm_sum_residual_add_multiply_cuda(torch::Tensor x, torch::Tensor y, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_bmm_inorm_residual_mul_ops_opt6",
    cpp_sources=fused_cpp_decl,
    cuda_sources=fused_cuda_src,
    functions=["bmm_instance_norm_sum_residual_add_multiply_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model: keep Linear, fuse per-sample normalization over feature dim + add + multiply.

    InstanceNorm2d is applied to x.unsqueeze(1).unsqueeze(1) where x is [B, C],
    which effectively normalizes over C only (per-sample). Affine/running stats not used here.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.custom_ops_lib = custom_ops_lib
        _ = momentum

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.bmm(x)

        if (not x.is_cuda) or (not y.is_cuda):
            raise RuntimeError("ModelNew expects CUDA tensors.")
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise RuntimeError("ModelNew expects float32 tensors.")
        if x.dim() != 2 or y.dim() != 2:
            raise RuntimeError("ModelNew expects 2D tensors [B, C].")
        if x.size(1) != self.out_features or y.size(1) != self.out_features:
            raise RuntimeError(f"Expected feature dim C={self.out_features}.")
        if x.size(0) != y.size(0):
            raise RuntimeError("Batch sizes of x and y must match.")

        x_contig = x.contiguous()
        y_contig = y.contiguous()
        return self.custom_ops_lib.bmm_instance_norm_sum_residual_add_multiply_cuda(x_contig, y_contig, self.eps)