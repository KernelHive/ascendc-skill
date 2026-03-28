import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: argmin_over_a_dimension (3D, float32) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>
#include <limits>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

static __device__ __forceinline__ void update_argmin(float v, int idx, float &best, int &best_idx) {
    // stable tie-break: smallest index on equal value
    if (v < best || (v == best && idx < best_idx)) {
        best = v;
        best_idx = idx;
    }
}

// Generic argmin over `rdim` in {0,1,2} for x of shape [D0, D1, D2] (contiguous, float32).
__global__ void argmin_reduce_3d_f32_generic_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ out,
    int D0, int D1, int D2,
    int rdim
) {
    int64_t out_numel;
    if (rdim == 0) out_numel = (int64_t)D1 * D2;
    else if (rdim == 1) out_numel = (int64_t)D0 * D2;
    else out_numel = (int64_t)D0 * D1;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t out_idx = tid; out_idx < out_numel; out_idx += stride) {
        float best_val = INFINITY;
        int best_i = 0;

        if (rdim == 0) {
            int j = (int)(out_idx / D2);
            int k = (int)(out_idx - (int64_t)j * D2);
            int64_t base = (int64_t)j * D2 + k;      // i=0 offset
            int64_t step = (int64_t)D1 * D2;         // i++
            for (int i = 0; i < D0; ++i) {
                float v = x[base + (int64_t)i * step];
                update_argmin(v, i, best_val, best_i);
            }
            out[out_idx] = (int64_t)best_i;
        } else if (rdim == 1) {
            int i = (int)(out_idx / D2);
            int k = (int)(out_idx - (int64_t)i * D2);
            int64_t base = ((int64_t)i * D1) * D2 + k; // j=0
            int64_t step = (int64_t)D2;                // j++
            for (int j = 0; j < D1; ++j) {
                float v = x[base + (int64_t)j * step];
                update_argmin(v, j, best_val, best_i);
            }
            out[out_idx] = (int64_t)best_i;
        } else {
            int i = (int)(out_idx / D1);
            int j = (int)(out_idx - (int64_t)i * D1);
            int64_t base = ((int64_t)i * D1 + j) * D2; // k=0
            for (int k = 0; k < D2; ++k) {
                float v = x[base + k];
                update_argmin(v, k, best_val, best_i);
            }
            out[out_idx] = (int64_t)best_i;
        }
    }
}

// Fast path for rdim == 1 (reduce over D1) for x [B, D1, D2] contiguous.
// Keep baseline "one thread scans j" but compute multiple k's per thread to increase ILP/MLP.
// No incorrect cross-dimension vector loads: each load is exactly x[b, j, k+p].
template<int VEC_K, int UNROLL_J>
__global__ __launch_bounds__(256, 2)
void argmin_dim1_vecK_kernel_f32(
    const float* __restrict__ x,
    int64_t* __restrict__ out,
    int B, int D1, int D2
) {
    // 2D grid: blockIdx.y = batch, blockIdx.x = k-tile
    const int b = (int)blockIdx.y;
    if (b >= B) return;

    const int tid = (int)threadIdx.x;
    const int k0 = ((int)blockIdx.x * (int)blockDim.x + tid) * VEC_K;
    if (k0 >= D2) return;

    float best[VEC_K];
    int best_j[VEC_K];
    #pragma unroll
    for (int p = 0; p < VEC_K; ++p) {
        best[p] = INFINITY;
        best_j[p] = 0;
    }

    const int64_t base_b = (int64_t)b * (int64_t)D1 * (int64_t)D2;

    // j loop: modest unroll + software prefetch of next step
    int j = 0;

    // Preload first iteration values (prefetch stage)
    float nxt[VEC_K];
    #pragma unroll
    for (int p = 0; p < VEC_K; ++p) nxt[p] = INFINITY;

    if (j < D1) {
        const int64_t row0 = base_b + (int64_t)j * (int64_t)D2 + (int64_t)k0;
        #pragma unroll
        for (int p = 0; p < VEC_K; ++p) {
            int k = k0 + p;
            nxt[p] = (k < D2) ? x[row0 + p] : INFINITY;
        }
    }

    for (; j + UNROLL_J <= D1; j += UNROLL_J) {
        #pragma unroll
        for (int u = 0; u < UNROLL_J; ++u) {
            // consume prefetched values for (j+u)
            const int jj = j + u;

            float cur[VEC_K];
            #pragma unroll
            for (int p = 0; p < VEC_K; ++p) cur[p] = nxt[p];

            // prefetch next (jj+1) values into nxt (unless u is last unroll and next block will handle)
            const int next_j = jj + 1;
            if (next_j < D1) {
                const int64_t rowN = base_b + (int64_t)next_j * (int64_t)D2 + (int64_t)k0;
                #pragma unroll
                for (int p = 0; p < VEC_K; ++p) {
                    int k = k0 + p;
                    nxt[p] = (k < D2) ? x[rowN + p] : INFINITY;
                }
            } else {
                #pragma unroll
                for (int p = 0; p < VEC_K; ++p) nxt[p] = INFINITY;
            }

            // update minima
            #pragma unroll
            for (int p = 0; p < VEC_K; ++p) {
                int k = k0 + p;
                if (k < D2) update_argmin(cur[p], jj, best[p], best_j[p]);
            }
        }
    }

    // Tail (no unroll), using direct loads
    for (; j < D1; ++j) {
        const int64_t row = base_b + (int64_t)j * (int64_t)D2 + (int64_t)k0;
        #pragma unroll
        for (int p = 0; p < VEC_K; ++p) {
            int k = k0 + p;
            if (k < D2) {
                float v = x[row + p];
                update_argmin(v, j, best[p], best_j[p]);
            }
        }
    }

    // store
    const int64_t out_base = (int64_t)b * (int64_t)D2 + (int64_t)k0;
    #pragma unroll
    for (int p = 0; p < VEC_K; ++p) {
        int k = k0 + p;
        if (k < D2) out[out_base + p] = (int64_t)best_j[p];
    }
}

torch::Tensor argmin_over_a_dimension_cuda(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor [D0, D1, D2]");
    TORCH_CHECK(dim >= -3 && dim <= 2, "dim must be in [-3, 2] for a 3D tensor");

    int rdim = (int)dim;
    if (rdim < 0) rdim += 3;
    TORCH_CHECK(rdim >= 0 && rdim < 3, "normalized dim must be in [0,2]");

    if (!x.is_contiguous()) x = x.contiguous();

    const int D0 = (int)x.size(0);
    const int D1 = (int)x.size(1);
    const int D2 = (int)x.size(2);

    torch::Tensor out;
    auto out_opts = x.options().dtype(torch::kInt64);
    if (rdim == 0) out = torch::empty({D1, D2}, out_opts);
    else if (rdim == 1) out = torch::empty({D0, D2}, out_opts);
    else out = torch::empty({D0, D1}, out_opts);

    at::cuda::CUDAGuard device_guard(x.device());

    if (rdim == 1) {
        // Heuristic: use VEC_K=4 for typical D2; keep threads=256.
        // 2D grid: x = k-tiles, y = batch (D0)
        const int threads = 256;
        constexpr int VEC_K = 4;
        constexpr int UNROLL_J = 4;

        const int64_t k_per_block = (int64_t)threads * VEC_K;
        int blocks_x = (int)((D2 + k_per_block - 1) / k_per_block);
        if (blocks_x < 1) blocks_x = 1;

        dim3 grid((unsigned)blocks_x, (unsigned)D0, 1);
        argmin_dim1_vecK_kernel_f32<VEC_K, UNROLL_J><<<grid, threads>>>(
            (const float*)x.data_ptr<float>(),
            (int64_t*)out.data_ptr<int64_t>(),
            D0, D1, D2
        );
        return out;
    }

    // Generic fallback for other dims.
    const int64_t out_numel = out.numel();
    int threads = 256;
    int blocks = (int)((out_numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    argmin_reduce_3d_f32_generic_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (int64_t*)out.data_ptr<int64_t>(),
        D0, D1, D2, rdim
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor argmin_over_a_dimension_cuda(torch::Tensor x, int64_t dim);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_argmin_reduce_dim_vecK_prefetch_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["argmin_over_a_dimension_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Model that performs argmin over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops_lib.argmin_over_a_dimension_cuda(x, self.dim)


# Keep original input helpers for compatibility with the provided scaffold.
batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [dim]