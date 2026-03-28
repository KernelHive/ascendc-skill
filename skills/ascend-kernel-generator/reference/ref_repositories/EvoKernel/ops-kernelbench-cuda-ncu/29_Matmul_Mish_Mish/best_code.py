import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float softplus_fast(float x) {
    // Stable and fast enough; branchy but avoids inf/nan and saves work for large |x|
    if (x > 20.0f) return x;
    if (x < -20.0f) return __expf(x);
    return log1pf(__expf(x));
}
__device__ __forceinline__ float mish_fast(float x) {
    float sp = softplus_fast(x);
    return x * tanhf(sp);
}

__device__ __forceinline__ bool aligned16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// Tile sizes (single 64x64 output tile per CTA)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// 128-thread CTA: 4 warps
constexpr int THREADS = 128;

// Per-thread micro-tile: 4x4 -> 16 accumulators
constexpr int TM = 4;
constexpr int TN = 4;

// Shared memory padding to reduce bank conflicts on K dimension
constexpr int PADK = 1;

__global__ __launch_bounds__(THREADS, 4)
void matmul_bias_mish_mish_tiled_f32_v5(
    const float* __restrict__ X,   // [M,K]
    const float* __restrict__ W,   // [N,K] (row-major, corresponds to W^T in linear)
    const float* __restrict__ b,   // [N] or nullptr
    float* __restrict__ Y,         // [M,N]
    int M, int K, int N
) {
    // Double-buffered shared memory
    __shared__ float sX[2][BM][BK + PADK];
    __shared__ float sW[2][BN][BK + PADK];

    const int block_n = blockIdx.x;
    const int block_m = blockIdx.y;

    const int m0 = block_m * BM;
    const int n0 = block_n * BN;

    const int tid = threadIdx.x;

    // 2D thread mapping for micro-tiles:
    // 16 threads across N (each produces 4 cols) -> 16*4 = 64
    // 8 threads across M  (each produces 4 rows) -> 8*4  = 32
    // We use 16x8 = 128 threads to cover BM=64 rows by having two "row groups":
    // group 0 computes rows [0..31], group 1 computes rows [32..63]
    const int tx = tid & 15;      // 0..15
    const int ty = (tid >> 4);    // 0..7

    const int row_group = 0; // implicit; we will compute two row groups by offsetting ty
    const int m_base0 = m0 + (ty * TM);          // rows 0..31
    const int m_base1 = m0 + 32 + (ty * TM);     // rows 32..63
    const int n_base  = n0 + (tx * TN);          // cols 0..63

    // Accumulators for two row-groups (each 4x4)
    float acc0[TM][TN];
    float acc1[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc0[i][j] = 0.0f;
            acc1[i][j] = 0.0f;
        }
    }

    // Bias for 4 columns (broadcast)
    float biasv[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) biasv[j] = 0.0f;
    if (b != nullptr) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int nc = n_base + j;
            biasv[j] = (nc < N) ? __ldg(b + nc) : 0.0f;
        }
    }

    auto load_tiles = [&](int buf, int k0) {
        // Cooperative load of sX(buf): [BM,BK] and sW(buf): [BN,BK]
        // Total floats = (BM+BN)*BK = 2048 floats
        // Use float4 where possible: 512 float4
        constexpr int total_f4 = (BM + BN) * BK / 4; // 512
        for (int i4 = tid; i4 < total_f4; i4 += THREADS) {
            int f = i4 * 4;
            if (f < BM * BK) {
                int r = f / BK;            // 0..63
                int c = f - r * BK;        // multiple of 4
                int gm = m0 + r;
                int gk = k0 + c;
                float4 v;
                const float* gptr = X + (int64_t)gm * K + gk;
                bool pred = (gm < M) && (gk + 3 < K);
                if (pred && aligned16(gptr) && aligned16(&sX[buf][r][c])) {
                    v = *reinterpret_cast<const float4*>(gptr);
                    *reinterpret_cast<float4*>(&sX[buf][r][c]) = v;
                } else {
                    sX[buf][r][c + 0] = (gm < M && gk + 0 < K) ? X[(int64_t)gm * K + (gk + 0)] : 0.0f;
                    sX[buf][r][c + 1] = (gm < M && gk + 1 < K) ? X[(int64_t)gm * K + (gk + 1)] : 0.0f;
                    sX[buf][r][c + 2] = (gm < M && gk + 2 < K) ? X[(int64_t)gm * K + (gk + 2)] : 0.0f;
                    sX[buf][r][c + 3] = (gm < M && gk + 3 < K) ? X[(int64_t)gm * K + (gk + 3)] : 0.0f;
                }
            } else {
                int f2 = f - BM * BK;      // 0..1020
                int r = f2 / BK;           // 0..63
                int c = f2 - r * BK;       // multiple of 4
                int gn = n0 + r;
                int gk = k0 + c;
                float4 v;
                const float* gptr = W + (int64_t)gn * K + gk;
                bool pred = (gn < N) && (gk + 3 < K);
                if (pred && aligned16(gptr) && aligned16(&sW[buf][r][c])) {
                    v = *reinterpret_cast<const float4*>(gptr);
                    *reinterpret_cast<float4*>(&sW[buf][r][c]) = v;
                } else {
                    sW[buf][r][c + 0] = (gn < N && gk + 0 < K) ? __ldg(W + (int64_t)gn * K + (gk + 0)) : 0.0f;
                    sW[buf][r][c + 1] = (gn < N && gk + 1 < K) ? __ldg(W + (int64_t)gn * K + (gk + 1)) : 0.0f;
                    sW[buf][r][c + 2] = (gn < N && gk + 2 < K) ? __ldg(W + (int64_t)gn * K + (gk + 2)) : 0.0f;
                    sW[buf][r][c + 3] = (gn < N && gk + 3 < K) ? __ldg(W + (int64_t)gn * K + (gk + 3)) : 0.0f;
                }
            }
        }
    };

    int buf = 0;
    // Preload first tile
    load_tiles(buf, 0);
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BK) {
        int next_k0 = k0 + BK;
        int next_buf = buf ^ 1;

        // Start loading next tile early (software pipeline)
        if (next_k0 < K) {
            load_tiles(next_buf, next_k0);
        }

        // Compute on current tile
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Load W vectors for 4 cols
            float wv[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int nc = (n_base + j);
                wv[j] = (nc < N) ? sW[buf][(tx * TN + j)][kk] : 0.0f;
            }

            // Two row groups: rows [m_base0..m_base0+3] and [m_base1..m_base1+3]
            float xv0[TM];
            float xv1[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int mr0 = m_base0 + i;
                int mr1 = m_base1 + i;
                xv0[i] = (mr0 < M) ? sX[buf][(ty * TM + i)][kk] : 0.0f;
                xv1[i] = (mr1 < M) ? sX[buf][(32 + ty * TM + i)][kk] : 0.0f;
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc0[i][j] = fmaf(xv0[i], wv[j], acc0[i][j]);
                    acc1[i][j] = fmaf(xv1[i], wv[j], acc1[i][j]);
                }
            }
        }

        __syncthreads(); // ensure next tile finished loading before swap
        buf = next_buf;
    }

    // Epilogue: bias + mish(mish(.)) and store
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int mr0 = m_base0 + i;
        int mr1 = m_base1 + i;

        if (mr0 < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int nc = n_base + j;
                if (nc < N) {
                    float v = acc0[i][j] + biasv[j];
                    v = mish_fast(mish_fast(v));
                    Y[(int64_t)mr0 * N + nc] = v;
                }
            }
        }
        if (mr1 < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int nc = n_base + j;
                if (nc < N) {
                    float v = acc1[i][j] + biasv[j];
                    v = mish_fast(mish_fast(v));
                    Y[(int64_t)mr1 * N + nc] = v;
                }
            }
        }
    }
}

torch::Tensor matmul_mish_mish_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    CHECK_CUDA(X); CHECK_CUDA(W);
    CHECK_CONTIGUOUS(X); CHECK_CONTIGUOUS(W);
    CHECK_FLOAT(X); CHECK_FLOAT(W);

    TORCH_CHECK(X.dim() == 2, "X must be 2D [M,K]");
    TORCH_CHECK(W.dim() == 2, "W must be 2D [N,K] (nn.Linear.weight)");
    TORCH_CHECK(X.size(1) == W.size(1), "K mismatch: X.size(1) must equal W.size(1)");

    int64_t M64 = X.size(0);
    int64_t K64 = X.size(1);
    int64_t N64 = W.size(0);

    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        CHECK_CUDA(b);
        CHECK_CONTIGUOUS(b);
        CHECK_FLOAT(b);
        TORCH_CHECK(b.dim() == 1 && b.size(0) == N64, "bias must be [N] matching W.size(0)");
        bptr = b.data_ptr<float>();
    }

    auto Y = torch::empty({M64, N64}, X.options());

    int M = (int)M64;
    int K = (int)K64;
    int N = (int)N64;

    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto stream = at::cuda::getDefaultCUDAStream();
    matmul_bias_mish_mish_tiled_f32_v5<<<grid, block, 0, stream>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        bptr,
        Y.data_ptr<float>(),
        M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
"""

cpp_src = r"""
torch::Tensor matmul_mish_mish_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_mish_mish_fwd_tiled_v5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_mish_mish_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-maxrregcount=80"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Linear + Mish + Mish fused into a custom CUDA forward kernel:
      y = mish(mish(x @ W^T + b))
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1.0 / (in_features ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous() if self.bias is not None else torch.empty(0, device=w.device, dtype=w.dtype)
        return custom_ops_lib.matmul_mish_mish_forward_cuda(x, w, b)