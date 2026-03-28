import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized custom CUDA: shallow-wide MLP forward (2 hidden ReLU layers).
# Keeps same external API as baseline but improves:
#  - shared-memory layout for W tile (store as [BK][BN]) to reduce bank conflicts
#  - vectorized float4 loads for X and W tiles with in-kernel safe gating
#  - __launch_bounds__ to help compiler trade registers/occupancy
# -----------------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

__device__ __forceinline__ float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

__device__ __forceinline__ bool ptr_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// X: [M,K] row-major
// W: [N,K] row-major (we compute X * W^T)
// b: [N]
// Y: [M,N] row-major
template<int BM, int BN, int BK, int TM, int TN, bool DO_RELU>
__global__ __launch_bounds__(256, 2)
void linear_bias_act_fwd_kernel_v2(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ Y,
    int M, int K, int N
) {
    // Shared tiles:
    //  As: [BM, BK]  (X tile)
    //  BsT: [BK, BN] (W tile stored transposed for better shared reads)
    __shared__ float As[BM][BK];
    __shared__ float BsT[BK][BN];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    const int row0 = (int)blockIdx.y * BM + ty * TM; // TM rows per thread
    const int col0 = (int)blockIdx.x * BN + tx * TN; // TN cols per thread

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    // In-kernel alignment checks (safe; no host/device mismatch).
    const bool x_aligned = ptr_aligned_16(X);
    const bool w_aligned = ptr_aligned_16(W);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ----------------------------
        // Load A tile: As[BM][BK]
        // ----------------------------
        // Use float4 when possible: along K dimension.
        // Map threads to cover BM*BK elements. We load As in row-major order.
        // Each thread loads multiple (row, col) pairs.
        if (x_aligned && ((k0 & 3) == 0)) {
            // Load As as float4 chunks: BK is multiple of 4 (BK=32).
            // Distribute over (ty, tx): each thread handles some rows and c4 groups.
            for (int r = ty; r < BM; r += blockDim.y) {
                const int gr = (int)blockIdx.y * BM + r;
                const int base = gr * K + k0;
                for (int c4 = tx; c4 < (BK / 4); c4 += blockDim.x) {
                    const int gc = k0 + c4 * 4;
                    float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (gr < M && (gc + 3) < K) {
                        const float4* p4 = reinterpret_cast<const float4*>(X + (int64_t)base + c4 * 4);
                        v4 = *p4;
                    }
                    As[r][c4 * 4 + 0] = v4.x;
                    As[r][c4 * 4 + 1] = v4.y;
                    As[r][c4 * 4 + 2] = v4.z;
                    As[r][c4 * 4 + 3] = v4.w;
                }
            }
        } else {
            for (int r = ty; r < BM; r += blockDim.y) {
                const int gr = (int)blockIdx.y * BM + r;
                for (int c = tx; c < BK; c += blockDim.x) {
                    const int gc = k0 + c;
                    float v = 0.0f;
                    if (gr < M && gc < K) v = X[(int64_t)gr * K + gc];
                    As[r][c] = v;
                }
            }
        }

        // ----------------------------
        // Load B tile: BsT[BK][BN] from W[BN][BK]
        // ----------------------------
        // We'll load W rows (output features) for this block's BN columns,
        // and store transposed in shared: BsT[kk][cc] where cc within BN.
        // Use float4 loads along K for each W row.
        if (w_aligned && ((k0 & 3) == 0)) {
            for (int cc = ty; cc < BN; cc += blockDim.y) {
                const int gn = (int)blockIdx.x * BN + cc; // W row index in [0,N)
                const int wrow_base = gn * K + k0;
                for (int k4 = tx; k4 < (BK / 4); k4 += blockDim.x) {
                    const int kk = k4 * 4;
                    const int gc = k0 + kk;
                    float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (gn < N && (gc + 3) < K) {
                        const float4* p4 = reinterpret_cast<const float4*>(W + (int64_t)wrow_base + kk);
                        v4 = *p4;
                    }
                    BsT[kk + 0][cc] = v4.x;
                    BsT[kk + 1][cc] = v4.y;
                    BsT[kk + 2][cc] = v4.z;
                    BsT[kk + 3][cc] = v4.w;
                }
            }
        } else {
            for (int cc = ty; cc < BN; cc += blockDim.y) {
                const int gn = (int)blockIdx.x * BN + cc;
                for (int kk = tx; kk < BK; kk += blockDim.x) {
                    const int gc = k0 + kk;
                    float v = 0.0f;
                    if (gn < N && gc < K) v = W[(int64_t)gn * K + gc];
                    BsT[kk][cc] = v;
                }
            }
        }

        __syncthreads();

        // ----------------------------
        // Compute
        // ----------------------------
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int rr = ty * TM + i;
                a_frag[i] = As[rr][kk];
            }

            // Read BsT[kk][cc] for each cc (within BN)
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int cc = tx * TN + j;
                const float bval = BsT[kk][cc];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    acc[i][j] = fmaf(a_frag[i], bval, acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------
    // Epilogue
    // ----------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + j;
            if (c >= N) continue;
            float v = acc[i][j] + b[c];
            if (DO_RELU) v = relu_f(v);
            Y[(int64_t)r * N + c] = v;
        }
    }
}

static inline dim3 make_block() {
    // BM=128, BN=128, TM=8, TN=8 => 16x16 = 256 threads
    return dim3(16, 16, 1);
}

torch::Tensor linear_relu_fwd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(), "linear_relu_fwd_cuda: all inputs must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "linear_relu_fwd_cuda: only float32 supported");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && b.is_contiguous(), "linear_relu_fwd_cuda: inputs must be contiguous");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2 && b.dim() == 1, "linear_relu_fwd_cuda: bad dims");
    TORCH_CHECK(X.size(1) == W.size(1), "linear_relu_fwd_cuda: K mismatch");
    TORCH_CHECK(W.size(0) == b.size(0), "linear_relu_fwd_cuda: N mismatch");

    const int M = (int)X.size(0);
    const int K = (int)X.size(1);
    const int N = (int)W.size(0);

    auto Y = torch::empty({M, N}, X.options());

    constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;
    dim3 block = make_block();
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    linear_bias_act_fwd_kernel_v2<BM, BN, BK, TM, TN, true><<<grid, block>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), Y.data_ptr<float>(), M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor linear_fwd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(), "linear_fwd_cuda: all inputs must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "linear_fwd_cuda: only float32 supported");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && b.is_contiguous(), "linear_fwd_cuda: inputs must be contiguous");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2 && b.dim() == 1, "linear_fwd_cuda: bad dims");
    TORCH_CHECK(X.size(1) == W.size(1), "linear_fwd_cuda: K mismatch");
    TORCH_CHECK(W.size(0) == b.size(0), "linear_fwd_cuda: N mismatch");

    const int M = (int)X.size(0);
    const int K = (int)X.size(1);
    const int N = (int)W.size(0);

    auto Y = torch::empty({M, N}, X.options());

    constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;
    dim3 block = make_block();
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    linear_bias_act_fwd_kernel_v2<BM, BN, BK, TM, TN, false><<<grid, block>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), Y.data_ptr<float>(), M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor shallow_wide_mlp_fwd_cuda(
    torch::Tensor X,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3
) {
    TORCH_CHECK(X.is_cuda(), "shallow_wide_mlp_fwd_cuda: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "shallow_wide_mlp_fwd_cuda: only float32 supported");
    TORCH_CHECK(X.is_contiguous(), "shallow_wide_mlp_fwd_cuda: X must be contiguous");
    TORCH_CHECK(X.dim() == 2, "shallow_wide_mlp_fwd_cuda: X must be 2D");

    auto check_wb = [](const char* name, torch::Tensor W, torch::Tensor b) {
        TORCH_CHECK(W.is_cuda() && b.is_cuda(), "%s: W and b must be CUDA", name);
        TORCH_CHECK(W.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "%s: only float32 supported", name);
        TORCH_CHECK(W.is_contiguous() && b.is_contiguous(), "%s: W and b must be contiguous", name);
        TORCH_CHECK(W.dim() == 2 && b.dim() == 1, "%s: bad dims", name);
        TORCH_CHECK(W.size(0) == b.size(0), "%s: N mismatch", name);
    };

    check_wb("layer1", W1, b1);
    check_wb("layer2", W2, b2);
    check_wb("layer3", W3, b3);

    TORCH_CHECK(X.size(1) == W1.size(1), "shallow_wide_mlp_fwd_cuda: X K != W1 K");
    TORCH_CHECK(W1.size(0) == W2.size(1), "shallow_wide_mlp_fwd_cuda: W1 N != W2 K");
    TORCH_CHECK(W2.size(0) == W3.size(1), "shallow_wide_mlp_fwd_cuda: W2 N != W3 K");

    auto Y1 = linear_relu_fwd_cuda(X, W1, b1);
    auto Y2 = linear_relu_fwd_cuda(Y1, W2, b2);
    auto Y3 = linear_fwd_cuda(Y2, W3, b3);
    return Y3;
}
"""

cpp_source = r"""
torch::Tensor linear_relu_fwd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
torch::Tensor linear_fwd_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
torch::Tensor shallow_wide_mlp_fwd_cuda(
    torch::Tensor X,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    torch::Tensor W3, torch::Tensor b3
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_shallow_wide_mlp_fused_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "linear_relu_fwd_cuda",
        "linear_fwd_cuda",
        "shallow_wide_mlp_fwd_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Shallow-wide MLP (2 hidden layers) using custom CUDA kernels:
      - linear + bias + ReLU for hidden layers
      - linear + bias for output layer
    Fast path requires CUDA float32 contiguous tensors.
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        assert isinstance(hidden_layer_sizes, (list, tuple)) and len(hidden_layer_sizes) == 2, \
            "ModelNew expects exactly 2 hidden layers for shallow_wide_mlp"

        h1 = int(hidden_layer_sizes[0])
        h2 = int(hidden_layer_sizes[1])
        in_f = int(input_size)
        out_f = int(output_size)

        self.custom_ops_lib = custom_ops_lib

        self.fc1 = nn.Linear(in_f, h1, bias=True)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.fc3 = nn.Linear(h2, out_f, bias=True)

    def _can_use_cuda(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous()):
            return False
        for fc in (self.fc1, self.fc2, self.fc3):
            w = fc.weight
            b = fc.bias
            if b is None:
                return False
            if not (w.is_cuda and b.is_cuda and w.dtype == torch.float32 and b.dtype == torch.float32):
                return False
            if not (w.is_contiguous() and b.is_contiguous()):
                return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_use_cuda(x):
            return self.custom_ops_lib.shallow_wide_mlp_fwd_cuda(
                x,
                self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias,
                self.fc3.weight, self.fc3.bias,
            )

        x = torch.relu(torch.addmm(self.fc1.bias, x, self.fc1.weight.t()))
        x = torch.relu(torch.addmm(self.fc2.bias, x, self.fc2.weight.t()))
        x = torch.addmm(self.fc3.bias, x, self.fc3.weight.t())
        return x