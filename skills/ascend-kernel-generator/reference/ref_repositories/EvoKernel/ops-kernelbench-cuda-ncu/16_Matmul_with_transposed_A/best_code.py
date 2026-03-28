import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Optimized extension for: C = A^T @ B
# A: [K, M] row-major contiguous
# B: [K, N] row-major contiguous
# C: [M, N]
#
# Fast path: cuBLASLt TF32 TensorCore matmul (no explicit transpose):
#   C[M,N] = (A[K,M])^T @ B[K,N]
#   Map to Lt: (A_lt = B) [M=???] careful:
#     We want: (M x K) @ (K x N)
#     Use A_lt = A_as_Boperand?:
#       Let Lt compute: D = Op(A0) * Op(B0)
#       Choose:
#         A0 = B (shape K x N), Op(A0)=T => N x K  (not desired)
#       Better:
#         A0 = A (shape K x M), Op(A0)=T => M x K (desired left operand)
#         B0 = B (shape K x N), Op(B0)=N => K x N (desired right operand)
#       So: TRANSA=T, TRANSB=N, with layouts:
#         A0 rows=K cols=M row-major (lda=M)
#         B0 rows=K cols=N row-major (ldb=N)
#         C rows=M cols=N row-major (ldc=N)
#
# Fallback: lighter FP32 shared-memory tiled kernel.
# ------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include <stdint.h>
#include <unordered_map>
#include <mutex>
#include <vector>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// --------------------
// Utilities
// --------------------
static inline void checkCublasLt(cublasStatus_t st, const char* msg) {
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, msg, " (cublas status=", (int)st, ")");
}

__host__ __device__ __forceinline__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// --------------------
// Fallback kernel (reduced register pressure):
// C[M,N] = A^T[M,K] @ B[K,N]
// A input is [K,M] row-major; B is [K,N] row-major.
// Tile: BMxBN, K by BK.
// 256 threads (16x16). Each thread computes TMxTN (small).
// --------------------
template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256, 3)
void matmul_at_b_fallback_kernel(
    const float* __restrict__ A, // [K,M]
    const float* __restrict__ B, // [K,N]
    float* __restrict__ C,       // [M,N]
    int K, int M, int N
) {
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;

    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    __shared__ float As[BK][BM]; // stage A tile: [k, m]
    __shared__ float Bs[BK][BN]; // stage B tile: [k, n]

    const int out_m0 = block_m + ty * TM;
    const int out_n0 = block_n + tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load As and Bs
        // As: BK*BM floats, Bs: BK*BN floats
        // Total floats = BK*(BM+BN). With BK=16, BM=64, BN=64 => 2048 floats => 8 floats/thread.
        int base = tid * 8;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            int idx = base + t; // 0..2047
            if (idx < BK * BM) {
                int kk = idx / BM;      // 0..BK-1
                int mm = idx - kk * BM; // 0..BM-1
                int gk = k0 + kk;
                int gm = block_m + mm;
                float v = 0.0f;
                if (gk < K && gm < M) v = ldg_f32(A + (int64_t)gk * M + gm);
                As[kk][mm] = v;
            } else {
                int bidx = idx - BK * BM; // 0..BK*BN-1
                int kk = bidx / BN;
                int nn = bidx - kk * BN;
                int gk = k0 + kk;
                int gn = block_n + nn;
                float v = 0.0f;
                if (gk < K && gn < N) v = ldg_f32(B + (int64_t)gk * N + gn);
                Bs[kk][nn] = v;
            }
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float b_frag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int nn = tx * TN + j;
                b_frag[j] = Bs[kk][nn];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                int mm = ty * TM + i;
                float a = As[kk][mm];
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a, b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = out_m0 + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gn = out_n0 + j;
                if (gn < N) C[(int64_t)gm * N + gn] = acc[i][j];
            }
        }
    }
}

// --------------------
// cuBLASLt persistent state
// --------------------
struct AlgoKey {
    int device;
    int M, N, K;
    bool operator==(const AlgoKey& o) const {
        return device == o.device && M == o.M && N == o.N && K == o.K;
    }
};
struct AlgoKeyHash {
    size_t operator()(const AlgoKey& k) const {
        size_t h = (size_t)k.device;
        h = h * 1315423911u + (size_t)k.M;
        h = h * 1315423911u + (size_t)k.N;
        h = h * 1315423911u + (size_t)k.K;
        return h;
    }
};
struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    bool valid;
};

static std::mutex g_mutex;
static cublasLtHandle_t g_lt = nullptr;

struct DeviceWs { void* ptr = nullptr; size_t size = 0; };
static std::unordered_map<int, DeviceWs> g_ws;
static std::unordered_map<AlgoKey, CachedAlgo, AlgoKeyHash> g_cache;

static void ensureLt() {
    if (g_lt == nullptr) checkCublasLt(cublasLtCreate(&g_lt), "cublasLtCreate failed");
}

static void ensureWorkspace(int device, size_t bytes) {
    auto &ws = g_ws[device];
    if (ws.size >= bytes) return;
    if (ws.ptr) {
        cudaFree(ws.ptr);
        ws.ptr = nullptr;
        ws.size = 0;
    }
    const size_t align = 2u * 1024u * 1024u;
    size_t newSize = ((bytes + align - 1) / align) * align;
    cudaError_t err = cudaMalloc(&ws.ptr, newSize);
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc workspace failed: ", cudaGetErrorString(err));
    ws.size = newSize;
}

static bool run_cublaslt_tf32_atb(
    const float* A, const float* B, float* C,
    int K, int M, int N,
    cudaStream_t stream,
    int device
) {
    ensureLt();

    // We want: C(M,N) = A^T(M,K) @ B(K,N)
    // Use: TRANSA = T on A0=A (K,M) => Op(A0)=(M,K)
    //      TRANSB = N on B0=B (K,N) => Op(B0)=(K,N)

    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t aLayout, bLayout, cLayout;

    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    cudaDataType_t scaleType = CUDA_R_32F;
    checkCublasLt(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType),
                  "MatmulDescCreate failed");

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    checkCublasLt(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
                  "set TRANSA failed");
    checkCublasLt(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)),
                  "set TRANSB failed");

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    int64_t lda = (int64_t)M; // A is KxM row-major
    int64_t ldb = (int64_t)N; // B is KxN row-major
    int64_t ldc = (int64_t)N; // C is MxN row-major

    checkCublasLt(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_32F, K, M, lda), "aLayout failed");
    checkCublasLt(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_32F, K, N, ldb), "bLayout failed");
    checkCublasLt(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, M, N, ldc), "cLayout failed");

    checkCublasLt(cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  "a order failed");
    checkCublasLt(cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  "b order failed");
    checkCublasLt(cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  "c order failed");

    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulPreference_t pref;
    checkCublasLt(cublasLtMatmulPreferenceCreate(&pref), "PreferenceCreate failed");
    size_t wsCap = 32u * 1024u * 1024u;
    checkCublasLt(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsCap, sizeof(wsCap)),
        "set ws cap failed");

    AlgoKey key{device, M, N, K};

    CachedAlgo ca{};
    bool haveCached = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_cache.find(key);
        if (it != g_cache.end() && it->second.valid) {
            ca = it->second;
            haveCached = true;
        }
    }

    cublasLtMatmulAlgo_t algo{};
    size_t algoWs = 0;
    bool haveAlgo = false;

    if (haveCached) {
        algo = ca.algo;
        algoWs = ca.workspaceSize;
        haveAlgo = true;
    } else {
        const int maxAlgos = 16;
        std::vector<cublasLtMatmulHeuristicResult_t> results(maxAlgos);
        int returned = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
            g_lt, opDesc, aLayout, bLayout, cLayout, cLayout,
            pref, maxAlgos, results.data(), &returned
        );
        if (st == CUBLAS_STATUS_SUCCESS && returned > 0) {
            algo = results[0].algo;
            algoWs = results[0].workspaceSize;
            haveAlgo = true;
            std::lock_guard<std::mutex> lock(g_mutex);
            g_cache[key] = CachedAlgo{algo, algoWs, true};
        }
    }

    void* workspace = nullptr;
    if (haveAlgo && algoWs > 0) {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensureWorkspace(device, algoWs);
        workspace = g_ws[device].ptr;
    }

    bool ok = false;
    if (haveAlgo) {
        cublasStatus_t st = cublasLtMatmul(
            g_lt,
            opDesc,
            &alpha,
            A, aLayout,
            B, bLayout,
            &beta,
            C, cLayout,
            C, cLayout,
            &algo,
            workspace, algoWs,
            stream
        );
        ok = (st == CUBLAS_STATUS_SUCCESS);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    cublasLtMatmulDescDestroy(opDesc);

    return ok;
}

torch::Tensor matmul_with_transposed_a_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (K,M)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,N)");
    TORCH_CHECK(A.size(0) == B.size(0), "K must match: A.shape[0] == B.shape[0]");

    at::cuda::CUDAGuard device_guard(A.device());
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    const int64_t K64 = A_c.size(0);
    const int64_t M64 = A_c.size(1);
    const int64_t N64 = B_c.size(1);
    TORCH_CHECK(K64 <= INT_MAX && M64 <= INT_MAX && N64 <= INT_MAX, "Sizes too large for int indexing");

    const int K = (int)K64;
    const int M = (int)M64;
    const int N = (int)N64;

    auto out = torch::empty({M, N}, A_c.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int device = -1;
    cudaGetDevice(&device);

    // cuBLASLt TF32 fast path
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensureLt();
    }
    bool ran = run_cublaslt_tf32_atb(
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        out.data_ptr<float>(),
        K, M, N,
        stream, device
    );

    if (!ran) {
        // FP32 fallback (lower register pressure than baseline)
        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 16;
        constexpr int TM = 2;
        constexpr int TN = 2;

        dim3 block(16, 16);
        dim3 grid(div_up_int(N, BN), div_up_int(M, BM));
        matmul_at_b_fallback_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
            A_c.data_ptr<float>(),
            B_c.data_ptr<float>(),
            out.data_ptr<float>(),
            K, M, N
        );
    }

    return out;
}
"""

cpp_source = r"""
torch::Tensor matmul_with_transposed_a_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_with_transposed_a_cublaslt_tf32_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_with_transposed_a_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model: C = A.T @ B, where A is provided as [K,M] and B as [K,N].
    Uses cuBLASLt TF32 TensorCore matmul fast path with caching + persistent workspace,
    with an FP32 fallback kernel for robustness.
    """
    def __init__(self):
        super().__init__()
        self.custom_ops = custom_ops_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.matmul_with_transposed_a_cuda(A, B)