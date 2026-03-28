import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------
# Custom CUDA op (v6-successor):
#   cuBLASLt GEMM (FP16 IO, FP32 compute) with BIAS epilogue -> FP16 Y
#   then fused sigmoid + row-sum epilogue (half2 + unroll)
#
# Key improvements vs baseline v5:
# - Use cuBLASLt with BIAS epilogue to remove bias reads in custom kernel.
# - Persistent per-device cuBLASLt workspace buffer (reused, grown as needed).
# - Cache best cuBLASLt algorithm per (device,B,H,K,types) to avoid repeated heuristics.
# - Lighter epilogue kernel: sigmoid+rowsum only, with half2 vectorization + ILP unroll.
# ---------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <unordered_map>
#include <mutex>
#include <vector>
#include <stdint.h>

#ifndef TORCH_CUDABLAS_CHECK
#define TORCH_CUDABLAS_CHECK(EXPR)                                      \
  do {                                                                  \
    cublasStatus_t __err = (EXPR);                                      \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS, "cuBLAS/cuBLASLt error: ", (int)__err); \
  } while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {                                  \
  cudaError_t err = cudaGetLastError();                                      \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err)); \
} while (0)
#endif

// ---------------- reductions ----------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        out = (lane < nwarps) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

// ---------------- sigmoid ----------------
__device__ __forceinline__ half2 sigmoid_half2(half2 x) {
    float2 xf = __half22float2(x);
    // fast math enabled at compilation; __expf ok
    float a = 1.0f / (1.0f + __expf(-xf.x));
    float b = 1.0f / (1.0f + __expf(-xf.y));
    return __floats2half2_rn(a, b);
}

// ---------------- fused sigmoid+rowsum ----------------
// Yh: (B,H) half row-major contiguous, already includes bias
__global__ __launch_bounds__(256, 3) void sigmoid_rowsum_half2_kernel(
    const half* __restrict__ Yh,
    float* __restrict__ out, // (B)
    int B, int H
) {
    int b = (int)blockIdx.x;
    if (b >= B) return;

    const int64_t row_base = (int64_t)b * (int64_t)H;

    float acc = 0.0f;

    // H even required
    int H2 = H >> 1;
    const half2* __restrict__ y2p = reinterpret_cast<const half2*>(Yh + row_base);

    // Small unroll for ILP without extra sync
    constexpr int UNROLL = 4;

    int idx = (int)threadIdx.x;
    int stride = (int)blockDim.x;

    int i2 = idx;
    int limit = H2 - (UNROLL - 1) * stride;
    for (; i2 < limit; i2 += UNROLL * stride) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            half2 y2 = y2p[i2 + u * stride];
            half2 s2 = sigmoid_half2(y2);
            float2 sf = __half22float2(s2);
            acc += sf.x + sf.y;
        }
    }
    for (; i2 < H2; i2 += stride) {
        half2 y2 = y2p[i2];
        half2 s2 = sigmoid_half2(y2);
        float2 sf = __half22float2(s2);
        acc += sf.x + sf.y;
    }

    float sum = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[b] = sum;
}

// ---------------- cuBLASLt caching ----------------
struct AlgoKey {
    int device;
    int B, H, K;
    int64_t w_ptr;
    // We only cache by shape/device (and pointer for safety); weight can change rarely.
    bool operator==(const AlgoKey& o) const {
        return device == o.device && B == o.B && H == o.H && K == o.K && w_ptr == o.w_ptr;
    }
};

struct AlgoKeyHash {
    std::size_t operator()(AlgoKey const& s) const noexcept {
        // simple hash combine
        auto h = (uint64_t)s.device;
        h = h * 1315423911u + (uint64_t)s.B;
        h = h * 1315423911u + (uint64_t)s.H;
        h = h * 1315423911u + (uint64_t)s.K;
        h = h * 1315423911u + (uint64_t)s.w_ptr;
        return (size_t)h;
    }
};

struct CachedAlgo {
    cublasLtMatmulHeuristicResult_t result;
    bool valid{false};
    size_t workspace_size{0};
};

static std::mutex g_mutex;
static std::unordered_map<AlgoKey, CachedAlgo, AlgoKeyHash> g_algo_cache;

// persistent per-device workspace
static std::unordered_map<int, torch::Tensor> g_workspace;
static std::unordered_map<int, size_t> g_workspace_size;

// Ensure workspace tensor of at least bytes on this device.
static void* get_workspace(int device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_workspace.find(device);
    size_t cur = 0;
    if (it != g_workspace.end()) cur = g_workspace_size[device];
    if (it == g_workspace.end() || cur < bytes) {
        // grow (round up to 2MB)
        size_t rounded = ((bytes + (2<<20) - 1) / (2<<20)) * (2<<20);
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device));
        g_workspace[device] = torch::empty({(long long)rounded}, opts);
        g_workspace_size[device] = rounded;
    }
    return g_workspace[device].data_ptr();
}

// ---------------- entry point ----------------
// x: float32 (B,K) contiguous
// w_h: half (H,K) contiguous
// b_h: half (H) contiguous
torch::Tensor matmul_sigmoid_sum_cuda(torch::Tensor x,
                                      torch::Tensor w_h,
                                      torch::Tensor b_h,
                                      int64_t workspace_cap_bytes,
                                      int heuristic_count) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_h.is_cuda(), "w_h must be CUDA");
    TORCH_CHECK(b_h.is_cuda(), "b_h must be CUDA");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w_h.scalar_type() == torch::kFloat16, "w_h must be float16");
    TORCH_CHECK(b_h.scalar_type() == torch::kFloat16, "b_h must be float16");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_h.is_contiguous(), "w_h must be contiguous");
    TORCH_CHECK(b_h.is_contiguous(), "b_h must be contiguous");

    TORCH_CHECK(x.dim() == 2, "x must be 2D (B,K)");
    TORCH_CHECK(w_h.dim() == 2, "w_h must be 2D (H,K)");
    TORCH_CHECK(b_h.dim() == 1, "b_h must be 1D (H)");

    const int B = (int)x.size(0);
    const int K = (int)x.size(1);
    const int H = (int)w_h.size(0);

    TORCH_CHECK((int)w_h.size(1) == K, "w_h must be (H,K)");
    TORCH_CHECK((int)b_h.size(0) == H, "b_h must be (H)");
    TORCH_CHECK((H % 2) == 0, "H must be even for half2 epilogue");

    if (workspace_cap_bytes <= 0) workspace_cap_bytes = (32ll << 20); // 32MB default
    if (heuristic_count <= 0) heuristic_count = 32;

    c10::cuda::CUDAGuard device_guard(x.device());
    int device = x.get_device();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Convert x to FP16 for GEMM
    auto xh = x.to(torch::kFloat16);

    // GEMM output in FP16: Yh is (B,H) row-major contiguous.
    auto yh = torch::empty({B, H}, xh.options());

    // output float32 (B,1)
    auto out = torch::empty({B, 1}, x.options());
    auto out_flat = out.view({B});

    // cuBLASLt setup
    cublasLtHandle_t lt;
    TORCH_CUDABLAS_CHECK(cublasLtCreate(&lt));

    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;

    // We use column-major descriptors with the same trick as baseline:
    // Interpret row-major (B,K) as column-major (K,B) with leading dim K.
    // Compute Y^T(H,B) = W(H,K) * X^T(K,B); store into Y with leading dim H.
    // This yields Y as row-major (B,H) in memory.
    cublasLtMatmulDesc_t matmulDesc;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Set epilogue bias
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    const half* bias_ptr = (const half*)b_h.data_ptr<at::Half>();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

    // Layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;

    // A: W (H,K) col-major ld=H
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, Atype, H, K, H));
    // B: X (K,B) col-major ld=K  (xh is (B,K) row-major)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, Btype, K, B, K));
    // C: Y (H,B) col-major ld=H  (yh is (B,H) row-major)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, H, B, H));

    // Prefer Tensor Cores
    cublasLtMatmulPreference_t pref;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_cap_bytes, sizeof(workspace_cap_bytes)));

    // Cache key
    AlgoKey key;
    key.device = device;
    key.B = B; key.H = H; key.K = K;
    key.w_ptr = (int64_t)w_h.data_ptr();

    CachedAlgo cached;
    bool have_algo = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_algo_cache.find(key);
        if (it != g_algo_cache.end() && it->second.valid) {
            cached = it->second;
            have_algo = true;
        }
    }

    // Choose algo if not cached
    if (!have_algo) {
        std::vector<cublasLtMatmulHeuristicResult_t> results(heuristic_count);
        int returned = 0;

        cublasLtMatmulAlgo_t algo;
        // Let cuBLASLt pick via heuristics
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            lt, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, pref,
            heuristic_count, results.data(), &returned));

        TORCH_CHECK(returned > 0, "cuBLASLt heuristic returned no results");

        // Pick first valid result (should already be sorted)
        CachedAlgo best;
        best.valid = false;
        for (int i = 0; i < returned; ++i) {
            if (results[i].state == CUBLAS_STATUS_SUCCESS) {
                best.result = results[i];
                best.valid = true;
                best.workspace_size = results[i].workspaceSize;
                break;
            }
        }
        TORCH_CHECK(best.valid, "cuBLASLt found no valid algo");

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_algo_cache[key] = best;
        }
        cached = best;
        have_algo = true;
    }

    // Workspace
    void* workspace = nullptr;
    size_t ws = cached.workspace_size;
    if (ws > 0) {
        if ((int64_t)ws > workspace_cap_bytes) {
            // Shouldn't happen due to preference, but guard
            ws = (size_t)workspace_cap_bytes;
        }
        workspace = get_workspace(device, ws);
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const half* A = (const half*)w_h.data_ptr<at::Half>();
    const half* Bp = (const half*)xh.data_ptr<at::Half>();
    half* C = (half*)yh.data_ptr<at::Half>();

    // Execute matmul
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(
        lt,
        matmulDesc,
        &alpha,
        A, Adesc,
        Bp, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        &cached.result.algo,
        workspace, ws,
        stream
    ));

    // Cleanup descriptors (lt handle destroyed at end)
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    TORCH_CUDABLAS_CHECK(cublasLtDestroy(lt));

    // Fused sigmoid+rowsum
    const int threads = 256;
    sigmoid_rowsum_half2_kernel<<<(unsigned)B, threads, 0, stream>>>(
        (const half*)yh.data_ptr<at::Half>(),
        (float*)out_flat.data_ptr<float>(),
        B, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_source = r"""
torch::Tensor matmul_sigmoid_sum_cuda(torch::Tensor x,
                                      torch::Tensor w_h,
                                      torch::Tensor b_h,
                                      int64_t workspace_cap_bytes,
                                      int heuristic_count);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_sigmoid_sum_v6lt",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_sigmoid_sum_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized CUDA implementation for:
      linear(x) -> sigmoid -> sum(dim=1, keepdim=True)

    Fast path:
      - x: CUDA float32 contiguous (B,K)
      - weight/bias cached as FP16 contiguous buffers
      - hidden_size must be even (half2 epilogue)
      - cuBLASLt used with bias epilogue; custom kernel does sigmoid+rowsum only
    """
    def __init__(self, input_size, hidden_size, workspace_cap_mb: int = 32, heuristic_count: int = 32):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.custom_ops_lib = custom_ops_lib

        self.workspace_cap_bytes = int(workspace_cap_mb) * (1 << 20)
        self.heuristic_count = int(heuristic_count)

        # FP16 cached parameters (device-specific)
        self._w_h_cache = None
        self._b_h_cache = None
        self._cache_key = None  # identity/metadata key

    @torch.no_grad()
    def _maybe_refresh_fp16_cache(self):
        w = self.linear.weight
        b = self.linear.bias
        if b is None:
            b = torch.zeros((w.size(0),), device=w.device, dtype=w.dtype)

        key = (
            int(w.data_ptr()),
            int(b.data_ptr()),
            str(w.device),
            tuple(w.shape),
            tuple(b.shape),
            tuple(w.stride()),
            tuple(b.stride()),
            w.dtype,
            b.dtype,
        )
        if self._cache_key == key and self._w_h_cache is not None and self._b_h_cache is not None:
            return

        w_src = w.contiguous() if not w.is_contiguous() else w
        if w_src.dtype != torch.float32:
            w_src = w_src.float()

        b_src = b.contiguous() if not b.is_contiguous() else b
        if b_src.dtype != torch.float32:
            b_src = b_src.float()

        self._w_h_cache = w_src.to(dtype=torch.float16)
        self._b_h_cache = b_src.to(dtype=torch.float16)
        self._cache_key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda) or x.dtype != torch.float32:
            y = self.linear(x)
            y = torch.sigmoid(y)
            return torch.sum(y, dim=1, keepdim=True)

        if not x.is_contiguous():
            x = x.contiguous()

        H = self.linear.weight.size(0)
        if (H % 2) != 0:
            y = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
            y = torch.sigmoid(y)
            return torch.sum(y, dim=1, keepdim=True)

        self._maybe_refresh_fp16_cache()

        return self.custom_ops_lib.matmul_sigmoid_sum_cuda(
            x, self._w_h_cache, self._b_h_cache, self.workspace_cap_bytes, self.heuristic_count
        )