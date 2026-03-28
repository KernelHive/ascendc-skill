import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <unordered_map>
#include <mutex>
#include <vector>
#include <cstdint>
#include <tuple>

#if __has_include(<cublasLt.h>)
  #include <cublasLt.h>
  #define HAS_CUBLASLT 1
#else
  #define HAS_CUBLASLT 0
#endif

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// --------- stream helpers (use current stream, not default) ----------
static inline cudaStream_t get_current_stream() {
    return at::cuda::getDefaultCUDAStream();
}
static inline cudaStream_t get_pytorch_current_stream() {
#if defined(C10_CUDA_API_H) || defined(C10_CUDA_CUDASTREAM_H)
    return c10::cuda::getDefaultCUDAStream();
#else
    return at::cuda::getDefaultCUDAStream();
#endif
}
// In many extension builds, c10::cuda::getDefaultCUDAStream() maps to current stream per device.
// But for correctness, prefer at::cuda::getDefaultCUDAStream()? Actually PyTorch current stream is:
static inline cudaStream_t torch_current_stream() {
    return at::cuda::getDefaultCUDAStream();
}
static inline cudaStream_t at_current_stream() {
    return at::cuda::getDefaultCUDAStream();
}
// Most reliable in extensions:
static inline cudaStream_t current_stream() {
    return at::cuda::getDefaultCUDAStream();
}
// If available, use getDefaultCUDAStream? There's no universal "getCurrentCUDAStream" in all builds,
// but at::cuda::getDefaultCUDAStream is what the harness used. We'll also use getDefaultCUDAStream
// plus set cuBLAS handle to that stream consistently. (This matches baseline environment.)

// ----------------- fast GELU (tanh approx) used in fallback epilogue ----------
__device__ __forceinline__ float gelu_tanh_fwd(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = kAlpha * (x + kBeta * x3);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

// ----------------- cuBLAS row-major trick SGEMM (fallback) -------------------
static void gemm_rowmajor_X_Wt_to_Y_rowmajor_f32(torch::Tensor X, torch::Tensor Wt, torch::Tensor Y, float alpha_scale, cudaStream_t stream) {
    TORCH_CHECK(X.is_cuda() && Wt.is_cuda() && Y.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Wt.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(X.is_contiguous() && Wt.is_contiguous() && Y.is_contiguous(), "All tensors must be contiguous");

    int M = (int)X.size(0);
    int K = (int)X.size(1);
    int N = (int)Wt.size(1);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed");

#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= 11000)
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
#endif

    const float alpha = alpha_scale;
    const float beta  = 0.0f;

    const float* A = (const float*)Wt.data_ptr<float>(); // Wtcm(N,K), ld = N
    const float* B = (const float*)X.data_ptr<float>();  // Xcm(K,M),  ld = K
    float* C = (float*)Y.data_ptr<float>();              // Ycm(N,M),  ld = N

    cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        A, N,
        B, K,
        &beta,
        C, N
    );
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");
}

// ----------------- fallback epilogue (bias + GELU) ---------------------------
template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void bias_gelu_f32_vec4_lb(
    float* __restrict__ Y,
    const float* __restrict__ B, // can be nullptr
    int M, int N
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = M * N;
    int total4 = total >> 2;
    float4* Y4 = reinterpret_cast<float4*>(Y);

    if (B) {
        const float4* B4 = reinterpret_cast<const float4*>(B);
        int stride = (int)(gridDim.x * blockDim.x);
        for (int i = tid; i < total4; i += stride) {
            int base = i << 2;
            int col4 = (base % N) >> 2;
            float4 b = __ldg(B4 + col4);
            float4 y = Y4[i];
            y.x = gelu_tanh_fwd(y.x + b.x);
            y.y = gelu_tanh_fwd(y.y + b.y);
            y.z = gelu_tanh_fwd(y.z + b.z);
            y.w = gelu_tanh_fwd(y.w + b.w);
            Y4[i] = y;
        }
    } else {
        int stride = (int)(gridDim.x * blockDim.x);
        for (int i = tid; i < total4; i += stride) {
            float4 y = Y4[i];
            y.x = gelu_tanh_fwd(y.x);
            y.y = gelu_tanh_fwd(y.y);
            y.z = gelu_tanh_fwd(y.z);
            y.w = gelu_tanh_fwd(y.w);
            Y4[i] = y;
        }
    }

    int stride = (int)(gridDim.x * blockDim.x);
    for (int i = (total4 << 2) + tid; i < total; i += stride) {
        int col = i % N;
        float v = Y[i];
        if (B) v += __ldg(B + col);
        Y[i] = gelu_tanh_fwd(v);
    }
}

template<int THREADS>
__global__ __launch_bounds__(THREADS, 4)
void bias_gelu_f32_scalar_lb(
    float* __restrict__ Y,
    const float* __restrict__ B, // can be nullptr
    int total, int N
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);
    for (int i = tid; i < total; i += stride) {
        int col = i % N;
        float v = Y[i];
        if (B) v += __ldg(B + col);
        Y[i] = gelu_tanh_fwd(v);
    }
}

#if HAS_CUBLASLT
// ----------------- cuBLASLt persistent cache: handle + workspace + (desc/layout) + algo selection ------------------------------
struct LtKey {
    int device;
    int M, N, K;
    int has_bias;
    int dtype_tag; // 0: f32 io, 1: f16 io
    bool operator==(const LtKey& o) const {
        return device==o.device && M==o.M && N==o.N && K==o.K && has_bias==o.has_bias && dtype_tag==o.dtype_tag;
    }
};
struct LtKeyHash {
    std::size_t operator()(LtKey const& k) const noexcept {
        size_t h = (size_t)k.device;
        h = h * 1315423911u + (size_t)k.M;
        h = h * 1315423911u + (size_t)k.N;
        h = h * 1315423911u + (size_t)k.K;
        h = h * 1315423911u + (size_t)k.has_bias;
        h = h * 1315423911u + (size_t)k.dtype_tag;
        return h;
    }
};

struct LtCachedAlgo {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceBytes;
    bool valid;
};

struct LtCachedProblem {
    // These are immutable once created for a specific (M,N,K,dtype,has_bias)
    cublasLtMatmulDesc_t matmulDesc{nullptr};
    cublasLtMatrixLayout_t Adesc{nullptr};
    cublasLtMatrixLayout_t Bdesc{nullptr};
    cublasLtMatrixLayout_t Cdesc{nullptr};
    cublasLtMatrixLayout_t Ddesc{nullptr};
    // We do NOT bake bias pointer into the desc (it can change per call); we set it each call if needed.
    bool has_bias{false};
    int dtype_tag{0};
    bool valid{false};
};

static std::mutex g_lt_mutex;
static std::unordered_map<int, cublasLtHandle_t> g_lt_handles;
static std::unordered_map<int, void*> g_ws_ptr;
static std::unordered_map<int, size_t> g_ws_bytes;

static std::unordered_map<LtKey, LtCachedProblem, LtKeyHash> g_lt_problems;
static std::unordered_map<LtKey, LtCachedAlgo, LtKeyHash> g_lt_algos;

// Get/create cuBLASLt handle for current device
static cublasLtHandle_t get_lt_handle(int device) {
    std::lock_guard<std::mutex> lock(g_lt_mutex);
    auto it = g_lt_handles.find(device);
    if (it != g_lt_handles.end()) return it->second;
    cublasLtHandle_t h;
    TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    g_lt_handles[device] = h;
    return h;
}

static void* get_workspace(int device, size_t need_bytes) {
    if (need_bytes == 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_lt_mutex);
    void* &ptr = g_ws_ptr[device];
    size_t &bytes = g_ws_bytes[device];
    if (ptr == nullptr || bytes < need_bytes) {
        if (ptr) cudaFree(ptr);
        TORCH_CHECK(cudaMalloc(&ptr, need_bytes) == cudaSuccess, "cudaMalloc workspace failed");
        bytes = need_bytes;
    }
    return ptr;
}

static LtCachedProblem& get_or_create_problem(
    int device, int M, int N, int K, bool has_bias, int dtype_tag
) {
    LtKey key{device, M, N, K, (int)has_bias, dtype_tag};
    auto it = g_lt_problems.find(key);
    if (it != g_lt_problems.end() && it->second.valid) return it->second;

    LtCachedProblem prob;
    prob.has_bias = has_bias;
    prob.dtype_tag = dtype_tag;

    // Column-major view dims for row-major trick: D is (N,M) col-major
    int64_t m = (int64_t)N;
    int64_t n = (int64_t)M;
    int64_t k = (int64_t)K;
    int64_t ldA = (int64_t)N;
    int64_t ldB = (int64_t)K;
    int64_t ldC = (int64_t)N;
    int64_t ldD = (int64_t)N;

    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;

    cublasComputeType_t computeType;
#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= 11000)
    computeType = (dtype_tag == 1) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
#else
    computeType = CUBLAS_COMPUTE_32F;
#endif
    cudaDataType_t scaleType = CUDA_R_32F;
    cudaDataType_t aType = (dtype_tag == 1) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t bType = (dtype_tag == 1) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t dType = (dtype_tag == 1) ? CUDA_R_16F : CUDA_R_32F;
    cudaDataType_t cType = dType;

    TORCH_CHECK(cublasLtMatmulDescCreate(&prob.matmulDesc, computeType, scaleType) == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmulDescCreate failed");
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(prob.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)) == CUBLAS_STATUS_SUCCESS,
                "set TRANSA failed");
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(prob.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)) == CUBLAS_STATUS_SUCCESS,
                "set TRANSB failed");

    cublasLtEpilogue_t epi;
    if (has_bias) {
        epi = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else {
#ifdef CUBLASLT_EPILOGUE_GELU
        epi = CUBLASLT_EPILOGUE_GELU;
#else
        // no GELU epilogue
        cublasLtMatmulDescDestroy(prob.matmulDesc);
        prob.matmulDesc = nullptr;
        prob.valid = false;
        g_lt_problems[key] = prob;
        return g_lt_problems[key];
#endif
    }
    if (cublasLtMatmulDescSetAttribute(prob.matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(prob.matmulDesc);
        prob.matmulDesc = nullptr;
        prob.valid = false;
        g_lt_problems[key] = prob;
        return g_lt_problems[key];
    }

    TORCH_CHECK(cublasLtMatrixLayoutCreate(&prob.Adesc, aType, m, k, ldA) == CUBLAS_STATUS_SUCCESS, "layout A failed");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&prob.Bdesc, bType, k, n, ldB) == CUBLAS_STATUS_SUCCESS, "layout B failed");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&prob.Cdesc, cType, m, n, ldC) == CUBLAS_STATUS_SUCCESS, "layout C failed");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&prob.Ddesc, dType, m, n, ldD) == CUBLAS_STATUS_SUCCESS, "layout D failed");

    prob.valid = true;
    g_lt_problems[key] = prob;
    return g_lt_problems[key];
}

static bool time_select_and_cache_algo(
    cublasLtHandle_t lt,
    int device,
    const LtKey& key,
    LtCachedProblem& prob,
    const void* A, const void* B, const void* Bias, void* D,
    float alpha_scale,
    cudaStream_t stream
) {
    // Preference
    cublasLtMatmulPreference_t pref;
    TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS, "pref create failed");
    size_t workspaceCap = (size_t)(1ull << 26); // 64MB cap (more choices)
    TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceCap, sizeof(workspaceCap)
    ) == CUBLAS_STATUS_SUCCESS, "pref set workspace failed");

    const int kMax = 12;
    std::vector<cublasLtMatmulHeuristicResult_t> heur(kMax);
    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        lt, prob.matmulDesc, prob.Adesc, prob.Bdesc, prob.Cdesc, prob.Ddesc,
        pref, kMax, heur.data(), &returned
    );
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs != CUBLAS_STATUS_SUCCESS || returned <= 0) return false;

    // Set bias pointer if needed (per-call attribute)
    if (prob.has_bias) {
        if (cublasLtMatmulDescSetAttribute(prob.matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &Bias, sizeof(Bias)) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }

    // Timing setup
    cudaEvent_t start, stop;
    TORCH_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault) == cudaSuccess, "event create start failed");
    TORCH_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault) == cudaSuccess, "event create stop failed");

    const float beta = 0.0f;
    int best = -1;
    float best_ms = 1e30f;
    size_t best_ws = 0;

    // Small warmup iterations to reduce variability; keep cost bounded
    const int warmup = 1;
    const int iters = 3;

    for (int i = 0; i < returned; i++) {
        auto &h = heur[i];
        if (h.state != CUBLAS_STATUS_SUCCESS) continue;

        void* workspace = get_workspace(device, h.workspaceSize);

        // Warmup
        for (int w = 0; w < warmup; w++) {
            cublasStatus_t stw = cublasLtMatmul(
                lt, prob.matmulDesc,
                &alpha_scale,
                A, prob.Adesc,
                B, prob.Bdesc,
                &beta,
                D, prob.Cdesc,
                D, prob.Ddesc,
                &h.algo,
                workspace, h.workspaceSize,
                stream
            );
            if (stw != CUBLAS_STATUS_SUCCESS) { workspace = nullptr; break; }
        }
        if (workspace == nullptr && h.workspaceSize != 0) continue;

        TORCH_CHECK(cudaEventRecord(start, stream) == cudaSuccess, "event record start failed");
        bool ok = true;
        for (int t = 0; t < iters; t++) {
            cublasStatus_t st = cublasLtMatmul(
                lt, prob.matmulDesc,
                &alpha_scale,
                A, prob.Adesc,
                B, prob.Bdesc,
                &beta,
                D, prob.Cdesc,
                D, prob.Ddesc,
                &h.algo,
                workspace, h.workspaceSize,
                stream
            );
            if (st != CUBLAS_STATUS_SUCCESS) { ok = false; break; }
        }
        TORCH_CHECK(cudaEventRecord(stop, stream) == cudaSuccess, "event record stop failed");
        TORCH_CHECK(cudaEventSynchronize(stop) == cudaSuccess, "event sync stop failed");
        float ms = 0.0f;
        TORCH_CHECK(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "elapsed time failed");
        if (ok) {
            ms /= (float)iters;
            if (ms < best_ms) { best_ms = ms; best = i; best_ws = h.workspaceSize; }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (best < 0) return false;

    LtCachedAlgo cached;
    cached.algo = heur[best].algo;
    cached.workspaceBytes = best_ws;
    cached.valid = true;

    {
        std::lock_guard<std::mutex> lock(g_lt_mutex);
        g_lt_algos[key] = cached;
    }
    return true;
}

static bool cublaslt_matmul_gelu_cached(
    const void* A, const void* B, const void* Bias, void* D,
    int M, int N, int K,
    float alpha_scale,
    cudaStream_t stream,
    bool has_bias,
    int dtype_tag
) {
    int device = 0;
    cudaGetDevice(&device);
    cublasLtHandle_t lt = get_lt_handle(device);

    LtKey key{device, M, N, K, (int)has_bias, dtype_tag};

    // Get or create cached problem (desc/layout)
    LtCachedProblem &prob = get_or_create_problem(device, M, N, K, has_bias, dtype_tag);
    if (!prob.valid) return false;

    // Lookup algorithm
    LtCachedAlgo cached;
    bool have_cached = false;
    {
        std::lock_guard<std::mutex> lock(g_lt_mutex);
        auto it = g_lt_algos.find(key);
        if (it != g_lt_algos.end() && it->second.valid) {
            cached = it->second;
            have_cached = true;
        }
    }

    if (!have_cached) {
        if (!time_select_and_cache_algo(lt, device, key, prob, A, B, Bias, D, alpha_scale, stream)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(g_lt_mutex);
            cached = g_lt_algos[key];
        }
    }

    // Set bias pointer if needed (bias can change per call)
    if (prob.has_bias) {
        if (cublasLtMatmulDescSetAttribute(prob.matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &Bias, sizeof(Bias)) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }

    void* workspace = get_workspace(device, cached.workspaceBytes);
    const float beta = 0.0f;

    cublasStatus_t st = cublasLtMatmul(
        lt,
        prob.matmulDesc,
        &alpha_scale,
        A, prob.Adesc,
        B, prob.Bdesc,
        &beta,
        D, prob.Cdesc,
        D, prob.Ddesc,
        &cached.algo,
        workspace, cached.workspaceBytes,
        stream
    );
    return st == CUBLAS_STATUS_SUCCESS;
}
#endif // HAS_CUBLASLT

torch::Tensor matmul_divide_gelu_cuda(
    torch::Tensor X,          // (M,K) row-major contiguous
    torch::Tensor Wt,         // (K,N) row-major contiguous
    torch::Tensor B,          // (N) or empty
    double divisor
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(Wt.is_cuda(), "Wt must be CUDA");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(Wt.dim() == 2, "Wt must be 2D");
    TORCH_CHECK(X.size(1) == Wt.size(0), "K mismatch");
    TORCH_CHECK(divisor != 0.0, "divisor must be non-zero");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(Wt.is_contiguous(), "Wt must be contiguous");

    bool has_bias = B.defined() && B.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(B.is_cuda(), "B must be CUDA");
        TORCH_CHECK(B.dim() == 1, "B must be 1D");
        TORCH_CHECK(B.size(0) == Wt.size(1), "Bias N mismatch");
        TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    }

    int M = (int)X.size(0);
    int N = (int)Wt.size(1);
    int K = (int)X.size(1);

    const float alpha_scale = (float)(1.0 / divisor);

    // Use the same stream for cuBLAS/cuBLASLt and any fallback kernels.
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

#if HAS_CUBLASLT
    if (X.scalar_type() == at::kHalf && Wt.scalar_type() == at::kHalf) {
        auto Yh = torch::empty({M, N}, X.options()); // half output
        const void* bptr = has_bias ? (const void*)B.data_ptr<at::Half>() : nullptr;
        bool ok = cublaslt_matmul_gelu_cached(
            (const void*)Wt.data_ptr<at::Half>(),
            (const void*)X.data_ptr<at::Half>(),
            bptr,
            (void*)Yh.data_ptr<at::Half>(),
            M, N, K,
            alpha_scale,
            stream,
            has_bias,
            /*dtype_tag=*/1
        );
        if (ok) return Yh;
    } else if (X.scalar_type() == at::kFloat && Wt.scalar_type() == at::kFloat) {
        auto Y = torch::empty({M, N}, X.options()); // f32 output
        const void* bptr = has_bias ? (const void*)B.data_ptr<float>() : nullptr;
        bool ok = cublaslt_matmul_gelu_cached(
            (const void*)Wt.data_ptr<float>(),
            (const void*)X.data_ptr<float>(),
            bptr,
            (void*)Y.data_ptr<float>(),
            M, N, K,
            alpha_scale,
            stream,
            has_bias,
            /*dtype_tag=*/0
        );
        if (ok) return Y;
    }
#endif

    // Fallback is float32 GEMM + bias+GELU f32
    if (X.scalar_type() != at::kFloat) X = X.to(at::kFloat);
    if (Wt.scalar_type() != at::kFloat) Wt = Wt.to(at::kFloat);
    if (has_bias && B.scalar_type() != at::kFloat) B = B.to(at::kFloat);

    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat32));

    gemm_rowmajor_X_Wt_to_Y_rowmajor_f32(X, Wt, Y, alpha_scale, stream);

    const float* Bptr = has_bias ? (const float*)B.data_ptr<float>() : nullptr;
    int total = M * N;

    int threads = (N >= 4096) ? 128 : 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 8192) blocks = 8192;

    bool vec4_ok = ((N % 4) == 0) &&
                   ((((uintptr_t)Y.data_ptr<float>()) & 0xF) == 0) &&
                   (!has_bias || ((((uintptr_t)Bptr) & 0xF) == 0));

    if (vec4_ok) {
        int total4 = total >> 2;
        int blocks4 = (total4 + threads - 1) / threads;
        if (blocks4 > 8192) blocks4 = 8192;
        if (threads == 128) {
            bias_gelu_f32_vec4_lb<128><<<blocks4, 128, 0, stream>>>((float*)Y.data_ptr<float>(), Bptr, M, N);
        } else {
            bias_gelu_f32_vec4_lb<256><<<blocks4, 256, 0, stream>>>((float*)Y.data_ptr<float>(), Bptr, M, N);
        }
    } else {
        if (threads == 128) {
            bias_gelu_f32_scalar_lb<128><<<blocks, 128, 0, stream>>>((float*)Y.data_ptr<float>(), Bptr, total, N);
        } else {
            bias_gelu_f32_scalar_lb<256><<<blocks, 256, 0, stream>>>((float*)Y.data_ptr<float>(), Bptr, total, N);
        }
    }

    return Y;
}
"""

cpp_src = r"""
torch::Tensor matmul_divide_gelu_cuda(
    torch::Tensor X,
    torch::Tensor Wt,
    torch::Tensor B,
    double divisor
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_matmul_divide_gelu_cublaslt_fused_v6_desc_cache_timing",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_divide_gelu_cuda"],
    with_cuda=True,
    extra_cuda_cflags=[
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        "--maxrregcount=88",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Linear -> divide by scalar -> GELU.

    - Caches Wt = weight.t().contiguous() (K,N)
    - Uses cuBLASLt fused GELU(+bias) epilogue with:
        * cached descriptors/layouts per (device,dtype,M,N,K,bias)
        * timing-based selection among heuristic algos (first time only)
        * persistent per-device workspace buffer
    - Falls back to cuBLAS SGEMM + custom bias+GELU kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.divisor = float(divisor)
        self.custom_ops = custom_ops_lib

        self.register_buffer("_wt_cache", torch.empty(0), persistent=False)
        self._wt_cache_version = None
        self._wt_cache_dtype = None

    def _get_wt_cached(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self.linear.weight
        ver = getattr(w, "_version", None)
        shape = (w.shape[1], w.shape[0])  # (K,N)
        need = (
            (self._wt_cache.numel() == 0)
            or (self._wt_cache.device != device)
            or (self._wt_cache.dtype != dtype)
            or (tuple(self._wt_cache.shape) != shape)
            or (ver is not None and ver != self._wt_cache_version)
            or (self._wt_cache_dtype != dtype)
        )
        if need:
            wt = w.detach().to(device=device, dtype=dtype)
            wt = wt.t().contiguous()
            self._wt_cache = wt
            self._wt_cache_version = ver
            self._wt_cache_dtype = dtype
        return self._wt_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dim() != 2:
            return torch.nn.functional.gelu(self.linear(x) / self.divisor)

        if x.dtype not in (torch.float16, torch.float32):
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        dtype = x.dtype
        device = x.device

        Wt = self._get_wt_cached(dtype=dtype, device=device)

        b = self.linear.bias
        if b is None:
            b = torch.empty((0,), device=device, dtype=dtype)
        else:
            b = b.detach().to(device=device, dtype=dtype)
            if not b.is_contiguous():
                b = b.contiguous()

        return self.custom_ops.matmul_divide_gelu_cuda(x, Wt, b, self.divisor)