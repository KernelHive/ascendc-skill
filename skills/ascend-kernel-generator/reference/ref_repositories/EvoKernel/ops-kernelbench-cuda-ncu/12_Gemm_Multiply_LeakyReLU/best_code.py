import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// -------------------- helpers --------------------

__device__ __forceinline__ float leaky_relu_f(float v, float negative_slope) {
    return v >= 0.0f ? v : v * negative_slope;
}

// -------------------- cuBLAS SGEMM wrapper --------------------
// Y_rm(M,N) = alpha * (X_rm(M,K) @ W_rm(K,N))
// We pass Wt = W.t().contiguous() in Python, which is shape (K,N) row-major.
// Use column-major trick to avoid explicit transposes.

static void gemm_rowmajor_X_Wt_to_Y_rowmajor_sgemm(torch::Tensor X, torch::Tensor Wt, torch::Tensor Y, float alpha) {
    TORCH_CHECK(X.is_cuda() && Wt.is_cuda() && Y.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Wt.dtype() == torch::kFloat32 && Y.dtype() == torch::kFloat32, "All tensors must be float32");
    TORCH_CHECK(X.is_contiguous() && Wt.is_contiguous() && Y.is_contiguous(), "All tensors must be contiguous");

    int M = (int)X.size(0);
    int K = (int)X.size(1);
    int N = (int)Wt.size(1);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed");

    const float beta  = 0.0f;

    // Column-major mapping:
    // Y_rm(M,N) == Y_cm(N,M)
    // Wt_rm(K,N) == Wt_cm(N,K)
    // X_rm(M,K) == X_cm(K,M)
    const float* A = (const float*)Wt.data_ptr<float>(); // ld = N
    const float* B = (const float*)X.data_ptr<float>();  // ld = K
    float* C = (float*)Y.data_ptr<float>();              // ld = N

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

// -------------------- cuBLASLt fused epilogue (ReLU + bias) --------------------

struct WorkspacePerDevice { void* ptr=nullptr; size_t bytes=0; };
static WorkspacePerDevice g_ws[16];

static void* get_or_alloc_workspace(size_t bytes, int device) {
    if (device < 0 || device >= 16) return nullptr;
    auto& ws = g_ws[device];
    if (bytes == 0) return nullptr;
    if (ws.ptr && ws.bytes >= bytes) return ws.ptr;
    if (ws.ptr) { cudaFree(ws.ptr); ws.ptr=nullptr; ws.bytes=0; }
    void* p=nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    ws.ptr=p; ws.bytes=bytes;
    return p;
}

struct LtAlgoCacheEntry {
    int64_t M=0,N=0,K=0;
    int epi_kind=0; // 1 relu_bias
    int valid=0;
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize=0;
};
static LtAlgoCacheEntry g_algo_cache[8];

static int find_cache_slot(int64_t M,int64_t N,int64_t K,int epi_kind){
    for(int i=0;i<8;i++){
        if(g_algo_cache[i].valid && g_algo_cache[i].M==M && g_algo_cache[i].N==N && g_algo_cache[i].K==K && g_algo_cache[i].epi_kind==epi_kind) return i;
    }
    for(int i=0;i<8;i++) if(!g_algo_cache[i].valid) return i;
    return 0;
}

static bool gemm_rowmajor_cublaslt_fused_relu_bias(
    torch::Tensor X, torch::Tensor Wt, torch::Tensor BiasScaled, torch::Tensor Y, float alpha
){
    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = Wt.size(1);

    cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    cublasLtMatmulDesc_t matmulDesc;
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
#else
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
#endif
    cudaDataType_t scaleType = CUDA_R_32F;

    if (cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType) != CUBLAS_STATUS_SUCCESS) return false;

    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
    if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(matmulDesc); return false;
    }

    const void* biasPtr = (const void*)BiasScaled.data_ptr<float>();
    if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(matmulDesc); return false;
    }

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    if (cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, (uint64_t)M, (uint64_t)K, (int64_t)K) != CUBLAS_STATUS_SUCCESS) { cublasLtMatmulDescDestroy(matmulDesc); return false; }
    if (cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, (uint64_t)K, (uint64_t)N, (int64_t)N) != CUBLAS_STATUS_SUCCESS) { cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(matmulDesc); return false; }
    if (cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, (uint64_t)M, (uint64_t)N, (int64_t)N) != CUBLAS_STATUS_SUCCESS) { cublasLtMatrixLayoutDestroy(Bdesc); cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(matmulDesc); return false; }

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    const float beta = 0.0f;

    int slot = find_cache_slot(M,N,K,1);
    cublasLtMatmulAlgo_t algo;
    size_t wsSize = 0;
    bool have_algo = false;

    if (g_algo_cache[slot].valid && g_algo_cache[slot].M==M && g_algo_cache[slot].N==N && g_algo_cache[slot].K==K && g_algo_cache[slot].epi_kind==1) {
        algo = g_algo_cache[slot].algo;
        wsSize = g_algo_cache[slot].workspaceSize;
        have_algo = true;
    } else {
        cublasLtMatmulPreference_t pref;
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(Cdesc); cublasLtMatrixLayoutDestroy(Bdesc); cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(matmulDesc);
            return false;
        }
        size_t maxWs = (size_t)1 << 22; // 4MB
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

        cublasLtMatmulHeuristicResult_t heur;
        int returned = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, matmulDesc,
            Adesc, Bdesc,
            Cdesc, Cdesc,
            pref,
            1, &heur, &returned
        );
        cublasLtMatmulPreferenceDestroy(pref);

        if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
            cublasLtMatrixLayoutDestroy(Cdesc); cublasLtMatrixLayoutDestroy(Bdesc); cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatmulDescDestroy(matmulDesc);
            return false;
        }
        algo = heur.algo;
        wsSize = heur.workspaceSize;

        g_algo_cache[slot].M=M; g_algo_cache[slot].N=N; g_algo_cache[slot].K=K;
        g_algo_cache[slot].epi_kind=1;
        g_algo_cache[slot].algo=algo;
        g_algo_cache[slot].workspaceSize=wsSize;
        g_algo_cache[slot].valid=1;
        have_algo = true;
    }

    int device=0;
    cudaGetDevice(&device);
    void* workspace = nullptr;
    size_t workspaceSize = wsSize;
    if (workspaceSize > 0) {
        workspace = get_or_alloc_workspace(workspaceSize, device);
        if (!workspace) workspaceSize = 0;
    }

    const void* A = (const void*)X.data_ptr<float>();
    const void* Bp = (const void*)Wt.data_ptr<float>();
    void* C = (void*)Y.data_ptr<float>();

    cublasStatus_t st = cublasLtMatmul(
        ltHandle,
        matmulDesc,
        &alpha,
        A, Adesc,
        Bp, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        have_algo ? &algo : nullptr,
        workspace, workspaceSize,
        stream
    );

    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(matmulDesc);

    return st == CUBLAS_STATUS_SUCCESS;
}

// -------------------- Optimized epilogue: 2D tiled, shared bias --------------------

// Y = leaky_relu( (Y + bias) * mul )   (Y is GEMM output with alpha=1)
// For our common N=8192: use COLS=1024 (256 float4s), ROWS_PER_CTA tuned to amortize bias loads.

template<int COLS, int ROWS_PER_CTA>
__global__ void __launch_bounds__(256, 3)
bias_mul_leaky_relu_tile_vec4(
    float* __restrict__ Y,
    const float* __restrict__ B,
    int M, int N,
    float mul,
    float negative_slope
){
    constexpr int VEC = 4;
    constexpr int COLS4 = COLS / VEC;
    static_assert((COLS % 4) == 0, "COLS must be multiple of 4");

    int col_tile = (int)blockIdx.x;              // over N
    int row_base = (int)blockIdx.y * ROWS_PER_CTA;

    int tid = (int)threadIdx.x;
    int lane4 = tid;                            // 0..255

    int col4_base = col_tile * COLS4;           // in float4 units
    int N4 = N / 4;

    __shared__ float4 sB4[COLS4];

    // Cooperative bias load into shared memory (bias is reused across ROWS_PER_CTA rows)
    if (lane4 < COLS4) {
        const float4* B4 = reinterpret_cast<const float4*>(B);
#if __CUDA_ARCH__ >= 350
        float4 bv;
        bv.x = __ldg((const float*)&B4[col4_base + lane4].x);
        bv.y = __ldg((const float*)&B4[col4_base + lane4].y);
        bv.z = __ldg((const float*)&B4[col4_base + lane4].z);
        bv.w = __ldg((const float*)&B4[col4_base + lane4].w);
#else
        float4 bv = B4[col4_base + lane4];
#endif
        sB4[lane4] = bv;
    }
    __syncthreads();

    float4* Y4 = reinterpret_cast<float4*>(Y);
    int y4_col = col4_base + lane4;

#pragma unroll
    for (int r = 0; r < ROWS_PER_CTA; r++) {
        int row = row_base + r;
        if (row >= M) break;

        int y4_idx = row * N4 + y4_col;

        float4 yv = Y4[y4_idx];
        float4 bv = sB4[lane4];

        // y = leaky_relu((y + b) * mul)
        float x0 = (yv.x + bv.x) * mul;
        float x1 = (yv.y + bv.y) * mul;
        float x2 = (yv.z + bv.z) * mul;
        float x3 = (yv.w + bv.w) * mul;

        yv.x = leaky_relu_f(x0, negative_slope);
        yv.y = leaky_relu_f(x1, negative_slope);
        yv.z = leaky_relu_f(x2, negative_slope);
        yv.w = leaky_relu_f(x3, negative_slope);

        Y4[y4_idx] = yv;
    }
}

__global__ void __launch_bounds__(256, 4)
bias_mul_leaky_relu_inplace_scalar(
    float* __restrict__ y,
    const float* __restrict__ b,
    int total, int N,
    float mul, float negative_slope
){
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    for (int i = tid; i < total; i += (int)(gridDim.x * blockDim.x)) {
        int col = i - (i / N) * N; // avoid % (often compiles better)
#if __CUDA_ARCH__ >= 350
        float bv = __ldg(b + col);
#else
        float bv = b[col];
#endif
        float v = (y[i] + bv) * mul;
        y[i] = leaky_relu_f(v, negative_slope);
    }
}

// -------------------- Entry --------------------

torch::Tensor gemm_multiply_leaky_relu_cuda_entry(
    torch::Tensor X,
    torch::Tensor Wt,
    torch::Tensor B,
    torch::Tensor Bscaled,
    double multiplier,
    double negative_slope
){
    TORCH_CHECK(X.is_cuda() && Wt.is_cuda() && B.is_cuda(), "X/Wt/B must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && Wt.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "X/Wt/B must be float32");
    TORCH_CHECK(X.dim() == 2 && Wt.dim() == 2, "X/Wt must be 2D");
    TORCH_CHECK(B.dim() == 1, "B must be 1D");
    TORCH_CHECK(X.size(1) == Wt.size(0), "K mismatch");
    TORCH_CHECK(Wt.size(1) == B.size(0), "N mismatch");

    if (!X.is_contiguous()) X = X.contiguous();
    if (!Wt.is_contiguous()) Wt = Wt.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();
    if (Bscaled.defined() && Bscaled.numel() > 0 && !Bscaled.is_contiguous()) Bscaled = Bscaled.contiguous();

    int M = (int)X.size(0);
    int N = (int)Wt.size(1);

    float mul = (float)multiplier;
    float ns  = (float)negative_slope;

    auto Y = torch::empty({M, N}, X.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Fast fused path for exact ReLU only
    if (ns == 0.0f && Bscaled.defined() && Bscaled.numel() == N) {
        bool ok = gemm_rowmajor_cublaslt_fused_relu_bias(X, Wt, Bscaled, Y, mul);
        if (ok) return Y;
    }

    // GEMM first (alpha=1), then epilogue does bias+mul+leaky.
    gemm_rowmajor_X_Wt_to_Y_rowmajor_sgemm(X, Wt, Y, 1.0f);

    bool alignedY = (((uintptr_t)Y.data_ptr<float>() & 0xF) == 0);
    bool alignedB = (((uintptr_t)B.data_ptr<float>() & 0xF) == 0);
    bool can_vec4 = alignedY && alignedB && ((N & 3) == 0);

    // Prefer tiled path when N is multiple of 1024 columns (in scalars).
    // For N=8192, this is 8 tiles, and each CTA processes ROWS_PER_CTA rows.
    if (can_vec4 && (N % 1024 == 0)) {
        constexpr int COLS = 1024;
        constexpr int ROWS_PER_CTA = 8; // higher reuse of bias; still keeps occupancy reasonable
        dim3 block(256, 1, 1);
        dim3 grid(N / COLS, (M + ROWS_PER_CTA - 1) / ROWS_PER_CTA, 1);

        bias_mul_leaky_relu_tile_vec4<COLS, ROWS_PER_CTA><<<grid, block, 0, stream>>>(
            (float*)Y.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            M, N, mul, ns
        );
    } else if (can_vec4 && (N % 512 == 0)) {
        // Smaller tile for other shapes (still avoids modulo and increases locality)
        constexpr int COLS = 512;
        constexpr int ROWS_PER_CTA = 8;
        // COLS=512 => COLS4=128; use 256 threads with 128 active for bias load and compute
        dim3 block(256, 1, 1);
        dim3 grid(N / COLS, (M + ROWS_PER_CTA - 1) / ROWS_PER_CTA, 1);
        bias_mul_leaky_relu_tile_vec4<COLS, ROWS_PER_CTA><<<grid, block, 0, stream>>>(
            (float*)Y.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            M, N, mul, ns
        );
    } else {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (blocks > 8192) blocks = 8192;
        bias_mul_leaky_relu_inplace_scalar<<<blocks, threads, 0, stream>>>(
            (float*)Y.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            total, N, mul, ns
        );
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_multiply_leaky_relu_cuda", &gemm_multiply_leaky_relu_cuda_entry, "GEMM + (mul + leaky_relu) optimized (CUDA)");
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gemm_multiply_leaky_relu_cuda_entry(
    torch::Tensor X,
    torch::Tensor Wt,
    torch::Tensor B,
    torch::Tensor Bscaled,
    double multiplier,
    double negative_slope
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_gemm_multiply_leaky_relu_v5_tiled_bias_shmem",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model:
      - caches Wt (= weight.t().contiguous()) to avoid per-forward transpose/copy
      - caches scaled bias (bias * multiplier) for exact cuBLASLt fused ReLU path when negative_slope == 0
      - otherwise: GEMM then improved 2D tiled epilogue (shared bias) for bias+mul+LeakyReLU
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        self.custom_ops_lib = custom_ops_lib

        self.register_buffer("_wt_cache", torch.empty(0), persistent=False)
        self._wt_cache_version = None

        self.register_buffer("_bscaled_cache", torch.empty(0), persistent=False)
        self._b_cache_version = None
        self._bscaled_cache_multiplier = None

    def _get_Wt_cached(self):
        w = self.gemm.weight
        ver = getattr(w, "_version", None)

        need = (
            (not self._wt_cache.is_cuda) or
            (self._wt_cache.numel() == 0) or
            (self._wt_cache.device != w.device) or
            (self._wt_cache.dtype != torch.float32) or
            (self._wt_cache.shape != (w.shape[1], w.shape[0])) or
            (ver is not None and ver != self._wt_cache_version)
        )
        if need:
            wt = w.detach()
            if not wt.is_cuda:
                wt = wt.cuda()
            if wt.dtype != torch.float32:
                wt = wt.float()
            wt = wt.t().contiguous()
            self._wt_cache = wt
            self._wt_cache_version = ver
        return self._wt_cache

    def _get_bias_contig(self, device):
        b = self.gemm.bias
        if b is None:
            raise RuntimeError("This optimization expects bias=True for exact match.")
        if not b.is_cuda:
            b = b.detach().to(device=device)
        if b.dtype != torch.float32:
            b = b.float()
        if not b.is_contiguous():
            b = b.contiguous()
        return b

    def _get_Bscaled_cached(self, b: torch.Tensor):
        ver = getattr(b, "_version", None)
        need = (
            (not self._bscaled_cache.is_cuda) or
            (self._bscaled_cache.numel() == 0) or
            (self._bscaled_cache.device != b.device) or
            (self._bscaled_cache.dtype != torch.float32) or
            (self._bscaled_cache.shape != b.shape) or
            (ver is not None and ver != self._b_cache_version) or
            (self._bscaled_cache_multiplier != self.multiplier)
        )
        if need:
            self._bscaled_cache = (b.detach() * self.multiplier).contiguous()
            self._b_cache_version = ver
            self._bscaled_cache_multiplier = self.multiplier
        return self._bscaled_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        Wt = self._get_Wt_cached()
        b = self._get_bias_contig(device=x.device)

        if self.negative_slope == 0.0:
            bscaled = self._get_Bscaled_cached(b)
        else:
            bscaled = torch.empty((0,), device=x.device, dtype=torch.float32)

        return self.custom_ops_lib.gemm_multiply_leaky_relu_cuda(
            x, Wt, b, bscaled, self.multiplier, self.negative_slope
        )


batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]