import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA extension: Nystrom attention forward (inference-oriented)
# Specialization: N=512, D=64, M=32, float32 contiguous, Q/K/V [B,H,N,D].
#
# Computes (per batch-head):
#   idx[j] = floor(j*(N-1)/(M-1))  (uniform landmarks like torch.linspace long)
#   QL[j,:] = Q[idx[j],:], KL[j,:] = K[idx[j],:]
#   A[n,j] = softmax_j( (Q[n]·KL[j]) / sqrt(D) )
#   B[i,j] = softmax_j( (QL[i]·KL[j]) / sqrt(D) )
#   Bs = 0.5*(B + B^T) + eps*I  (symmetrize + regularize)
#   Solve Bs * X = Y via Cholesky (SPD) where:
#       Y = (A^T @ V)  => [M,D]
#       Also y1 = (A^T @ 1) => [M]
#   Then:
#       Out[n,:] = (A[n,:] @ X) / (A[n,:] @ x1 + eps)
# where x1 solves Bs*x1 = y1.
#
# Notes:
# - Avoids explicit inversion/pinv and Gauss-Jordan to prevent NaNs.
# - Uses stable softmax with block-wide max/sum reductions.
# - Uses a single block per (bh) for small-matrix ops (B factor/solve),
#   and separate grids for A computation and output application.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}
__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

// floor(j*(N-1)/(M-1)) for j=0..M-1
__device__ __forceinline__ int landmark_index(int j, int N, int M) {
    if (M <= 1) return 0;
    long long num = (long long)j * (long long)(N - 1);
    long long den = (long long)(M - 1);
    return (int)(num / den);
}

// Gather Q_landmarks and K_landmarks: [BH,M,D] each
__global__ void gather_landmarks_kernel(
    const float* __restrict__ Q, // [BH,N,D]
    const float* __restrict__ K, // [BH,N,D]
    float* __restrict__ QL,      // [BH,M,D]
    float* __restrict__ KL,      // [BH,M,D]
    int N, int D, int M
) {
    int bh = (int)blockIdx.x;
    int j  = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int idx = landmark_index(j, N, M);

    const float* qsrc = Q + ((size_t)bh * (size_t)N + (size_t)idx) * (size_t)D;
    const float* ksrc = K + ((size_t)bh * (size_t)N + (size_t)idx) * (size_t)D;
    float* qdst = QL + ((size_t)bh * (size_t)M + (size_t)j) * (size_t)D;
    float* kdst = KL + ((size_t)bh * (size_t)M + (size_t)j) * (size_t)D;

    for (int d = tid; d < D; d += (int)blockDim.x) {
        qdst[d] = qsrc[d];
        kdst[d] = ksrc[d];
    }
}

// A: [BH,N,M] softmax over M of (Q[n] dot KL[j]) / sqrt(D)
__global__ void compute_A_kernel(
    const float* __restrict__ Q,  // [BH,N,D]
    const float* __restrict__ KL, // [BH,M,D]
    float* __restrict__ A,        // [BH,N,M]
    int N, int D, int M, float inv_sqrt_d
) {
    int bh = (int)blockIdx.x;
    int n  = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    const float* q = Q + ((size_t)bh * (size_t)N + (size_t)n) * (size_t)D;
    const float* kl_bh = KL + (size_t)bh * (size_t)M * (size_t)D;
    float* arow = A + ((size_t)bh * (size_t)N + (size_t)n) * (size_t)M;

    float local_max = -INFINITY;
    for (int j = tid; j < M; j += (int)blockDim.x) {
        const float* k = kl_bh + (size_t)j * (size_t)D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) dot = fmaf(q[d], k[d], dot);
        float logit = dot * inv_sqrt_d;
        arow[j] = logit;
        local_max = fmaxf(local_max, logit);
    }
    float mx = block_reduce_max(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < M; j += (int)blockDim.x) {
        float e = __expf(arow[j] - mx);
        arow[j] = e;
        local_sum += e;
    }
    float denom = block_reduce_sum(local_sum);
    __syncthreads();

    float inv = 1.0f / fmaxf(denom, 1e-20f);
    for (int j = tid; j < M; j += (int)blockDim.x) arow[j] *= inv;
}

// B: [BH,M,M] softmax over M (rowwise) of (QL[i] dot KL[j]) / sqrt(D)
__global__ void compute_B_kernel(
    const float* __restrict__ QL, // [BH,M,D]
    const float* __restrict__ KL, // [BH,M,D]
    float* __restrict__ B,        // [BH,M,M]
    int D, int M, float inv_sqrt_d
) {
    int bh = (int)blockIdx.x;
    int i  = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    const float* ql_bh = QL + (size_t)bh * (size_t)M * (size_t)D;
    const float* kl_bh = KL + (size_t)bh * (size_t)M * (size_t)D;
    const float* q = ql_bh + (size_t)i * (size_t)D;
    float* brow = B + ((size_t)bh * (size_t)M + (size_t)i) * (size_t)M;

    float local_max = -INFINITY;
    for (int j = tid; j < M; j += (int)blockDim.x) {
        const float* k = kl_bh + (size_t)j * (size_t)D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) dot = fmaf(q[d], k[d], dot);
        float logit = dot * inv_sqrt_d;
        brow[j] = logit;
        local_max = fmaxf(local_max, logit);
    }
    float mx = block_reduce_max(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < M; j += (int)blockDim.x) {
        float e = __expf(brow[j] - mx);
        brow[j] = e;
        local_sum += e;
    }
    float denom = block_reduce_sum(local_sum);
    __syncthreads();

    float inv = 1.0f / fmaxf(denom, 1e-20f);
    for (int j = tid; j < M; j += (int)blockDim.x) brow[j] *= inv;
}

// Compute Y = A^T @ V : [BH,M,D]
__global__ void compute_AT_V_kernel(
    const float* __restrict__ A, // [BH,N,M]
    const float* __restrict__ V, // [BH,N,D]
    float* __restrict__ Y,       // [BH,M,D]
    int N, int M, int D
) {
    int bh = (int)blockIdx.x;
    int m  = (int)blockIdx.y;
    int d  = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x;
    if (d >= D) return;

    const float* Abh = A + (size_t)bh * (size_t)N * (size_t)M;
    const float* Vbh = V + (size_t)bh * (size_t)N * (size_t)D;
    float* Ybh = Y + (size_t)bh * (size_t)M * (size_t)D;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float a = Abh[(size_t)n * (size_t)M + (size_t)m];
        float v = Vbh[(size_t)n * (size_t)D + (size_t)d];
        acc = fmaf(a, v, acc);
    }
    Ybh[(size_t)m * (size_t)D + (size_t)d] = acc;
}

// Compute y1 = A^T @ 1 : [BH,M]
__global__ void compute_AT_ones_kernel(
    const float* __restrict__ A, // [BH,N,M]
    float* __restrict__ y1,      // [BH,M]
    int N, int M
) {
    int bh = (int)blockIdx.x;
    int m  = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    const float* Abh = A + (size_t)bh * (size_t)N * (size_t)M;

    float partial = 0.0f;
    for (int n = tid; n < N; n += (int)blockDim.x) {
        partial += Abh[(size_t)n * (size_t)M + (size_t)m];
    }
    float sum = block_reduce_sum(partial);
    if (tid == 0) y1[(size_t)bh * (size_t)M + (size_t)m] = sum;
}

// Cholesky factorization (lower) and solves on Bs (M=32).
// Input: B [BH,M,M] row-stochastic (softmax).
// Output: X [BH,M,D] solving Bs*X = Y, and x1 [BH,M] solving Bs*x1 = y1.
__global__ void chol_solve_kernel_M32(
    const float* __restrict__ B,   // [BH,32,32]
    const float* __restrict__ Y,   // [BH,32,64]
    const float* __restrict__ y1,  // [BH,32]
    float* __restrict__ X,         // [BH,32,64]
    float* __restrict__ x1,        // [BH,32]
    float eps
) {
    constexpr int M = 32;
    constexpr int D = 64;
    int bh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    __shared__ float L[M*M];   // lower triangular stored full
    __shared__ float rhs[D*M]; // Y copy (32*64=2048 floats) -> 8KB
    __shared__ float rhs1[M];  // y1 copy

    const float* Bbh = B + (size_t)bh * M * M;
    const float* Ybh = Y + (size_t)bh * M * D;
    const float* y1bh = y1 + (size_t)bh * M;
    float* Xbh = X + (size_t)bh * M * D;
    float* x1bh = x1 + (size_t)bh * M;

    // Load and symmetrize + regularize into L buffer (temporarily as Bs)
    for (int idx = tid; idx < M*M; idx += (int)blockDim.x) {
        int i = idx / M;
        int j = idx - i * M;
        float bij = Bbh[(size_t)i * M + (size_t)j];
        float bji = Bbh[(size_t)j * M + (size_t)i];
        float v = 0.5f * (bij + bji);
        if (i == j) v += eps;
        L[idx] = v;
    }
    // Load RHS
    for (int idx = tid; idx < M*D; idx += (int)blockDim.x) rhs[idx] = Ybh[idx];
    for (int i = tid; i < M; i += (int)blockDim.x) rhs1[i] = y1bh[i];
    __syncthreads();

    // Cholesky (single thread for stability/simplicity; M=32 small)
    if (tid == 0) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j <= i; ++j) {
                float sum = L[i*M + j];
                for (int k = 0; k < j; ++k) sum -= L[i*M + k] * L[j*M + k];
                if (i == j) {
                    sum = fmaxf(sum, 1e-8f);
                    L[i*M + j] = sqrtf(sum);
                } else {
                    L[i*M + j] = sum / L[j*M + j];
                }
            }
            for (int j = i + 1; j < M; ++j) L[i*M + j] = 0.0f;
        }
    }
    __syncthreads();

    // Solve L * Z = rhs (forward) for each d in parallel
    for (int d = tid; d < D; d += (int)blockDim.x) {
        // forward solve for column d
        for (int i = 0; i < M; ++i) {
            float sum = rhs[i*D + d];
            for (int k = 0; k < i; ++k) sum -= L[i*M + k] * rhs[k*D + d];
            rhs[i*D + d] = sum / L[i*M + i];
        }
        // backward solve L^T * X = Z (reuse rhs as Z, write Xbh)
        for (int i = M - 1; i >= 0; --i) {
            float sum = rhs[i*D + d];
            for (int k = i + 1; k < M; ++k) sum -= L[k*M + i] * Xbh[k*D + d];
            Xbh[i*D + d] = sum / L[i*M + i];
        }
    }

    // Solve for x1: L*z = rhs1; L^T*x1 = z (single thread)
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < M; ++i) {
            float sum = rhs1[i];
            for (int k = 0; k < i; ++k) sum -= L[i*M + k] * rhs1[k];
            rhs1[i] = sum / L[i*M + i];
        }
        for (int i = M - 1; i >= 0; --i) {
            float sum = rhs1[i];
            for (int k = i + 1; k < M; ++k) sum -= L[k*M + i] * x1bh[k];
            x1bh[i] = sum / L[i*M + i];
        }
    }
}

// Out[n,d] = (A[n,:] @ X[:,d]) / (A[n,:] @ x1 + eps)
__global__ void apply_AX_kernel(
    const float* __restrict__ A,  // [BH,N,M]
    const float* __restrict__ X,  // [BH,M,D]
    const float* __restrict__ x1, // [BH,M]
    float* __restrict__ Out,      // [BH,N,D]
    int N, int M, int D,
    float eps
) {
    int bh = (int)blockIdx.x;
    int n  = (int)blockIdx.y;
    int d  = (int)blockIdx.z * (int)blockDim.x + (int)threadIdx.x;
    if (d >= D) return;

    const float* Abh = A + (size_t)bh * (size_t)N * (size_t)M;
    const float* Xbh = X + (size_t)bh * (size_t)M * (size_t)D;
    const float* x1bh = x1 + (size_t)bh * (size_t)M;
    float* Obh = Out + (size_t)bh * (size_t)N * (size_t)D;

    const float* arow = Abh + (size_t)n * (size_t)M;

    // denom computed redundantly across d threads in block; acceptable for small M.
    float denom = 0.0f;
    for (int m = 0; m < M; ++m) denom = fmaf(arow[m], x1bh[m], denom);
    float inv = 1.0f / (denom + eps);

    float acc = 0.0f;
    for (int m = 0; m < M; ++m) acc = fmaf(arow[m], Xbh[(size_t)m * (size_t)D + (size_t)d], acc);
    Obh[(size_t)n * (size_t)D + (size_t)d] = acc * inv;
}

torch::Tensor nystrom_attention_forward_cuda(
    torch::Tensor Q, // [B,H,N,D] float32 contiguous
    torch::Tensor K, // [B,H,N,D]
    torch::Tensor V, // [B,H,N,D]
    int64_t num_landmarks,
    double eps
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,D]");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int D = (int)Q.size(3);
    int M = (int)num_landmarks;

    TORCH_CHECK(N == 512, "Optimized kernel supports only N=512");
    TORCH_CHECK(D == 64,  "Optimized kernel supports only D=64");
    TORCH_CHECK(M == 32,  "Optimized kernel supports only num_landmarks M=32");

    int BH = B * H;
    auto opts = Q.options();

    auto Qf = Q.view({BH, N, D});
    auto Kf = K.view({BH, N, D});
    auto Vf = V.view({BH, N, D});

    auto QL = torch::empty({BH, M, D}, opts);
    auto KL = torch::empty({BH, M, D}, opts);

    // gather landmarks
    {
        dim3 grid((unsigned)BH, (unsigned)M, 1u);
        int threads = 128;
        gather_landmarks_kernel<<<grid, threads, 0>>>(
            Qf.data_ptr<float>(), Kf.data_ptr<float>(),
            QL.data_ptr<float>(), KL.data_ptr<float>(),
            N, D, M
        );
    }

    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    auto A = torch::empty({BH, N, M}, opts);
    auto Bmat = torch::empty({BH, M, M}, opts);

    // A
    {
        dim3 grid((unsigned)BH, (unsigned)N, 1u);
        int threads = 128;
        compute_A_kernel<<<grid, threads, 0>>>(
            Qf.data_ptr<float>(), KL.data_ptr<float>(), A.data_ptr<float>(),
            N, D, M, inv_sqrt_d
        );
    }

    // B
    {
        dim3 grid((unsigned)BH, (unsigned)M, 1u);
        int threads = 128;
        compute_B_kernel<<<grid, threads, 0>>>(
            QL.data_ptr<float>(), KL.data_ptr<float>(), Bmat.data_ptr<float>(),
            D, M, inv_sqrt_d
        );
    }

    // Y = A^T @ V
    auto Y = torch::empty({BH, M, D}, opts);
    {
        int threads = 128;
        dim3 grid((unsigned)BH, (unsigned)M, (unsigned)((D + threads - 1) / threads));
        compute_AT_V_kernel<<<grid, threads, 0>>>(
            A.data_ptr<float>(), Vf.data_ptr<float>(), Y.data_ptr<float>(),
            N, M, D
        );
    }

    // y1 = A^T @ ones
    auto y1 = torch::empty({BH, M}, opts);
    {
        dim3 grid((unsigned)BH, (unsigned)M, 1u);
        int threads = 256;
        compute_AT_ones_kernel<<<grid, threads, 0>>>(
            A.data_ptr<float>(), y1.data_ptr<float>(),
            N, M
        );
    }

    // Solve Bs*X=Y and Bs*x1=y1
    auto X = torch::empty({BH, M, D}, opts);
    auto x1 = torch::empty({BH, M}, opts);
    {
        int threads = 256;
        chol_solve_kernel_M32<<<(unsigned)BH, threads, 0>>>(
            Bmat.data_ptr<float>(),
            Y.data_ptr<float>(),
            y1.data_ptr<float>(),
            X.data_ptr<float>(),
            x1.data_ptr<float>(),
            (float)eps
        );
    }

    // Out = apply A @ X with normalization by A @ x1
    auto Out = torch::empty({BH, N, D}, opts);
    {
        int threads = 128;
        dim3 grid((unsigned)BH, (unsigned)N, (unsigned)((D + threads - 1) / threads));
        apply_AX_kernel<<<grid, threads, 0>>>(
            A.data_ptr<float>(),
            X.data_ptr<float>(),
            x1.data_ptr<float>(),
            Out.data_ptr<float>(),
            N, M, D,
            (float)eps
        );
    }

    return Out.view({B, H, N, D});
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor nystrom_attention_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int64_t num_landmarks, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_nystrom_attention_fwd_n512_d64_m32_chol_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["nystrom_attention_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Nystrom Attention with a CUDA-accelerated forward (inference fast path).
    Falls back to reference PyTorch for unsupported cases.
    """
    def __init__(self, d_model, n_heads, num_landmarks=32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_k = self.d_model // self.n_heads
        self.num_landmarks = int(num_landmarks)
        self.eps = 1e-6

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        B, N, _ = x.size()

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        use_fast = (
            (not self.training)
            and x.is_cuda
            and Q.is_cuda and K.is_cuda and V.is_cuda
            and Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32
            and Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
            and Q.dim() == 4
            and N == 512 and self.d_k == 64
            and self.num_landmarks == 32
            and N > self.num_landmarks
        )

        if use_fast:
            out = custom_ops_lib.nystrom_attention_forward_cuda(
                Q, K, V, int(self.num_landmarks), float(self.eps)
            )
        else:
            if N <= self.num_landmarks:
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                attn_weights = F.softmax(scores, dim=-1)
                out = torch.matmul(attn_weights, V)
            else:
                landmark_indices = torch.linspace(
                    0, N - 1, self.num_landmarks, dtype=torch.long, device=x.device
                )
                Q_landmarks = Q[:, :, landmark_indices]
                K_landmarks = K[:, :, landmark_indices]

                A = torch.matmul(Q, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
                A = F.softmax(A, dim=-1)

                Bmat = torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
                Bmat = F.softmax(Bmat, dim=-1)

                eye = torch.eye(self.num_landmarks, device=Bmat.device, dtype=Bmat.dtype).unsqueeze(0).unsqueeze(0)
                B_inv = torch.pinverse(Bmat + self.eps * eye)

                attn_approx = torch.matmul(torch.matmul(A, B_inv), A.transpose(-2, -1))
                attn_approx = attn_approx / (attn_approx.sum(dim=-1, keepdim=True) + self.eps)
                out = torch.matmul(attn_approx, V)

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)
        return out