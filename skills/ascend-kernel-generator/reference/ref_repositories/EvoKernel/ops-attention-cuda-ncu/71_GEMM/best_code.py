import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA op: GEMM (Y = X @ W^T) for BF16 using Tensor Cores (WMMA)
# Fast path (typical LLM dims multiples of 16): 64x64 output tile per CTA (4 warps)
# - WMMA BF16 fragments, FP32 accumulate, BF16 output
# - Shared-memory tiling with DOUBLE-BUFFERING (ping-pong) for A/B panels
# - Optional 16B vectorized global loads (int4) with host-side alignment gating
# Fallback scalar kernel for arbitrary shapes
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)

static __device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) { return __float2bfloat16_rn(x); }
static __device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }

// Scalar fallback (correctness for arbitrary shapes)
__global__ void gemm_fwd_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,   // [M,K]
    const __nv_bfloat16* __restrict__ w,   // [N,K]
    __nv_bfloat16* __restrict__ y,         // [M,N]
    int M, int N, int K
) {
    int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (m >= M || n >= N) return;

    const __nv_bfloat16* x_row = x + (int64_t)m * K;
    const __nv_bfloat16* w_row = w + (int64_t)n * K;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(bf16_to_f32(x_row[k]), bf16_to_f32(w_row[k]), acc);
    }
    y[(int64_t)m * N + n] = f32_to_bf16(acc);
}

// 64x64 tile per CTA, 4 warps (128 threads), K step = 16
// Shared memory double-buffering for A and B panels:
//   A panel: 64 x 16 bf16
//   B panel: 64 x 16 bf16  (we store as [n-major][k] and interpret as col-major for WMMA)
//
// Each warp computes a 32x32 quadrant via 4 WMMA ops per K-step.
template<bool VEC>
__global__ void gemm_fwd_wmma_64x64_db_kernel(
    const __nv_bfloat16* __restrict__ x,   // [M,K] row-major
    const __nv_bfloat16* __restrict__ w,   // [N,K] row-major (represents B = W^T with col-major access)
    __nv_bfloat16* __restrict__ y,         // [M,N] row-major
    int M, int N, int K
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const int block_m = (int)blockIdx.y;
    const int block_n = (int)blockIdx.x;
    const int m0 = block_m * 64;
    const int n0 = block_n * 64;

    const int tid = (int)threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane = tid & 31;

    // Warp layout: 2x2 over 32x32 quadrants
    const int warp_m = (warp_id >> 1) * 32; // 0 or 32
    const int warp_n = (warp_id & 1) * 32;  // 0 or 32

    // Shared memory layout:
    //   shA[2][64*16] bf16
    //   shB[2][64*16] bf16  (stored as n-major contiguous blocks: shB[buf][n*16 + k])
    //   shF[4][256] float per-warp scratch for stores
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* shA = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* shB = shA + 2 * (64 * 16);
    float* shF = (float*)(shB + 2 * (64 * 16));

    auto gA = [&](int gm, int gk) -> const __nv_bfloat16* {
        return x + (int64_t)gm * K + gk;
    };
    auto gW = [&](int gn, int gk) -> const __nv_bfloat16* {
        return w + (int64_t)gn * K + gk;  // W[gn, gk]
    };

    // Accumulators: warp computes 4 tiles (2x2 of 16x16) within its 32x32 quadrant
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc00, acc01, acc10, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    // Cooperative loader:
    // We load A panel (64x16) and B panel (64x16) per k0.
    // Each thread loads 8 bf16 (16 bytes) for A and 8 bf16 for B.
    auto load_panel = [&](int k0, int buf) {
        // A mapping
        int a_elem0 = tid * 8;     // bf16 index in 64x16 panel
        int ar = a_elem0 >> 4;     // /16 => row 0..63
        int ac = a_elem0 & 15;     // col 0..15
        int gm = m0 + ar;
        int gk = k0 + ac;
        __nv_bfloat16* dstA = shA + buf * (64 * 16) + ar * 16 + ac;

        // B mapping: view panel as [n(0..63), k(0..15)]
        int b_elem0 = tid * 8;
        int bn = b_elem0 >> 4;     // 0..63
        int bk = b_elem0 & 15;     // 0..15
        int gn = n0 + bn;
        int gk2 = k0 + bk;
        __nv_bfloat16* dstB = shB + buf * (64 * 16) + bn * 16 + bk;

        // Because fast path is multiples-of-16, K is aligned to 16, but edge tiles for M/N may be partial.
        // For typical LLM configs M and N are multiples of 64 too; still guard generally.
        if constexpr (VEC) {
            // vectorized: require we only do it if we have 8 contiguous bf16 and row in bounds.
            if (gm < M && (gk + 7) < K) {
                *(int4*)dstA = *(const int4*)gA(gm, gk);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int kk = gk + i;
                    dstA[i] = (gm < M && kk < K) ? *gA(gm, kk) : __float2bfloat16_rn(0.0f);
                }
            }
            if (gn < N && (gk2 + 7) < K) {
                *(int4*)dstB = *(const int4*)gW(gn, gk2);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int kk = gk2 + i;
                    dstB[i] = (gn < N && kk < K) ? *gW(gn, kk) : __float2bfloat16_rn(0.0f);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int kkA = gk + i;
                dstA[i] = (gm < M && kkA < K) ? *gA(gm, kkA) : __float2bfloat16_rn(0.0f);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int kkB = gk2 + i;
                dstB[i] = (gn < N && kkB < K) ? *gW(gn, kkB) : __float2bfloat16_rn(0.0f);
            }
        }
    };

    // Prefetch first panel into buf=0
    int buf = 0;
    load_panel(0, buf);
    __syncthreads();

#pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += 16) {
        int next_k0 = k0 + 16;
        int next_buf = buf ^ 1;

        // Prefetch next panel early (overlaps with current compute as much as possible)
        if (next_k0 < K) {
            load_panel(next_k0, next_buf);
        }

        // Compute on current buf
        const __nv_bfloat16* A_base = shA + buf * (64 * 16);
        const __nv_bfloat16* B_base = shB + buf * (64 * 16);

        // For WMMA:
        // A is row_major 16x16 with ld=16. Pick rows (warp_m + tile_rm)
        // B is col_major 16x16. We have B stored as n-major blocks of length 16 (k),
        // which is exactly col-major with ld=16 if we set pointer to (n_offset*16) and ld=16.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;

        const __nv_bfloat16* a_ptr0 = A_base + (warp_m + 0) * 16;
        const __nv_bfloat16* a_ptr1 = A_base + (warp_m + 16) * 16;

        const __nv_bfloat16* b_ptr0 = B_base + (warp_n + 0) * 16;
        const __nv_bfloat16* b_ptr1 = B_base + (warp_n + 16) * 16;

        wmma::load_matrix_sync(a0, a_ptr0, 16);
        wmma::load_matrix_sync(a1, a_ptr1, 16);
        wmma::load_matrix_sync(b0, b_ptr0, 16);
        wmma::load_matrix_sync(b1, b_ptr1, 16);

        wmma::mma_sync(acc00, a0, b0, acc00);
        wmma::mma_sync(acc01, a0, b1, acc01);
        wmma::mma_sync(acc10, a1, b0, acc10);
        wmma::mma_sync(acc11, a1, b1, acc11);

        // Finish prefetch and swap buffers
        __syncthreads();
        buf = next_buf;
    }

    // Store: per-warp scratch (256 floats)
    float* warp_scratch = shF + warp_id * 256;

    auto store_tile = [&](wmma::fragment<wmma::accumulator,16,16,16,float>& acc, int tile_rm, int tile_cn) {
        wmma::store_matrix_sync(warp_scratch, acc, 16, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = lane + i * 32; // 0..255
            int r = idx >> 4;
            int c = idx & 15;
            int gm = m0 + warp_m + tile_rm + r;
            int gn = n0 + warp_n + tile_cn + c;
            if (gm < M && gn < N) {
                y[(int64_t)gm * N + gn] = f32_to_bf16(warp_scratch[idx]);
            }
        }
    };

    store_tile(acc00, 0, 0);
    store_tile(acc01, 0, 16);
    store_tile(acc10, 16, 0);
    store_tile(acc11, 16, 16);
#else
    (void)x; (void)w; (void)y; (void)M; (void)N; (void)K;
#endif
}

torch::Tensor gemm_fwd_cuda(torch::Tensor x, torch::Tensor w) {
    CHECK_INPUT_BF16(x);
    CHECK_INPUT_BF16(w);

    TORCH_CHECK(x.dim() == 2, "x must be [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be [N,K]");

    int64_t M64 = x.size(0);
    int64_t K64 = x.size(1);
    int64_t N64 = w.size(0);
    int64_t K2_64 = w.size(1);
    TORCH_CHECK(K64 == K2_64, "K mismatch: x is [M,K] but w is [N,K2]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "sizes too large for int32 indexing");

    int M = (int)M64;
    int N = (int)N64;
    int K = (int)K64;

    auto y = torch::empty({M, N}, torch::TensorOptions().dtype(at::kBFloat16).device(x.device()));

    const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr<at::BFloat16>();
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w.data_ptr<at::BFloat16>();
    __nv_bfloat16* yp = (__nv_bfloat16*)y.data_ptr<at::BFloat16>();

    // Fast path: WMMA requires multiples of 16
    if ((M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0)) {
        dim3 block(128, 1, 1);
        dim3 grid((N + 64 - 1) / 64, (M + 64 - 1) / 64, 1);

        // Shared memory:
        // shA: 2*(64*16) bf16
        // shB: 2*(64*16) bf16
        // shF: 4*256 float
        size_t shmem_bytes =
            (2 * (64 * 16) + 2 * (64 * 16)) * sizeof(__nv_bfloat16) +
            (4 * 256) * sizeof(float);

        // Host-side alignment gating for int4 vector loads:
        // We need base pointers 16B aligned and K multiple of 8 elements (16 bytes) for row loads;
        // K is multiple of 16 here => ok. Base alignment is typically >=256B but don't assume.
        uintptr_t x_addr = (uintptr_t)xp;
        uintptr_t w_addr = (uintptr_t)wp;
        bool vec_ok = ((x_addr & 0xF) == 0) && ((w_addr & 0xF) == 0);

        if (vec_ok) {
            gemm_fwd_wmma_64x64_db_kernel<true><<<grid, block, shmem_bytes>>>(xp, wp, yp, M, N, K);
        } else {
            gemm_fwd_wmma_64x64_db_kernel<false><<<grid, block, shmem_bytes>>>(xp, wp, yp, M, N, K);
        }
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "gemm_fwd_wmma_64x64_db_kernel launch failed: ", cudaGetErrorString(err));
        return y;
    }

    // Fallback scalar
    dim3 block2(16, 16);
    dim3 grid2((N + block2.x - 1) / block2.x, (M + block2.y - 1) / block2.y);
    gemm_fwd_scalar_kernel<<<grid2, block2>>>(xp, wp, yp, M, N, K);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_fwd_scalar_kernel launch failed: ", cudaGetErrorString(err));
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor gemm_fwd_cuda(torch::Tensor x, torch::Tensor w);
"""

custom_ops_lib = load_inline(
    name="custom_gemm_ops_wmma_db64_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gemm_fwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    GEMM using a custom CUDA kernel.
    Semantics match:
        Y = X @ W^T
    where weight is stored as [N, K] (row-major).
    """

    def __init__(self, K, N):
        super().__init__()
        self.K = int(K)
        self.N = int(N)
        self.weight = nn.Parameter(torch.randn(self.N, self.K, dtype=torch.bfloat16))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and self.weight.is_cuda
            and self.weight.dtype == torch.bfloat16
        ):
            return self.custom_ops_lib.gemm_fwd_cuda(x.contiguous(), self.weight.contiguous())
        return torch.matmul(x, self.weight.t())