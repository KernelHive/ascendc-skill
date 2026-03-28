import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# grouped_gemm optimized v9:
# - Coherent fast path: ping-pong shared-memory RHS tiling with ONE __syncthreads()
#   per K2 tile transition (software pipeline, no cp.async).
# - Cheaper coherence detection: warp-vote + warp0 aggregation (no atomics).
# - Smem padding on K2 dimension to reduce bank conflicts.
# - Keeps warp-level LHS broadcast (lane0 load + shfl).
# - Fallback baseline kernel for odd-K / unaligned pointers.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")
#define CHECK_I32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_INPUT_BF16(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_BF16(x)
#define CHECK_INPUT_I32(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_I32(x)

static __device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) { return __float2bfloat16(x); }

#if __CUDA_ARCH__ >= 350
static __device__ __forceinline__ uint32_t ro_load_u32(const uint32_t* p) { return __ldg(p); }
static __device__ __forceinline__ __nv_bfloat16 ro_load_bf16(const __nv_bfloat16* p) { return __ldg(p); }
#else
static __device__ __forceinline__ uint32_t ro_load_u32(const uint32_t* p) { return *p; }
static __device__ __forceinline__ __nv_bfloat16 ro_load_bf16(const __nv_bfloat16* p) { return *p; }
#endif

// --------------------------- Baseline direct-load kernel ---------------------------
template<int VEC_N, int WARPS_PER_BLOCK, int UNROLL_K2>
__global__ void grouped_gemm_warp_lhs_broadcast_fwd(
    const __nv_bfloat16* __restrict__ lhs,        // [M, K]
    const __nv_bfloat16* __restrict__ rhs,        // [G, N, K]
    const int32_t* __restrict__ m_indices,        // [M]
    __nv_bfloat16* __restrict__ out,              // [M, N]
    int M, int N, int K, int G
) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x & (WARP - 1);
    const int warp_id = threadIdx.x >> 5;

    const int row = (int)blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (row >= M) return;

    int g = m_indices[row];
    if ((unsigned)g >= (unsigned)G) g = 0;

    constexpr int TILE_N = WARP * VEC_N;
    const int cols_start = (int)blockIdx.x * TILE_N;
    const int n_base = cols_start + lane * VEC_N;

    const __nv_bfloat16* lhs_row = lhs + (int64_t)row * (int64_t)K;
    const __nv_bfloat16* rhs_group = rhs + (int64_t)g * (int64_t)N * (int64_t)K;

    float acc[VEC_N];
#pragma unroll
    for (int j = 0; j < VEC_N; ++j) acc[j] = 0.0f;

    if ((K & 1) == 0) {
        const int K2 = K >> 1;
        for (int k2 = 0; k2 < K2; k2 += UNROLL_K2) {
#pragma unroll
            for (int u = 0; u < UNROLL_K2; ++u) {
                int kk2 = k2 + u;
                if (kk2 >= K2) continue;
                int k0 = kk2 << 1;

                uint32_t a_pack = 0u;
                if (lane == 0) {
                    a_pack = ro_load_u32(reinterpret_cast<const uint32_t*>(lhs_row + k0));
                }
                a_pack = __shfl_sync(0xFFFFFFFFu, a_pack, 0);

                uint16_t alo = (uint16_t)(a_pack & 0xFFFFu);
                uint16_t ahi = (uint16_t)((a_pack >> 16) & 0xFFFFu);
                __nv_bfloat16 a0 = *reinterpret_cast<__nv_bfloat16*>(&alo);
                __nv_bfloat16 a1 = *reinterpret_cast<__nv_bfloat16*>(&ahi);
                float fa0 = bf16_to_f32(a0);
                float fa1 = bf16_to_f32(a1);

#pragma unroll
                for (int j = 0; j < VEC_N; ++j) {
                    int n = n_base + j;
                    if (n < N) {
                        const __nv_bfloat16* rhs_row = rhs_group + (int64_t)n * (int64_t)K + (int64_t)k0;
                        uint32_t b_pack = ro_load_u32(reinterpret_cast<const uint32_t*>(rhs_row));
                        uint16_t blo = (uint16_t)(b_pack & 0xFFFFu);
                        uint16_t bhi = (uint16_t)((b_pack >> 16) & 0xFFFFu);
                        __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&blo);
                        __nv_bfloat16 b1 = *reinterpret_cast<__nv_bfloat16*>(&bhi);
                        float fb0 = bf16_to_f32(b0);
                        float fb1 = bf16_to_f32(b1);
                        acc[j] = fmaf(fa0, fb0, acc[j]);
                        acc[j] = fmaf(fa1, fb1, acc[j]);
                    }
                }
            }
        }
    } else {
        for (int k = 0; k < K; ++k) {
            __nv_bfloat16 a_b;
            if (lane == 0) a_b = lhs_row[k];
            uint32_t a_u = (lane == 0) ? (uint32_t)(*reinterpret_cast<uint16_t*>(&a_b)) : 0u;
            a_u = __shfl_sync(0xFFFFFFFFu, a_u, 0);
            uint16_t au16 = (uint16_t)(a_u & 0xFFFFu);
            __nv_bfloat16 a = *reinterpret_cast<__nv_bfloat16*>(&au16);
            float fa = bf16_to_f32(a);

#pragma unroll
            for (int j = 0; j < VEC_N; ++j) {
                int n = n_base + j;
                if (n < N) {
                    float fb = bf16_to_f32(ro_load_bf16(rhs_group + (int64_t)n * (int64_t)K + (int64_t)k));
                    acc[j] = fmaf(fa, fb, acc[j]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < VEC_N; ++j) {
        int n = n_base + j;
        if (n < N) {
            out[(int64_t)row * (int64_t)N + (int64_t)n] = f32_to_bf16(acc[j]);
        }
    }
}

// --------------------------- Smem RHS-tiling kernel with single-barrier pipeline ---------------------------
// Only uses smem when all WARPS_PER_BLOCK rows share same expert g (block-coherent).
// Otherwise, falls back to direct loads for that block (still K-even aligned path).
template<int VEC_N, int WARPS_PER_BLOCK, int K2_TILE, int K2_PAD, int UNROLL_K2>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void grouped_gemm_rhs_smem_coherent_pipelined_fwd(
    const __nv_bfloat16* __restrict__ lhs,        // [M, K]
    const __nv_bfloat16* __restrict__ rhs,        // [G, N, K]
    const int32_t* __restrict__ m_indices,        // [M]
    __nv_bfloat16* __restrict__ out,              // [M, N]
    int M, int N, int K, int G
) {
    constexpr int WARP = 32;
    constexpr int TILE_N = WARP * VEC_N;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

    const int tid = (int)threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp_id = tid >> 5;

    const int row = (int)blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (row >= M) return;

    const int block_row0 = (int)blockIdx.y * WARPS_PER_BLOCK;

    // Load g0 (expert id for first row in this block) and broadcast.
    int g0 = 0;
    if (tid == 0) {
        int r0 = block_row0;
        int t = (r0 < M) ? m_indices[r0] : 0;
        if ((unsigned)t >= (unsigned)G) t = 0;
        g0 = t;
    }
    __shared__ int sh_g0;
    if (tid == 0) sh_g0 = g0;
    __syncthreads();
    g0 = sh_g0;

    // Warp-level coherence check: does this warp's row use g0?
    int g = m_indices[row];
    if ((unsigned)g >= (unsigned)G) g = 0;
    int warp_ok = (g == g0);

    // Aggregate across warps using warp0 ballot + shared:
    __shared__ int sh_all_ok;
    if (tid == 0) sh_all_ok = 1;
    __syncthreads();

    if (warp_id == 0) {
        // Each lane corresponds to a potential warp_id; only first WARPS_PER_BLOCK lanes are used.
        int ok_lane = 1;
        if (lane < WARPS_PER_BLOCK) {
            int r = block_row0 + lane;
            if (r < M) {
                int gg = m_indices[r];
                if ((unsigned)gg >= (unsigned)G) gg = 0;
                ok_lane = (gg == g0);
            }
        }
        unsigned mask = __ballot_sync(FULL_MASK, ok_lane);
        if (lane == 0) {
            unsigned want = (WARPS_PER_BLOCK == 32) ? 0xFFFFFFFFu : ((1u << WARPS_PER_BLOCK) - 1u);
            sh_all_ok = ((mask & want) == want) ? 1 : 0;
        }
    }
    __syncthreads();

    bool coherent = (sh_all_ok != 0);

    constexpr int TILE_COLS = TILE_N;
    const int cols_start = (int)blockIdx.x * TILE_COLS;
    const int n_base = cols_start + lane * VEC_N;

    const __nv_bfloat16* lhs_row = lhs + (int64_t)row * (int64_t)K;
    const __nv_bfloat16* rhs_group_g = rhs + (int64_t)g * (int64_t)N * (int64_t)K;
    const __nv_bfloat16* rhs_group_g0 = rhs + (int64_t)g0 * (int64_t)N * (int64_t)K;

    float acc[VEC_N];
#pragma unroll
    for (int j = 0; j < VEC_N; ++j) acc[j] = 0.0f;

    const int K2 = K >> 1;

    // Incoherent fallback (still K-even, aligned): direct loads.
    if (!coherent) {
        for (int k2 = 0; k2 < K2; k2 += UNROLL_K2) {
#pragma unroll
            for (int u = 0; u < UNROLL_K2; ++u) {
                int kk2 = k2 + u;
                if (kk2 >= K2) continue;
                int k0 = kk2 << 1;

                uint32_t a_pack = 0u;
                if (lane == 0) a_pack = ro_load_u32(reinterpret_cast<const uint32_t*>(lhs_row + k0));
                a_pack = __shfl_sync(FULL_MASK, a_pack, 0);

                uint16_t alo = (uint16_t)(a_pack & 0xFFFFu);
                uint16_t ahi = (uint16_t)((a_pack >> 16) & 0xFFFFu);
                __nv_bfloat16 a0 = *reinterpret_cast<__nv_bfloat16*>(&alo);
                __nv_bfloat16 a1 = *reinterpret_cast<__nv_bfloat16*>(&ahi);
                float fa0 = bf16_to_f32(a0);
                float fa1 = bf16_to_f32(a1);

#pragma unroll
                for (int j = 0; j < VEC_N; ++j) {
                    int n = n_base + j;
                    if (n < N) {
                        const __nv_bfloat16* rhs_row = rhs_group_g + (int64_t)n * (int64_t)K + (int64_t)k0;
                        uint32_t b_pack = ro_load_u32(reinterpret_cast<const uint32_t*>(rhs_row));
                        uint16_t blo = (uint16_t)(b_pack & 0xFFFFu);
                        uint16_t bhi = (uint16_t)((b_pack >> 16) & 0xFFFFu);
                        __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&blo);
                        __nv_bfloat16 b1 = *reinterpret_cast<__nv_bfloat16*>(&bhi);
                        float fb0 = bf16_to_f32(b0);
                        float fb1 = bf16_to_f32(b1);
                        acc[j] = fmaf(fa0, fb0, acc[j]);
                        acc[j] = fmaf(fa1, fb1, acc[j]);
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < VEC_N; ++j) {
            int n = n_base + j;
            if (n < N) out[(int64_t)row * (int64_t)N + (int64_t)n] = f32_to_bf16(acc[j]);
        }
        return;
    }

    // Coherent fast path: stage RHS for g0 into shared memory and reuse for all WARPS_PER_BLOCK rows.
    // Smem layout: two buffers, each [TILE_N][K2_TILE + K2_PAD] of packed u32
    extern __shared__ uint32_t smem_u32[];
    const int STRIDE = (K2_TILE + K2_PAD);
    const int BUF_ELEMS = TILE_N * STRIDE;
    uint32_t* smem0 = smem_u32;
    uint32_t* smem1 = smem_u32 + BUF_ELEMS;

    auto stage = [&](uint32_t* dst, int k2_base) {
        // total u32 to load for this tile
        int total = TILE_N * K2_TILE;
        for (int t = tid; t < total; t += (int)blockDim.x) {
            int n_in = t / K2_TILE;
            int k2i = t - n_in * K2_TILE;
            int n = cols_start + n_in;
            int kk2 = k2_base + k2i;
            uint32_t v = 0u;
            if (n < N && kk2 < K2) {
                int k0 = kk2 << 1;
                const __nv_bfloat16* p = rhs_group_g0 + (int64_t)n * (int64_t)K + (int64_t)k0;
                v = ro_load_u32(reinterpret_cast<const uint32_t*>(p));
            }
            dst[n_in * STRIDE + k2i] = v;
        }
        // No need to clear padded columns; we never read them.
    };

    int k2_base = 0;

    // Initial fill
    stage(smem0, 0);
    __syncthreads();

    for (k2_base = 0; k2_base < K2; k2_base += K2_TILE) {
        uint32_t* cur = ((k2_base / K2_TILE) & 1) ? smem1 : smem0;
        uint32_t* nxt = ((k2_base / K2_TILE) & 1) ? smem0 : smem1;

        int next_k2 = k2_base + K2_TILE;

        // Compute on current tile while the "next" tile is not yet needed.
        int k2_end = k2_base + K2_TILE;
        if (k2_end > K2) k2_end = K2;

        for (int k2 = k2_base; k2 < k2_end; k2 += UNROLL_K2) {
#pragma unroll
            for (int u = 0; u < UNROLL_K2; ++u) {
                int kk2 = k2 + u;
                if (kk2 >= k2_end) continue;
                int k0 = kk2 << 1;

                uint32_t a_pack = 0u;
                if (lane == 0) a_pack = ro_load_u32(reinterpret_cast<const uint32_t*>(lhs_row + k0));
                a_pack = __shfl_sync(FULL_MASK, a_pack, 0);

                uint16_t alo = (uint16_t)(a_pack & 0xFFFFu);
                uint16_t ahi = (uint16_t)((a_pack >> 16) & 0xFFFFu);
                __nv_bfloat16 a0 = *reinterpret_cast<__nv_bfloat16*>(&alo);
                __nv_bfloat16 a1 = *reinterpret_cast<__nv_bfloat16*>(&ahi);
                float fa0 = bf16_to_f32(a0);
                float fa1 = bf16_to_f32(a1);

                int k2i = kk2 - k2_base;

#pragma unroll
                for (int j = 0; j < VEC_N; ++j) {
                    int n = n_base + j;
                    if (n < N) {
                        int n_in = n - cols_start;
                        uint32_t b_pack = cur[n_in * STRIDE + k2i];
                        uint16_t blo = (uint16_t)(b_pack & 0xFFFFu);
                        uint16_t bhi = (uint16_t)((b_pack >> 16) & 0xFFFFu);
                        __nv_bfloat16 b0 = *reinterpret_cast<__nv_bfloat16*>(&blo);
                        __nv_bfloat16 b1 = *reinterpret_cast<__nv_bfloat16*>(&bhi);
                        float fb0 = bf16_to_f32(b0);
                        float fb1 = bf16_to_f32(b1);
                        acc[j] = fmaf(fa0, fb0, acc[j]);
                        acc[j] = fmaf(fa1, fb1, acc[j]);
                    }
                }
            }
        }

        if (next_k2 < K2) {
            // Stage next tile then single barrier to make it visible before next iteration uses it.
            stage(nxt, next_k2);
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < VEC_N; ++j) {
        int n = n_base + j;
        if (n < N) out[(int64_t)row * (int64_t)N + (int64_t)n] = f32_to_bf16(acc[j]);
    }
}

static void cuda_check_last(const char* msg) {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, msg, ": ", cudaGetErrorString(e));
}

torch::Tensor grouped_gemm_fwd_cuda(torch::Tensor lhs, torch::Tensor rhs, torch::Tensor m_indices) {
    CHECK_INPUT_BF16(lhs);
    CHECK_INPUT_BF16(rhs);
    CHECK_INPUT_I32(m_indices);

    TORCH_CHECK(lhs.dim() == 2, "lhs must be [M,K]");
    TORCH_CHECK(rhs.dim() == 3, "rhs must be [G,N,K]");
    TORCH_CHECK(m_indices.dim() == 1, "m_indices must be [M]");

    int64_t M64 = lhs.size(0);
    int64_t K64 = lhs.size(1);
    int64_t G64 = rhs.size(0);
    int64_t N64 = rhs.size(1);
    int64_t K2_64 = rhs.size(2);

    TORCH_CHECK(K64 == K2_64, "K mismatch between lhs and rhs");
    TORCH_CHECK(m_indices.size(0) == M64, "m_indices length must equal M");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX && G64 <= INT_MAX,
                "sizes too large for int32 kernel indexing");

    int M = (int)M64;
    int N = (int)N64;
    int K = (int)K64;
    int G = (int)G64;

    auto out = torch::empty({M, N}, torch::TensorOptions().dtype(at::kBFloat16).device(lhs.device()));

    constexpr int VEC_N = 4;                 // warp covers 128 columns
    constexpr int WARPS_PER_BLOCK = 4;       // 4 rows per block
    constexpr int UNROLL_K2 = 2;

    constexpr int TILE_N = 32 * VEC_N;
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    bool k_even = ((K & 1) == 0);
    uintptr_t lhs_ptr = (uintptr_t)lhs.data_ptr<at::BFloat16>();
    uintptr_t rhs_ptr = (uintptr_t)rhs.data_ptr<at::BFloat16>();
    uintptr_t out_ptr = (uintptr_t)out.data_ptr<at::BFloat16>();
    bool aligned4 = ((lhs_ptr | rhs_ptr | out_ptr) & 0x3) == 0;

    if (k_even && aligned4) {
        // Tune tile: smaller improves staging latency; padding reduces bank conflicts.
        constexpr int K2_TILE = 16;     // was 32 in v8
        constexpr int K2_PAD  = 1;      // pad stride to mitigate bank conflicts on reads

        // smem: 2 buffers * TILE_N * (K2_TILE + K2_PAD) * 4 bytes
        size_t smem_bytes = (size_t)2 * (size_t)TILE_N * (size_t)(K2_TILE + K2_PAD) * 4ull;

        grouped_gemm_rhs_smem_coherent_pipelined_fwd<VEC_N, WARPS_PER_BLOCK, K2_TILE, K2_PAD, UNROLL_K2>
            <<<grid, block, smem_bytes>>>(
                (const __nv_bfloat16*)lhs.data_ptr<at::BFloat16>(),
                (const __nv_bfloat16*)rhs.data_ptr<at::BFloat16>(),
                (const int32_t*)m_indices.data_ptr<int32_t>(),
                (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
                M, N, K, G
            );
        cuda_check_last("grouped_gemm smem-coherent-pipelined kernel launch failed");
        return out;
    }

    // Fallback: baseline kernel (works for odd K / unaligned)
    constexpr int WARPS_FALLBACK = 2;
    dim3 block2(WARPS_FALLBACK * 32);
    dim3 grid2((N + TILE_N - 1) / TILE_N, (M + WARPS_FALLBACK - 1) / WARPS_FALLBACK);

    grouped_gemm_warp_lhs_broadcast_fwd<VEC_N, WARPS_FALLBACK, UNROLL_K2><<<grid2, block2>>>(
        (const __nv_bfloat16*)lhs.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)rhs.data_ptr<at::BFloat16>(),
        (const int32_t*)m_indices.data_ptr<int32_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        M, N, K, G
    );
    cuda_check_last("grouped_gemm baseline kernel launch failed");
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor grouped_gemm_fwd_cuda(torch::Tensor lhs, torch::Tensor rhs, torch::Tensor m_indices);
"""

custom_ops_lib = load_inline(
    name="custom_grouped_gemm_ops_opt9_pipelined_singlebarrier",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["grouped_gemm_fwd_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Grouped GEMM for MoE FFN layers using an optimized custom CUDA kernel.

    Semantics:
      out[i] = lhs[i] @ rhs[m_indices[i]].T

    Optimization: optional stable sort by m_indices to make expert indices contiguous,
    enabling the CUDA kernel's block-level coherent RHS shared-memory reuse.
    """

    def __init__(self, num_groups, N, K, enable_sort: bool = True, sort_min_M: int = 2048):
        super().__init__()
        self.num_groups = int(num_groups)
        self.N = int(N)
        self.K = int(K)
        self.enable_sort = bool(enable_sort)
        self.sort_min_M = int(sort_min_M)
        self.rhs = nn.Parameter(torch.randn(num_groups, N, K, dtype=torch.bfloat16))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, lhs: torch.Tensor, m_indices: torch.Tensor) -> torch.Tensor:
        if not (
            lhs.is_cuda
            and m_indices.is_cuda
            and lhs.dtype == torch.bfloat16
            and self.rhs.dtype == torch.bfloat16
            and m_indices.dtype in (torch.int32, torch.int)
        ):
            rhs_sel = self.rhs.index_select(0, m_indices.to(torch.long)).to(lhs.dtype)
            out = torch.bmm(rhs_sel, lhs.unsqueeze(-1)).squeeze(-1)
            return out

        lhs_c = lhs.contiguous()
        rhs_c = self.rhs.contiguous()
        idx_c = m_indices.contiguous()
        if idx_c.dtype != torch.int32:
            idx_c = idx_c.to(torch.int32)

        M = int(lhs_c.shape[0])

        if self.enable_sort and M >= self.sort_min_M:
            perm = torch.argsort(idx_c, stable=True)
            inv_perm = torch.empty_like(perm)
            inv_perm.scatter_(0, perm, torch.arange(M, device=perm.device, dtype=perm.dtype))

            lhs_s = lhs_c.index_select(0, perm)
            idx_s = idx_c.index_select(0, perm)

            out_s = self.custom_ops_lib.grouped_gemm_fwd_cuda(lhs_s, rhs_c, idx_s)
            out = out_s.index_select(0, inv_perm)
            return out

        return self.custom_ops_lib.grouped_gemm_fwd_cuda(lhs_c, rhs_c, idx_c)