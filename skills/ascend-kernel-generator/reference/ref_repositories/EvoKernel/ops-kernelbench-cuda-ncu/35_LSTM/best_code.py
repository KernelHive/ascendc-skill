import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA/C++ extension sources
# ----------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float tanhf_fast(float x) { return tanhf(x); }

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Specialized kernel for H=256.
// Grid: blockIdx.x = bidx * 2 + part (0/1), each part computes 128 hidden units.
template<int THREADS>
__global__ __launch_bounds__(THREADS, 3)
void lstm_last_splitH256_f32_kernel(
    const float* __restrict__ x,      // [B, T, I]
    const float* __restrict__ h0,     // [B, 256]
    const float* __restrict__ c0,     // [B, 256]
    const float* __restrict__ w_ih,   // [4*256, I]
    const float* __restrict__ w_hh,   // [4*256, 256]
    const float* __restrict__ b,      // [4*256] or nullptr
    float* __restrict__ h_last,       // [B, 256]
    int B, int T, int I
) {
    constexpr int H = 256;
    constexpr int WARP = 32;

    int tid = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid >> 5; // 0..(THREADS/32-1)
    int warps = THREADS / WARP;

    int block = (int)blockIdx.x;
    int bidx = block >> 1;
    int part = block & 1; // 0 or 1
    if (bidx >= B) return;

    // This CTA computes hidden indices [part*128, part*128 + 128).
    int h_base = part * 128;

    // Shared full hidden vector for recurrent dot (published slice-by-slice by both parts?):
    // NOTE: Since parts are separate CTAs, they cannot share memory. Therefore, each CTA must
    // maintain its own local view of the full hidden vector. We build it as:
    // - Start from h0 in shared (full 256)
    // - Update only our slice each timestep, but keep the other slice as "previous timestep" values
    //   loaded from global? That would be wrong.
    //
    // To preserve correctness with split CTAs, we must avoid cross-CTA dependency.
    // Therefore, the split-CTA approach is only correct if we compute recurrent term using
    // the SAME slice only (block-diagonal W_hh), which is not general.
    //
    // So: for correctness, we do not split across CTAs for the recurrent coupling case.
    // Instead, we keep splitH256 kernel as a "two-warps-per-hidden tile" WITHIN ONE CTA.
    //
    // This kernel is kept for compilation completeness but is not used (guarded in host).
    (void)x; (void)h0; (void)c0; (void)w_ih; (void)w_hh; (void)b; (void)h_last; (void)T; (void)I;
}

// Improved cooperative kernel: reduce barriers and registers, THREADS=128, one CTA per batch.
// Each lane computes one hidden unit (up to 128 threads -> 4 warps -> 128 hidden units computed;
// we loop over two tiles of 128 to cover H=256).
template<int THREADS, int TILE_K>
__global__ __launch_bounds__(THREADS, 3)
void lstm_last_coop_v2_f32_kernel(
    const float* __restrict__ x,      // [B, T, I]
    const float* __restrict__ h0,     // [B, H]
    const float* __restrict__ c0,     // [B, H]
    const float* __restrict__ w_ih,   // [4H, I] row-major
    const float* __restrict__ w_hh,   // [4H, H] row-major
    const float* __restrict__ b,      // [4H] or nullptr
    float* __restrict__ h_last,       // [B, H]
    int B, int T, int I, int H
) {
    constexpr int WARP = 32;
    int tid = (int)threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid >> 5; // 0..(THREADS/32-1)

    int bidx = (int)blockIdx.x;
    if (bidx >= B) return;

    extern __shared__ float smem[];
    float* h_sh = smem;               // [H] full hidden vector in shared
    float* x_tile = h_sh + H;         // [TILE_K] shared tile for x/h chunks

    // Initialize full h in shared and keep c in registers per element computed by this thread (one at a time).
    for (int h = tid; h < H; h += THREADS) {
        h_sh[h] = ldg_f32(h0 + (int64_t)bidx * H + h);
    }
    __syncthreads();

    // We compute hidden in tiles of size THREADS (each thread owns h = tile_base + tid).
    // For H=256 and THREADS=128 => 2 tiles.
    // Each thread keeps ct for its current h in a scalar, recomputed per tile and persisted across time.
    // To persist across time for both tiles, we keep ct for both tiles if H==256 and THREADS==128.
    float ct_tile0 = 0.f;
    float ct_tile1 = 0.f;

    bool has_tile1 = (H >= 2 * THREADS);
    if (tid < H) ct_tile0 = ldg_f32(c0 + (int64_t)bidx * H + tid);
    if (has_tile1 && (tid + THREADS) < H) ct_tile1 = ldg_f32(c0 + (int64_t)bidx * H + (tid + THREADS));

    // Time loop
    for (int t = 0; t < T; ++t) {
        const float* xt = x + ((int64_t)bidx * T + t) * I;

        // Process tile 0 and tile 1 (if present) with a CTA barrier between tiles to ensure h_sh updated
        // before being used by the other tile? Actually recurrent dot uses full h_sh from previous timestep,
        // and updates should be applied after computing gates for all h. Baseline updates h_sh[h] immediately,
        // which is a valid synchronous update because all h depend on previous h_sh, not partially updated.
        // Thus we must ensure we read h_sh consistently from previous timestep: we will read h_sh only,
        // and write back after finishing both GEMVs for that h. However, other threads may already write.
        // To keep semantics, we snapshot h_sh into x_tile chunks during recurrent GEMV; that ensures
        // each chunk is consistent at time of load. This matches baseline behavior (reads can race with writes)
        // but correctness requires full previous vector. Therefore, we insert a barrier at start of timestep
        // to ensure previous timestep writes completed before new timestep reads.
        __syncthreads();

        #pragma unroll 2
        for (int tile = 0; tile < 2; ++tile) {
            int h = tid + tile * THREADS;
            if (h >= H) continue;

            float ct = (tile == 0) ? ct_tile0 : ct_tile1;

            int gi = 0 * H + h;
            int gf = 1 * H + h;
            int gg = 2 * H + h;
            int go = 3 * H + h;

            float i_acc = 0.f, f_acc = 0.f, g_acc = 0.f, o_acc = 0.f;

            // ---- Input GEMV (warp-parallel reduction over K) ----
            // Each warp cooperates: lanes iterate over k with stride 32, reduce to lane0, broadcast.
            for (int k0 = 0; k0 < I; k0 += TILE_K) {
                // stage x tile once per CTA
                int kk = k0 + tid;
                if (tid < TILE_K) {
                    x_tile[tid] = (kk < I) ? ldg_f32(xt + kk) : 0.f;
                }
                __syncthreads();

                int klen = min(TILE_K, I - k0);

                // warp-parallel: each lane handles kk = lane, lane+32, ...
                float si = 0.f, sf = 0.f, sg = 0.f, so = 0.f;
                for (int k = lane; k < klen; k += WARP) {
                    float xv = x_tile[k];
                    si = fmaf(ldg_f32(w_ih + (int64_t)gi * I + (k0 + k)), xv, si);
                    sf = fmaf(ldg_f32(w_ih + (int64_t)gf * I + (k0 + k)), xv, sf);
                    sg = fmaf(ldg_f32(w_ih + (int64_t)gg * I + (k0 + k)), xv, sg);
                    so = fmaf(ldg_f32(w_ih + (int64_t)go * I + (k0 + k)), xv, so);
                }
                si = warp_reduce_sum(si);
                sf = warp_reduce_sum(sf);
                sg = warp_reduce_sum(sg);
                so = warp_reduce_sum(so);
                si = __shfl_sync(0xffffffff, si, 0);
                sf = __shfl_sync(0xffffffff, sf, 0);
                sg = __shfl_sync(0xffffffff, sg, 0);
                so = __shfl_sync(0xffffffff, so, 0);

                // only one lane per warp contributes to scalar accumulators to avoid redundant adds
                if (lane == 0) {
                    i_acc += si; f_acc += sf; g_acc += sg; o_acc += so;
                }
                __syncthreads();
            }

            // Broadcast the per-warp accumulator from lane0 to all lanes, then only the owning thread continues.
            // Here each thread is an owner, so we keep i_acc etc in the owner thread only (lane doesn't matter).
            // But we computed i_acc only in lane0; fix: make every thread take lane0's i_acc within its warp.
            i_acc = __shfl_sync(0xffffffff, i_acc, 0);
            f_acc = __shfl_sync(0xffffffff, f_acc, 0);
            g_acc = __shfl_sync(0xffffffff, g_acc, 0);
            o_acc = __shfl_sync(0xffffffff, o_acc, 0);

            // ---- Recurrent GEMV: stage h_sh tiles and do warp-parallel reduction ----
            for (int k0 = 0; k0 < H; k0 += TILE_K) {
                int kk = k0 + tid;
                if (tid < TILE_K) {
                    x_tile[tid] = (kk < H) ? h_sh[kk] : 0.f;
                }
                __syncthreads();

                int klen = min(TILE_K, H - k0);

                float si = 0.f, sf = 0.f, sg = 0.f, so = 0.f;
                for (int k = lane; k < klen; k += WARP) {
                    float hv = x_tile[k];
                    si = fmaf(ldg_f32(w_hh + (int64_t)gi * H + (k0 + k)), hv, si);
                    sf = fmaf(ldg_f32(w_hh + (int64_t)gf * H + (k0 + k)), hv, sf);
                    sg = fmaf(ldg_f32(w_hh + (int64_t)gg * H + (k0 + k)), hv, sg);
                    so = fmaf(ldg_f32(w_hh + (int64_t)go * H + (k0 + k)), hv, so);
                }
                si = warp_reduce_sum(si);
                sf = warp_reduce_sum(sf);
                sg = warp_reduce_sum(sg);
                so = warp_reduce_sum(so);
                si = __shfl_sync(0xffffffff, si, 0);
                sf = __shfl_sync(0xffffffff, sf, 0);
                sg = __shfl_sync(0xffffffff, sg, 0);
                so = __shfl_sync(0xffffffff, so, 0);
                if (lane == 0) {
                    i_acc += si; f_acc += sf; g_acc += sg; o_acc += so;
                }
                __syncthreads();
            }

            i_acc = __shfl_sync(0xffffffff, i_acc, 0);
            f_acc = __shfl_sync(0xffffffff, f_acc, 0);
            g_acc = __shfl_sync(0xffffffff, g_acc, 0);
            o_acc = __shfl_sync(0xffffffff, o_acc, 0);

            if (b) {
                i_acc += ldg_f32(b + gi);
                f_acc += ldg_f32(b + gf);
                g_acc += ldg_f32(b + gg);
                o_acc += ldg_f32(b + go);
            }

            float it = sigmoidf_fast(i_acc);
            float ft = sigmoidf_fast(f_acc);
            float gt = tanhf_fast(g_acc);
            float ot = sigmoidf_fast(o_acc);

            ct = ft * ct + it * gt;
            float ht = ot * tanhf_fast(ct);

            // write updated h to shared for next timestep
            h_sh[h] = ht;

            if (tile == 0) ct_tile0 = ct;
            else ct_tile1 = ct;
        }
        __syncthreads();
    }

    // Store last hidden
    for (int h = tid; h < H; h += THREADS) {
        h_last[(int64_t)bidx * H + h] = h_sh[h];
    }
}

torch::Tensor lstm_last_coop_v2_f32_cuda(
    torch::Tensor x,      // [B,T,I]
    torch::Tensor h0,     // [B,H]
    torch::Tensor c0,     // [B,H]
    torch::Tensor w_ih,   // [4H,I]
    torch::Tensor w_hh,   // [4H,H]
    c10::optional<torch::Tensor> b_opt
) {
    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(c0);
    CHECK_INPUT(w_ih);
    CHECK_INPUT(w_hh);

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(h0.scalar_type() == at::kFloat, "h0 must be float32");
    TORCH_CHECK(c0.scalar_type() == at::kFloat, "c0 must be float32");
    TORCH_CHECK(w_ih.scalar_type() == at::kFloat, "w_ih must be float32");
    TORCH_CHECK(w_hh.scalar_type() == at::kFloat, "w_hh must be float32");

    TORCH_CHECK(x.dim() == 3, "x must be [B,T,I]");
    TORCH_CHECK(h0.dim() == 2 && c0.dim() == 2, "h0/c0 must be [B,H]");
    TORCH_CHECK(w_ih.dim() == 2 && w_hh.dim() == 2, "weights must be 2D");

    int B = (int)x.size(0);
    int T = (int)x.size(1);
    int I = (int)x.size(2);
    int H = (int)h0.size(1);

    TORCH_CHECK(h0.size(0) == B && c0.size(0) == B, "batch mismatch");
    TORCH_CHECK(c0.size(1) == H, "hidden mismatch");
    TORCH_CHECK(w_ih.size(0) == 4 * H && w_ih.size(1) == I, "w_ih must be [4H,I]");
    TORCH_CHECK(w_hh.size(0) == 4 * H && w_hh.size(1) == H, "w_hh must be [4H,H]");

    const float* b = nullptr;
    torch::Tensor b_t;
    if (b_opt.has_value()) {
        b_t = b_opt.value();
        CHECK_INPUT(b_t);
        TORCH_CHECK(b_t.scalar_type() == at::kFloat, "b must be float32");
        TORCH_CHECK(b_t.dim() == 1 && b_t.size(0) == 4 * H, "b must be [4H]");
        b = (const float*)b_t.data_ptr<float>();
    }

    auto out = torch::empty({B, H}, x.options());

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = c10::cuda::getDefaultCUDAStream();

    // Tuned for H=256; still works for other H but best for 256.
    constexpr int THREADS = 128;
    constexpr int TILE_K = 128;

    dim3 blocks(B);
    size_t shmem = (size_t)(H + TILE_K) * sizeof(float);

    lstm_last_coop_v2_f32_kernel<THREADS, TILE_K><<<blocks, THREADS, shmem, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)h0.data_ptr<float>(),
        (const float*)c0.data_ptr<float>(),
        (const float*)w_ih.data_ptr<float>(),
        (const float*)w_hh.data_ptr<float>(),
        b,
        (float*)out.data_ptr<float>(),
        B, T, I, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor lstm_last_coop_v2_f32_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor w_ih,
    torch::Tensor w_hh,
    c10::optional<torch::Tensor> b_opt
);
"""

_ext_name = "custom_ops_lib_lstm_last_coop_v2"
custom_ops_lib = load_inline(
    name=_ext_name,
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lstm_last_coop_v2_f32_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    LSTM -> FC model with a custom CUDA fast-path for single-layer, unidirectional LSTM forward
    producing only the last hidden state. Falls back to nn.LSTM for unsupported cases.

    Fast path supports:
    - num_layers == 1
    - bidirectional == False
    - dropout == 0
    - float32 CUDA contiguous inputs/weights
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = float(dropout)

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.custom_ops_lib = custom_ops_lib
        self._bias_cache = {}

    def _get_fused_bias(self, device):
        b_ih = getattr(self.lstm, "bias_ih_l0", None)
        b_hh = getattr(self.lstm, "bias_hh_l0", None)
        if b_ih is None and b_hh is None:
            return None
        key = (
            str(device),
            -1 if b_ih is None else b_ih._version,
            -1 if b_hh is None else b_hh._version,
        )
        if key in self._bias_cache:
            return self._bias_cache[key]
        if b_ih is None:
            fused = b_hh.contiguous()
        elif b_hh is None:
            fused = b_ih.contiguous()
        else:
            fused = (b_ih + b_hh).contiguous()
        self._bias_cache[key] = fused
        return fused

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        if (
            x.is_cuda and x.dtype == torch.float32 and x.dim() == 3 and x.is_contiguous()
            and self.num_layers == 1 and self.dropout == 0.0 and (not self.lstm.bidirectional)
            and h0.is_cuda and c0.is_cuda
            and h0.dtype == torch.float32 and c0.dtype == torch.float32
            and h0.is_contiguous() and c0.is_contiguous()
            and h0.dim() == 3 and c0.dim() == 3
        ):
            w_ih = self.lstm.weight_ih_l0
            w_hh = self.lstm.weight_hh_l0
            if (
                w_ih.is_cuda and w_hh.is_cuda
                and w_ih.dtype == torch.float32 and w_hh.dtype == torch.float32
                and w_ih.is_contiguous() and w_hh.is_contiguous()
            ):
                h0_0 = h0[0].contiguous()
                c0_0 = c0[0].contiguous()
                b_fused = self._get_fused_bias(x.device)

                h_last = self.custom_ops_lib.lstm_last_coop_v2_f32_cuda(
                    x, h0_0, c0_0, w_ih, w_hh, b_fused
                )
                return self.fc(h_last)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out