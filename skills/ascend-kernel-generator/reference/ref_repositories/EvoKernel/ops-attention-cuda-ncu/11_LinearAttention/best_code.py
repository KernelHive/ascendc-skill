import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized CUDA: two-stage linear attention forward for N=512, D=64 (float32)
# Stage1 (stats): compute Ksum[D] and KV[D,D] per (B,H) without atomics.
# Stage2 (out): compute Out[B,H,N,D], computing denom once per token and
#               producing 4 output channels per thread.
# -----------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

enum FeatureMapType : int {
    FM_ELU_PLUS_ONE = 0,
    FM_RELU = 1,
    FM_IDENTITY = 2
};

__device__ __forceinline__ float phi(float x, int fm_type) {
    if (fm_type == FM_ELU_PLUS_ONE) {
        return (x > 0.0f) ? (x + 1.0f) : __expf(x);
    } else if (fm_type == FM_RELU) {
        return (x > 0.0f) ? x : 0.0f;
    } else {
        return x;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    // full mask
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void linattn_stats_bh_N512_D64(
    const float* __restrict__ K,   // [B,H,N,D]
    const float* __restrict__ V,   // [B,H,N,D]
    float* __restrict__ Ksum,      // [B,H,D]
    float* __restrict__ KV,        // [B,H,D,D]
    int B, int H,
    int fm_type
) {
    constexpr int N = 512;
    constexpr int D = 64;

    int bh = (int)blockIdx.x;
    int b = bh / H;
    int h = bh - b * H;

    // Thread mapping:
    // tid in [0,1023] => dk = tid / 16 (0..63), group = tid % 16
    // group splits dv in chunks of 4: dv_base = group * 4 (0..60)
    int tid = (int)threadIdx.x;
    int dk = tid >> 4;          // /16
    int grp = tid & 15;         // %16
    int dv_base = grp * 4;

    const int base = ((b * H + h) * N) * D;

    float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
    float ksum_acc = 0.f;

    // Alignment check for float4 loads
    uintptr_t v_addr = (uintptr_t)(V + base);
    bool v_aligned = (v_addr % 16u) == 0u;

    #pragma unroll 4
    for (int n = 0; n < N; n++) {
        float kf = phi(__ldg(K + base + n * D + dk), fm_type);
        ksum_acc += kf;

        const float* vptr = V + base + n * D + dv_base;
        if (v_aligned) {
            float4 vv = *reinterpret_cast<const float4*>(vptr);
            kv0 = fmaf(kf, vv.x, kv0);
            kv1 = fmaf(kf, vv.y, kv1);
            kv2 = fmaf(kf, vv.z, kv2);
            kv3 = fmaf(kf, vv.w, kv3);
        } else {
            kv0 = fmaf(kf, vptr[0], kv0);
            kv1 = fmaf(kf, vptr[1], kv1);
            kv2 = fmaf(kf, vptr[2], kv2);
            kv3 = fmaf(kf, vptr[3], kv3);
        }
    }

    // Write Ksum and KV (no atomics: unique ownership)
    if (dv_base == 0) {
        Ksum[(bh * D) + dk] = ksum_acc;
    }

    float* kv_row = KV + (bh * D * D) + dk * D + dv_base;
    // KV is contiguous, should be aligned, but keep safe
    uintptr_t kv_addr = (uintptr_t)kv_row;
    if ((kv_addr % 16u) == 0u) {
        *reinterpret_cast<float4*>(kv_row) = make_float4(kv0, kv1, kv2, kv3);
    } else {
        kv_row[0] = kv0; kv_row[1] = kv1; kv_row[2] = kv2; kv_row[3] = kv3;
    }
}

__global__ void linattn_out_bhn_N512_D64(
    const float* __restrict__ Q,    // [B,H,N,D]
    const float* __restrict__ Ksum, // [B,H,D]
    const float* __restrict__ KV,   // [B,H,D,D]
    float* __restrict__ Out,        // [B,H,N,D]
    int B, int H,
    float eps,
    int fm_type
) {
    constexpr int N = 512;
    constexpr int D = 64;

    // 2D grid: x = bh, y = token n
    int bh = (int)blockIdx.x;
    int n  = (int)blockIdx.y;

    // blockDim.x = 256 threads = 8 warps
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0..7

    int b = bh / H;
    int h = bh - b * H;

    const int base = ((b * H + h) * N) * D;

    // Each thread computes 4 output channels dv_base derived from tid: 0..252 step 4
    int dv_base = (tid * 4); // 0..1020
    if (dv_base >= D) return; // only first 16 threads active; but keep 256 threads for denom work

    // Compute denom via warp-parallel dot over dk:
    // Each warp handles 8 dk indices: dk = lane % 8 + warp*8 => 0..63 across 8 warps.
    int dk = (lane & 7) + (warp * 8);
    float qf = phi(__ldg(Q + base + n * D + dk), fm_type);
    float ks = __ldg(Ksum + bh * D + dk);
    float prod = qf * ks;

    // Reduce within warp
    float wsum = warp_sum(prod);
    // Sum warp results (8 warps) using shared
    __shared__ float sWarp[8];
    if (lane == 0) sWarp[warp] = wsum;
    __syncthreads();

    float denom = 0.0f;
    if (tid < 8) denom = sWarp[tid];
    // reduce denom across first warp lanes using shuffle
    // broadcast to all threads via shared
    float denom_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < 8) ? denom : 0.0f;
        v = warp_sum(v);
        if (lane == 0) sWarp[0] = v;
    }
    __syncthreads();
    denom_sum = sWarp[0];

    float inv = 1.0f / (denom_sum + eps);

    // Now compute 4 outputs for dv_base..dv_base+3 using full dk loop (64).
    // Only 16 threads do stores, but all blocks still help denom (better latency hiding).
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    // Vectorize KV loads for each dk
    const float* kv_bh = KV + bh * D * D;

    #pragma unroll 8
    for (int dk2 = 0; dk2 < D; dk2++) {
        float qf2 = phi(__ldg(Q + base + n * D + dk2), fm_type);
        const float* kv_ptr = kv_bh + dk2 * D + dv_base;

        // KV is aligned; but keep safe
        uintptr_t kv_addr = (uintptr_t)kv_ptr;
        if ((kv_addr % 16u) == 0u) {
            float4 kvv = *reinterpret_cast<const float4*>(kv_ptr);
            acc0 = fmaf(qf2, kvv.x, acc0);
            acc1 = fmaf(qf2, kvv.y, acc1);
            acc2 = fmaf(qf2, kvv.z, acc2);
            acc3 = fmaf(qf2, kvv.w, acc3);
        } else {
            acc0 = fmaf(qf2, kv_ptr[0], acc0);
            acc1 = fmaf(qf2, kv_ptr[1], acc1);
            acc2 = fmaf(qf2, kv_ptr[2], acc2);
            acc3 = fmaf(qf2, kv_ptr[3], acc3);
        }
    }

    float* out_ptr = Out + base + n * D + dv_base;
    uintptr_t out_addr = (uintptr_t)out_ptr;
    float4 o4 = make_float4(acc0 * inv, acc1 * inv, acc2 * inv, acc3 * inv);
    if ((out_addr % 16u) == 0u) {
        *reinterpret_cast<float4*>(out_ptr) = o4;
    } else {
        out_ptr[0] = o4.x; out_ptr[1] = o4.y; out_ptr[2] = o4.z; out_ptr[3] = o4.w;
    }
}

std::vector<torch::Tensor> linear_attention_forward_cuda_v2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double eps,
    int64_t feature_map_type
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q/K/V must be [B,H,N,D]");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V must have same shape");

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int N = (int)Q.size(2);
    int D = (int)Q.size(3);

    TORCH_CHECK(N == 512, "Optimized kernel supports only seq_len N=512");
    TORCH_CHECK(D == 64, "Optimized kernel supports only head dim D=64");
    TORCH_CHECK(feature_map_type == 0 || feature_map_type == 1 || feature_map_type == 2, "Invalid feature_map_type");

    auto opts = Q.options();
    auto Out = torch::empty_like(Q);
    auto Ksum = torch::empty({B, H, D}, opts);
    auto KV = torch::empty({B, H, D, D}, opts);

    // Stage 1: stats per (b,h)
    {
        dim3 grid((unsigned)(B * H), 1, 1);
        dim3 block(1024, 1, 1); // 64*16 threads
        linattn_stats_bh_N512_D64<<<grid, block>>>(
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Ksum.data_ptr<float>(),
            (float*)KV.data_ptr<float>(),
            B, H,
            (int)feature_map_type
        );
    }

    // Stage 2: output per (b,h,n)
    {
        dim3 grid((unsigned)(B * H), (unsigned)N, 1);
        dim3 block(256, 1, 1);
        linattn_out_bhn_N512_D64<<<grid, block>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)Ksum.data_ptr<float>(),
            (const float*)KV.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H,
            (float)eps,
            (int)feature_map_type
        );
    }

    return {Out};
}
"""

cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> linear_attention_forward_cuda_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double eps, int64_t feature_map_type);
"""

custom_ops_lib = load_inline(
    name="custom_linear_attention_fwd_n512_d64_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["linear_attention_forward_cuda_v2"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, d_model, n_heads, feature_map="elu"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_k = self.d_model // self.n_heads
        self.eps = 1e-6

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        if feature_map == "elu":
            self.feature_map_type = 0
        elif feature_map == "relu":
            self.feature_map_type = 1
        elif feature_map == "identity":
            self.feature_map_type = 2
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        use_fast = (
            (not self.training)
            and Q.is_cuda and K.is_cuda and V.is_cuda
            and Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32
            and Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
            and Q.dim() == 4 and seq_len == 512 and self.d_k == 64
        )

        if use_fast:
            (out,) = custom_ops_lib.linear_attention_forward_cuda_v2(
                Q, K, V, float(self.eps), int(self.feature_map_type)
            )
        else:
            if self.feature_map_type == 0:
                phi = lambda t: F.elu(t) + 1
            elif self.feature_map_type == 1:
                phi = F.relu
            else:
                phi = lambda t: t

            Qm = phi(Q)
            Km = phi(K)
            KV = torch.matmul(Km.transpose(-2, -1), V)
            Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Qm, Km.sum(dim=2)).unsqueeze(-1) + self.eps)
            out = torch.matmul(Qm, KV) * Z

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out