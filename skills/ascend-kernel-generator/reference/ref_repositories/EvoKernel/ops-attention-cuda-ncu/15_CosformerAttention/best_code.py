import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# v2: Keep stats kernel (writes global Ksum/KV), improve OUT kernel:
#  - block = 64 threads (2 warps) per (bh,n)
#  - stage Ksum (64 floats) + KV (64*64 floats) into shared once per block
#  - load q as 16 float4 once into registers (warp1 only)
#  - warp0 computes denom via warp reduction (no shared multi-warp reduce)
#  - use constant-memory decay LUT to remove expf from hot loops
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

__device__ __forceinline__ float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// decay LUT for N=512
__device__ __constant__ float c_decay512[512];

__global__ void cosformer_stats_bh_N512_D64(
    const float* __restrict__ K,   // [B,H,N,D]
    const float* __restrict__ V,   // [B,H,N,D]
    float* __restrict__ Ksum,      // [B,H,D]
    float* __restrict__ KV,        // [B,H,D,D]  (dv-major: [dv][dk])
    int B, int H
) {
    constexpr int N = 512;
    constexpr int D = 64;

    int bh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    int dv = tid >> 4;        // 0..63
    int grp = tid & 15;       // 0..15
    int dk_base = grp * 4;    // 0..60

    int b = bh / H;
    int h = bh - b * H;
    const int base = ((b * H + h) * N) * D;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float ksum0 = 0.f, ksum1 = 0.f, ksum2 = 0.f, ksum3 = 0.f;

    #pragma unroll 4
    for (int n = 0; n < N; n++) {
        float dec = c_decay512[n];

        const float* kptr = K + base + n * D + dk_base;
        float k0 = relu_f(__ldg(kptr + 0)) * dec;
        float k1 = relu_f(__ldg(kptr + 1)) * dec;
        float k2 = relu_f(__ldg(kptr + 2)) * dec;
        float k3 = relu_f(__ldg(kptr + 3)) * dec;

        float v = __ldg(V + base + n * D + dv);

        acc0 = fmaf(k0, v, acc0);
        acc1 = fmaf(k1, v, acc1);
        acc2 = fmaf(k2, v, acc2);
        acc3 = fmaf(k3, v, acc3);

        if (dv == 0) {
            ksum0 += k0; ksum1 += k1; ksum2 += k2; ksum3 += k3;
        }
    }

    float* kv_ptr = KV + (bh * D * D) + dv * D + dk_base;
    uintptr_t kv_addr = (uintptr_t)kv_ptr;
    if ((kv_addr % 16u) == 0u) {
        *reinterpret_cast<float4*>(kv_ptr) = make_float4(acc0, acc1, acc2, acc3);
    } else {
        kv_ptr[0] = acc0; kv_ptr[1] = acc1; kv_ptr[2] = acc2; kv_ptr[3] = acc3;
    }

    if (dv == 0) {
        float* ks_ptr = Ksum + (bh * D) + dk_base;
        uintptr_t ks_addr = (uintptr_t)ks_ptr;
        float4 ks4 = make_float4(ksum0, ksum1, ksum2, ksum3);
        if ((ks_addr % 16u) == 0u) {
            *reinterpret_cast<float4*>(ks_ptr) = ks4;
        } else {
            ks_ptr[0] = ks4.x; ks_ptr[1] = ks4.y; ks_ptr[2] = ks4.z; ks_ptr[3] = ks4.w;
        }
    }
}

// OUT kernel v2: block=64 threads (2 warps)
// Shared: Ksum[64] + KV[64*64] floats
__global__ void cosformer_out_bhn_N512_D64_v2(
    const float* __restrict__ Q,    // [B,H,N,D]
    const float* __restrict__ Ksum, // [B,H,D]
    const float* __restrict__ KV,   // [B,H,D,D] dv-major
    float* __restrict__ Out,        // [B,H,N,D]
    int B, int H,
    float eps
) {
    constexpr int N = 512;
    constexpr int D = 64;

    int bh = (int)blockIdx.x;
    int n  = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0,1

    int b = bh / H;
    int h = bh - b * H;
    const int base = ((b * H + h) * N) * D;

    extern __shared__ float smem[];
    float* sKsum = smem;                  // 64
    float* sKV   = smem + D;              // 4096

    // Stage summaries into shared (coalesced)
    // 64 threads load Ksum: each thread loads one element
    if (tid < D) {
        sKsum[tid] = __ldg(Ksum + bh * D + tid);
    }
    // Load KV: 64 threads load 4096 floats => 64 floats per thread
    const float* kv_g = KV + bh * D * D;
    #pragma unroll 4
    for (int i = tid; i < D * D; i += 64) {
        sKV[i] = __ldg(kv_g + i);
    }
    __syncthreads();

    float dec = c_decay512[n];

    // Warp0 computes denom
    __shared__ float sInv;
    if (warp == 0) {
        int dk0 = lane;
        int dk1 = lane + 32;

        float q0 = relu_f(__ldg(Q + base + n * D + dk0)) * dec;
        float q1 = relu_f(__ldg(Q + base + n * D + dk1)) * dec;

        float part = fmaf(q0, sKsum[dk0], q1 * sKsum[dk1]);
        float denom = warp_sum(part);

        if (lane == 0) sInv = 1.0f / (denom + eps);
    }

    // Warp1 computes output using q cached in regs as float4[16]
    if (warp == 1) {
        float inv = sInv;

        // load q once as 16 float4 (all 64 dims)
        float4 q4[16];
        const float* qptr = Q + base + n * D;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float4 t = *reinterpret_cast<const float4*>(qptr + i * 4);
            // relu+decay
            t.x = relu_f(t.x) * dec;
            t.y = relu_f(t.y) * dec;
            t.z = relu_f(t.z) * dec;
            t.w = relu_f(t.w) * dec;
            q4[i] = t;
        }

        // each lane handles one dv in 0..31, and also dv+32
        int dvA = lane;
        int dvB = lane + 32;

        float accA = 0.f;
        float accB = 0.f;

        // dot with KV rows; KV is dv-major: row starts at dv*64
        const float* rowA = sKV + dvA * D;
        const float* rowB = sKV + dvB * D;

        // accumulate in float4 chunks
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float4 kvA = *reinterpret_cast<const float4*>(rowA + i * 4);
            float4 kvB = *reinterpret_cast<const float4*>(rowB + i * 4);
            float4 qq  = q4[i];

            accA = fmaf(qq.x, kvA.x, accA);
            accA = fmaf(qq.y, kvA.y, accA);
            accA = fmaf(qq.z, kvA.z, accA);
            accA = fmaf(qq.w, kvA.w, accA);

            accB = fmaf(qq.x, kvB.x, accB);
            accB = fmaf(qq.y, kvB.y, accB);
            accB = fmaf(qq.z, kvB.z, accB);
            accB = fmaf(qq.w, kvB.w, accB);
        }

        // store out (coalesced across lanes)
        float* outp = Out + base + n * D;
        outp[dvA] = accA * inv;
        outp[dvB] = accB * inv;
    }
}

static void upload_decay_lut_512() {
    static bool inited = false;
    if (inited) return;
    float host[512];
    for (int n = 0; n < 512; n++) host[n] = expf(-(float)n / 512.0f);
    cudaMemcpyToSymbol(c_decay512, host, sizeof(host));
    inited = true;
}

std::vector<torch::Tensor> cosformer_attention_forward_cuda_v2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double eps
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

    TORCH_CHECK(N == 512, "Optimized cosformer kernel supports only seq_len N=512");
    TORCH_CHECK(D == 64, "Optimized cosformer kernel supports only head dim D=64");

    upload_decay_lut_512();

    auto opts = Q.options();
    auto Out  = torch::empty_like(Q);
    auto Ksum = torch::empty({B, H, D}, opts);
    auto KV   = torch::empty({B, H, D, D}, opts);

    // Stage 1: stats per (b,h)
    {
        dim3 grid((unsigned)(B * H), 1, 1);
        dim3 block(1024, 1, 1);
        cosformer_stats_bh_N512_D64<<<grid, block>>>(
            (const float*)K.data_ptr<float>(),
            (const float*)V.data_ptr<float>(),
            (float*)Ksum.data_ptr<float>(),
            (float*)KV.data_ptr<float>(),
            B, H
        );
    }

    // Stage 2: out per (b,h,n) with shared staging
    {
        dim3 grid((unsigned)(B * H), (unsigned)N, 1);
        dim3 block(64, 1, 1);
        size_t shmem = (size_t)((64 + 64 * 64) * sizeof(float)); // 256 + 16384 = 16640 bytes
        cosformer_out_bhn_N512_D64_v2<<<grid, block, shmem>>>(
            (const float*)Q.data_ptr<float>(),
            (const float*)Ksum.data_ptr<float>(),
            (const float*)KV.data_ptr<float>(),
            (float*)Out.data_ptr<float>(),
            B, H,
            (float)eps
        );
    }

    return {Out};
}
"""

cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> cosformer_attention_forward_cuda_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double eps);
"""

custom_ops_lib = load_inline(
    name="custom_cosformer_attention_fwd_n512_d64_v2_outopt",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["cosformer_attention_forward_cuda_v2"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Cosformer Attention with an optimized CUDA forward path for (seq_len=512, d_k=64, fp32).
    Falls back to the original PyTorch formulation otherwise.
    """

    def __init__(self, d_model, n_heads):
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
            and seq_len == 512 and self.d_k == 64
        )

        if use_fast:
            (out,) = custom_ops_lib.cosformer_attention_forward_cuda_v2(Q, K, V, float(self.eps))
        else:
            Qm = F.relu(Q)
            Km = F.relu(K)

            position_indices = torch.arange(seq_len, device=x.device, dtype=Qm.dtype).view(1, 1, seq_len, 1)
            decay = torch.exp(-position_indices / float(seq_len))
            Qm = Qm * decay
            Km = Km * decay

            KV = torch.matmul(Km.transpose(-2, -1), V)
            Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Qm, Km.sum(dim=2)).unsqueeze(-1) + self.eps)
            out = torch.matmul(Qm, KV) * Z

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out