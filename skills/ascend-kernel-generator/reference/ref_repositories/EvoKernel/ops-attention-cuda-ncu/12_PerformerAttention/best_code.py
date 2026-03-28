import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

// Ksum/KV from already-transformed K features:
// Kf: [BH,N,F], V: [BH,N,D]
// Ksum[bh,f] = sum_n Kf
// KV[bh,f,d] = sum_n Kf * V
__global__ void ksum_kv_from_features(
    const float* __restrict__ Kf,   // [BH,N,F]
    const float* __restrict__ V,    // [BH,N,D]
    float* __restrict__ Ksum_g,     // [BH,F]
    float* __restrict__ KV_g,       // [BH,F,D]
    int N, int F, int D
){
    int bh = (int)blockIdx.x;
    int f  = (int)blockIdx.y;

    const float* Kf_bh = Kf + ((size_t)bh * N * F);
    const float* V_bh  = V  + ((size_t)bh * N * D);

    // One thread computes Ksum (cheap, avoids atomics)
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int n = 0; n < N; ++n) {
            sum += Kf_bh[n * F + f];
        }
        Ksum_g[bh * F + f] = sum;
    }

    // KV accumulation over D (vectorize float4 when possible)
    int tid = (int)threadIdx.x;
    if ((D & 3) == 0) {
        int D4 = D >> 2;
        int d4 = (int)blockIdx.z * (int)blockDim.x + tid;
        if (d4 >= D4) return;

        float4 acc4 = {0.f, 0.f, 0.f, 0.f};
        for (int n = 0; n < N; ++n) {
            float k = Kf_bh[n * F + f];
            const float4* v4 = reinterpret_cast<const float4*>(V_bh + (size_t)n * D);
            float4 vv = v4[d4];
            acc4.x = fmaf(k, vv.x, acc4.x);
            acc4.y = fmaf(k, vv.y, acc4.y);
            acc4.z = fmaf(k, vv.z, acc4.z);
            acc4.w = fmaf(k, vv.w, acc4.w);
        }
        float4* out4 = reinterpret_cast<float4*>(KV_g + ((size_t)(bh * F + f) * D));
        out4[d4] = acc4;
    } else {
        int d = (int)blockIdx.z * (int)blockDim.x + tid;
        if (d >= D) return;

        float acc = 0.f;
        for (int n = 0; n < N; ++n) {
            float k = Kf_bh[n * F + f];
            float vv = V_bh[(size_t)n * D + d];
            acc = fmaf(k, vv, acc);
        }
        KV_g[((size_t)(bh * F + f) * D) + d] = acc;
    }
}

// Fused output kernel from features with shared-memory Q caching:
// Qf: [BH,N,F], Ksum: [BH,F], KV: [BH,F,D] -> Out: [BH,N,D]
__global__ void out_fused_from_features(
    const float* __restrict__ Qf,      // [BH,N,F]
    const float* __restrict__ Ksum_g,  // [BH,F]
    const float* __restrict__ KV_g,    // [BH,F,D]
    float* __restrict__ Out,           // [BH,N,D]
    int N, int F, int D,
    float eps_out
){
    int bh = (int)blockIdx.x;
    int n  = (int)blockIdx.y;

    const float* Qf_row = Qf + ((size_t)(bh * N + n) * F);
    const float* Ksum   = Ksum_g + (size_t)bh * F;
    const float* KV     = KV_g + (size_t)(bh * F) * D;
    float* out_row      = Out  + ((size_t)(bh * N + n) * D);

    extern __shared__ float qsh[]; // size F floats

    // Load Q feature vector into shared (coalesced)
    for (int i = threadIdx.x; i < F; i += blockDim.x) {
        qsh[i] = __ldg(Qf_row + i);
    }
    __syncthreads();

    // Denominator computed once per block
    __shared__ float denom_sh;
    if (threadIdx.x == 0) {
        float denom = 0.f;
        for (int f = 0; f < F; ++f) {
            denom = fmaf(qsh[f], Ksum[f], denom);
        }
        denom_sh = denom;
    }
    __syncthreads();
    float z = 1.f / (denom_sh + eps_out);

    // Compute output over D (vectorize float4 when possible)
    if ((D & 3) == 0) {
        int D4 = D >> 2;
        int d4 = (int)threadIdx.x + (int)blockIdx.z * (int)blockDim.x;
        if (d4 >= D4) return;

        float4 acc4 = {0.f, 0.f, 0.f, 0.f};
        for (int f = 0; f < F; ++f) {
            float q = qsh[f];
            const float4* kv4 = reinterpret_cast<const float4*>(KV + (size_t)f * D);
            float4 vv = kv4[d4];
            acc4.x = fmaf(q, vv.x, acc4.x);
            acc4.y = fmaf(q, vv.y, acc4.y);
            acc4.z = fmaf(q, vv.z, acc4.z);
            acc4.w = fmaf(q, vv.w, acc4.w);
        }
        acc4.x *= z; acc4.y *= z; acc4.z *= z; acc4.w *= z;
        float4* out4 = reinterpret_cast<float4*>(out_row);
        out4[d4] = acc4;
    } else {
        int d = (int)threadIdx.x + (int)blockIdx.z * (int)blockDim.x;
        if (d >= D) return;

        float acc = 0.f;
        for (int f = 0; f < F; ++f) {
            acc = fmaf(qsh[f], KV[(size_t)f * D + d], acc);
        }
        out_row[d] = acc * z;
    }
}

torch::Tensor performer_attention_cuda(
    torch::Tensor Q_features,         // [B,H,N,F]   (already transformed)
    torch::Tensor K_features,         // [B,H,N,F]   (already transformed)
    torch::Tensor V,                  // [B,H,N,D]
    double eps
) {
    CHECK_INPUT(Q_features);
    CHECK_INPUT(K_features);
    CHECK_INPUT(V);

    TORCH_CHECK(Q_features.dim() == 4, "Q_features must be [B,H,N,F]");
    TORCH_CHECK(K_features.dim() == 4, "K_features must be [B,H,N,F]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,N,D]");

    int B = (int)Q_features.size(0);
    int H = (int)Q_features.size(1);
    int N = (int)Q_features.size(2);
    int F = (int)Q_features.size(3);
    int D = (int)V.size(3);

    TORCH_CHECK(K_features.size(0)==B && K_features.size(1)==H && K_features.size(2)==N && K_features.size(3)==F, "K_features shape mismatch");
    TORCH_CHECK(V.size(0)==B && V.size(1)==H && V.size(2)==N, "V shape mismatch");

    auto opts = Q_features.options();
    auto Out  = torch::empty({B, H, N, D}, opts);

    int BH = B * H;

    auto Ksum = torch::empty({BH, F}, opts);
    auto KV   = torch::empty({BH, F, D}, opts);

    // Stage 1: Ksum + KV from feature-space K
    int threads1 = 256;
    dim3 block1(threads1);
    int zdim1 = ((D & 3) == 0) ? (((D >> 2) + threads1 - 1) / threads1)
                              : ((D + threads1 - 1) / threads1);
    dim3 grid1(BH, F, zdim1);

    // Flatten [B,H,...] as BH in memory (contiguous in last dims)
    ksum_kv_from_features<<<grid1, block1>>>(
        K_features.data_ptr<float>(),
        V.data_ptr<float>(),
        Ksum.data_ptr<float>(),
        KV.data_ptr<float>(),
        N, F, D
    );

    // Stage 2: Out from feature-space Q
    int threads2 = 128;
    dim3 block2(threads2);
    int zdim2 = ((D & 3) == 0) ? (((D >> 2) + threads2 - 1) / threads2)
                              : ((D + threads2 - 1) / threads2);
    dim3 grid2(BH, N, zdim2);

    size_t shmem = (size_t)F * sizeof(float);

    out_fused_from_features<<<grid2, block2, shmem>>>(
        Q_features.data_ptr<float>(),
        Ksum.data_ptr<float>(),
        KV.data_ptr<float>(),
        Out.data_ptr<float>(),
        N, F, D,
        (float)eps
    );

    return Out;
}
"""

cpp_src = r"""
torch::Tensor performer_attention_cuda(
    torch::Tensor Q_features,
    torch::Tensor K_features,
    torch::Tensor V,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_performer_attention_opt6_features",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["performer_attention_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Performer FAVOR+ attention with optimized CUDA core.
    Important: CUDA op consumes already-transformed Q/K feature tensors (same semantics as reference).
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nb_features = self.d_k
        self.eps = 1e-6

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        projection = torch.randn(self.nb_features, self.d_k)
        self.register_buffer("projection_matrix", projection)

        self.custom_ops = custom_ops_lib

    def softmax_kernel_transformation(self, data, projection_matrix):
        # data: [B,H,N,D]
        data_normalizer = 1.0 / math.sqrt(math.sqrt(self.d_k))
        data = data * data_normalizer
        data_proj = torch.matmul(data, projection_matrix.T)
        data_proj = F.relu(data_proj) + 1e-6
        return data_proj

    def forward(self, x):
        B, N, _ = x.size()

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        P = self.projection_matrix
        if x.is_cuda:
            P = P.to(device=x.device, dtype=torch.float32)
        P = P.contiguous()

        Qf = self.softmax_kernel_transformation(Q, P).contiguous()  # [B,H,N,F]
        Kf = self.softmax_kernel_transformation(K, P).contiguous()  # [B,H,N,F]

        out = self.custom_ops.performer_attention_cuda(Qf, Kf, V, float(self.eps))
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)
        return out