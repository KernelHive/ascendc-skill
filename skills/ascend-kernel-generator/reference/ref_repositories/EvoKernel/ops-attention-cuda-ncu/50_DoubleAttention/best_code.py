import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Optimized CUDA op for double-attention core:
#   attB = softmax(B over S)
#   attV = softmax(V over S)
#   G    = A @ attB^T         [B,Cm,Cn]
#   Z    = G @ attV           [B,Cm,S] -> [B,Cm,H,W]
#
# Optimizations:
#  - Warp-level softmax for S<=64 (typical S=49)
#  - Tiled small-GEMM kernels for G and Z using shared memory
#  - Specialized unrolling for S=49 fast path (kept safe and correct)
# ------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// One warp per (b,n). blockDim.x must be 32. Works best for S<=64.
__global__ void softmax_warp_lastdim_kernel(
    const float* __restrict__ x,  // [B, Cn, S]
    float* __restrict__ y,        // [B, Cn, S]
    int B, int Cn, int S
) {
    int bn = (int)blockIdx.x;
    int b = bn / Cn;
    int n = bn - b * Cn;
    if (b >= B) return;

    const float* in = x + ((b * Cn + n) * S);
    float* out = y + ((b * Cn + n) * S);

    int lane = (int)(threadIdx.x & 31);

    float maxv = -INFINITY;
    for (int s = lane; s < S; s += 32) {
        maxv = fmaxf(maxv, in[s]);
    }
    maxv = warp_reduce_max(maxv);
    maxv = __shfl_sync(0xffffffff, maxv, 0);

    float sum = 0.0f;
    for (int s = lane; s < S; s += 32) {
        float e = __expf(in[s] - maxv);
        out[s] = e;
        sum += e;
    }
    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffff, sum, 0) + 1e-20f;
    float inv = 1.0f / sum;

    for (int s = lane; s < S; s += 32) {
        out[s] *= inv;
    }
}

// Generic block softmax (fallback for S>64).
// One block per (b,n). blockDim.x = 256.
__global__ void softmax_block_lastdim_kernel(
    const float* __restrict__ x,  // [B, Cn, S]
    float* __restrict__ y,        // [B, Cn, S]
    int B, int Cn, int S
) {
    int bn = (int)blockIdx.x;
    int b = bn / Cn;
    int n = bn - b * Cn;
    if (b >= B) return;

    const float* in = x + ((b * Cn + n) * S);
    float* out = y + ((b * Cn + n) * S);

    float tmax = -INFINITY;
    for (int s = threadIdx.x; s < S; s += blockDim.x) tmax = fmaxf(tmax, in[s]);

    __shared__ float smax[256];
    smax[threadIdx.x] = tmax;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
        __syncthreads();
    }
    float maxv = smax[0];

    float tsum = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float e = expf(in[s] - maxv);
        out[s] = e;
        tsum += e;
    }

    __shared__ float ssum[256];
    ssum[threadIdx.x] = tsum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    float sum = ssum[0] + 1e-20f;
    float inv = 1.0f / sum;

    for (int s = threadIdx.x; s < S; s += blockDim.x) out[s] *= inv;
}

// ------------------------
// G = A @ attB^T
// A:    [B, Cm, S]
// attB: [B, Cn, S]
// G:    [B, Cm, Cn]
// Tiled kernel: block computes tile (TM x TN) for one batch b.
// ------------------------
template<int TM, int TN, int KS, bool UNROLL49>
__global__ void gemm_A_attBt_kernel(
    const float* __restrict__ A,      // [B,Cm,S]
    const float* __restrict__ attB,   // [B,Cn,S]
    float* __restrict__ G,            // [B,Cm,Cn]
    int B, int Cm, int Cn, int S
) {
    int b = (int)blockIdx.z;
    int m0 = (int)blockIdx.y * TM;
    int n0 = (int)blockIdx.x * TN;

    int tid = (int)threadIdx.x; // 0..255
    int mi = tid / TN;          // 0..TM-1
    int ni = tid - mi * TN;     // 0..TN-1

    int m = m0 + mi;
    int n = n0 + ni;

    if (b >= B) return;

    const float* Ap_base = A + ((int64_t)b * (int64_t)Cm) * (int64_t)S;
    const float* Bp_base = attB + ((int64_t)b * (int64_t)Cn) * (int64_t)S;

    __shared__ float As[TM][KS]; // TM*KS floats
    __shared__ float Bs[TN][KS]; // TN*KS floats

    float acc = 0.0f;

    if (UNROLL49) {
        // Specialized path for S==49; one K tile covers all (KS=49).
        // Load As and Bs cooperatively.
        // Total elements to load: TM*KS + TN*KS = (TM+TN)*49.
        // Use all threads with strided linear indexing.
        int totalA = TM * KS;
        int totalB = TN * KS;
        for (int idx = tid; idx < totalA; idx += blockDim.x) {
            int lm = idx / KS;
            int lk = idx - lm * KS;
            int gm = m0 + lm;
            As[lm][lk] = (gm < Cm) ? Ap_base[(int64_t)gm * (int64_t)S + lk] : 0.0f;
        }
        for (int idx = tid; idx < totalB; idx += blockDim.x) {
            int ln = idx / KS;
            int lk = idx - ln * KS;
            int gn = n0 + ln;
            Bs[ln][lk] = (gn < Cn) ? Bp_base[(int64_t)gn * (int64_t)S + lk] : 0.0f;
        }
        __syncthreads();

        if (m < Cm && n < Cn) {
            #pragma unroll
            for (int k = 0; k < 49; ++k) {
                acc = fmaf(As[mi][k], Bs[ni][k], acc);
            }
            G[((int64_t)b * (int64_t)Cm + m) * (int64_t)Cn + n] = acc;
        }
        return;
    }

    // Generic tiled reduction over k in chunks of KS
    for (int k0 = 0; k0 < S; k0 += KS) {
        // Load A tile
        int totalA = TM * KS;
        for (int idx = tid; idx < totalA; idx += blockDim.x) {
            int lm = idx / KS;
            int lk = idx - lm * KS;
            int gm = m0 + lm;
            int gk = k0 + lk;
            As[lm][lk] = (gm < Cm && gk < S) ? Ap_base[(int64_t)gm * (int64_t)S + gk] : 0.0f;
        }
        // Load B tile (note: we need attB[n,k] contiguous in k)
        int totalB = TN * KS;
        for (int idx = tid; idx < totalB; idx += blockDim.x) {
            int ln = idx / KS;
            int lk = idx - ln * KS;
            int gn = n0 + ln;
            int gk = k0 + lk;
            Bs[ln][lk] = (gn < Cn && gk < S) ? Bp_base[(int64_t)gn * (int64_t)S + gk] : 0.0f;
        }
        __syncthreads();

        if (m < Cm && n < Cn) {
            #pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                acc = fmaf(As[mi][kk], Bs[ni][kk], acc);
            }
        }
        __syncthreads();
    }

    if (m < Cm && n < Cn) {
        G[((int64_t)b * (int64_t)Cm + m) * (int64_t)Cn + n] = acc;
    }
}

// ------------------------
// Z = G @ attV
// G:    [B, Cm, Cn]
// attV: [B, Cn, S]
// Z:    [B, Cm, S]
// Tiled kernel: block computes tile (TM x TS) for one batch b, reduce over n.
// ------------------------
template<int TM, int TS, int KN>
__global__ void gemm_G_attV_kernel(
    const float* __restrict__ G,     // [B,Cm,Cn]
    const float* __restrict__ attV,  // [B,Cn,S]
    float* __restrict__ Z,           // [B,Cm,S]
    int B, int Cm, int Cn, int S
) {
    int b = (int)blockIdx.z;
    int m0 = (int)blockIdx.y * TM;
    int s0 = (int)blockIdx.x * TS;

    int tid = (int)threadIdx.x; // 0..255
    int mi = tid / TS;          // 0..TM-1
    int si = tid - mi * TS;     // 0..TS-1

    int m = m0 + mi;
    int s = s0 + si;

    if (b >= B) return;

    const float* Gp_base = G + ((int64_t)b * (int64_t)Cm) * (int64_t)Cn;
    const float* Vp_base = attV + ((int64_t)b * (int64_t)Cn) * (int64_t)S;

    __shared__ float Gs[TM][KN];
    __shared__ float Vs[KN][TS];

    float acc = 0.0f;

    for (int n0 = 0; n0 < Cn; n0 += KN) {
        // Load G tile
        int totalG = TM * KN;
        for (int idx = tid; idx < totalG; idx += blockDim.x) {
            int lm = idx / KN;
            int ln = idx - lm * KN;
            int gm = m0 + lm;
            int gn = n0 + ln;
            Gs[lm][ln] = (gm < Cm && gn < Cn) ? Gp_base[(int64_t)gm * (int64_t)Cn + gn] : 0.0f;
        }
        // Load V tile
        int totalV = KN * TS;
        for (int idx = tid; idx < totalV; idx += blockDim.x) {
            int ln = idx / TS;
            int ls = idx - ln * TS;
            int gn = n0 + ln;
            int gs = s0 + ls;
            Vs[ln][ls] = (gn < Cn && gs < S) ? Vp_base[(int64_t)gn * (int64_t)S + gs] : 0.0f;
        }
        __syncthreads();

        if (m < Cm && s < S) {
            #pragma unroll
            for (int nn = 0; nn < KN; ++nn) {
                acc = fmaf(Gs[mi][nn], Vs[nn][si], acc);
            }
        }
        __syncthreads();
    }

    if (m < Cm && s < S) {
        Z[((int64_t)b * (int64_t)Cm + m) * (int64_t)S + s] = acc;
    }
}

torch::Tensor double_attention_fused_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor V) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && V.is_cuda(), "A,B,V must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && V.dtype() == torch::kFloat32, "A,B,V must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && V.is_contiguous(), "A,B,V must be contiguous");
    TORCH_CHECK(A.dim() == 4 && B.dim() == 4 && V.dim() == 4, "A,B,V must be 4D");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == V.size(0), "batch mismatch");
    TORCH_CHECK(A.size(2) == B.size(2) && A.size(2) == V.size(2), "H mismatch");
    TORCH_CHECK(A.size(3) == B.size(3) && A.size(3) == V.size(3), "W mismatch");
    TORCH_CHECK(B.size(1) == V.size(1), "Cn mismatch");

    int Bsz = (int)A.size(0);
    int Cm  = (int)A.size(1);
    int Cn  = (int)B.size(1);
    int H   = (int)A.size(2);
    int W   = (int)A.size(3);
    int S   = H * W;

    auto A3 = A.view({Bsz, Cm, S});
    auto B3 = B.view({Bsz, Cn, S});
    auto V3 = V.view({Bsz, Cn, S});

    auto attB = torch::empty_like(B3);
    auto attV = torch::empty_like(V3);
    auto G    = torch::empty({Bsz, Cm, Cn}, A.options());
    auto Z3   = torch::empty({Bsz, Cm, S}, A.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Softmax
    int sm_blocks = Bsz * Cn;
    if (S <= 64) {
        dim3 block(32, 1, 1);
        dim3 grid((unsigned int)sm_blocks, 1, 1);
        softmax_warp_lastdim_kernel<<<grid, block, 0, stream>>>(B3.data_ptr<float>(), attB.data_ptr<float>(), Bsz, Cn, S);
        softmax_warp_lastdim_kernel<<<grid, block, 0, stream>>>(V3.data_ptr<float>(), attV.data_ptr<float>(), Bsz, Cn, S);
    } else {
        dim3 block(256, 1, 1);
        dim3 grid((unsigned int)sm_blocks, 1, 1);
        softmax_block_lastdim_kernel<<<grid, block, 0, stream>>>(B3.data_ptr<float>(), attB.data_ptr<float>(), Bsz, Cn, S);
        softmax_block_lastdim_kernel<<<grid, block, 0, stream>>>(V3.data_ptr<float>(), attV.data_ptr<float>(), Bsz, Cn, S);
    }

    // G = A @ attB^T
    // Tile sizes chosen to keep blocks=256 threads and shared memory modest.
    // TM=16, TN=16, KS=49 for S==49 (fast path) else KS=32.
    {
        const int TM = 16;
        const int TN = 16;
        dim3 block(256, 1, 1);
        dim3 grid((unsigned int)div_up_int(Cn, TN),
                  (unsigned int)div_up_int(Cm, TM),
                  (unsigned int)Bsz);

        if (S == 49) {
            gemm_A_attBt_kernel<TM, TN, 49, true><<<grid, block, 0, stream>>>(
                A3.data_ptr<float>(), attB.data_ptr<float>(), G.data_ptr<float>(),
                Bsz, Cm, Cn, S
            );
        } else {
            gemm_A_attBt_kernel<TM, TN, 32, false><<<grid, block, 0, stream>>>(
                A3.data_ptr<float>(), attB.data_ptr<float>(), G.data_ptr<float>(),
                Bsz, Cm, Cn, S
            );
        }
    }

    // Z = G @ attV
    // Tile TM=16, TS=16, KN=32 => 256 threads, shared mem ~ (16*32 + 32*16)=1024 floats = 4KB.
    {
        const int TM = 16;
        const int TS = 16;
        const int KN = 32;
        dim3 block(256, 1, 1);
        dim3 grid((unsigned int)div_up_int(S, TS),
                  (unsigned int)div_up_int(Cm, TM),
                  (unsigned int)Bsz);

        gemm_G_attV_kernel<TM, TS, KN><<<grid, block, 0, stream>>>(
            G.data_ptr<float>(), attV.data_ptr<float>(), Z3.data_ptr<float>(),
            Bsz, Cm, Cn, S
        );
    }

    return Z3.view({Bsz, Cm, H, W});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_attention_fused_cuda", &double_attention_fused_cuda, "double attention fused forward (CUDA, optimized tiled)");
}
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_double_attention_opt_tiled",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Double Attention (A2-Net) with optimized CUDA core for the attention part.
    Convs remain in PyTorch; attention core is computed by custom CUDA kernels.
    """
    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        if x.is_cuda and x.dtype == torch.float32:
            Z = self.custom_ops.double_attention_fused_cuda(
                A.contiguous(), B.contiguous(), V.contiguous()
            )
        else:
            tmpA = A.view(b, self.c_m, -1)
            attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)
            attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)
            global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))
            tmpZ = global_descriptors.matmul(attention_vectors)
            Z = tmpZ.view(b, self.c_m, h, w)

        if self.reconstruct:
            Z = self.conv_reconstruct(Z)
        return Z


# Input helpers (kept compatible with original signature)
batch_size = 128
in_channels = 512
height = 7
width = 7
c_m = 128
c_n = 128
reconstruct = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512, 128, 128, True]