import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.f / (1.f + __expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
    return v;
}

// Welford accumulator
struct WelfordAcc {
    int n;
    float mean;
    float m2;
};

__device__ __forceinline__ WelfordAcc welford_combine(const WelfordAcc& a, const WelfordAcc& b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordAcc o;
    o.n = a.n + b.n;
    float delta = b.mean - a.mean;
    o.mean = a.mean + delta * ((float)b.n / (float)o.n);
    o.m2 = a.m2 + b.m2 + delta * delta * ((float)a.n * (float)b.n / (float)o.n);
    return o;
}

__device__ __forceinline__ WelfordAcc warp_welford_reduce(WelfordAcc v) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordAcc b;
        b.n    = __shfl_down_sync(mask, v.n, offset);
        b.mean = __shfl_down_sync(mask, v.mean, offset);
        b.m2   = __shfl_down_sync(mask, v.m2, offset);
        v = welford_combine(v, b);
    }
    return v;
}

// Reduce across warps; supports up to 4 warps (128 threads)
__device__ __forceinline__ WelfordAcc block_welford_reduce_128(WelfordAcc v) {
    __shared__ int   sh_n[4];
    __shared__ float sh_mean[4];
    __shared__ float sh_m2[4];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5; // 0..3

    v = warp_welford_reduce(v);
    if (lane == 0) {
        sh_n[warp]    = v.n;
        sh_mean[warp] = v.mean;
        sh_m2[warp]   = v.m2;
    }
    __syncthreads();

    WelfordAcc out{0, 0.f, 0.f};
    if (warp == 0) {
        WelfordAcc t;
        t.n    = (lane < 4) ? sh_n[lane] : 0;
        t.mean = (lane < 4) ? sh_mean[lane] : 0.f;
        t.m2   = (lane < 4) ? sh_m2[lane] : 0.f;
        t = warp_welford_reduce(t);
        if (lane == 0) out = t;
    }

    __shared__ int   shN0;
    __shared__ float shM0, shS0;
    if (threadIdx.x == 0) {
        shN0 = out.n;
        shM0 = out.mean;
        shS0 = out.m2;
    }
    __syncthreads();

    WelfordAcc r;
    r.n = shN0; r.mean = shM0; r.m2 = shS0;
    return r;
}

extern "C" __global__ void sge_fused_hw49_kernel_v4(
    const float* __restrict__ x,
    const float* __restrict__ weight, // [G]
    const float* __restrict__ bias,   // [G]
    float* __restrict__ out,
    int B, int C, int /*H*/, int /*W*/, int G
) {
    constexpr int HW = 49;
    // blockDim must be 128
    int bg = (int)blockIdx.x;
    int b = bg / G;
    int g = bg - b * G;

    int cpg = C / G;
    int c_start = g * cpg;

    const float* x_bg = x + ((b * C + c_start) * HW);
    float* out_bg = out + ((b * C + c_start) * HW);

    // shared: means[cpg] + gates[49]
    extern __shared__ float shmem[];
    float* sh_means = shmem;
    float* sh_gates = shmem + cpg;

    // 1) per-channel mean via warps (same as baseline-ish)
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5; // 0..3

    for (int ci = warp; ci < cpg; ci += 4) {
        const float* x_c = x_bg + ci * HW;
        float sum = 0.f;
        for (int p = lane; p < HW; p += 32) sum += __ldg(x_c + p);
        sum = warp_sum(sum);
        if (lane == 0) sh_means[ci] = sum * (1.0f / (float)HW);
    }
    __syncthreads();

    // 2) compute s(p) for p in [0,48] using first 64 threads (2 warps) and Welford
    // Map: tid < 64 => computes one p; each does dot across channels
    int tid = (int)threadIdx.x;
    int p = tid; // 0..127
    bool has_p = (p < HW);

    float s = 0.f;
    if (has_p) {
        // common cpg=64 unroll; fallback otherwise
        if (cpg == 64) {
            #pragma unroll 8
            for (int ci0 = 0; ci0 < 64; ci0 += 8) {
                float m0 = sh_means[ci0 + 0];
                float m1 = sh_means[ci0 + 1];
                float m2 = sh_means[ci0 + 2];
                float m3 = sh_means[ci0 + 3];
                float m4 = sh_means[ci0 + 4];
                float m5 = sh_means[ci0 + 5];
                float m6 = sh_means[ci0 + 6];
                float m7 = sh_means[ci0 + 7];
                s += __ldg(x_bg + (ci0 + 0) * HW + p) * m0;
                s += __ldg(x_bg + (ci0 + 1) * HW + p) * m1;
                s += __ldg(x_bg + (ci0 + 2) * HW + p) * m2;
                s += __ldg(x_bg + (ci0 + 3) * HW + p) * m3;
                s += __ldg(x_bg + (ci0 + 4) * HW + p) * m4;
                s += __ldg(x_bg + (ci0 + 5) * HW + p) * m5;
                s += __ldg(x_bg + (ci0 + 6) * HW + p) * m6;
                s += __ldg(x_bg + (ci0 + 7) * HW + p) * m7;
            }
        } else {
            for (int ci = 0; ci < cpg; ++ci) {
                s += __ldg(x_bg + ci * HW + p) * sh_means[ci];
            }
        }
    }

    WelfordAcc acc;
    if (has_p) {
        acc.n = 1;
        acc.mean = s;
        acc.m2 = 0.f;
    } else {
        acc.n = 0;
        acc.mean = 0.f;
        acc.m2 = 0.f;
    }

    WelfordAcc tot = block_welford_reduce_128(acc);
    float mean = tot.mean;
    float var = tot.m2 * (1.0f / (float)HW); // population variance
    float inv_std = rsqrtf(var + 1e-5f);

    float wg = __ldg(weight + g);
    float bgias = __ldg(bias + g);

    // 3) build gates in shared (only p<49 threads write)
    if (has_p) {
        float t = (s - mean) * inv_std;
        sh_gates[p] = sigmoidf_fast(t * wg + bgias);
    }
    __syncthreads();

    // 4) apply in channel-major order for coalesced HW writes:
    // each warp owns channels: ci = warp + k*4, and lanes iterate p in [0,48] stride 32
    for (int ci = warp; ci < cpg; ci += 4) {
        const float* in_c = x_bg + ci * HW;
        float* out_c = out_bg + ci * HW;
        // first 32 positions
        if (lane < 32) {
            int pp = lane;
            if (pp < HW) out_c[pp] = __ldg(in_c + pp) * sh_gates[pp];
            pp = lane + 32;
            if (pp < HW) out_c[pp] = __ldg(in_c + pp) * sh_gates[pp];
        }
    }
}

extern "C" __global__ void sge_fused_generic_kernel_v2(
    const float* __restrict__ x,
    const float* __restrict__ weight, // [G]
    const float* __restrict__ bias,   // [G]
    float* __restrict__ out,
    int B, int C, int H, int W, int G
) {
    int bg = (int)blockIdx.x;
    int b = bg / G;
    int g = bg - b * G;

    int hw = H * W;
    int cpg = C / G;
    int c_start = g * cpg;

    const float* x_bg = x + ((b * C + c_start) * hw);
    float* out_bg = out + ((b * C + c_start) * hw);

    extern __shared__ float sh_means[]; // cpg floats

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    // means
    for (int ci = warp; ci < cpg; ci += nwarps) {
        const float* x_c = x_bg + ci * hw;
        float sum = 0.f;
        for (int p = lane; p < hw; p += 32) sum += __ldg(x_c + p);
        sum = warp_sum(sum);
        if (lane == 0) sh_means[ci] = sum / (float)hw;
    }
    __syncthreads();

    // Welford over s(p)
    WelfordAcc acc{0, 0.f, 0.f};
    for (int p = threadIdx.x; p < hw; p += blockDim.x) {
        float s = 0.f;
        for (int ci = 0; ci < cpg; ++ci) s += __ldg(x_bg + ci * hw + p) * sh_means[ci];
        acc.n += 1;
        float delta = s - acc.mean;
        acc.mean += delta / (float)acc.n;
        float delta2 = s - acc.mean;
        acc.m2 += delta * delta2;
    }

    // reduce across warps (up to 4 warps used by default)
    __shared__ int   sh_n[8];
    __shared__ float sh_mean2[8];
    __shared__ float sh_m2[8];

    WelfordAcc v = warp_welford_reduce(acc);
    if (lane == 0) {
        sh_n[warp] = v.n;
        sh_mean2[warp] = v.mean;
        sh_m2[warp] = v.m2;
    }
    __syncthreads();

    WelfordAcc outw{0,0.f,0.f};
    if (warp == 0) {
        WelfordAcc t;
        if (lane < nwarps) {
            t.n = sh_n[lane];
            t.mean = sh_mean2[lane];
            t.m2 = sh_m2[lane];
        } else {
            t.n = 0; t.mean = 0.f; t.m2 = 0.f;
        }
        t = warp_welford_reduce(t);
        if (lane == 0) outw = t;
    }
    __shared__ float shMeanFinal, shM2Final;
    if (threadIdx.x == 0) { shMeanFinal = outw.mean; shM2Final = outw.m2; }
    __syncthreads();

    float mean = shMeanFinal;
    float var = shM2Final / (float)hw;
    float inv_std = rsqrtf(var + 1e-5f);

    float wg = __ldg(weight + g);
    float bgias = __ldg(bias + g);

    // Safe vectorization: require hw%4==0 and (x_bg + ci*hw) 16B-aligned for all ci.
    // Since hw varies, easiest safe guard is: hw%4==0 and hw*4 is multiple of 16 -> hw%4==0,
    // and x_bg 16B aligned. Then every channel base is aligned too.
    bool vec = ((hw & 3) == 0) && ((((uintptr_t)x_bg) & 15) == 0) && ((((uintptr_t)out_bg) & 15) == 0);

    int hw4 = hw >> 2;
    if (vec) {
        for (int p4 = threadIdx.x; p4 < hw4; p4 += blockDim.x) {
            int p = p4 << 2;

            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int ci = 0; ci < cpg; ++ci) {
                const float* base = x_bg + ci * hw + p; // aligned because x_bg aligned and hw%4==0
                float4 xv = *reinterpret_cast<const float4*>(base);
                float m = sh_means[ci];
                s0 += xv.x * m; s1 += xv.y * m; s2 += xv.z * m; s3 += xv.w * m;
            }

            float t0 = (s0 - mean) * inv_std;
            float t1 = (s1 - mean) * inv_std;
            float t2 = (s2 - mean) * inv_std;
            float t3 = (s3 - mean) * inv_std;

            float g0 = sigmoidf_fast(t0 * wg + bgias);
            float g1 = sigmoidf_fast(t1 * wg + bgias);
            float g2 = sigmoidf_fast(t2 * wg + bgias);
            float g3 = sigmoidf_fast(t3 * wg + bgias);

            for (int ci = 0; ci < cpg; ++ci) {
                const float* in_ptr = x_bg + ci * hw + p;
                float* out_ptr = out_bg + ci * hw + p;
                float4 xv = *reinterpret_cast<const float4*>(in_ptr);
                float4 ov{ xv.x*g0, xv.y*g1, xv.z*g2, xv.w*g3 };
                *reinterpret_cast<float4*>(out_ptr) = ov;
            }
        }
    } else {
        for (int p = threadIdx.x; p < hw; p += blockDim.x) {
            float s = 0.f;
            for (int ci = 0; ci < cpg; ++ci) s += __ldg(x_bg + ci * hw + p) * sh_means[ci];
            float t = (s - mean) * inv_std;
            float gate = sigmoidf_fast(t * wg + bgias);
            for (int ci = 0; ci < cpg; ++ci) {
                int idx = ci * hw + p;
                out_bg[idx] = __ldg(x_bg + idx) * gate;
            }
        }
    }
}

torch::Tensor spatial_group_enhance_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t groups) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA");
    TORCH_CHECK(weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "weight/bias must be float32");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "weight/bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int G = (int)groups;

    TORCH_CHECK(G > 0, "groups must be > 0");
    TORCH_CHECK(C % G == 0, "C must be divisible by groups");
    TORCH_CHECK(weight.numel() == G && bias.numel() == G, "weight/bias must have numel == groups");

    auto out = torch::empty_like(x);

    int cpg = C / G;
    int hw = H * W;

    dim3 blocks(B * G);

    if (hw == 49) {
        int threads = 128;
        // means[cpg] + gates[49]
        size_t shmem = ((size_t)cpg + 49u) * sizeof(float);
        sge_fused_hw49_kernel_v4<<<blocks, threads, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)weight.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, H, W, G
        );
    } else {
        int threads = 128;
        size_t shmem = (size_t)cpg * sizeof(float);
        sge_fused_generic_kernel_v2<<<blocks, threads, shmem>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)weight.data_ptr<float>(),
            (const float*)bias.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            B, C, H, W, G
        );
    }
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor spatial_group_enhance_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t groups);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_sge_fused_hw49_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["spatial_group_enhance_forward_cuda"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.zeros(1, self.groups, 1, 1, device="cuda", dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, self.groups, 1, 1, device="cuda", dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight.contiguous().view(-1)
        b = self.bias.contiguous().view(-1)
        return custom_ops_lib.spatial_group_enhance_forward_cuda(x, w, b, self.groups)