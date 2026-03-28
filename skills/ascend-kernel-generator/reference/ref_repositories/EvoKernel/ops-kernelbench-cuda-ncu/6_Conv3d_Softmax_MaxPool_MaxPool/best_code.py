import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused: conv3d (stride=1,pad=0) + softmax(dim=1) + maxpool3d(P) + maxpool3d(P)
# Fast path specialized for:
#   - float32 contiguous NCDHW
#   - Cin=3, K=3, P=2
#   - Cout<=32, bias present
# Optimization vs current baseline fast path:
#   - Stage the required input tile (for each output voxel) into shared memory ONCE per voxel
#     and reuse across all Cout lanes => removes massive redundant global loads of x.
#   - Multi-warp per block + grid-stride for better occupancy/latency hiding.
#   - No unsafe vector loads; alignment-safe scalar staging.

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cmath>
#include <limits>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

static inline __device__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_max(float v, unsigned mask=0xffffffffu) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(mask, v, off));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask=0xffffffffu) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
    return v;
}

// ---------------- Generic fallback kernel (correct for general params; slow) ----------------
__global__ void conv3d_softmax_pool2_fwd_kernel_generic(
    const float* __restrict__ x,      // [N, Cin, D, H, W]
    const float* __restrict__ w,      // [Cout, Cin, K, K, K]
    const float* __restrict__ b,      // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, D2, H2, W2]
    int N, int Cin, int D, int H, int W,
    int Cout, int K,
    int P,
    int D0, int H0, int W0,
    int D1, int H1, int W1,
    int D2, int H2, int W2
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long long total = (long long)N * (long long)Cout * (long long)D2 * (long long)H2 * (long long)W2;
    if ((long long)idx >= total) return;

    int ow2 = idx % W2; idx /= W2;
    int oh2 = idx % H2; idx /= H2;
    int od2 = idx % D2; idx /= D2;
    int co  = idx % Cout; idx /= Cout;
    int n   = idx;

    int d1_base = od2 * P;
    int h1_base = oh2 * P;
    int w1_base = ow2 * P;

    float out_max = -INFINITY;

    const int x_HW = H * W;
    const int x_DHW = D * x_HW;
    const int w_KKK = K * K * K;
    const int w_Cin_KKK = Cin * w_KKK;

    for (int pd2 = 0; pd2 < P; ++pd2) {
        int d1 = d1_base + pd2;
        if ((unsigned)d1 >= (unsigned)D1) continue;
        for (int ph2 = 0; ph2 < P; ++ph2) {
            int h1 = h1_base + ph2;
            if ((unsigned)h1 >= (unsigned)H1) continue;
            for (int pw2 = 0; pw2 < P; ++pw2) {
                int w1 = w1_base + pw2;
                if ((unsigned)w1 >= (unsigned)W1) continue;

                int d0_base = d1 * P;
                int h0_base = h1 * P;
                int w0_base = w1 * P;

                for (int pd1 = 0; pd1 < P; ++pd1) {
                    int d0 = d0_base + pd1;
                    if ((unsigned)d0 >= (unsigned)D0) continue;
                    for (int ph1 = 0; ph1 < P; ++ph1) {
                        int h0 = h0_base + ph1;
                        if ((unsigned)h0 >= (unsigned)H0) continue;
                        for (int pw1 = 0; pw1 < P; ++pw1) {
                            int w0c = w0_base + pw1;
                            if ((unsigned)w0c >= (unsigned)W0) continue;

                            float maxv = -INFINITY;

                            for (int c = 0; c < Cout; ++c) {
                                float acc = (b != nullptr) ? ldg_f(b + c) : 0.0f;
                                int x_n_base = n * Cin * x_DHW;
                                int w_c_base = c * w_Cin_KKK;

                                for (int ci = 0; ci < Cin; ++ci) {
                                    int x_nc_base = x_n_base + ci * x_DHW;
                                    int w_ci_base = w_c_base + ci * w_KKK;
                                    for (int kd = 0; kd < K; ++kd) {
                                        int id = d0 + kd;
                                        int x_d_base = x_nc_base + id * x_HW;
                                        int w_kd_base = w_ci_base + kd * K * K;
                                        for (int kh = 0; kh < K; ++kh) {
                                            int ih = h0 + kh;
                                            int x_h_base = x_d_base + ih * W;
                                            int w_kh_base = w_kd_base + kh * K;
#pragma unroll 1
                                            for (int kw = 0; kw < K; ++kw) {
                                                int iw = w0c + kw;
                                                float xv = ldg_f(x + x_h_base + iw);
                                                float wv = ldg_f(w + w_kh_base + kw);
                                                acc = fmaf(xv, wv, acc);
                                            }
                                        }
                                    }
                                }
                                if (acc > maxv) maxv = acc;
                            }

                            float sum = 0.0f;
                            for (int c = 0; c < Cout; ++c) {
                                float acc = (b != nullptr) ? ldg_f(b + c) : 0.0f;
                                int x_n_base = n * Cin * x_DHW;
                                int w_c_base = c * w_Cin_KKK;

                                for (int ci = 0; ci < Cin; ++ci) {
                                    int x_nc_base = x_n_base + ci * x_DHW;
                                    int w_ci_base = w_c_base + ci * w_KKK;
                                    for (int kd = 0; kd < K; ++kd) {
                                        int id = d0 + kd;
                                        int x_d_base = x_nc_base + id * x_HW;
                                        int w_kd_base = w_ci_base + kd * K * K;
                                        for (int kh = 0; kh < K; ++kh) {
                                            int ih = h0 + kh;
                                            int x_h_base = x_d_base + ih * W;
                                            int w_kh_base = w_kd_base + kh * K;
#pragma unroll 1
                                            for (int kw = 0; kw < K; ++kw) {
                                                int iw = w0c + kw;
                                                float xv = ldg_f(x + x_h_base + iw);
                                                float wv = ldg_f(w + w_kh_base + kw);
                                                acc = fmaf(xv, wv, acc);
                                            }
                                        }
                                    }
                                }
                                sum += __expf(acc - maxv);
                            }

                            float acc = (b != nullptr) ? ldg_f(b + co) : 0.0f;
                            int x_n_base = n * Cin * x_DHW;
                            int w_c_base = co * w_Cin_KKK;

                            for (int ci = 0; ci < Cin; ++ci) {
                                int x_nc_base = x_n_base + ci * x_DHW;
                                int w_ci_base = w_c_base + ci * w_KKK;
                                for (int kd = 0; kd < K; ++kd) {
                                    int id = d0 + kd;
                                    int x_d_base = x_nc_base + id * x_HW;
                                    int w_kd_base = w_ci_base + kd * K * K;
                                    for (int kh = 0; kh < K; ++kh) {
                                        int ih = h0 + kh;
                                        int x_h_base = x_d_base + ih * W;
                                        int w_kh_base = w_kd_base + kh * K;
#pragma unroll 1
                                        for (int kw = 0; kw < K; ++kw) {
                                            int iw = w0c + kw;
                                            float xv = ldg_f(x + x_h_base + iw);
                                            float wv = ldg_f(w + w_kh_base + kw);
                                            acc = fmaf(xv, wv, acc);
                                        }
                                    }
                                }
                            }
                            float soft = __expf(acc - maxv) / sum;
                            out_max = fmaxf(out_max, soft);
                        }
                    }
                }
            }
        }
    }

    long long y_idx = (((((long long)n * Cout + co) * D2 + od2) * H2 + oh2) * W2 + ow2);
    y[y_idx] = out_max;
}

// ---------------- Fast path: Cin=3, K=3, P=2, Cout<=32, bias required ----------------
//
// One warp computes one output voxel (n,od2,oh2,ow2); lane==channel.
// Stage the full required input tile into shared memory:
//   We need x indices covering conv starts w0c in {w1p*2 + 0..1} where w1p in {ow2*2 + 0..1} => w0c in [4*ow2 .. 4*ow2+3] (size 4)
//   Each conv uses kw=0..2 => x width needed: [w0c .. w0c+2] => overall width range size 6.
// Similarly for H and D: overall needed ranges size 6.
// So tile is 6x6x6xCin=3 => 648 floats (~2.6KB) per warp.
// This is reused for all 64 sites and all channels.
//
// Shared memory per block = warps_per_block * 648 floats.
template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK*32, 2) void conv3d_softmax_pool2_fwd_fast_smemtile_3_3_2(
    const float* __restrict__ x,  // [N,3,D,H,W]
    const float* __restrict__ w,  // [Cout,3,3,3,3]
    const float* __restrict__ b,  // [Cout]
    float* __restrict__ y,        // [N,Cout,D2,H2,W2]
    int N, int D, int H, int W,
    int Cout,
    int D0, int H0, int W0,
    int D1, int H1, int W1,
    int D2, int H2, int W2
) {
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    unsigned mask = 0xffffffffu;

    long long total_vox = (long long)N * (long long)D2 * (long long)H2 * (long long)W2;
    long long warp_global = (long long)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    const int x_HW = H * W;
    const int x_DHW = D * x_HW;

    // shared tile base
    extern __shared__ float smem[];
    // per-warp tile: [3][6][6][6] flattened
    float* tile = smem + warp_in_block * (3 * 6 * 6 * 6);

    for (; warp_global < total_vox; warp_global += (long long)gridDim.x * WARPS_PER_BLOCK) {
        long long t = warp_global;
        int ow2 = (int)(t % W2); t /= W2;
        int oh2 = (int)(t % H2); t /= H2;
        int od2 = (int)(t % D2); t /= D2;
        int n   = (int)t;

        // Determine tile origin in input x:
        // Conv start w0c ranges [4*ow2 .. 4*ow2+3], kw adds [0..2] => xw range [4*ow2 .. 4*ow2+5]
        // Likewise for h,d.
        int base_w = (ow2 << 2);
        int base_h = (oh2 << 2);
        int base_d = (od2 << 2);

        // Stage tile: total 648 floats. Use all threads in warp to cooperatively load.
        // Each lane loads multiple elements strided by 32.
        int n_base = n * 3 * x_DHW;

        for (int idx = lane; idx < 3 * 6 * 6 * 6; idx += 32) {
            int tmp = idx;
            int w6 = tmp % 6; tmp /= 6;
            int h6 = tmp % 6; tmp /= 6;
            int d6 = tmp % 6; tmp /= 6;
            int ci = tmp; // 0..2

            int gd = base_d + d6;
            int gh = base_h + h6;
            int gw = base_w + w6;

            float v = 0.0f;
            // All sites needed by conv are guaranteed within input bounds for valid conv outputs,
            // but for safety on unusual shapes, guard.
            if ((unsigned)gd < (unsigned)D && (unsigned)gh < (unsigned)H && (unsigned)gw < (unsigned)W) {
                long long x_idx = (long long)n_base + (long long)ci * x_DHW + (long long)gd * x_HW + (long long)gh * W + gw;
                v = ldg_f(x + x_idx);
            }
            tile[idx] = v;
        }

        __syncwarp(mask);

        int c = lane;
        float out_max = -INFINITY;

        // pool window mapping
        int d1_base = od2 << 1;
        int h1_base = oh2 << 1;
        int w1_base = ow2 << 1;

        // Iterate 64 sites; compute conv logits using shared tile.
#pragma unroll
        for (int s = 0; s < 64; ++s) {
            int tmp = s;
            int pw1 = tmp & 1; tmp >>= 1;
            int ph1 = tmp & 1; tmp >>= 1;
            int pd1 = tmp & 1; tmp >>= 1;
            int pw2 = tmp & 1; tmp >>= 1;
            int ph2 = tmp & 1; tmp >>= 1;
            int pd2 = tmp & 1;

            int d1 = d1_base + pd2;
            int h1 = h1_base + ph2;
            int w1p = w1_base + pw2;

            // For output sizes from formula, these should be valid; still guard for safety.
            if ((unsigned)d1 >= (unsigned)D1 || (unsigned)h1 >= (unsigned)H1 || (unsigned)w1p >= (unsigned)W1) {
                // invalid site => contribute nothing
                float v = (c < Cout) ? -INFINITY : -INFINITY;
                float vmax = warp_reduce_max(v, mask);
                vmax = __shfl_sync(mask, vmax, 0);
                float ev = 0.0f;
                float sumv = warp_reduce_sum(ev, mask);
                sumv = __shfl_sync(mask, sumv, 0);
                float soft = -INFINITY;
                out_max = fmaxf(out_max, soft);
                continue;
            }

            int d0 = (d1 << 1) + pd1;
            int h0 = (h1 << 1) + ph1;
            int w0c = (w1p << 1) + pw1;

            if ((unsigned)d0 >= (unsigned)D0 || (unsigned)h0 >= (unsigned)H0 || (unsigned)w0c >= (unsigned)W0) {
                float v = (c < Cout) ? -INFINITY : -INFINITY;
                float vmax = warp_reduce_max(v, mask);
                vmax = __shfl_sync(mask, vmax, 0);
                float ev = 0.0f;
                float sumv = warp_reduce_sum(ev, mask);
                sumv = __shfl_sync(mask, sumv, 0);
                float soft = -INFINITY;
                out_max = fmaxf(out_max, soft);
                continue;
            }

            float v = -INFINITY;
            if (c < Cout) {
                float acc = ldg_f(b + c);
                int w_c_base = c * 81;

                // Convert global (d0,h0,w0c) to tile-local offsets:
                int td0 = d0 - base_d;  // in 0..3
                int th0 = h0 - base_h;
                int tw0 = w0c - base_w;

                // Convolution: sum_{ci,kd,kh,kw} x * w
#pragma unroll
                for (int ci = 0; ci < 3; ++ci) {
                    int w_ci_base = w_c_base + ci * 27;
                    int tile_ci_base = ci * (6 * 6 * 6);

#pragma unroll
                    for (int kd = 0; kd < 3; ++kd) {
                        int w_kd_base = w_ci_base + kd * 9;
                        int tdz = td0 + kd; // 0..5
#pragma unroll
                        for (int kh = 0; kh < 3; ++kh) {
                            int w_kh_base = w_kd_base + kh * 3;
                            int thy = th0 + kh; // 0..5
                            int tile_base = tile_ci_base + (tdz * 6 + thy) * 6 + tw0; // points to kw=0
                            // kw=0..2 contiguous
                            float x0 = tile[tile_base + 0];
                            float x1 = tile[tile_base + 1];
                            float x2 = tile[tile_base + 2];

                            float w0 = ldg_f(w + (w_kh_base + 0));
                            float w1 = ldg_f(w + (w_kh_base + 1));
                            float w2 = ldg_f(w + (w_kh_base + 2));
                            acc = fmaf(x0, w0, acc);
                            acc = fmaf(x1, w1, acc);
                            acc = fmaf(x2, w2, acc);
                        }
                    }
                }
                v = acc;
            }

            // Warp softmax across channels for this site
            float vmax = warp_reduce_max(v, mask);
            vmax = __shfl_sync(mask, vmax, 0);
            float ev = (c < Cout) ? __expf(v - vmax) : 0.0f;
            float sumv = warp_reduce_sum(ev, mask);
            sumv = __shfl_sync(mask, sumv, 0);

            float soft = (c < Cout && sumv > 0.0f) ? (ev / sumv) : -INFINITY;
            out_max = fmaxf(out_max, soft);
        }

        if (c < Cout) {
            long long y_idx = (((((long long)n * Cout + c) * D2 + od2) * H2 + oh2) * W2 + ow2);
            y[y_idx] = out_max;
        }
    }
}

torch::Tensor conv3d_softmax_pool2_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    int64_t pool_kernel
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be [Cout,Cin,K,K,K]");
    TORCH_CHECK(w.size(2) == w.size(3) && w.size(3) == w.size(4), "kernel must be cubic");
    TORCH_CHECK(x.size(1) == w.size(1), "Cin mismatch");
    TORCH_CHECK(pool_kernel > 0, "pool_kernel must be > 0");

    bool has_bias = (b.defined() && b.numel() > 0);
    TORCH_CHECK(has_bias, "bias must be provided for this fused op");
    TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(b.dim() == 1, "bias must be 1D [Cout]");
    TORCH_CHECK(b.size(0) == w.size(0), "bias size must equal Cout");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int D   = (int)x.size(2);
    const int H   = (int)x.size(3);
    const int W   = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int K    = (int)w.size(2);
    const int P    = (int)pool_kernel;

    // conv stride=1, pad=0
    const int D0 = D - K + 1;
    const int H0 = H - K + 1;
    const int W0 = W - K + 1;
    TORCH_CHECK(D0 > 0 && H0 > 0 && W0 > 0, "Conv output size must be positive");

    // pool1
    const int D1 = (D0 - P) / P + 1;
    const int H1 = (H0 - P) / P + 1;
    const int W1 = (W0 - P) / P + 1;
    TORCH_CHECK(D1 > 0 && H1 > 0 && W1 > 0, "Pool1 output size must be positive");

    // pool2
    const int D2 = (D1 - P) / P + 1;
    const int H2 = (H1 - P) / P + 1;
    const int W2 = (W1 - P) / P + 1;
    TORCH_CHECK(D2 > 0 && H2 > 0 && W2 > 0, "Pool2 output size must be positive");

    auto y = torch::empty({N, Cout, D2, H2, W2}, x.options());

    // Fast path for benchmark-like config
    bool fast = (Cin == 3) && (K == 3) && (P == 2) && (Cout > 0) && (Cout <= 32);

    if (fast) {
        // Choose warps per block; tradeoff occupancy vs smem.
        // smem per warp: 3*6*6*6 floats = 648 floats = 2592 bytes.
        // WARPS=4 => ~10.1KB; WARPS=8 => ~20.2KB.
        constexpr int WARPS = 8;
        constexpr int THREADS = WARPS * 32;

        long long total_vox = (long long)N * (long long)D2 * (long long)H2 * (long long)W2;
        long long blocks_ll = (total_vox + WARPS - 1) / WARPS;

        // Use a reasonable grid cap.
        int blocks = (int)blocks_ll;
        if (blocks > 65535) blocks = 65535;
        if (blocks < 1) blocks = 1;

        size_t shmem = (size_t)WARPS * (size_t)(3 * 6 * 6 * 6) * sizeof(float);

        conv3d_softmax_pool2_fwd_fast_smemtile_3_3_2<WARPS><<<blocks, THREADS, shmem, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, D, H, W,
            Cout,
            D0, H0, W0,
            D1, H1, W1,
            D2, H2, W2
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }

    // Generic fallback
    const float* bptr = (const float*)b.data_ptr<float>();
    long long total = (long long)N * (long long)Cout * (long long)D2 * (long long)H2 * (long long)W2;
    const int threads = 128;
    const int blocks = (int)((total + threads - 1) / threads);

    conv3d_softmax_pool2_fwd_kernel_generic<<<blocks, threads, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        bptr,
        (float*)y.data_ptr<float>(),
        N, Cin, D, H, W,
        Cout, K,
        P,
        D0, H0, W0,
        D1, H1, W1,
        D2, H2, W2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv3d_softmax_pool2_forward_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t pool_kernel);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_softmax_pool2_smemtile_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_softmax_pool2_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Fused replacement for:
      x = Conv3d(x)
      x = softmax(x, dim=1)
      x = MaxPool3d(P)(x)
      x = MaxPool3d(P)(x)

    Fast path:
      - float32 CUDA contiguous NCDHW
      - Conv3d: stride=1, padding=0, dilation=1, groups=1
      - Cin=3, K=3, P=2
      - Cout<=32, bias present
    Generic fallback supports other cubic K / P but is slower.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        if not isinstance(kernel_size, int):
            raise ValueError("kernel_size must be int (cubic kernel).")
        if not isinstance(pool_kernel_size, int):
            raise ValueError("pool_kernel_size must be int.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.pool_kernel_size = int(pool_kernel_size)

        w = torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        b = torch.empty(self.out_channels)
        fan_in = self.in_channels * self.kernel_size * self.kernel_size * self.kernel_size
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(b, -bound, bound)
        self.bias = nn.Parameter(b)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        b = self.bias
        if not w.is_cuda:
            w = w.cuda()
        if not b.is_cuda:
            b = b.cuda()
        if w.dtype != torch.float32:
            w = w.float()
        if b.dtype != torch.float32:
            b = b.float()
        if not w.is_contiguous():
            w = w.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()

        return self.custom_ops_lib.conv3d_softmax_pool2_forward_cuda(x, w, b, self.pool_kernel_size)