import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv3d_hard_swish_group_norm_mean (fast-path single kernel + fallback) ---------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// Fixed-shape fast path constants: Cin=3, Cout=16, K=4 => 16*3*4*4*4 = 3072
__constant__ float CW[3072];
__constant__ float CB[16];

__device__ __forceinline__ float hardswish(float x) {
    float t = x + 3.0f;
    t = fminf(fmaxf(t, 0.0f), 6.0f);
    return x * (t * (1.0f / 6.0f));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

template<int Kd, int Kh, int Kw, bool UseConstWB>
__device__ __forceinline__ float conv3d_cin3_k4(
    const float* __restrict__ x,  // [N,3,D,H,W]
    const float* __restrict__ w,  // [Cout,3,4,4,4] (ignored if UseConstWB)
    const float* __restrict__ b,  // [Cout] or nullptr (ignored if UseConstWB)
    int n, int co, int od, int oh, int ow,
    int D, int H, int W
){
    float acc = 0.0f;
    if constexpr (UseConstWB) {
        acc = CB[co];
    } else {
        acc = b ? __ldg(b + co) : 0.0f;
    }

    const int HW  = H * W;
    const int DHW = D * HW;

    const int x_n_base = n * 3 * DHW;
    const int w_co_base = co * 3 * (Kd * Kh * Kw);

    #pragma unroll
    for (int ci = 0; ci < 3; ++ci) {
        const int x_ci_base = x_n_base + ci * DHW;
        const int w_ci_base = w_co_base + ci * (Kd * Kh * Kw);

        #pragma unroll
        for (int kd = 0; kd < Kd; ++kd) {
            const int id = od + kd;
            const int x_kd_base = x_ci_base + id * HW;
            const int w_kd_base = w_ci_base + kd * (Kh * Kw);

            #pragma unroll
            for (int kh = 0; kh < Kh; ++kh) {
                const int ih = oh + kh;
                const int x_kh_base = x_kd_base + ih * W;
                const int w_kh_base = w_kd_base + kh * Kw;

                #pragma unroll
                for (int kw = 0; kw < Kw; ++kw) {
                    const int iw = ow + kw;
                    float xv = __ldg(x + x_kh_base + iw);
                    float wv = 0.0f;
                    if constexpr (UseConstWB) {
                        wv = CW[w_kh_base + kw];
                    } else {
                        wv = __ldg(w + w_kh_base + kw);
                    }
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }
    return acc;
}

__device__ __forceinline__ float conv3d_generic(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int n, int co, int od, int oh, int ow,
    int Cin, int D, int H, int W,
    int Kd, int Kh, int Kw
){
    float acc = b ? __ldg(b + co) : 0.0f;
    const int HW  = H * W;
    const int DHW = D * HW;

    const int x_n_base = n * Cin * DHW;
    const int w_co_base = co * Cin * (Kd * Kh * Kw);

    for (int ci = 0; ci < Cin; ++ci) {
        const int x_ci_base = x_n_base + ci * DHW;
        const int w_ci_base = w_co_base + ci * (Kd * Kh * Kw);
        for (int kd = 0; kd < Kd; ++kd) {
            const int id = od + kd;
            const int x_kd_base = x_ci_base + id * HW;
            const int w_kd_base = w_ci_base + kd * (Kh * Kw);
            for (int kh = 0; kh < Kh; ++kh) {
                const int ih = oh + kh;
                const int x_kh_base = x_kd_base + ih * W;
                const int w_kh_base = w_kd_base + kh * Kw;
                #pragma unroll 1
                for (int kw = 0; kw < Kw; ++kw) {
                    const int iw = ow + kw;
                    float xv = __ldg(x + x_kh_base + iw);
                    float wv = __ldg(w + w_kh_base + kw);
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }
    return acc;
}

// ---- Fast path: one kernel does everything for Cin=3,Cout=16,K=4, groups=4 ----
// Grid: (N*G) blocks, each block handles one (n,g). 4 warps -> 4 channels.
// No intermediate global buffers; outputs [N,Cout].
__global__ __launch_bounds__(128, 4) void fastpath_single_kernel_3_16_k4_g4(
    const float* __restrict__ x,
    const float* __restrict__ w,   // unused when use_const_wb==1, but kept for API symmetry
    const float* __restrict__ b,   // unused when use_const_wb==1
    const float* __restrict__ gn_w,
    const float* __restrict__ gn_b,
    float* __restrict__ out,
    int N, int D, int H, int W,
    int Dout, int Hout, int Wout,
    float eps,
    int use_const_wb
){
    const int ng = (int)blockIdx.x;
    const int n = ng >> 2;         // G=4
    const int g = ng & 3;
    if (n >= N) return;

    const int Cg = 4;              // Cout/G = 16/4
    const int c0 = g * Cg;
    const int S = Dout * Hout * Wout;
    const int M = Cg * S;
    const float invM = 1.0f / (float)M;
    const float invS = 1.0f / (float)S;

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5; // 0..3
    if (warp >= 4) return;

    const int co = c0 + warp;

    float csum = 0.0f;
    float gsum_local = 0.0f;
    float gsumsq_local = 0.0f;

    // warp-stride loop over spatial elements
    for (int s = lane; s < S; s += 32) {
        int tmp = s;
        const int ow = tmp % Wout; tmp /= Wout;
        const int oh = tmp % Hout; tmp /= Hout;
        const int od = tmp;

        float acc;
        if (use_const_wb) {
            acc = conv3d_cin3_k4<4,4,4,true>(x, nullptr, nullptr, n, co, od, oh, ow, D, H, W);
        } else {
            acc = conv3d_cin3_k4<4,4,4,false>(x, w, b, n, co, od, oh, ow, D, H, W);
        }
        float hv = hardswish(acc);

        csum += hv;
        gsum_local += hv;
        gsumsq_local = fmaf(hv, hv, gsumsq_local);
    }

    // warp reduce each warp's sums
    csum = warp_sum(csum);
    gsum_local = warp_sum(gsum_local);
    gsumsq_local = warp_sum(gsumsq_local);

    __shared__ float sh_gsum[4];
    __shared__ float sh_gsumsq[4];
    __shared__ float sh_csum[4];

    if (lane == 0) {
        sh_gsum[warp] = gsum_local;
        sh_gsumsq[warp] = gsumsq_local;
        sh_csum[warp] = csum;
    }
    __syncthreads();

    float mean, inv_std;
    if (threadIdx.x == 0) {
        float gsum = sh_gsum[0] + sh_gsum[1] + sh_gsum[2] + sh_gsum[3];
        float gsumsq = sh_gsumsq[0] + sh_gsumsq[1] + sh_gsumsq[2] + sh_gsumsq[3];
        mean = gsum * invM;
        float ex2 = gsumsq * invM;
        float var = fmaxf(ex2 - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
        // stash back
        sh_gsum[0] = mean;
        sh_gsumsq[0] = inv_std;
    }
    __syncthreads();
    mean = sh_gsum[0];
    inv_std = sh_gsumsq[0];

    // finalize per-channel output mean after GN affine:
    // out = ((sum_hv - S*mean) * inv_std * gamma)/S + beta
    if (lane == 0) {
        float gamma = __ldg(gn_w + co);
        float beta  = __ldg(gn_b + co);
        float sum_hv = sh_csum[warp];
        float y_mean = ((sum_hv - (float)S * mean) * inv_std * gamma) * invS + beta;
        out[n * 16 + co] = y_mean;
    }
}

// ---------------- Fallback path (baseline three-kernel) ----------------

__global__ __launch_bounds__(256, 2) void partials_noatom_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ c_sum_part,
    float* __restrict__ g_sum_part,
    float* __restrict__ g_sumsq_part,
    int N, int Cin, int D, int H, int W,
    int Cout, int Kd, int Kh, int Kw,
    int G,
    int Dout, int Hout, int Wout,
    int TILES,
    int use_const_wb
){
    const int ng = (int)blockIdx.x;
    const int tile = (int)blockIdx.y;
    const int n = ng / G;
    const int g = ng - n * G;
    if (n >= N) return;
    if (tile >= TILES) return;

    const int Cg = Cout / G;
    const int c0 = g * Cg;
    const int S = Dout * Hout * Wout;

    const int tile_size = (S + TILES - 1) / TILES;
    const int s0 = tile * tile_size;
    const int s1 = min(S, s0 + tile_size);

    float gsum = 0.0f;
    float gsumsq = 0.0f;

    for (int i = 0; i < Cg; ++i) {
        const int co = c0 + i;
        float csum = 0.0f;

        for (int s = s0 + (int)threadIdx.x; s < s1; s += (int)blockDim.x) {
            int tmp = s;
            const int ow = tmp % Wout; tmp /= Wout;
            const int oh = tmp % Hout; tmp /= Hout;
            const int od = tmp;

            float acc;
            if (Cin == 3 && Kd == 4 && Kh == 4 && Kw == 4 && use_const_wb) {
                acc = conv3d_cin3_k4<4,4,4,true>(x, nullptr, nullptr, n, co, od, oh, ow, D, H, W);
            } else if (Cin == 3 && Kd == 4 && Kh == 4 && Kw == 4) {
                acc = conv3d_cin3_k4<4,4,4,false>(x, w, b, n, co, od, oh, ow, D, H, W);
            } else {
                acc = conv3d_generic(x, w, b, n, co, od, oh, ow, Cin, D, H, W, Kd, Kh, Kw);
            }

            float hv = hardswish(acc);
            csum += hv;
            gsum += hv;
            gsumsq = fmaf(hv, hv, gsumsq);
        }

        __shared__ float sh_csum[8];
        int lane = (int)threadIdx.x & 31;
        int warp = (int)threadIdx.x >> 5;
        csum = warp_sum(csum);
        if (lane == 0) sh_csum[warp] = csum;
        __syncthreads();

        if (warp == 0) {
            float bsum = (lane < ((int)blockDim.x >> 5)) ? sh_csum[lane] : 0.0f;
            bsum = warp_sum(bsum);
            if (lane == 0) {
                const int idx = ((n * Cout + co) * TILES + tile);
                c_sum_part[idx] = bsum;
            }
        }
        __syncthreads();
    }

    __shared__ float sh_gsum[8];
    __shared__ float sh_gsumsq[8];
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;

    gsum = warp_sum(gsum);
    gsumsq = warp_sum(gsumsq);
    if (lane == 0) { sh_gsum[warp] = gsum; sh_gsumsq[warp] = gsumsq; }
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < ((int)blockDim.x >> 5)) ? sh_gsum[lane] : 0.0f;
        float bsumsq = (lane < ((int)blockDim.x >> 5)) ? sh_gsumsq[lane] : 0.0f;
        bsum = warp_sum(bsum);
        bsumsq = warp_sum(bsumsq);
        if (lane == 0) {
            g_sum_part[ng * TILES + tile] = bsum;
            g_sumsq_part[ng * TILES + tile] = bsumsq;
        }
    }
}

__global__ __launch_bounds__(256, 2) void reduce_partials_kernel2(
    const float* __restrict__ c_sum_part,
    const float* __restrict__ g_sum_part,
    const float* __restrict__ g_sumsq_part,
    float* __restrict__ c_sum,
    float* __restrict__ g_mean,
    float* __restrict__ g_inv_std,
    int N, int Cout, int G,
    int Dout, int Hout, int Wout,
    int TILES, float eps
){
    const int domain = (int)blockIdx.y;
    const int idx = (int)blockIdx.x;

    if (domain == 0) {
        const int total = N * Cout;
        if (idx >= total) return;

        float sum = 0.0f;
        const float* base = c_sum_part + (idx * TILES);
        for (int t = (int)threadIdx.x; t < TILES; t += (int)blockDim.x) sum += base[t];

        __shared__ float sh_sum[8];
        int lane = (int)threadIdx.x & 31;
        int warp = (int)threadIdx.x >> 5;
        sum = warp_sum(sum);
        if (lane == 0) sh_sum[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float bsum = (lane < ((int)blockDim.x >> 5)) ? sh_sum[lane] : 0.0f;
            bsum = warp_sum(bsum);
            if (lane == 0) c_sum[idx] = bsum;
        }
    } else {
        const int total = N * G;
        if (idx >= total) return;

        float sum = 0.0f;
        float sumsq = 0.0f;
        const float* base1 = g_sum_part + idx * TILES;
        const float* base2 = g_sumsq_part + idx * TILES;
        for (int t = (int)threadIdx.x; t < TILES; t += (int)blockDim.x) {
            sum += base1[t];
            sumsq += base2[t];
        }

        __shared__ float sh_sum[8];
        __shared__ float sh_sumsq[8];
        int lane = (int)threadIdx.x & 31;
        int warp = (int)threadIdx.x >> 5;

        sum = warp_sum(sum);
        sumsq = warp_sum(sumsq);
        if (lane == 0) { sh_sum[warp] = sum; sh_sumsq[warp] = sumsq; }
        __syncthreads();

        if (warp == 0) {
            float bsum = (lane < ((int)blockDim.x >> 5)) ? sh_sum[lane] : 0.0f;
            float bsumsq = (lane < ((int)blockDim.x >> 5)) ? sh_sumsq[lane] : 0.0f;
            bsum = warp_sum(bsum);
            bsumsq = warp_sum(bsumsq);
            if (lane == 0) {
                const int Cg = Cout / G;
                const int S = Dout * Hout * Wout;
                const int M = Cg * S;
                const float invM = 1.0f / (float)M;
                const float mean = bsum * invM;
                float ex2 = bsumsq * invM;
                float var = fmaxf(ex2 - mean * mean, 0.0f);
                g_mean[idx] = mean;
                g_inv_std[idx] = rsqrtf(var + eps);
            }
        }
    }
}

__global__ void finalize_kernel3(
    const float* __restrict__ c_sum,
    const float* __restrict__ gn_w,
    const float* __restrict__ gn_b,
    const float* __restrict__ g_mean,
    const float* __restrict__ g_inv_std,
    float* __restrict__ out,
    int N, int Cout, int G,
    int Dout, int Hout, int Wout
){
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = N * Cout;
    if (idx >= total) return;

    int n = idx / Cout;
    int co = idx - n * Cout;
    int Cg = Cout / G;
    int g = co / Cg;
    int ng = n * G + g;

    float mean = __ldg(g_mean + ng);
    float inv_std = __ldg(g_inv_std + ng);
    float gamma = __ldg(gn_w + co);
    float beta  = __ldg(gn_b + co);

    int S = Dout * Hout * Wout;
    float sum_hv = c_sum[idx];
    float y_mean = ((sum_hv - (float)S * mean) * inv_std * gamma) * (1.0f / (float)S) + beta;
    out[idx] = y_mean;
}

static void maybe_load_const_wb(torch::Tensor w, c10::optional<torch::Tensor> b_opt) {
    if (w.numel() != 3072) return;
    cudaMemcpyToSymbol(CW, w.data_ptr<float>(), 3072 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    if (b_opt.has_value()) {
        auto b = b_opt.value();
        if (b.numel() == 16) {
            cudaMemcpyToSymbol(CB, b.data_ptr<float>(), 16 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        } else {
            float zero[16] = {0};
            cudaMemcpyToSymbol(CB, zero, 16 * sizeof(float), 0, cudaMemcpyHostToDevice);
        }
    } else {
        float zero[16] = {0};
        cudaMemcpyToSymbol(CB, zero, 16 * sizeof(float), 0, cudaMemcpyHostToDevice);
    }
}

std::vector<torch::Tensor> conv3d_hardswish_groupnorm_mean_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    torch::Tensor gn_w,
    torch::Tensor gn_b,
    int64_t num_groups,
    double eps
){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(gn_w.is_cuda() && gn_b.is_cuda(), "GroupNorm params must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(gn_w.dtype() == torch::kFloat32 && gn_b.dtype() == torch::kFloat32, "GroupNorm params must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be [Cout,Cin,Kd,Kh,Kw]");
    TORCH_CHECK(x.size(1) == w.size(1), "Cin mismatch");
    TORCH_CHECK(gn_w.numel() == w.size(0) && gn_b.numel() == w.size(0), "GroupNorm param size mismatch");
    TORCH_CHECK(num_groups > 0, "num_groups must be > 0");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();
    if (!gn_w.is_contiguous()) gn_w = gn_w.contiguous();
    if (!gn_b.is_contiguous()) gn_b = gn_b.contiguous();

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int D   = (int)x.size(2);
    const int H   = (int)x.size(3);
    const int Wd  = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int Kd = (int)w.size(2);
    const int Kh = (int)w.size(3);
    const int Kw = (int)w.size(4);

    TORCH_CHECK((int)num_groups <= Cout && (Cout % (int)num_groups) == 0, "num_groups must divide Cout");

    const int Dout = D - Kd + 1;
    const int Hout = H - Kh + 1;
    const int Wout = Wd - Kw + 1;
    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "Invalid output size");

    const float* b_ptr = nullptr;
    torch::Tensor b;
    if (b_opt.has_value()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.numel() == Cout, "bias size mismatch");
        if (!b.is_contiguous()) b = b.contiguous();
        b_ptr = (const float*)b.data_ptr<float>();
    }

    auto opts = x.options();
    auto out = torch::empty({N, Cout}, opts);

    // ---- Fast path trigger ----
    const int G = (int)num_groups;
    int use_const_wb = 0;
    const bool is_fixed = (Cin == 3 && Cout == 16 && Kd == 4 && Kh == 4 && Kw == 4 && G == 4);
    if (is_fixed) {
        use_const_wb = 1;
        maybe_load_const_wb(w, b_opt);

        const int threads = 128; // 4 warps
        const int blocks = N * 4; // N*G
        fastpath_single_kernel_3_16_k4_g4<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            b_ptr,
            (const float*)gn_w.data_ptr<float>(),
            (const float*)gn_b.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            N, D, H, Wd,
            Dout, Hout, Wout,
            (float)eps,
            use_const_wb
        );
        return {out};
    }

    // ---- Fallback (baseline) ----
    // Tiling over spatial S
    int TILES = 8;
    const int S = Dout * Hout * Wout;
    if (S < 2048) TILES = 4;
    if (S < 512)  TILES = 2;
    if (S < 128)  TILES = 1;

    // still can use constant mem in fallback when fixed weights shape, even if G!=4 etc.
    if (Cin == 3 && Cout == 16 && Kd == 4 && Kh == 4 && Kw == 4) {
        use_const_wb = 1;
        maybe_load_const_wb(w, b_opt);
    }

    auto c_sum_part   = torch::empty({N, Cout, TILES}, opts);
    auto g_sum_part   = torch::empty({N * G, TILES}, opts);
    auto g_sumsq_part = torch::empty({N * G, TILES}, opts);
    auto c_sum        = torch::empty({N, Cout}, opts);
    auto g_mean       = torch::empty({N, G}, opts);
    auto g_inv_std    = torch::empty({N, G}, opts);

    const int threads1 = 256;
    dim3 grid1((unsigned)(N * G), (unsigned)TILES, 1);
    partials_noatom_kernel<<<grid1, threads1>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)w.data_ptr<float>(),
        b_ptr,
        (float*)c_sum_part.data_ptr<float>(),
        (float*)g_sum_part.data_ptr<float>(),
        (float*)g_sumsq_part.data_ptr<float>(),
        N, Cin, D, H, Wd,
        Cout, Kd, Kh, Kw,
        G,
        Dout, Hout, Wout,
        TILES,
        use_const_wb
    );

    const int threads2 = 256;
    const int total_c = N * Cout;
    const int total_g = N * G;
    const int grid2x = (total_c > total_g) ? total_c : total_g;
    dim3 grid2((unsigned)grid2x, 2, 1);
    reduce_partials_kernel2<<<grid2, threads2>>>(
        (const float*)c_sum_part.data_ptr<float>(),
        (const float*)g_sum_part.data_ptr<float>(),
        (const float*)g_sumsq_part.data_ptr<float>(),
        (float*)c_sum.data_ptr<float>(),
        (float*)g_mean.data_ptr<float>(),
        (float*)g_inv_std.data_ptr<float>(),
        N, Cout, G,
        Dout, Hout, Wout,
        TILES, (float)eps
    );

    const int threads3 = 256;
    const int total = N * Cout;
    const int blocks3 = (total + threads3 - 1) / threads3;
    finalize_kernel3<<<blocks3, threads3>>>(
        (const float*)c_sum.data_ptr<float>(),
        (const float*)gn_w.data_ptr<float>(),
        (const float*)gn_b.data_ptr<float>(),
        (const float*)g_mean.data_ptr<float>(),
        (const float*)g_inv_std.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        N, Cout, G, Dout, Hout, Wout
    );

    return {out};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> conv3d_hardswish_groupnorm_mean_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    torch::Tensor gn_w,
    torch::Tensor gn_b,
    int64_t num_groups,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_hswish_gn_mean_fastpath_v2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_hardswish_groupnorm_mean_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized replacement model using CUDA extension.

    Fast path (common benchmark shape):
      - Cin=3, Cout=16, K=4, num_groups=4
      - Single kernel: Conv3d + HardSwish + GroupNorm stats + Mean over spatial (after GN affine)
      - No intermediate global buffers

    Fallback:
      - Uses prior deterministic multi-kernel approach (no atomics)

    Assumptions:
      - Conv3d stride=1, padding=0, dilation=1, groups=1
      - float32 CUDA
      - NCDHW contiguous
      - GroupNorm affine=True
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True, eps=1e-5):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_groups = int(num_groups)
        self.eps = float(eps)

        if isinstance(kernel_size, int):
            kd = kh = kw = int(kernel_size)
        else:
            kd, kh, kw = kernel_size
            kd, kh, kw = int(kd), int(kh), int(kw)
        self.kernel_size = (kd, kh, kw)

        if self.out_channels % self.num_groups != 0:
            raise ValueError("num_groups must divide out_channels")

        w = torch.empty(self.out_channels, self.in_channels, kd, kh, kw, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.empty(self.out_channels, dtype=torch.float32)
            fan_in = self.in_channels * kd * kh * kw
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

        self.gn_weight = nn.Parameter(torch.ones(self.out_channels, dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda:
            w = w.cuda()
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        gn_w = self.gn_weight
        gn_b = self.gn_bias
        if not gn_w.is_cuda:
            gn_w = gn_w.cuda()
        if not gn_b.is_cuda:
            gn_b = gn_b.cuda()
        if gn_w.dtype != torch.float32:
            gn_w = gn_w.float()
        if gn_b.dtype != torch.float32:
            gn_b = gn_b.float()
        if not gn_w.is_contiguous():
            gn_w = gn_w.contiguous()
        if not gn_b.is_contiguous():
            gn_b = gn_b.contiguous()

        b = self.bias
        if b is not None:
            if not b.is_cuda:
                b = b.cuda()
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()

        out_list = self.custom_ops_lib.conv3d_hardswish_groupnorm_mean_forward_cuda(
            x, w, b, gn_w, gn_b, self.num_groups, self.eps
        )
        return out_list[0]