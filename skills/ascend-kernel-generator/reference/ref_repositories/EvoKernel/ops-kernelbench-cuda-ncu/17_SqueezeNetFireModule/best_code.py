import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fire_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f(const float* p) { return *p; }
#endif

// ---------------------------
// Kernel 1: squeeze 1x1 + ReLU
// ---------------------------
__global__ __launch_bounds__(256, 2)
void squeeze1x1_relu_vec4(
    const float* __restrict__ x,     // [N,Cin,H,W]
    const float* __restrict__ w_sq,  // [Csq,Cin] flattened
    const float* __restrict__ b_sq,  // [Csq] or nullptr
    float* __restrict__ s,           // [N,Csq,H,W]
    int N, int Cin, int H, int W,
    int Csq
){
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int W4 = W >> 2;
    int total4 = N * Csq * H * W4;
    int total = N * Csq * H * W;

    const bool aligned = (((uintptr_t)x & 0xF) == 0) && (((uintptr_t)s & 0xF) == 0);

    if (aligned && W4 > 0) {
        if (tid >= total4) return;

        int tmp = tid;
        int w4 = tmp % W4; tmp /= W4;
        int h  = tmp % H;  tmp /= H;
        int csq= tmp % Csq;tmp /= Csq;
        int n  = tmp;

        int w = w4 * 4;

        float acc0 = b_sq ? ldg_f(b_sq + csq) : 0.f;
        float acc1 = acc0, acc2 = acc0, acc3 = acc0;

        int HW = H * W;
        int x_base = ((n * Cin) * H + h) * W + w;
        int w_base = csq * Cin;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            const float4* xp4 = (const float4*)(x + x_base + ic * HW);
            float4 xv4 = __ldg(xp4);
            float wv = ldg_f(w_sq + w_base + ic);
            acc0 = fmaf(xv4.x, wv, acc0);
            acc1 = fmaf(xv4.y, wv, acc1);
            acc2 = fmaf(xv4.z, wv, acc2);
            acc3 = fmaf(xv4.w, wv, acc3);
        }

        float4 out;
        out.x = relu_f(acc0);
        out.y = relu_f(acc1);
        out.z = relu_f(acc2);
        out.w = relu_f(acc3);

        float4* sp4 = (float4*)(s + ((n * Csq + csq) * H + h) * W + w);
        *sp4 = out;
        return;
    }

    if (tid >= total) return;
    int HW = H * W;
    int p = tid % HW;
    int tmp = tid / HW;
    int csq = tmp % Csq;
    int n   = tmp / Csq;
    int h = p / W;
    int w = p - h * W;

    float acc = b_sq ? ldg_f(b_sq + csq) : 0.f;
    int x_base = ((n * Cin) * H + h) * W + w;
    int w_base = csq * Cin;

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        float xv = ldg_f(x + x_base + ic * HW);
        float wv = ldg_f(w_sq + w_base + ic);
        acc = fmaf(xv, wv, acc);
    }
    s[((n * Csq + csq) * H + h) * W + w] = relu_f(acc);
}

// -----------------------------------------
// Baseline Kernel 2 (fallback): expand (1x1 or 3x3) + ReLU + concat
// -----------------------------------------
template<int BW, int BH>
__global__ __launch_bounds__(256, 2)
void expand_fused_nosync(
    const float* __restrict__ s,     // [N,Csq,H,W]
    const float* __restrict__ w_e1,  // [Ce1,Csq]
    const float* __restrict__ b_e1,  // [Ce1] or nullptr
    const float* __restrict__ w_e3,  // [Ce3,Csq,3,3]
    const float* __restrict__ b_e3,  // [Ce3] or nullptr
    float* __restrict__ y,           // [N,CeT,H,W]
    int N, int Csq, int H, int W,
    int Ce1, int Ce3
){
    int CeT = Ce1 + Ce3;

    int tz = (int)blockIdx.z;
    int n  = tz / CeT;
    int oc = tz - n * CeT;
    if (n >= N) return;

    int tile_x0 = (int)blockIdx.x * BW;
    int tile_y0 = (int)blockIdx.y * BH;

    int t = (int)threadIdx.x;
    constexpr int NUM_PIX = BW * BH;

    int py = t / BW;
    int px = t - py * BW;

    int oy = tile_y0 + py;
    int ox = tile_x0 + px;

    bool do_pix = (t < NUM_PIX) && ((unsigned)oy < (unsigned)H) && ((unsigned)ox < (unsigned)W);

    int HW = H * W;
    int s_n_base = n * Csq * HW;
    int y_n_base = n * CeT * HW;

    int out_hw = oy * W + ox;

    if (oc < Ce1) {
        if (!do_pix) return;
        float acc = b_e1 ? ldg_f(b_e1 + oc) : 0.f;
        int w_base = oc * Csq;
        #pragma unroll 1
        for (int csq = 0; csq < Csq; ++csq) {
            float sv = ldg_f(s + s_n_base + csq * HW + out_hw);
            float wv = ldg_f(w_e1 + w_base + csq);
            acc = fmaf(sv, wv, acc);
        }
        y[y_n_base + oc * HW + out_hw] = relu_f(acc);
        return;
    }

    if (!do_pix) return;
    int oc3 = oc - Ce1;

    float acc = b_e3 ? ldg_f(b_e3 + oc3) : 0.f;
    int w_oc_base = oc3 * (Csq * 9);

    #pragma unroll 1
    for (int csq = 0; csq < Csq; ++csq) {
        const float* sp = s + s_n_base + csq * HW;

        float c0 = 0.f, c1 = 0.f, c2 = 0.f;
        if ((unsigned)oy < (unsigned)H) {
            int base = oy * W + ox;
            c1 = ldg_f(sp + base);
            c0 = (ox > 0)     ? ldg_f(sp + base - 1) : 0.f;
            c2 = (ox + 1 < W) ? ldg_f(sp + base + 1) : 0.f;
        }

        float t0=0.f,t1=0.f,t2=0.f,b0=0.f,b1=0.f,b2=0.f;
        if (oy > 0) {
            int base = (oy - 1) * W + ox;
            t1 = ldg_f(sp + base);
            t0 = (ox > 0)     ? ldg_f(sp + base - 1) : 0.f;
            t2 = (ox + 1 < W) ? ldg_f(sp + base + 1) : 0.f;
        }
        if (oy + 1 < H) {
            int base = (oy + 1) * W + ox;
            b1 = ldg_f(sp + base);
            b0 = (ox > 0)     ? ldg_f(sp + base - 1) : 0.f;
            b2 = (ox + 1 < W) ? ldg_f(sp + base + 1) : 0.f;
        }

        const float* wp = w_e3 + w_oc_base + csq * 9;

        acc = fmaf(t0, ldg_f(wp + 0), acc);
        acc = fmaf(t1, ldg_f(wp + 1), acc);
        acc = fmaf(t2, ldg_f(wp + 2), acc);
        acc = fmaf(c0, ldg_f(wp + 3), acc);
        acc = fmaf(c1, ldg_f(wp + 4), acc);
        acc = fmaf(c2, ldg_f(wp + 5), acc);
        acc = fmaf(b0, ldg_f(wp + 6), acc);
        acc = fmaf(b1, ldg_f(wp + 7), acc);
        acc = fmaf(b2, ldg_f(wp + 8), acc);
    }

    y[y_n_base + oc * HW + out_hw] = relu_f(acc);
}

// -----------------------------------------
// Fast stage-2 specialized kernel for (Csq=6, Ce1=64, Ce3=64)
// Single launch, grid.z = N*2:
//   z%2==0 => expand1x1 (write [0..63])
//   z%2==1 => expand3x3 (write [64..127]) with shared-memory halo
// No warp-broadcast of weights (avoids prior correctness failure).
// Uses OC blocking (8) to increase ILP and reduce memory latency impact.
// -----------------------------------------
template<int BW, int BH, int OCB>
__global__ __launch_bounds__(128, 3)
void expand_fast_fire64(
    const float* __restrict__ s,     // [N,6,H,W]
    const float* __restrict__ w_e1,  // [64,6]
    const float* __restrict__ b_e1,  // [64] or nullptr
    const float* __restrict__ w_e3,  // [64,6,3,3]
    const float* __restrict__ b_e3,  // [64] or nullptr
    float* __restrict__ y,           // [N,128,H,W]
    int N, int H, int W
){
    int z = (int)blockIdx.z;
    int n = z >> 1;
    int which = z & 1;
    if (n >= N) return;

    int ox0 = (int)blockIdx.x * BW;
    int oy0 = (int)blockIdx.y * BH;

    int t = (int)threadIdx.x; // 0..127
    int tx = t & (BW - 1);    // BW=16
    int ty = t >> 4;          // 0..7
    int ox = ox0 + tx;
    int oy = oy0 + ty;

    int HW = H * W;
    long s_nb = (long)n * 6L * HW;
    long y_nb = (long)n * 128L * HW;
    int pix = oy * W + ox;

    if (which == 0) {
        if ((unsigned)ox >= (unsigned)W || (unsigned)oy >= (unsigned)H) return;

        float sv[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) sv[c] = ldg_f(s + s_nb + (long)c * HW + pix);

        // OC blocking by 8
        #pragma unroll 1
        for (int oc0 = 0; oc0 < 64; oc0 += OCB) {
            float acc[OCB];
            #pragma unroll
            for (int i = 0; i < OCB; ++i) acc[i] = b_e1 ? ldg_f(b_e1 + (oc0 + i)) : 0.f;

            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float xval = sv[c];
                const float* wp = w_e1 + (long)oc0 * 6L + c;
                #pragma unroll
                for (int i = 0; i < OCB; ++i) {
                    float wv = ldg_f(wp + (long)i * 6L);
                    acc[i] = fmaf(xval, wv, acc[i]);
                }
            }

            #pragma unroll
            for (int i = 0; i < OCB; ++i) {
                y[y_nb + (long)(oc0 + i) * HW + pix] = relu_f(acc[i]);
            }
        }
        return;
    }

    // 3x3 path with shared halo
    constexpr int SHW = BW + 2;
    constexpr int SHH = BH + 2;
    extern __shared__ float sh[]; // 6*SHH*SHW floats
    auto sh_index = [&](int c, int sy, int sx) -> int {
        return (c * SHH + sy) * SHW + sx;
    };

    int halo = SHW * SHH;
    for (int idx = t; idx < 6 * halo; idx += 128) {
        int c = idx / halo;
        int rem = idx - c * halo;
        int sy = rem / SHW;
        int sx = rem - sy * SHW;
        int iy = oy0 + sy - 1;
        int ix = ox0 + sx - 1;
        float v = 0.f;
        if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
            v = ldg_f(s + s_nb + (long)c * HW + (iy * W + ix));
        }
        sh[sh_index(c, sy, sx)] = v;
    }
    __syncthreads();

    if ((unsigned)ox >= (unsigned)W || (unsigned)oy >= (unsigned)H) return;

    int syc = ty + 1;
    int sxc = tx + 1;

    // OC blocking by 8, write to channels [64..127]
    #pragma unroll 1
    for (int oc0 = 0; oc0 < 64; oc0 += OCB) {
        float acc[OCB];
        #pragma unroll
        for (int i = 0; i < OCB; ++i) acc[i] = b_e3 ? ldg_f(b_e3 + (oc0 + i)) : 0.f;

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            float t0 = sh[sh_index(c, syc - 1, sxc - 1)];
            float t1 = sh[sh_index(c, syc - 1, sxc    )];
            float t2 = sh[sh_index(c, syc - 1, sxc + 1)];
            float c0 = sh[sh_index(c, syc    , sxc - 1)];
            float c1 = sh[sh_index(c, syc    , sxc    )];
            float c2 = sh[sh_index(c, syc    , sxc + 1)];
            float b0 = sh[sh_index(c, syc + 1, sxc - 1)];
            float b1 = sh[sh_index(c, syc + 1, sxc    )];
            float b2 = sh[sh_index(c, syc + 1, sxc + 1)];

            float xv[9] = {t0,t1,t2,c0,c1,c2,b0,b1,b2};

            long base = (long)oc0 * 54L + (long)c * 9L; // 6*9=54
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const float* wp = w_e3 + base + k;
                float xval = xv[k];
                #pragma unroll
                for (int i = 0; i < OCB; ++i) {
                    float wv = ldg_f(wp + (long)i * 54L);
                    acc[i] = fmaf(xval, wv, acc[i]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < OCB; ++i) {
            y[y_nb + (long)(64 + oc0 + i) * HW + pix] = relu_f(acc[i]);
        }
    }
}

torch::Tensor fire_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_sq, torch::Tensor b_sq,
    torch::Tensor w_e1, torch::Tensor b_e1,
    torch::Tensor w_e3, torch::Tensor b_e3
) {
    CHECK_CUDA(x); CHECK_CUDA(w_sq); CHECK_CUDA(w_e1); CHECK_CUDA(w_e3);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w_sq); CHECK_CONTIGUOUS(w_e1); CHECK_CONTIGUOUS(w_e3);
    CHECK_FLOAT(x); CHECK_FLOAT(w_sq); CHECK_FLOAT(w_e1); CHECK_FLOAT(w_e3);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w_sq.dim() == 4 && w_sq.size(2) == 1 && w_sq.size(3) == 1, "w_sq must be [Csq,Cin,1,1]");
    TORCH_CHECK(w_e1.dim() == 4 && w_e1.size(2) == 1 && w_e1.size(3) == 1, "w_e1 must be [Ce1,Csq,1,1]");
    TORCH_CHECK(w_e3.dim() == 4 && w_e3.size(2) == 3 && w_e3.size(3) == 3, "w_e3 must be [Ce3,Csq,3,3]");

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int H   = (int)x.size(2);
    int W   = (int)x.size(3);

    int Csq = (int)w_sq.size(0);
    TORCH_CHECK((int)w_sq.size(1) == Cin, "w_sq Cin mismatch");

    int Ce1 = (int)w_e1.size(0);
    TORCH_CHECK((int)w_e1.size(1) == Csq, "w_e1 Csq mismatch");

    int Ce3 = (int)w_e3.size(0);
    TORCH_CHECK((int)w_e3.size(1) == Csq, "w_e3 Csq mismatch");

    const float* b_sq_ptr = nullptr;
    const float* b_e1_ptr = nullptr;
    const float* b_e3_ptr = nullptr;

    if (b_sq.defined() && b_sq.numel() > 0) {
        CHECK_CUDA(b_sq); CHECK_CONTIGUOUS(b_sq); CHECK_FLOAT(b_sq);
        TORCH_CHECK(b_sq.dim() == 1 && (int)b_sq.size(0) == Csq, "b_sq must be [Csq]");
        b_sq_ptr = b_sq.data_ptr<float>();
    }
    if (b_e1.defined() && b_e1.numel() > 0) {
        CHECK_CUDA(b_e1); CHECK_CONTIGUOUS(b_e1); CHECK_FLOAT(b_e1);
        TORCH_CHECK(b_e1.dim() == 1 && (int)b_e1.size(0) == Ce1, "b_e1 must be [Ce1]");
        b_e1_ptr = b_e1.data_ptr<float>();
    }
    if (b_e3.defined() && b_e3.numel() > 0) {
        CHECK_CUDA(b_e3); CHECK_CONTIGUOUS(b_e3); CHECK_FLOAT(b_e3);
        TORCH_CHECK(b_e3.dim() == 1 && (int)b_e3.size(0) == Ce3, "b_e3 must be [Ce3]");
        b_e3_ptr = b_e3.data_ptr<float>();
    }

    int CeT = Ce1 + Ce3;
    auto y = torch::empty({N, CeT, H, W}, x.options());
    auto s = torch::empty({N, Csq, H, W}, x.options());

    // Kernel 1
    {
        int W4 = W >> 2;
        int total4 = N * Csq * H * W4;
        int total  = N * Csq * H * W;
        int launch_elems = (W4 > 0) ? total4 : total;

        dim3 block(256, 1, 1);
        dim3 grid((launch_elems + (int)block.x - 1) / (int)block.x, 1, 1);
        squeeze1x1_relu_vec4<<<grid, block>>>(
            x.data_ptr<float>(),
            w_sq.data_ptr<float>(), b_sq_ptr,
            s.data_ptr<float>(),
            N, Cin, H, W, Csq
        );
    }

    // Kernel 2
    {
        constexpr int BW = 16;
        constexpr int BH = 8;

        bool fast_ok = (Cin == 3 && Csq == 6 && Ce1 == 64 && Ce3 == 64);
        if (fast_ok) {
            dim3 block(128, 1, 1);
            dim3 grid((W + BW - 1) / BW, (H + BH - 1) / BH, N * 2);
            size_t smem = (size_t)6 * (size_t)(BW + 2) * (size_t)(BH + 2) * sizeof(float);

            // OCB=8 chosen to increase ILP while keeping registers reasonable.
            expand_fast_fire64<BW, BH, 8><<<grid, block, smem>>>(
                s.data_ptr<float>(),
                w_e1.data_ptr<float>(), b_e1_ptr,
                w_e3.data_ptr<float>(), b_e3_ptr,
                y.data_ptr<float>(),
                N, H, W
            );
        } else {
            dim3 block(256, 1, 1);
            dim3 grid((W + BW - 1) / BW, (H + BH - 1) / BH, N * CeT);
            expand_fused_nosync<BW, BH><<<grid, block, 0>>>(
                s.data_ptr<float>(),
                w_e1.data_ptr<float>(), b_e1_ptr,
                w_e3.data_ptr<float>(), b_e3_ptr,
                y.data_ptr<float>(),
                N, Csq, H, W, Ce1, Ce3
            );
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fire_forward_cuda: CUDA kernel launch failed: ", cudaGetErrorString(err));
    return y;
}
"""

fire_cpp_source = r"""
#include <torch/extension.h>
torch::Tensor fire_forward_cuda(
    torch::Tensor x,
    torch::Tensor w_sq, torch::Tensor b_sq,
    torch::Tensor w_e1, torch::Tensor b_e1,
    torch::Tensor w_e3, torch::Tensor b_e3
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_fire_twostage_opt_v5_fastz2",
    cpp_sources=fire_cpp_source,
    cuda_sources=fire_cuda_source,
    functions=["fire_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        Cin = in_channels
        Csq = squeeze_channels
        Ce1 = expand1x1_channels
        Ce3 = expand3x3_channels

        self.w_sq = nn.Parameter(torch.empty(Csq, Cin, 1, 1))
        self.b_sq = nn.Parameter(torch.empty(Csq))

        self.w_e1 = nn.Parameter(torch.empty(Ce1, Csq, 1, 1))
        self.b_e1 = nn.Parameter(torch.empty(Ce1))

        self.w_e3 = nn.Parameter(torch.empty(Ce3, Csq, 3, 3))
        self.b_e3 = nn.Parameter(torch.empty(Ce3))

        for w, b in [(self.w_sq, self.b_sq), (self.w_e1, self.b_e1), (self.w_e3, self.b_e3)]:
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
            fan_in = w.size(1) * w.size(2) * w.size(3)
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        x = x.contiguous()
        return custom_ops_lib.fire_forward_cuda(
            x,
            self.w_sq.contiguous(), self.b_sq.contiguous(),
            self.w_e1.contiguous(), self.b_e1.contiguous(),
            self.w_e3.contiguous(), self.b_e3.contiguous(),
        )