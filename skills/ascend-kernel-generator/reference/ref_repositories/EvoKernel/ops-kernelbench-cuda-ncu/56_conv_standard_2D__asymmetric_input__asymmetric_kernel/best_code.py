import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------- CUDA/C++ Extension: conv_standard2d_asymmetric_input_asymmetric_kernel ---------

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

// ---------------- Generic baseline (kept for correctness) ----------------
__global__ void conv2d_forward_nchw_f32_generic(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cout, Cin, Kh, Kw]
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int Hout, int Wout
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % Wout; tmp /= Wout;
    int oh = tmp % Hout; tmp /= Hout;
    int oc = tmp % Cout; tmp /= Cout;
    int n  = tmp;

    float acc = 0.0f;
    int in_h0 = oh * stride_h - pad_h;
    int in_w0 = ow * stride_w - pad_w;

    for (int ic = 0; ic < Cin; ++ic) {
        const float* w_oc_ic = w + ((oc * Cin + ic) * Kh * Kw);
        const float* x_n_ic  = x + ((n * Cin + ic) * Hin * Win);

        #pragma unroll 1
        for (int kh = 0; kh < Kh; ++kh) {
            int ih = in_h0 + kh * dil_h;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            #pragma unroll 1
            for (int kw = 0; kw < Kw; ++kw) {
                int iw = in_w0 + kw * dil_w;
                if ((unsigned)iw >= (unsigned)Win) continue;

                float xv = ldg_f32(x_n_ic + ih * Win + iw);
                float wv = ldg_f32(w_oc_ic + kh * Kw + kw);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[idx] = acc;
}

// ---------------- Specialized fast path: K=5x7, stride=1, dilation=1 ----------------
// Tile: OUT_W=32, OUT_H=4 -> shared input tile size: (4+4) x (32+6) = 8 x 38
// CTA: 128 threads (1D), each thread computes one output (ow,oh) for 4 output channels.
// Output-channel blocking: 4 output channels per block (oc..oc+3).
// IC tiling: IC_TILE=4; stage input tile for 4 ICs and weights for 4 OCs x 4 ICs into shared once per tile.
template<int OUT_W, int OUT_H, int IC_TILE>
__global__ __launch_bounds__(128, 3)
void conv2d_forward_nchw_f32_k5x7_s1d1_oc4_ictile_shmem(
    const float* __restrict__ x,   // [N,Cin,Hin,Win]
    const float* __restrict__ w,   // [Cout,Cin,5,7]
    float* __restrict__ y,         // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout,
    int pad_h, int pad_w,
    int Hout, int Wout,
    int tiles_per_n  // = ceil_div(Cout, 4)
){
    int tile_ow = (int)blockIdx.x;
    int tile_oh = (int)blockIdx.y;
    int gz      = (int)blockIdx.z;

    int n   = gz / tiles_per_n;
    int oc4 = gz - n * tiles_per_n;
    int oc0 = oc4 * 4;

    if (n >= N || oc0 >= Cout) return;

    int tid = (int)threadIdx.x; // 0..127

    // Map tid -> output element inside 32x4 tile: 128 threads cover exactly 128 outputs
    int t_oh = tid / OUT_W;      // 0..3
    int t_ow = tid - t_oh * OUT_W; // 0..31

    int ow = tile_ow * OUT_W + t_ow;
    int oh = tile_oh * OUT_H + t_oh;
    bool vout = (ow < Wout) && (oh < Hout);

    constexpr int SH_H = OUT_H + 4;  // 8
    constexpr int SH_W = OUT_W + 6;  // 38
    constexpr int KHW = 35;

    // Shared layout:
    // sh_in: [IC_TILE][SH_H][SH_W]
    // sh_w : [4][IC_TILE][KHW]
    extern __shared__ float shmem[];
    float* sh_in = shmem;
    float* sh_wt = sh_in + (IC_TILE * SH_H * SH_W);

    // Output accumulators for 4 output channels
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    // Base input coordinates for this tile
    int base_ih0 = tile_oh * OUT_H - pad_h;
    int base_iw0 = tile_ow * OUT_W - pad_w;

    // For each IC tile
    for (int ic0 = 0; ic0 < Cin; ic0 += IC_TILE) {
        int tile_ic = min(IC_TILE, Cin - ic0);

        // ---------------- Load input tile into shared ----------------
        // We load per-ic plane: [SH_H*SH_W] elements. Total per IC_TILE = tile_ic*SH_H*SH_W.
        // Cooperative: tid strides through the linearized region.
        int in_plane = SH_H * SH_W;
        int in_total = tile_ic * in_plane;

        // Try vectorized float4 loads for the "interior" contiguous W segments when aligned.
        // Because base_iw0 can be misaligned, we use safe scalar loads for simplicity in misaligned cases.
        for (int i = tid; i < in_total; i += 128) {
            int ic = i / in_plane;           // 0..tile_ic-1
            int rem = i - ic * in_plane;
            int r = rem / SH_W;              // 0..SH_H-1
            int c = rem - r * SH_W;          // 0..SH_W-1

            int ih = base_ih0 + r;
            int iw = base_iw0 + c;

            float v = 0.f;
            if ((unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
                const float* xptr = x + ((n * Cin + (ic0 + ic)) * Hin + ih) * Win + iw;
                v = ldg_f32(xptr);
            }
            sh_in[(ic * SH_H + r) * SH_W + c] = v;
        }

        // ---------------- Load weights for oc0..oc0+3 and ic0..ic0+tile_ic-1 into shared ----------------
        // Total weights: 4 * tile_ic * 35 floats
        int wt_total = 4 * tile_ic * KHW;
        for (int i = tid; i < wt_total; i += 128) {
            int oc = i / (tile_ic * KHW);        // 0..3
            int rem = i - oc * (tile_ic * KHW);
            int ic = rem / KHW;                  // 0..tile_ic-1
            int k  = rem - ic * KHW;             // 0..34

            int g_oc = oc0 + oc;
            float v = 0.f;
            if (g_oc < Cout) {
                const float* wptr = w + ((g_oc * Cin + (ic0 + ic)) * KHW + k);
                v = ldg_f32(wptr);
            }
            sh_wt[(oc * IC_TILE + ic) * KHW + k] = v;
        }

        __syncthreads();

        // ---------------- Compute using shared ----------------
        if (vout) {
            int sh_r0 = t_oh;
            int sh_c0 = t_ow;

            #pragma unroll
            for (int ic = 0; ic < IC_TILE; ++ic) {
                if (ic >= tile_ic) break;

                const float* w0 = sh_wt + (0 * IC_TILE + ic) * KHW;
                const float* w1 = sh_wt + (1 * IC_TILE + ic) * KHW;
                const float* w2 = sh_wt + (2 * IC_TILE + ic) * KHW;
                const float* w3 = sh_wt + (3 * IC_TILE + ic) * KHW;

                const float* in_base = sh_in + (ic * SH_H + sh_r0) * SH_W + sh_c0;

                #pragma unroll
                for (int kh = 0; kh < 5; ++kh) {
                    const float* in_row = in_base + kh * SH_W;
                    int wbase = kh * 7;

                    float x0 = in_row[0];
                    float x1v = in_row[1];
                    float x2v = in_row[2];
                    float x3v = in_row[3];
                    float x4v = in_row[4];
                    float x5v = in_row[5];
                    float x6v = in_row[6];

                    acc0 = fmaf(x0,  w0[wbase + 0], acc0);
                    acc0 = fmaf(x1v, w0[wbase + 1], acc0);
                    acc0 = fmaf(x2v, w0[wbase + 2], acc0);
                    acc0 = fmaf(x3v, w0[wbase + 3], acc0);
                    acc0 = fmaf(x4v, w0[wbase + 4], acc0);
                    acc0 = fmaf(x5v, w0[wbase + 5], acc0);
                    acc0 = fmaf(x6v, w0[wbase + 6], acc0);

                    acc1 = fmaf(x0,  w1[wbase + 0], acc1);
                    acc1 = fmaf(x1v, w1[wbase + 1], acc1);
                    acc1 = fmaf(x2v, w1[wbase + 2], acc1);
                    acc1 = fmaf(x3v, w1[wbase + 3], acc1);
                    acc1 = fmaf(x4v, w1[wbase + 4], acc1);
                    acc1 = fmaf(x5v, w1[wbase + 5], acc1);
                    acc1 = fmaf(x6v, w1[wbase + 6], acc1);

                    acc2 = fmaf(x0,  w2[wbase + 0], acc2);
                    acc2 = fmaf(x1v, w2[wbase + 1], acc2);
                    acc2 = fmaf(x2v, w2[wbase + 2], acc2);
                    acc2 = fmaf(x3v, w2[wbase + 3], acc2);
                    acc2 = fmaf(x4v, w2[wbase + 4], acc2);
                    acc2 = fmaf(x5v, w2[wbase + 5], acc2);
                    acc2 = fmaf(x6v, w2[wbase + 6], acc2);

                    acc3 = fmaf(x0,  w3[wbase + 0], acc3);
                    acc3 = fmaf(x1v, w3[wbase + 1], acc3);
                    acc3 = fmaf(x2v, w3[wbase + 2], acc3);
                    acc3 = fmaf(x3v, w3[wbase + 3], acc3);
                    acc3 = fmaf(x4v, w3[wbase + 4], acc3);
                    acc3 = fmaf(x5v, w3[wbase + 5], acc3);
                    acc3 = fmaf(x6v, w3[wbase + 6], acc3);
                }
            }
        }

        __syncthreads(); // safe reuse of shmem for next ic0
    }

    // ---------------- Store results ----------------
    if (vout) {
        int out_base = ((n * Cout + oc0) * Hout + oh) * Wout + ow;
        y[out_base] = acc0;
        if (oc0 + 1 < Cout) y[out_base + (Hout * Wout)] = acc1;
        if (oc0 + 2 < Cout) y[out_base + 2 * (Hout * Wout)] = acc2;
        if (oc0 + 3 < Cout) y[out_base + 3 * (Hout * Wout)] = acc3;
    }
}

torch::Tensor conv_standard2d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW (4D)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous (OIHW)");

    int N   = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Hin = (int)x.size(2);
    int Win = (int)x.size(3);

    int Cout = (int)w.size(0);
    int wCin = (int)w.size(1);
    int Kh   = (int)w.size(2);
    int Kw   = (int)w.size(3);

    TORCH_CHECK(wCin == Cin, "weight Cin must match input Cin");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be > 0");
    TORCH_CHECK(dil_h > 0 && dil_w > 0, "dilation must be > 0");

    int Hout = (int)((Hin + 2 * (int)pad_h - (int)dil_h * (Kh - 1) - 1) / (int)stride_h + 1);
    int Wout = (int)((Win + 2 * (int)pad_w - (int)dil_w * (Kw - 1) - 1) / (int)stride_w + 1);
    TORCH_CHECK(Hout >= 0 && Wout >= 0, "Invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    bool fast = (Kh == 5 && Kw == 7 &&
                 stride_h == 1 && stride_w == 1 &&
                 dil_h == 1 && dil_w == 1);

    if (fast) {
        constexpr int OUT_W = 32;
        constexpr int OUT_H = 4;
        constexpr int IC_TILE = 4;

        dim3 block(128, 1, 1);
        dim3 grid((unsigned)((Wout + OUT_W - 1) / OUT_W),
                  (unsigned)((Hout + OUT_H - 1) / OUT_H),
                  (unsigned)(N * ((Cout + 3) / 4)));

        // shared bytes:
        // input: IC_TILE * (OUT_H+4) * (OUT_W+6)
        // weights: 4 * IC_TILE * 35
        size_t sh_in = (size_t)IC_TILE * (size_t)(OUT_H + 4) * (size_t)(OUT_W + 6) * sizeof(float);
        size_t sh_wt = (size_t)4 * (size_t)IC_TILE * (size_t)35 * sizeof(float);
        size_t shmem_bytes = sh_in + sh_wt;

        conv2d_forward_nchw_f32_k5x7_s1d1_oc4_ictile_shmem<OUT_W, OUT_H, IC_TILE>
            <<<grid, block, shmem_bytes, stream>>>(
                (const float*)x.data_ptr<float>(),
                (const float*)w.data_ptr<float>(),
                (float*)y.data_ptr<float>(),
                N, Cin, Hin, Win,
                Cout,
                (int)pad_h, (int)pad_w,
                Hout, Wout,
                (int)((Cout + 3) / 4)
            );
    } else {
        int total = N * Cout * Hout * Wout;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        conv2d_forward_nchw_f32_generic<<<blocks, threads, 0, stream>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Kh, Kw,
            (int)stride_h, (int)stride_w,
            (int)pad_h, (int)pad_w,
            (int)dil_h, (int)dil_w,
            Hout, Wout
        );
    }

    return y;
}
"""

cpp_src = r"""
torch::Tensor conv_standard2d_asymmetric_input_asymmetric_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_standard2d_asym_v5_oc4_ictile4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv_standard2d_asymmetric_input_asymmetric_kernel_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------- Model using the custom op ---------

class ModelNew(nn.Module):
    """
    Custom CUDA forward for Conv2d.
    Assumptions: NCHW float32 CUDA tensors, groups=1, bias=False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if groups != 1:
            raise ValueError("ModelNew custom kernel supports groups=1 only")
        if bias:
            raise ValueError("ModelNew custom kernel supports bias=False only")

        self.custom_ops_lib = custom_ops_lib
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != torch.float32:
            raise ValueError("Input must be float32")
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.conv2d.weight
        if not w.is_cuda:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        sh, sw = self.conv2d.stride
        ph, pw = self.conv2d.padding
        dh, dw = self.conv2d.dilation

        return self.custom_ops_lib.conv_standard2d_asymmetric_input_asymmetric_kernel_cuda(
            x, w, int(sh), int(sw), int(ph), int(pw), int(dh), int(dw)
        )