import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static inline int calc_out_int(int in, int k, int pad, int stride, int dil) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// ---------------- Generic kernel (grid-stride) ----------------
__global__ void conv3d_forward_f32_generic_kernel(
    const float* __restrict__ x,        // [N, Cin, D, H, W]
    const float* __restrict__ w,        // [Cout, Cin/groups, kD, kH, kW]
    const float* __restrict__ b,        // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, Do, Ho, Wo]
    int N, int Cin, int D, int H, int W,
    int Cout,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW,
    int groups,
    int Do, int Ho, int Wo
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * Cout * Do * Ho * Wo;
    long long gstride = (long long)gridDim.x * blockDim.x;

    int CoutG = Cout / groups;
    int CinG  = Cin  / groups;

    for (long long linear = tid; linear < total; linear += gstride) {
        long long idx = linear;
        int ow = (int)(idx % Wo); idx /= Wo;
        int oh = (int)(idx % Ho); idx /= Ho;
        int od = (int)(idx % Do); idx /= Do;
        int oc = (int)(idx % Cout);
        int n  = (int)(idx / Cout);

        int g = oc / CoutG;
        int cin_start = g * CinG;

        float acc = (b != nullptr) ? ldg_f32(b + oc) : 0.0f;

        for (int icg = 0; icg < CinG; ++icg) {
            int ic = cin_start + icg;
            const float* x_base = x + (((((long long)n * Cin + ic) * D) * H) * W);
            const float* w_base = w + (((((long long)oc * CinG + icg) * kD) * kH) * kW);

#pragma unroll 1
            for (int kd = 0; kd < kD; ++kd) {
                int id = od * strideD - padD + kd * dilD;
                if ((unsigned)id >= (unsigned)D) continue;
#pragma unroll 1
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh * strideH - padH + kh * dilH;
                    if ((unsigned)ih >= (unsigned)H) continue;
#pragma unroll 1
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ow * strideW - padW + kw * dilW;
                        if ((unsigned)iw >= (unsigned)W) continue;

                        float xv = ldg_f32(x_base + ((id * H + ih) * W + iw));
                        float wv = ldg_f32(w_base + ((kd * kH + kh) * kW + kw));
                        acc = fmaf(xv, wv, acc);
                    }
                }
            }
        }

        y[((((long long)n * Cout + oc) * Do + od) * Ho + oh) * Wo + ow] = acc;
    }
}

// ---------------- Specialized fast path: fixed shape/params + OCx4 + SMEM weights ----------------
// Dominant model: k=(3,5,7), stride=1, pad=0, dil=1, groups=1, Cin=3.
// Each thread computes 4 output channels (oc..oc+3) for one (n,od,oh,ow).
// Weights for these 4 channels are staged into shared memory once per CTA.
template <bool HasBias, bool Full4>
__global__ __launch_bounds__(128, 4)
void conv3d_forward_f32_k357_s1p0d1_g1_cin3_oc4_smemw_kernel(
    const float* __restrict__ x,   // [N,3,D,H,W]
    const float* __restrict__ w,   // [Cout,3,3,5,7]
    const float* __restrict__ b,   // [Cout] (valid iff HasBias)
    float* __restrict__ y,         // [N,Cout,Do,Ho,Wo]
    int N, int D, int H, int W,
    int Cout,
    int Do, int Ho, int Wo
) {
    constexpr int kD = 3, kH = 5, kW = 7;
    constexpr int kHW = kH * kW;     // 35
    constexpr int kDHW = kD * kHW;   // 105
    constexpr int Cin = 3;

    const int oc0 = (int)blockIdx.y * 4;
    if (oc0 >= Cout) return;

    int valid = Cout - oc0;
    int oc1 = oc0 + 1;
    int oc2 = oc0 + 2;
    int oc3 = oc0 + 3;

    // shared weights: [4, 3, 105] contiguous
    extern __shared__ float smem[];
    float* sw = smem; // size 4*3*105 floats = 1260 floats = 5040 bytes

    // cooperative load weights (and benefit from L2 once, then SMEM hits)
    // total weights per tile = 4*3*105 = 1260
    int t = threadIdx.x;
    int stride = blockDim.x;
    // weight base for tile
    const long long tile_w_base = (long long)oc0 * (long long)Cin * (long long)kDHW;

    for (int i = t; i < 4 * Cin * kDHW; i += stride) {
        int tmp = i;
        int o = tmp / (Cin * kDHW); tmp -= o * (Cin * kDHW);
        int ic = tmp / kDHW;        tmp -= ic * kDHW;
        int tap = tmp;
        int oc = oc0 + o;
        // for tail tiles, load zeros for invalid channels so compute can stay branch-light
        float v = 0.f;
        if (Full4 || (o < valid)) {
            v = ldg_f32(w + (tile_w_base + (long long)o * (Cin * kDHW) + (long long)ic * kDHW + tap));
        }
        sw[i] = v;
    }
    __syncthreads();

    const long long HW  = (long long)H * W;
    const long long DHW = (long long)D * HW;
    const long long plane = (long long)Do * Ho * Wo;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_spatial = (long long)N * Do * Ho * Wo;
    long long gstride = (long long)gridDim.x * blockDim.x;

    for (long long linear = tid; linear < total_spatial; linear += gstride) {
        long long tt = linear;
        int ow = (int)(tt % Wo); tt /= Wo;
        int oh = (int)(tt % Ho); tt /= Ho;
        int od = (int)(tt % Do);
        int n  = (int)(tt / Do);

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        if constexpr (HasBias) {
            // for Full4, all valid; for tail, guard reads (minor, outside hot loops)
            acc0 = ldg_f32(b + oc0);
            if constexpr (Full4) {
                acc1 = ldg_f32(b + oc1);
                acc2 = ldg_f32(b + oc2);
                acc3 = ldg_f32(b + oc3);
            } else {
                if (valid > 1) acc1 = ldg_f32(b + oc1);
                if (valid > 2) acc2 = ldg_f32(b + oc2);
                if (valid > 3) acc3 = ldg_f32(b + oc3);
            }
        }

        const long long x_n_base = (long long)n * (long long)Cin * DHW;
        const long long out_sp = ((long long)od * Ho + oh) * Wo + ow;
        const long long y_n_base = (long long)n * (long long)Cout * plane;

#pragma unroll
        for (int ic = 0; ic < Cin; ++ic) {
            const long long x_c_base = x_n_base + (long long)ic * DHW;
            const float* __restrict__ sw_ic0 = sw + (0 * Cin + ic) * kDHW;
            const float* __restrict__ sw_ic1 = sw + (1 * Cin + ic) * kDHW;
            const float* __restrict__ sw_ic2 = sw + (2 * Cin + ic) * kDHW;
            const float* __restrict__ sw_ic3 = sw + (3 * Cin + ic) * kDHW;

#pragma unroll
            for (int kd = 0; kd < kD; ++kd) {
                int id = od + kd;
                const long long x_d = x_c_base + (long long)id * HW;
                const int w_kd = kd * kHW;

#pragma unroll
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh + kh;
                    const long long x_h = x_d + (long long)ih * W + ow; // points at kw=0
                    const int w_kh = w_kd + kh * kW;

                    // lightweight prefetch pipeline for x
                    float xv = ldg_f32(x + x_h);
#pragma unroll
                    for (int kw = 0; kw < kW; ++kw) {
                        float xv_next = (kw + 1 < kW) ? ldg_f32(x + (x_h + kw + 1)) : 0.f;

                        int tap = w_kh + kw;
                        float w0 = sw_ic0[tap];
                        float w1 = sw_ic1[tap];
                        float w2 = sw_ic2[tap];
                        float w3 = sw_ic3[tap];

                        acc0 = fmaf(xv, w0, acc0);
                        acc1 = fmaf(xv, w1, acc1);
                        acc2 = fmaf(xv, w2, acc2);
                        acc3 = fmaf(xv, w3, acc3);

                        xv = xv_next;
                    }
                }
            }
        }

        const long long out_base = y_n_base + out_sp;
        y[out_base + (long long)oc0 * plane] = acc0;
        if constexpr (Full4) {
            y[out_base + (long long)oc1 * plane] = acc1;
            y[out_base + (long long)oc2 * plane] = acc2;
            y[out_base + (long long)oc3 * plane] = acc3;
        } else {
            if (valid > 1) y[out_base + (long long)oc1 * plane] = acc1;
            if (valid > 2) y[out_base + (long long)oc2 * plane] = acc2;
            if (valid > 3) y[out_base + (long long)oc3 * plane] = acc3;
        }
    }
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dilD, int64_t dilH, int64_t dilW,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");
    TORCH_CHECK(w.dim() == 5, "w must be 5D (Cout,Cin/groups,kD,kH,kW)");
    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!w.is_contiguous()) w = w.contiguous();

    const int N   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int D   = (int)x.size(2);
    const int H   = (int)x.size(3);
    const int W   = (int)x.size(4);

    const int Cout = (int)w.size(0);
    const int CinG = (int)w.size(1);
    const int kD   = (int)w.size(2);
    const int kH   = (int)w.size(3);
    const int kW   = (int)w.size(4);

    TORCH_CHECK(Cin % (int)groups == 0, "Cin must be divisible by groups");
    TORCH_CHECK(Cout % (int)groups == 0, "Cout must be divisible by groups");
    TORCH_CHECK(CinG == Cin / (int)groups, "w.size(1) must equal Cin/groups");

    const int Do = calc_out_int(D, kD, (int)padD, (int)strideD, (int)dilD);
    const int Ho = calc_out_int(H, kH, (int)padH, (int)strideH, (int)dilH);
    const int Wo = calc_out_int(W, kW, (int)padW, (int)strideW, (int)dilW);
    TORCH_CHECK(Do > 0 && Ho > 0 && Wo > 0, "Invalid output shape");

    torch::Tensor y = torch::empty({N, Cout, Do, Ho, Wo}, x.options());

    torch::Tensor b;
    const float* b_ptr = nullptr;
    bool has_bias = false;
    if (b_opt.has_value()) {
        b = b_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias must be 1D of shape [Cout]");
        if (!b.is_contiguous()) b = b.contiguous();
        b_ptr = (const float*)b.data_ptr<float>();
        has_bias = true;
    }

    const float* x_ptr = (const float*)x.data_ptr<float>();
    const float* w_ptr = (const float*)w.data_ptr<float>();
    float* y_ptr = (float*)y.data_ptr<float>();

    const int sD = (int)strideD, sH = (int)strideH, sW = (int)strideW;
    const int pD = (int)padD, pH = (int)padH, pW = (int)padW;
    const int dD = (int)dilD, dH = (int)dilH, dW = (int)dilW;
    const int g = (int)groups;

    bool fast_k357 =
        (g == 1) &&
        (Cin == 3) &&
        (kD == 3 && kH == 5 && kW == 7) &&
        (sD == 1 && sH == 1 && sW == 1) &&
        (pD == 0 && pH == 0 && pW == 0) &&
        (dD == 1 && dH == 1 && dW == 1);

    if (fast_k357) {
        long long total_spatial = (long long)N * Do * Ho * Wo;

        int threads = 128;
        int blocks_x = (int)((total_spatial + threads - 1) / threads);
        if (blocks_x > 65535) blocks_x = 65535;

        int oc_quads = (Cout + 3) / 4;
        dim3 grid((unsigned)blocks_x, (unsigned)oc_quads, 1);

        // dynamic shared mem bytes for weights
        size_t smem_bytes = (size_t)(4 * 3 * 3 * 5 * 7) * sizeof(float); // 5040

        bool full4_all = ((Cout % 4) == 0);

        // Use Full4 kernel for all tiles when divisible by 4; otherwise use Full4 for all but last tile.
        if (has_bias) {
            if (full4_all) {
                conv3d_forward_f32_k357_s1p0d1_g1_cin3_oc4_smemw_kernel<true, true><<<grid, threads, smem_bytes>>>(
                    x_ptr, w_ptr, b_ptr, y_ptr, N, D, H, W, Cout, Do, Ho, Wo
                );
            } else {
                // launch: for all tiles we still use same grid; Full4=false safely handles tail,
                // but it adds a couple of guards only in stores/bias loads, not in hot loops (weights already zeroed).
                conv3d_forward_f32_k357_s1p0d1_g1_cin3_oc4_smemw_kernel<true, false><<<grid, threads, smem_bytes>>>(
                    x_ptr, w_ptr, b_ptr, y_ptr, N, D, H, W, Cout, Do, Ho, Wo
                );
            }
        } else {
            if (full4_all) {
                conv3d_forward_f32_k357_s1p0d1_g1_cin3_oc4_smemw_kernel<false, true><<<grid, threads, smem_bytes>>>(
                    x_ptr, w_ptr, nullptr, y_ptr, N, D, H, W, Cout, Do, Ho, Wo
                );
            } else {
                conv3d_forward_f32_k357_s1p0d1_g1_cin3_oc4_smemw_kernel<false, false><<<grid, threads, smem_bytes>>>(
                    x_ptr, w_ptr, nullptr, y_ptr, N, D, H, W, Cout, Do, Ho, Wo
                );
            }
        }
        return y;
    }

    long long total = (long long)N * Cout * Do * Ho * Wo;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    conv3d_forward_f32_generic_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, Cin, D, H, W,
        Cout,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        dD, dH, dW,
        g,
        Do, Ho, Wo
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv3d_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b_opt,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dilD, int64_t dilH, int64_t dilW,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_asym_sqinput_oc4_smemw_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Conv3d forward replaced with a custom CUDA kernel.
    Fast path specialized for: groups=1, Cin=3, k=(3,5,7), stride=1, padding=0, dilation=1.
    This version stages weights for each oc-tile into shared memory, and uses templated variants
    to avoid oc-tail checks in the hot loops, targeting lower register pressure and higher occupancy.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.custom_ops_lib = custom_ops_lib

    @staticmethod
    def _triple(v):
        if isinstance(v, (tuple, list)):
            if len(v) != 3:
                raise ValueError("expected triple")
            return int(v[0]), int(v[1]), int(v[2])
        v = int(v)
        return v, v, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()

        w = self.conv3d.weight
        b = self.conv3d.bias if self.conv3d.bias is not None else None

        strideD, strideH, strideW = self._triple(self.conv3d.stride)
        padD, padH, padW = self._triple(self.conv3d.padding)
        dilD, dilH, dilW = self._triple(self.conv3d.dilation)
        groups = int(self.conv3d.groups)

        return self.custom_ops_lib.conv3d_forward_cuda(
            x, w, b,
            strideD, strideH, strideW,
            padD, padH, padW,
            dilD, dilH, dilW,
            groups
        )