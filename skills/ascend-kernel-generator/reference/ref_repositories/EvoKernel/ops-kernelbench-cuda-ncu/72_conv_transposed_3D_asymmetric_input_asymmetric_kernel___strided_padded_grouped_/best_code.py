import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif

static __host__ __device__ __forceinline__ int div_up_int(int a, int b) {
    return (a + b - 1) / b;
}

// ------------------------- Generic kernel (fallback) -------------------------
__global__ __launch_bounds__(256, 2)
void conv_transpose3d_grouped_fwd_kernel_generic(
    const float* __restrict__ x,     // [N, Cin, Din, Hin, Win]
    const float* __restrict__ w,     // [Cin, CoutPerG, kD, kH, kW]
    const float* __restrict__ bias,  // [Cout] or nullptr
    float* __restrict__ y,           // [N, Cout, Dout, Hout, Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Dout, int Hout, int Wout,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int groups
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cout * Dout * Hout * Wout;
    if (idx >= total) return;

    int64_t tmp = idx;
    int ow = (int)(tmp % Wout); tmp /= Wout;
    int oh = (int)(tmp % Hout); tmp /= Hout;
    int od = (int)(tmp % Dout); tmp /= Dout;
    int oc = (int)(tmp % Cout); tmp /= Cout;
    int n  = (int)(tmp);

    int CoutPerG = Cout / groups;
    int CinPerG  = Cin / groups;
    int g = oc / CoutPerG;
    int ocg = oc - g * CoutPerG;

    float acc = 0.0f;
    if (bias) acc = LDG(bias + oc);

    for (int icg = 0; icg < CinPerG; ++icg) {
        int ic = g * CinPerG + icg;

        for (int kd = 0; kd < kD; ++kd) {
            int tD = od + pD - kd;
            if (tD < 0) continue;
            int rD = tD - (tD / sD) * sD;
            if (rD != 0) continue;
            int id = tD / sD;
            if ((unsigned)id >= (unsigned)Din) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int tH = oh + pH - kh;
                if (tH < 0) continue;
                int rH = tH - (tH / sH) * sH;
                if (rH != 0) continue;
                int ih = tH / sH;
                if ((unsigned)ih >= (unsigned)Hin) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int tW = ow + pW - kw;
                    if (tW < 0) continue;
                    int rW = tW - (tW / sW) * sW;
                    if (rW != 0) continue;
                    int iw = tW / sW;
                    if ((unsigned)iw >= (unsigned)Win) continue;

                    int64_t x_off = (((int64_t)n * Cin + ic) * Din + id) * (int64_t)Hin * Win
                                  + (int64_t)ih * Win + iw;

                    int64_t w_off = (((int64_t)ic * CoutPerG + ocg) * kD + kd) * (int64_t)kH * kW
                                  + (int64_t)kh * kW + kw;

                    acc = fmaf(LDG(x + x_off), LDG(w + w_off), acc);
                }
            }
        }
    }

    int64_t y_off = (((int64_t)n * Cout + oc) * Dout + od) * (int64_t)Hout * Wout
                  + (int64_t)oh * Wout + ow;
    y[y_off] = acc;
}

// ------------------------- Specialized fast kernel -------------------------
// Specialized for: kD=3,kH=5,kW=7; s=2; p=(1,2,3); op=(1,1,1)
// Computes OCx2 per thread (within the same group), stages weights for kW into shared memory per (ic,kd,kh).
__global__ __launch_bounds__(128, 4)
void convt3d_k3k5k7_s2_p123_op111_grouped_oc2_smemw(
    const float* __restrict__ x,     // [N, Cin, Din, Hin, Win]
    const float* __restrict__ w,     // [Cin, CoutPerG, 3, 5, 7]
    const float* __restrict__ b,     // [Cout] or nullptr
    float* __restrict__ y,           // [N, Cout, Dout, Hout, Wout]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Dout, int Hout, int Wout,
    int groups,
    int has_bias
) {
    constexpr int kD = 3, kH = 5, kW = 7;
    constexpr int sD = 2, sH = 2, sW = 2;
    constexpr int pD = 1, pH = 2, pW = 3;

    const int CoutPerG = Cout / groups;
    const int CinPerG  = Cin / groups;
    const int ocPairsPerG = (CoutPerG >> 1); // requires even CoutPerG for fast path

    const int oh = (int)blockIdx.y;
    if (oh >= Hout) return;

    // z = n * (Dout*groups*ocPairsPerG) + od*(groups*ocPairsPerG) + g*(ocPairsPerG) + ocPairInG
    int z = (int)blockIdx.z;
    const int ocPairInG = z % ocPairsPerG; z /= ocPairsPerG;
    const int g = z % groups; z /= groups;
    const int od = z % Dout; z /= Dout;
    const int n  = z;
    if (n >= N) return;

    const int oc0g = ocPairInG * 2;
    const int oc1g = oc0g + 1;
    const int oc0 = g * CoutPerG + oc0g;
    const int oc1 = oc0 + 1;

    const float b0 = has_bias ? LDG(b + oc0) : 0.f;
    const float b1 = has_bias ? LDG(b + oc1) : 0.f;

    const int od_p = od + pD;
    const int oh_p = oh + pH;

    // bounds for kd/kh that can map back into input for stride=2
    int kd_lo = od_p - (Din - 1) * sD; if (kd_lo < 0) kd_lo = 0;
    int kd_hi = od_p;                  if (kd_hi > (kD - 1)) kd_hi = (kD - 1);

    int kh_lo = oh_p - (Hin - 1) * sH; if (kh_lo < 0) kh_lo = 0;
    int kh_hi = oh_p;                  if (kh_hi > (kH - 1)) kh_hi = (kH - 1);

    // parity-adjusted starts
    const int kd0 = kd_lo + ((od_p - kd_lo) & 1);
    const int kh0 = kh_lo + ((oh_p - kh_lo) & 1);

    __shared__ float2 smw[kW]; // weights for kw=0..6, (w_oc0, w_oc1)

    const int ci_start = g * CinPerG;
    const int64_t spatial = (int64_t)Dout * Hout * Wout;

    // grid-stride over ow
    for (int ow = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
         ow < Wout;
         ow += (int)blockDim.x * (int)gridDim.x) {

        float acc0 = b0;
        float acc1 = b1;

        const int ow_p = ow + pW;
        // kw must satisfy parity: (ow_p - kw) even -> kw parity == ow_p parity
        const int kw0 = (ow_p & 1); // 0 or 1
        // We'll iterate kw = kw0, kw0+2, ... <=6
        // Also need iw = (ow_p - kw)/2 in [0, Win)
        // precompute iw for each possible kw to reduce integer work in inner loops
        int iw_arr[4]; // max 4 terms when kw0=0: 0,2,4,6; when kw0=1: 1,3,5
        int kw_arr[4];
        int nterms = 0;
        #pragma unroll
        for (int kw = 0; kw < kW; ++kw) {
            // compile-time unrolled filter by parity
            if (((kw & 1) == kw0)) {
                int tW = ow_p - kw;
                if (tW >= 0) {
                    int iw = tW >> 1;
                    if ((unsigned)iw < (unsigned)Win) {
                        kw_arr[nterms] = kw;
                        iw_arr[nterms] = iw;
                        nterms++;
                    }
                }
            }
        }

        // if no kw terms are valid, output is just bias
        if (nterms == 0) {
            int64_t out0 = ((((int64_t)n * Cout + oc0) * Dout + od) * Hout + oh) * Wout + ow;
            y[out0] = acc0;
            y[out0 + spatial] = acc1;
            continue;
        }

        // Iterate IC in group (keep unroll modest to avoid reg blowup)
        for (int icg = 0; icg < CinPerG; ++icg) {
            const int ci = ci_start + icg;

            const float* __restrict__ x_base = x + (((int64_t)n * Cin + ci) * (int64_t)Din * Hin * Win);
            const float* __restrict__ w0_ci  = w + (((int64_t)ci * CoutPerG + oc0g) * (kD * kH * kW));
            const float* __restrict__ w1_ci  = w + (((int64_t)ci * CoutPerG + oc1g) * (kD * kH * kW));

            // kd loop (at most 2 iterations for kD=3 with stride=2 parity)
            #pragma unroll
            for (int t0 = 0; t0 < 2; ++t0) {
                const int kd = kd0 + (t0 << 1);
                if (kd > kd_hi) break;
                const int id = (od_p - kd) >> 1;

                const float* __restrict__ x_d   = x_base + (int64_t)id * Hin * Win;
                const float* __restrict__ w0_kd = w0_ci + (int64_t)kd * (kH * kW);
                const float* __restrict__ w1_kd = w1_ci + (int64_t)kd * (kH * kW);

                // kh loop (parity => up to 3 iterations: 0/2/4 or 1/3)
                // keep as small unroll of 3
                #pragma unroll
                for (int kh = 0; kh < kH; ++kh) {
                    if (kh < kh0) continue;
                    if (kh > kh_hi) continue;
                    if (((kh - kh0) & 1) != 0) continue;

                    const int ih = (oh_p - kh) >> 1;
                    if ((unsigned)ih >= (unsigned)Hin) continue;

                    const float* __restrict__ x_dh  = x_d + (int64_t)ih * Win;
                    const float* __restrict__ w0_kh = w0_kd + (int64_t)kh * kW;
                    const float* __restrict__ w1_kh = w1_kd + (int64_t)kh * kW;

                    // cooperative load kW float2 weights into smem: first 7 threads
                    if ((int)threadIdx.x < kW) {
                        int t = (int)threadIdx.x;
                        float2 vv;
                        vv.x = LDG(w0_kh + t);
                        vv.y = LDG(w1_kh + t);
                        smw[t] = vv;
                    }
                    __syncthreads();

                    // accumulate over valid kw terms
                    float p0 = 0.f;
                    float p1 = 0.f;
                    #pragma unroll
                    for (int it = 0; it < 4; ++it) {
                        if (it >= nterms) break;
                        const int kw = kw_arr[it];
                        const int iw = iw_arr[it];
                        const float xv = LDG(x_dh + iw);
                        const float2 ww = smw[kw];
                        p0 = fmaf(xv, ww.x, p0);
                        p1 = fmaf(xv, ww.y, p1);
                    }
                    acc0 += p0;
                    acc1 += p1;

                    __syncthreads();
                }
            }
        }

        int64_t out0 = ((((int64_t)n * Cout + oc0) * Dout + od) * Hout + oh) * Wout + ow;
        y[out0] = acc0;
        y[out0 + spatial] = acc1;
    }
}

torch::Tensor conv_transpose3d_grouped_cuda_opt3(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t opD, int64_t opH, int64_t opW,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "w must be [Cin, CoutPerG, kD, kH, kW]");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int64_t N   = x_c.size(0);
    const int64_t Cin = x_c.size(1);
    const int64_t Din = x_c.size(2);
    const int64_t Hin = x_c.size(3);
    const int64_t Win = x_c.size(4);

    TORCH_CHECK(groups >= 1, "groups must be >= 1");
    TORCH_CHECK(Cin % groups == 0, "Cin must be divisible by groups");
    TORCH_CHECK(w_c.size(0) == Cin, "w.size(0) must equal Cin");

    const int64_t CoutPerG = w_c.size(1);
    const int64_t kD = w_c.size(2);
    const int64_t kH = w_c.size(3);
    const int64_t kW = w_c.size(4);

    const int64_t Cout = CoutPerG * groups;

    TORCH_CHECK(sD >= 1 && sH >= 1 && sW >= 1, "stride must be >= 1");
    TORCH_CHECK(pD >= 0 && pH >= 0 && pW >= 0, "padding must be >= 0");
    TORCH_CHECK(opD >= 0 && opH >= 0 && opW >= 0, "output_padding must be >= 0");

    const int64_t Dout = (Din - 1) * sD - 2 * pD + kD + opD;
    const int64_t Hout = (Hin - 1) * sH - 2 * pH + kH + opH;
    const int64_t Wout = (Win - 1) * sW - 2 * pW + kW + opW;
    TORCH_CHECK(Dout > 0 && Hout > 0 && Wout > 0, "computed output size must be positive");

    auto y = torch::zeros({N, Cout, Dout, Hout, Wout}, x_c.options());

    torch::Tensor b_c;
    const float* b_ptr = nullptr;
    int has_bias = 0;
    if (bias_opt.has_value() && bias_opt.value().defined() && bias_opt.value().numel() > 0) {
        b_c = bias_opt.value().contiguous();
        TORCH_CHECK(b_c.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b_c.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b_c.dim() == 1 && b_c.size(0) == Cout, "bias must be [Cout]");
        b_ptr = b_c.data_ptr<float>();
        has_bias = 1;
    }

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int device = x_c.get_device();
    cudaDeviceProp prop;
    int sm_count = 80;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) sm_count = prop.multiProcessorCount;

    const bool fast_path =
        (kD == 3 && kH == 5 && kW == 7) &&
        (sD == 2 && sH == 2 && sW == 2) &&
        (pD == 1 && pH == 2 && pW == 3) &&
        (opD == 1 && opH == 1 && opW == 1) &&
        ((CoutPerG & 1) == 0) &&
        ((CoutPerG >> 1) > 0);

    if (fast_path) {
        const int threads = 128;
        dim3 block(threads, 1, 1);

        int gx = div_up_int((int)Wout, threads);
        int max_gx = sm_count * 16;
        if (gx > max_gx) gx = max_gx;
        if (gx < 1) gx = 1;

        dim3 grid;
        grid.x = (unsigned)gx;
        grid.y = (unsigned)Hout;

        int ocPairsPerG_i = (int)(CoutPerG >> 1);
        int64_t z_total = N * Dout * (int64_t)groups * (int64_t)ocPairsPerG_i;
        TORCH_CHECK(z_total <= (int64_t)2147483647, "grid.z too large for fast path");
        grid.z = (unsigned)z_total;

        convt3d_k3k5k7_s2_p123_op111_grouped_oc2_smemw<<<grid, block, 0, stream>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
            (int)Cout, (int)Dout, (int)Hout, (int)Wout,
            (int)groups,
            has_bias
        );
    } else {
        int64_t total = N * Cout * Dout * Hout * Wout;
        const int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        int max_blocks = sm_count * 32;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        conv_transpose3d_grouped_fwd_kernel_generic<<<blocks, threads, 0, stream>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            b_ptr,
            y.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
            (int)Cout, (int)Dout, (int)Hout, (int)Wout,
            (int)kD, (int)kH, (int)kW,
            (int)sD, (int)sH, (int)sW,
            (int)pD, (int)pH, (int)pW,
            (int)groups
        );
    }

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose3d_grouped_cuda_opt3(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t opD, int64_t opH, int64_t opW,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_asym_opt3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_grouped_cuda_opt3"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-maxrregcount=72",
    ],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Drop-in replacement using an optimized custom CUDA kernel for ConvTranspose3d forward.
    Weight layout: [Cin, Cout/groups, kD, kH, kW] (matches PyTorch ConvTranspose3d).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.output_padding = tuple(output_padding)
        self.groups = int(groups)

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        kD, kH, kW = self.kernel_size
        cout_per_g = self.out_channels // self.groups
        self.weight = nn.Parameter(torch.empty(self.in_channels, cout_per_g, kD, kH, kW, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(self.out_channels, dtype=torch.float32)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * kD * kH * kW
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        opD, opH, opW = self.output_padding
        return self.custom_ops.conv_transpose3d_grouped_cuda_opt3(
            x,
            self.weight,
            self.bias,
            int(sD), int(sH), int(sW),
            int(pD), int(pH), int(pW),
            int(opD), int(opH), int(opW),
            int(self.groups),
        )