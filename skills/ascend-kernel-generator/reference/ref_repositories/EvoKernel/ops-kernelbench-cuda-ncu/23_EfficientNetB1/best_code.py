import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float relu_f(float x) { return x > 0.f ? x : 0.f; }

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ldg_f(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ldg_f(const float* p) { return *p; }
#endif

// -----------------------------
// STEM kernel (baseline)
// -----------------------------
__global__ void stem_conv3x3s2p1_cin3_cout32_vec4_bn_relu(
    const float* __restrict__ x,     // [N,3,240,240]
    const float* __restrict__ w,     // [32,3,3,3]
    const float* __restrict__ alpha, // [32]
    const float* __restrict__ beta,  // [32]
    float* __restrict__ y,           // [N,32,120,120]
    int N,
    int do_relu
) {
    constexpr int Cin = 3;
    constexpr int Hin = 240;
    constexpr int Win = 240;
    constexpr int Hout = 120;
    constexpr int Wout = 120;
    constexpr int Vec = 4;
    constexpr int Wvec = Wout / Vec; // 30

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 32 * Hout * Wvec;
    if (idx >= total) return;

    int owv = idx % Wvec;
    int t1 = idx / Wvec;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % 32;
    int n  = t2 / 32;

    int ow0 = owv * Vec;

    int in_y0 = oh * 2 - 1;
    int in_x0 = ow0 * 2 - 1;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    int w_oc_base = oc * Cin * 9;

    #pragma unroll
    for (int ic = 0; ic < Cin; ++ic) {
        int w_ic_base = w_oc_base + ic * 9;
        int x_ic_base = ((n * Cin + ic) * Hin) * Win;

        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_ic_base + iy * Win;

            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                float wv = ldg_f(w + w_ic_base + ky * 3 + kx);

                int ix0 = in_x0 + kx;
                int ix1 = ix0 + 2;
                int ix2 = ix0 + 4;
                int ix3 = ix0 + 6;

                if ((unsigned)ix0 < (unsigned)Win) acc0 = fmaf(x[x_row + ix0], wv, acc0);
                if ((unsigned)ix1 < (unsigned)Win) acc1 = fmaf(x[x_row + ix1], wv, acc1);
                if ((unsigned)ix2 < (unsigned)Win) acc2 = fmaf(x[x_row + ix2], wv, acc2);
                if ((unsigned)ix3 < (unsigned)Win) acc3 = fmaf(x[x_row + ix3], wv, acc3);
            }
        }
    }

    float a = ldg_f(alpha + oc);
    float b = ldg_f(beta + oc);

    float o0 = fmaf(acc0, a, b);
    float o1 = fmaf(acc1, a, b);
    float o2 = fmaf(acc2, a, b);
    float o3 = fmaf(acc3, a, b);

    if (do_relu) {
        o0 = relu_f(o0);
        o1 = relu_f(o1);
        o2 = relu_f(o2);
        o3 = relu_f(o3);
    }

    int out_base = (((n * 32 + oc) * Hout + oh) * Wout + ow0);
    *reinterpret_cast<float4*>(y + out_base) = make_float4(o0, o1, o2, o3);
}

// -----------------------------
// Generic fallback kernels (baseline)
// -----------------------------
__global__ void conv2d_fwd_nchw_k3_bn_relu(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad,
    int do_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int in_y0 = oh * stride - pad;
    int in_x0 = ow * stride - pad;

    float acc = 0.0f;
    int w_oc_base = oc * Cin * 9;

    for (int ic = 0; ic < Cin; ++ic) {
        int w_ic_base = w_oc_base + ic * 9;
        int x_ic_base = ((n * Cin + ic) * Hin) * Win;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            int iy = in_y0 + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            int x_row = x_ic_base + iy * Win;
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int ix = in_x0 + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;
                float xv = x[x_row + ix];
                float wv = ldg_f(w + w_ic_base + ky * 3 + kx);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    float outv = fmaf(acc, ldg_f(alpha + oc), ldg_f(beta + oc));
    if (do_relu) outv = relu_f(outv);

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = outv;
}

__global__ void conv2d_fwd_nchw_k1_bn_relu(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride,
    int do_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    int iy = oh * stride;
    int ix = ow * stride;

    float acc = 0.0f;
    int w_oc_base = oc * Cin;

    int x_base = ((n * Cin) * Hin + iy) * Win + ix;
    int stride_ic = Hin * Win;

    int ic = 0;
    for (; ic + 3 < Cin; ic += 4) {
        float xv0 = x[x_base + (ic + 0) * stride_ic];
        float xv1 = x[x_base + (ic + 1) * stride_ic];
        float xv2 = x[x_base + (ic + 2) * stride_ic];
        float xv3 = x[x_base + (ic + 3) * stride_ic];

        float wv0 = ldg_f(w + w_oc_base + ic + 0);
        float wv1 = ldg_f(w + w_oc_base + ic + 1);
        float wv2 = ldg_f(w + w_oc_base + ic + 2);
        float wv3 = ldg_f(w + w_oc_base + ic + 3);

        acc = fmaf(xv0, wv0, acc);
        acc = fmaf(xv1, wv1, acc);
        acc = fmaf(xv2, wv2, acc);
        acc = fmaf(xv3, wv3, acc);
    }
    for (; ic < Cin; ++ic) {
        float xv = x[x_base + ic * stride_ic];
        float wv = ldg_f(w + w_oc_base + ic);
        acc = fmaf(xv, wv, acc);
    }

    float outv = fmaf(acc, ldg_f(alpha + oc), ldg_f(beta + oc));
    if (do_relu) outv = relu_f(outv);

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = outv;
}

// -----------------------------
// Head: fused GAP + GEMV + BN + ReLU (optimized)
// Warp-specialized: 1 warp computes 1 output channel for fixed n.
// Weight tile staged to shared once per block.
// Persistent loop over oc tiles to reduce grid overhead.
// -----------------------------
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

template<int OC_TILE, int WARPS>
__global__ void head_gap_gemv_bn_relu_fused_tiled(
    const float* __restrict__ x,     // [N,Cin,H,W]
    const float* __restrict__ w,     // [Cout,Cin]
    const float* __restrict__ alpha, // [Cout]
    const float* __restrict__ beta,  // [Cout]
    float* __restrict__ y,           // [N,Cout]
    int N, int Cin, int H, int W, int Cout
) {
    int n = (int)blockIdx.y;
    if (n >= N) return;

    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    constexpr int THREADS = WARPS * 32;
    static_assert(THREADS <= 1024, "too many threads");

    int HW = H * W;
    float invHW = 1.0f / (float)HW;

    // Persistent tiles over output channels
    for (int tile0 = (int)blockIdx.x * OC_TILE; tile0 < Cout; tile0 += (int)gridDim.x * OC_TILE) {
        int oc = tile0 + warp;

        // stage weights for this tile: [OC_TILE, Cin]
        extern __shared__ float sW[];
        // cooperative load: linear over OC_TILE*Cin
        int lin = tid;
        int total = OC_TILE * Cin;
        for (; lin < total; lin += THREADS) {
            int toc = lin / Cin;
            int ic  = lin - toc * Cin;
            int goc = tile0 + toc;
            sW[toc * Cin + ic] = (goc < Cout) ? ldg_f(w + goc * Cin + ic) : 0.0f;
        }
        __syncthreads();

        if (warp < OC_TILE && oc < Cout) {
            float acc = 0.f;

            // loop over input channels
            for (int ic = 0; ic < Cin; ++ic) {
                const float* xptr = x + ((n * Cin + ic) * HW);

                // compute sum over HW using only this warp
                float sum = 0.f;

                if (HW == 64) {
                    // fixed 8x8 tail common for EfficientNet-B1; use 16 float4 loads per channel
                    // map lanes 0..15 to float4 chunks, others idle
                    if (lane < 16) {
                        float4 v = *reinterpret_cast<const float4*>(xptr + lane * 4);
                        sum = v.x + v.y + v.z + v.w;
                    }
                    sum = warp_sum(sum);
                } else {
                    // generic: stride loop over HW
                    for (int p = lane; p < HW; p += 32) sum += xptr[p];
                    sum = warp_sum(sum);
                }

                float mean = sum * invHW;
                acc = fmaf(mean, sW[warp * Cin + ic], acc);
            }

            float outv = fmaf(acc, ldg_f(alpha + oc), ldg_f(beta + oc));
            outv = relu_f(outv);
            if (lane == 0) y[n * Cout + oc] = outv;
        }

        __syncthreads();
    }
}

// -----------------------------
// Launch helpers
// -----------------------------
static inline void launch_stem_special(
    torch::Tensor x, torch::Tensor w,
    torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor y,
    int do_relu
) {
    int N = (int)x.size(0);
    int total = N * 32 * 120 * (120/4);
    int block = 256;
    int grid = (total + block - 1) / block;
    stem_conv3x3s2p1_cin3_cout32_vec4_bn_relu<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        do_relu
    );
}

static inline void launch_k3(
    torch::Tensor x, torch::Tensor w,
    torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor y,
    int stride, int pad, int do_relu
) {
    int N = (int)x.size(0), Cin=(int)x.size(1), Hin=(int)x.size(2), Win=(int)x.size(3);
    int Cout=(int)w.size(0);
    int Hout=(int)y.size(2), Wout=(int)y.size(3);
    int total = N*Cout*Hout*Wout;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_fwd_nchw_k3_bn_relu<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(),
        alpha.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout,
        stride, pad, do_relu
    );
}

static inline void launch_k1(
    torch::Tensor x, torch::Tensor w2d,
    torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor y,
    int stride, int do_relu
) {
    int N = (int)x.size(0), Cin=(int)x.size(1), Hin=(int)x.size(2), Win=(int)x.size(3);
    int Cout=(int)w2d.size(0);
    int Hout=(int)y.size(2), Wout=(int)y.size(3);
    int total = N*Cout*Hout*Wout;

    int block = 256;
    int grid = (total + block - 1) / block;

    conv2d_fwd_nchw_k1_bn_relu<<<grid, block>>>(
        x.data_ptr<float>(), w2d.data_ptr<float>(),
        alpha.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Hout, Wout,
        stride, do_relu
    );
}

// -----------------------------
// PyTorch entrypoints
// -----------------------------
torch::Tensor conv_bn_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor alpha,
    torch::Tensor beta,
    int64_t stride,
    int64_t pad,
    bool do_relu
) {
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(alpha); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(x); CHECK_FLOAT(w); CHECK_FLOAT(alpha); CHECK_FLOAT(beta);

    TORCH_CHECK(x.dim()==4, "x must be NCHW");
    TORCH_CHECK(w.dim()==4, "w must be OIHW");
    TORCH_CHECK(alpha.dim()==1 && beta.dim()==1, "alpha/beta must be 1D");

    int64_t N = x.size(0), Cin = x.size(1), Hin = x.size(2), Win = x.size(3);
    int64_t Cout = w.size(0);
    int64_t Kh = w.size(2), Kw = w.size(3);

    TORCH_CHECK(w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK(alpha.numel() == Cout && beta.numel() == Cout, "alpha/beta size mismatch");
    TORCH_CHECK(stride == 1 || stride == 2, "only stride 1 or 2 supported");
    TORCH_CHECK((Kh==3 && Kw==3) || (Kh==1 && Kw==1), "only 3x3 or 1x1 supported");
    if (Kh==1) TORCH_CHECK(pad == 0, "pad must be 0 for 1x1");
    if (Kh==3) TORCH_CHECK(pad == 1, "pad must be 1 for 3x3");

    int64_t Hout = (Hin + 2 * pad - Kh) / stride + 1;
    int64_t Wout = (Win + 2 * pad - Kw) / stride + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    bool is_stem = (Kh==3 && Kw==3 && stride==2 && pad==1 &&
                    Cin==3 && Cout==32 && Hin==240 && Win==240 &&
                    Hout==120 && Wout==120);

    if (is_stem) {
        launch_stem_special(x, w, alpha, beta, y, do_relu ? 1 : 0);
        return y;
    }

    if (Kh == 3) {
        launch_k3(x, w, alpha, beta, y, (int)stride, (int)pad, do_relu ? 1 : 0);
    } else {
        launch_k1(x, w.view({Cout, Cin}), alpha, beta, y, (int)stride, do_relu ? 1 : 0);
    }
    return y;
}

// Head entrypoint: fused GAP+GEMV+BN+ReLU (returns [N,Cout])
torch::Tensor head_conv_bn_relu_gap_forward_cuda(
    torch::Tensor x,      // [N,Cin,H,W]
    torch::Tensor w2d,    // [Cout,Cin]
    torch::Tensor alpha,  // [Cout]
    torch::Tensor beta,   // [Cout]
    bool do_relu
) {
    (void)do_relu; // always relu in this fused head
    CHECK_CUDA(x); CHECK_CUDA(w2d); CHECK_CUDA(alpha); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w2d); CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(x); CHECK_FLOAT(w2d); CHECK_FLOAT(alpha); CHECK_FLOAT(beta);

    TORCH_CHECK(x.dim()==4, "x must be NCHW");
    TORCH_CHECK(w2d.dim()==2, "w2d must be [Cout,Cin]");
    TORCH_CHECK(alpha.dim()==1 && beta.dim()==1, "alpha/beta must be 1D");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int Cout = (int)w2d.size(0);
    TORCH_CHECK((int)w2d.size(1) == Cin, "w2d Cin mismatch");
    TORCH_CHECK((int)alpha.numel() == Cout && (int)beta.numel() == Cout, "alpha/beta size mismatch");

    auto y = torch::empty({N, Cout}, x.options());

    // Tuned: 8 warps/block => 256 threads. Each warp computes one oc.
    constexpr int WARPS = 8;
    constexpr int OC_TILE = 8;

    // Keep grid.x moderate; persistent tiling in-kernel handles remaining tiles.
    // Choose grid_x based on Cout and N to balance occupancy without overlaunch.
    int grid_x = (Cout + OC_TILE - 1) / OC_TILE;
    if (grid_x > 64) grid_x = 64; // cap to reduce launch overhead; persistence will cover all tiles
    if (grid_x < 1) grid_x = 1;

    dim3 grid((unsigned)grid_x, (unsigned)N, 1);
    int block = WARPS * 32;
    size_t shmem = (size_t)OC_TILE * (size_t)Cin * sizeof(float);

    head_gap_gemv_bn_relu_fused_tiled<OC_TILE, WARPS><<<grid, block, shmem>>>(
        x.data_ptr<float>(),
        w2d.data_ptr<float>(),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, H, W, Cout
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_bn_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor alpha,
    torch::Tensor beta,
    int64_t stride,
    int64_t pad,
    bool do_relu
);

torch::Tensor head_conv_bn_relu_gap_forward_cuda(
    torch::Tensor x,
    torch::Tensor w2d,
    torch::Tensor alpha,
    torch::Tensor beta,
    bool do_relu
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_efficientnetb1_conv_bn_relu_headgap_v7_persistent",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_bn_relu_forward_cuda", "head_conv_bn_relu_gap_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)

class _BNFoldCache:
    _cache = {}

    @classmethod
    def get(cls, bn: nn.BatchNorm2d, device: torch.device, dtype: torch.dtype):
        key = (id(bn), device.index if device.type == "cuda" else -1, int(dtype))
        entry = cls._cache.get(key, None)
        if entry is not None:
            return entry
        with torch.no_grad():
            gamma = bn.weight.detach().to(device=device, dtype=dtype)
            beta = bn.bias.detach().to(device=device, dtype=dtype)
            mean = bn.running_mean.detach().to(device=device, dtype=dtype)
            var = bn.running_var.detach().to(device=device, dtype=dtype)
            invstd = torch.rsqrt(var + bn.eps)
            alpha = (gamma * invstd).contiguous()
            beta2 = (beta - mean * alpha).contiguous()
        cls._cache[key] = (alpha, beta2)
        return alpha, beta2

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        can_fuse = (
            x.is_cuda and x.dtype == torch.float32 and (not self.training) and
            self.conv1.weight.is_cuda and self.conv2.weight.is_cuda
        )

        if can_fuse:
            x = x.contiguous()
            a1, b1 = _BNFoldCache.get(self.bn1, x.device, x.dtype)
            x = custom_ops_lib.conv_bn_relu_forward_cuda(
                x,
                self.conv1.weight.contiguous(),
                a1, b1,
                2, 1,
                True
            )
        else:
            x = F.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        if can_fuse and x.is_contiguous():
            a2, b2 = _BNFoldCache.get(self.bn2, x.device, x.dtype)
            x = custom_ops_lib.head_conv_bn_relu_gap_forward_cuda(
                x,
                self.conv2.weight.view(self.conv2.out_channels, self.conv2.in_channels).contiguous(),
                a2, b2,
                True
            )
        else:
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        x = self.fc(x)
        return x