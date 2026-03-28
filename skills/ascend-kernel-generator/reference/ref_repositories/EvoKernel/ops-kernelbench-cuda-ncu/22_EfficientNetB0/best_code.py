import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA extension:
#  - Generic fused conv3x3 + BN(infer) + ReLU (NCHW FP32)
#  - Generic fused conv1x1 + BN(infer) + ReLU (NCHW FP32)
#  - Head-specialized fused conv1x1 (Cin=320,Cout=1280,H=W=7,stride=1) with
#    OC×spatial tiling + shared-memory input staging + float4 loads
# -------------------------

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

__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

// ------------------------------
// Generic conv3x3 + BN(infer) + ReLU, NCHW FP32
// ------------------------------
__global__ __launch_bounds__(256, 2)
void conv2d_bn_relu_fwd_nchw_k3(
    const float* __restrict__ x,      // [N,Cin,H,W]
    const float* __restrict__ w,      // [Cout,Cin,3,3]
    const float* __restrict__ alpha,  // [Cout]
    const float* __restrict__ beta,   // [Cout]
    float* __restrict__ y,            // [N,Cout,Hout,Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int stride, int pad
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;

    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
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

        #pragma unroll 1
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
                    float wv = w[w_ic_base + ky * 3 + kx];
                    acc = fmaf(xv, wv, acc);
                }
            }
        }

        float a = __ldg(alpha + oc);
        float b = __ldg(beta + oc);
        float v = relu_f(fmaf(acc, a, b));
        y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = v;
    }
}

// ------------------------------
// Generic conv1x1 + BN(infer) + ReLU, NCHW FP32 (scalar)
// ------------------------------
__global__ __launch_bounds__(256, 2)
void conv2d_bn_relu_fwd_nchw_k1_scalar(
    const float* __restrict__ x,      // [N,Cin,H,W]
    const float* __restrict__ w,      // [Cout,Cin]
    const float* __restrict__ alpha,  // [Cout]
    const float* __restrict__ beta,   // [Cout]
    float* __restrict__ y,            // [N,Cout,H,W]
    int N, int Cin, int H, int W,
    int Cout
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int M = H * W;
    int total = N * Cout * M;

    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        int m  = idx % M;         // spatial
        int t1 = idx / M;
        int oc = t1 % Cout;
        int n  = t1 / Cout;

        float acc = 0.0f;
        const float* xptr = x + n * Cin * M + m; // x[n, :, m]
        const float* wptr = w + oc * Cin;

        #pragma unroll 1
        for (int ic = 0; ic < Cin; ++ic) {
            acc = fmaf(xptr[ic * M], wptr[ic], acc);
        }

        float a = __ldg(alpha + oc);
        float b = __ldg(beta + oc);
        float v = relu_f(fmaf(acc, a, b));
        y[(n * Cout + oc) * M + m] = v;
    }
}

// ------------------------------
// Head-specialized conv1x1 + BN(infer) + ReLU for:
//   Cin=320, Cout=1280, H=W=7, stride=1, pad=0, N arbitrary
//
// Tiling:
//   - OC tile = 8
//   - Spatial tile = 16 (m positions in [0..48])
// Mapping:
//   blockIdx.x = spatial tile index
//   blockIdx.y = oc tile index
//   blockIdx.z = batch n
// Threads:
//   256 threads = 8 (oc) * 16 (spatial) * 2 (lanes) ; each output computed by 2 threads reduction
// Shared memory:
//   smem_x[SP][Cin] staged once (vectorized float4 loads)
//
// This increases reuse: each x value reused across 8 output channels.
// ------------------------------
__global__ __launch_bounds__(256, 2)
void conv1x1_head_320_1280_7x7_bn_relu(
    const float* __restrict__ x,      // [N,320,7,7] contiguous NCHW
    const float* __restrict__ w,      // [1280,320] contiguous
    const float* __restrict__ alpha,  // [1280]
    const float* __restrict__ beta,   // [1280]
    float* __restrict__ y             // [N,1280,7,7]
) {
    constexpr int Cin = 320;
    constexpr int Cout = 1280;
    constexpr int H = 7;
    constexpr int W = 7;
    constexpr int M = H * W; // 49
    constexpr int SP = 16;   // spatial tile
    constexpr int OC = 8;    // output channel tile
    constexpr int LANES = 2; // 2 threads per output for reduction
    constexpr int VEC = 4;   // float4
    constexpr int Cin4 = Cin / VEC; // 80

    int m_tile = (int)blockIdx.x; // 0..ceil(49/16)-1 => 0..3
    int oc_tile = (int)blockIdx.y; // 0..(1280/8)-1 => 0..159
    int n = (int)blockIdx.z;

    int tid = (int)threadIdx.x; // 0..255
    int lane = tid & (LANES - 1);         // 0..1
    int t = tid >> 1;                     // 0..127
    int sp = t & (SP - 1);                // 0..15
    int oc_in_tile = (t >> 4) & (OC - 1); // 0..7
    int oc = oc_tile * OC + oc_in_tile;
    int m = m_tile * SP + sp;

    extern __shared__ float smem[];
    // layout: [SP][Cin] = SP*Cin floats
    float* smem_x = smem;

    // Cooperative load of input tile into shared memory (float4 over Cin)
    // total float4 elements per tile = SP * Cin4 = 16*80 = 1280
    int total4 = SP * Cin4;

    // base pointers
    const float* x_base = x + n * Cin * M; // x[n] flattened to [Cin, M]
    // We store x as x[c*M + m]
    for (int i = tid; i < total4; i += 256) {
        int sp_i = i / Cin4;
        int c4 = i - sp_i * Cin4;   // 0..79
        int m_i = m_tile * SP + sp_i;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((unsigned)m_i < (unsigned)M) {
            const float* src = x_base + (c4 * VEC) * M + m_i;
            // load 4 channels at same spatial m_i (stride M between channels)
            v.x = src[0 * M];
            v.y = src[1 * M];
            v.z = src[2 * M];
            v.w = src[3 * M];
        }
        // store to smem_x[sp_i, c4*4 .. c4*4+3]
        float* dst = smem_x + sp_i * Cin + c4 * VEC;
        dst[0] = v.x;
        dst[1] = v.y;
        dst[2] = v.z;
        dst[3] = v.w;
    }
    __syncthreads();

    if ((unsigned)m >= (unsigned)M) return;

    // 2-way reduction over Cin: each lane handles half of Cin4 chunks
    float acc = 0.0f;
    const float* w_oc = w + oc * Cin;

    int c4_start = lane * (Cin4 / LANES);
    int c4_end = (lane + 1) * (Cin4 / LANES);

    // Unroll a bit for ILP without blowing registers
    #pragma unroll 4
    for (int c4 = c4_start; c4 < c4_end; ++c4) {
        // load x from smem, 4 floats contiguous
        const float* xv = smem_x + sp * Cin + c4 * VEC;
        float x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];

        // load w via read-only cache; w is contiguous along Cin
        const float* wv = w_oc + c4 * VEC;
        float w0 = __ldg(wv + 0);
        float w1 = __ldg(wv + 1);
        float w2 = __ldg(wv + 2);
        float w3 = __ldg(wv + 3);

        acc = fmaf(x0, w0, acc);
        acc = fmaf(x1, w1, acc);
        acc = fmaf(x2, w2, acc);
        acc = fmaf(x3, w3, acc);
    }

    // Reduce within the 2-lane group for each output (oc_in_tile, sp)
    // Use warp shuffles: lanes are adjacent (0/1)
    unsigned mask = 0xffffffffu;
    float acc_other = __shfl_xor_sync(mask, acc, 1, 32);
    acc += acc_other;

    if (lane == 0) {
        float a = __ldg(alpha + oc);
        float b = __ldg(beta + oc);
        float outv = relu_f(fmaf(acc, a, b));
        y[(n * Cout + oc) * M + m] = outv;
    }
}

static inline int clamp_grid_1d(int grid) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int maxGrid = prop.multiProcessorCount * 20;
    return grid > maxGrid ? maxGrid : grid;
}

torch::Tensor conv_bn_relu_forward_cuda(
    torch::Tensor x,            // [N,Cin,H,W]
    torch::Tensor w,            // [Cout,Cin,Kh,Kw]
    torch::Tensor alpha,        // [Cout]
    torch::Tensor beta,         // [Cout]
    int64_t stride,
    int64_t pad,
    int64_t kh,
    int64_t kw
) {
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(alpha); CHECK_CUDA(beta);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(beta);
    CHECK_FLOAT(x); CHECK_FLOAT(w); CHECK_FLOAT(alpha); CHECK_FLOAT(beta);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cout,Cin,Kh,Kw]");
    TORCH_CHECK(alpha.dim() == 1 && beta.dim() == 1, "alpha/beta must be 1D");
    TORCH_CHECK(kh == w.size(2) && kw == w.size(3), "kh/kw must match weight");
    TORCH_CHECK((kh == 3 && kw == 3) || (kh == 1 && kw == 1), "only 3x3 or 1x1 supported");
    TORCH_CHECK(stride == 1 || stride == 2, "stride must be 1 or 2");
    if (kh == 3) TORCH_CHECK(pad == 1, "3x3 expects pad=1");
    if (kh == 1) TORCH_CHECK(pad == 0, "1x1 expects pad=0");

    int N = (int)x.size(0), Cin = (int)x.size(1), Hin = (int)x.size(2), Win = (int)x.size(3);
    int Cout = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == Cin, "weight Cin mismatch");
    TORCH_CHECK((int)alpha.numel() == Cout && (int)beta.numel() == Cout, "alpha/beta must be [Cout]");

    int Hout = (Hin + 2 * (int)pad - (int)kh) / (int)stride + 1;
    int Wout = (Win + 2 * (int)pad - (int)kw) / (int)stride + 1;

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    // Head specialized path: exact match (N,320,7,7) -> (N,1280,7,7), k=1,s=1,p=0
    if (kh == 1 && kw == 1 && pad == 0 && stride == 1 &&
        Cin == 320 && Cout == 1280 && Hin == 7 && Win == 7 && Hout == 7 && Wout == 7) {

        auto w2 = w.view({Cout, Cin}).contiguous();
        dim3 block(256);
        // grid: x = ceil(49/16)=4, y = 1280/8=160, z = N
        dim3 grid(4, 160, (unsigned)N);
        size_t smem_bytes = (size_t)(16 * 320) * sizeof(float); // SP*Cin
        conv1x1_head_320_1280_7x7_bn_relu<<<grid, block, smem_bytes>>>(
            x.data_ptr<float>(),
            w2.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>()
        );
        return y;
    }

    // Generic fused fallbacks
    if (kh == 3) {
        int total = N * Cout * Hout * Wout;
        int block = 256;
        int grid = clamp_grid_1d((total + block - 1) / block);
        conv2d_bn_relu_fwd_nchw_k3<<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Hout, Wout,
            (int)stride, (int)pad
        );
        return y;
    } else {
        TORCH_CHECK(stride == 1, "generic 1x1 only supports stride=1 in this extension");
        TORCH_CHECK(Hout == Hin && Wout == Win, "1x1 stride=1 expects Hout==Hin and Wout==Win");
        auto w2 = w.view({Cout, Cin}).contiguous();

        int M = Hout * Wout;
        int total = N * Cout * M;
        int block = 256;
        int grid = clamp_grid_1d((total + block - 1) / block);

        conv2d_bn_relu_fwd_nchw_k1_scalar<<<grid, block>>>(
            x.data_ptr<float>(),
            w2.data_ptr<float>(),
            alpha.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, Cin, Hout, Wout, Cout
        );
        return y;
    }
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
    int64_t kh,
    int64_t kw
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_efficientnetb0_convbnrelu_v4_headtiling",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_bn_relu_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)

# -------------------------
# Original blocks (kept identical for correctness)
# -------------------------

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x

# -------------------------
# Model using custom CUDA fused ops (conv+bn+relu for stem/head)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, num_classes)

        # Cache BN inference params for eval mode
        self._bn_cache = {}

    @staticmethod
    def _bn_alpha_beta_infer(bn: nn.BatchNorm2d, device, dtype):
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
        mean = bn.running_mean.to(device=device, dtype=dtype)
        var = bn.running_var.to(device=device, dtype=dtype)
        invstd = torch.rsqrt(var + bn.eps)
        alpha = gamma * invstd
        beta2 = beta - mean * alpha
        return alpha.contiguous(), beta2.contiguous()

    def _get_bn_ab_cached(self, bn: nn.BatchNorm2d, x: torch.Tensor):
        key = (
            id(bn),
            x.device,
            x.dtype,
            int(bn.running_mean.data_ptr()) if bn.running_mean.is_cuda else -1,
            int(bn.running_var.data_ptr()) if bn.running_var.is_cuda else -1,
            int(bn.weight.data_ptr()) if (bn.weight is not None and bn.weight.is_cuda) else -1,
            int(bn.bias.data_ptr()) if (bn.bias is not None and bn.bias.is_cuda) else -1,
        )
        cached = self._bn_cache.get(key)
        if cached is None:
            cached = self._bn_alpha_beta_infer(bn, x.device, x.dtype)
            self._bn_cache[key] = cached
        return cached

    def _conv_bn_relu_fused(self, x, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        if self.training:
            return F.relu(bn(conv(x)))

        if not (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()):
            return F.relu(bn(conv(x)))

        if not (conv.weight.is_cuda and bn.weight.is_cuda and bn.running_mean.is_cuda and bn.running_var.is_cuda):
            return F.relu(bn(conv(x)))

        w = conv.weight.contiguous()
        alpha, beta = self._get_bn_ab_cached(bn, x)

        kh = int(w.size(2))
        kw = int(w.size(3))
        stride = int(conv.stride[0])
        pad = int(conv.padding[0])

        if not ((kh == 3 and kw == 3 and pad == 1) or (kh == 1 and kw == 1 and pad == 0)):
            return F.relu(bn(conv(x)))

        return custom_ops_lib.conv_bn_relu_forward_cuda(x, w, alpha, beta, stride, pad, kh, kw)

    def forward(self, x):
        x = x.contiguous()
        x = self._conv_bn_relu_fused(x, self.conv1, self.bn1)
        x = self.blocks(x)
        x = self._conv_bn_relu_fused(x, self.conv2, self.bn2)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x