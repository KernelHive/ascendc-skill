import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif

// ---------------- Generic fallback ----------------
__global__ void conv_transpose3d_forward_generic(
    const float* __restrict__ x,      // [N, Cin, Di, Hi, Wi]
    const float* __restrict__ w,      // [Cin, Cout, kD, kH, kW]
    float* __restrict__ y,            // [N, Cout, Do, Ho, Wo]
    int N, int Cin, int Di, int Hi, int Wi,
    int Cout, int kD, int kH, int kW,
    int stride, int padding, int dilation,
    int Do, int Ho, int Wo
) {
    int64_t spatial = (int64_t)N * Do * Ho * Wo;
    int64_t sidx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int oc = (int)blockIdx.y;
    if (sidx >= spatial || oc >= Cout) return;

    int64_t t = sidx;
    int ow = (int)(t % Wo); t /= Wo;
    int oh = (int)(t % Ho); t /= Ho;
    int od = (int)(t % Do); t /= Do;
    int n  = (int)t;

    float acc = 0.0f;

    for (int ic = 0; ic < Cin; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int num_d = od + padding - kd * dilation;
            if (num_d % stride != 0) continue;
            int id = num_d / stride;
            if ((unsigned)id >= (unsigned)Di) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int num_h = oh + padding - kh * dilation;
                if (num_h % stride != 0) continue;
                int ih = num_h / stride;
                if ((unsigned)ih >= (unsigned)Hi) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int num_w = ow + padding - kw * dilation;
                    if (num_w % stride != 0) continue;
                    int iw = num_w / stride;
                    if ((unsigned)iw >= (unsigned)Wi) continue;

                    int64_t x_off = (((((int64_t)n * Cin + ic) * Di + id) * Hi + ih) * Wi + iw);
                    int64_t w_off = (((((int64_t)ic * Cout + oc) * kD + kd) * kH + kh) * kW + kw);

                    acc = fmaf(LDG(x + x_off), LDG(w + w_off), acc);
                }
            }
        }
    }

    int64_t y_off = (((((int64_t)n * Cout + oc) * Do + od) * Ho + oh) * Wo + ow);
    y[y_off] = acc;
}

// ---------------- Specialized (k=3,s=2,d=2,p=1): odd-lattice + OC tiling + weights in shared ----------------
//
// For k=3,s=2,d=2,p=1,dil=2:
// num = o + 1 - 2*k; stride=2 => num%2==0 iff o is odd. If o even => no contributions.
// For odd o: id = (o+1)/2 - k, k in {0,1,2}.
//
// We compute only odd outputs by mapping lattice indices: od = 2*ld+1, etc.
//
template<int OC_TILE>
__global__ __launch_bounds__(128, 2)
void conv_transpose3d_forward_k3_s2_d2_p1_tiled(
    const float* __restrict__ x,   // [N,Cin,Di,Hi,Wi]
    const float* __restrict__ w,   // [Cin,Cout,3,3,3]
    float* __restrict__ y,         // [N,Cout,Do,Ho,Wo]
    int N, int Cin, int Di, int Hi, int Wi,
    int Cout,
    int Do, int Ho, int Wo
) {
    // odd-lattice sizes
    int Do2 = Do >> 1;
    int Ho2 = Ho >> 1;
    int Wo2 = Wo >> 1;

    int oc_base = (int)blockIdx.y * OC_TILE;
    if (oc_base >= Cout) return;

    // Each thread computes two ow positions in odd-lattice: lw0 and lw1 = lw0+1
    // Flattened lattice index per "ow-pair":
    // pair_count = N * Do2 * Ho2 * ceil(Wo2/2)
    int Wo2_pairs = (Wo2 + 1) >> 1;
    int64_t pairs_total = (int64_t)N * Do2 * Ho2 * Wo2_pairs;

    int tid = threadIdx.x;
    int64_t pair_linear0 = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t pair_stride = (int64_t)gridDim.x * blockDim.x;

    extern __shared__ float smem[];
    // layout: [OC_TILE][27]
    float* wsh = smem;

    for (int64_t pair_linear = pair_linear0; pair_linear < pairs_total; pair_linear += pair_stride) {
        int64_t t = pair_linear;
        int lpw = (int)(t % Wo2_pairs); t /= Wo2_pairs;
        int lh  = (int)(t % Ho2); t /= Ho2;
        int ld  = (int)(t % Do2); t /= Do2;
        int n   = (int)t;

        int lw0 = (lpw << 1);
        int lw1 = lw0 + 1;

        // actual output coords (odd)
        int od = (ld << 1) + 1;
        int oh = (lh << 1) + 1;
        int ow0 = (lw0 << 1) + 1;
        int ow1 = (lw1 << 1) + 1;

        // base input indices for odd outputs:
        int bd = (od + 1) >> 1; // = ld+1
        int bh = (oh + 1) >> 1; // = lh+1
        int bw0 = (ow0 + 1) >> 1; // = lw0+1
        int bw1 = (ow1 + 1) >> 1; // = lw1+1

        // per-oc accumulators (small OC_TILE)
        float acc0[OC_TILE];
        float acc1[OC_TILE];
        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }

        // x base pointer for batch n
        const float* __restrict__ x_n = x + ((int64_t)n * Cin) * (int64_t)Di * Hi * Wi;

        // for each input channel: stage weights for OC tile and accumulate
        for (int ic = 0; ic < Cin; ++ic) {
            const float* __restrict__ w_ic = w + ((int64_t)ic * Cout + oc_base) * 27;

            // cooperative load of OC_TILE*27 weights
            int loads = OC_TILE * 27;
            for (int i = tid; i < loads; i += blockDim.x) {
                int oc_t = i / 27;
                int k = i - oc_t * 27;
                int oc = oc_base + oc_t;
                float val = 0.0f;
                if (oc < Cout) val = LDG(w_ic + (int64_t)oc_t * 27 + k);
                wsh[i] = val;
            }
            __syncthreads();

            const float* __restrict__ x_ic = x_n + ((int64_t)ic * Di) * (int64_t)Hi * Wi;

            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                int id = bd - kd;
                if ((unsigned)id >= (unsigned)Di) continue;

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = bh - kh;
                    if ((unsigned)ih >= (unsigned)Hi) continue;

                    const float* __restrict__ x_row =
                        x_ic + ((int64_t)id * Hi + ih) * (int64_t)Wi;

                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw0 = bw0 - kw;
                        int iw1 = bw1 - kw;

                        float xv0 = 0.0f, xv1 = 0.0f;
                        if ((unsigned)iw0 < (unsigned)Wi) xv0 = LDG(x_row + iw0);
                        if (lw1 < Wo2 && (unsigned)iw1 < (unsigned)Wi) xv1 = LDG(x_row + iw1);

                        int wi = kd * 9 + kh * 3 + kw; // 0..26
                        // accumulate OC_TILE outputs
                        #pragma unroll
                        for (int j = 0; j < OC_TILE; ++j) {
                            float wv = wsh[j * 27 + wi];
                            acc0[j] = fmaf(xv0, wv, acc0[j]);
                            acc1[j] = fmaf(xv1, wv, acc1[j]);
                        }
                    }
                }
            }

            __syncthreads();
        }

        // store outputs for OC tile at (od,oh,ow0/ow1)
        int64_t y_base0 = (((((int64_t)n * Cout + oc_base) * Do + od) * Ho + oh) * (int64_t)Wo + ow0);
        int64_t y_base1 = y_base0 + 2; // next odd ow is +2 in full output
        #pragma unroll
        for (int j = 0; j < OC_TILE; ++j) {
            int oc = oc_base + j;
            if (oc < Cout) {
                y[y_base0 + (int64_t)j * Do * (int64_t)Ho * Wo] = acc0[j];
                if (lw1 < Wo2) {
                    y[y_base1 + (int64_t)j * Do * (int64_t)Ho * Wo] = acc1[j];
                }
            }
        }
    }
}

torch::Tensor conv_transpose3d_square_input_square_kernel_padded_dilated_strided_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);

    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,Cin,Di,Hi,Wi)");
    TORCH_CHECK(w.dim() == 5, "w must be 5D (Cin,Cout,kD,kH,kW)");

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Di = (int)x.size(2);
    int Hi = (int)x.size(3);
    int Wi = (int)x.size(4);

    TORCH_CHECK((int)w.size(0) == Cin, "w.size(0) must equal Cin");
    int Cout = (int)w.size(1);
    int kD = (int)w.size(2);
    int kH = (int)w.size(3);
    int kW = (int)w.size(4);

    TORCH_CHECK(kD == kH && kH == kW, "kernel must be cubic (k,k,k)");
    TORCH_CHECK(stride > 0 && dilation > 0, "stride and dilation must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    int Do = (int)((Di - 1) * (int)stride - 2 * (int)padding + (int)dilation * (kD - 1) + 1);
    int Ho = (int)((Hi - 1) * (int)stride - 2 * (int)padding + (int)dilation * (kH - 1) + 1);
    int Wo = (int)((Wi - 1) * (int)stride - 2 * (int)padding + (int)dilation * (kW - 1) + 1);

    TORCH_CHECK(Do > 0 && Ho > 0 && Wo > 0, "computed output size must be positive");

    auto y = torch::zeros({N, Cout, Do, Ho, Wo}, x.options());

    const float* xp = x.data_ptr<float>();
    const float* wp = w.data_ptr<float>();
    float* yp = y.data_ptr<float>();

    // Specialized path: (k=3,s=2,d=2,p=1)
    if (kD == 3 && (int)stride == 2 && (int)dilation == 2 && (int)padding == 1) {
        constexpr int OC_TILE = 8;
        constexpr int THREADS = 128;

        int Do2 = Do >> 1;
        int Ho2 = Ho >> 1;
        int Wo2 = Wo >> 1;
        int Wo2_pairs = (Wo2 + 1) >> 1;

        int64_t pairs_total = (int64_t)N * Do2 * Ho2 * Wo2_pairs;

        // x-grid dimension: enough blocks to cover pairs_total; cap to avoid huge grids
        int grid_x = (int)((pairs_total + THREADS - 1) / THREADS);
        grid_x = max(1, min(grid_x, 32768)); // practical cap
        dim3 grid(grid_x, (unsigned)((Cout + OC_TILE - 1) / OC_TILE), 1);
        dim3 block(THREADS, 1, 1);

        size_t shmem = (size_t)(OC_TILE * 27) * sizeof(float);

        conv_transpose3d_forward_k3_s2_d2_p1_tiled<OC_TILE><<<grid, block, shmem>>>(
            xp, wp, yp, N, Cin, Di, Hi, Wi, Cout, Do, Ho, Wo
        );
    } else {
        // Generic: one oc per grid.y
        int64_t spatial = (int64_t)N * Do * Ho * Wo;
        constexpr int threads = 128;
        dim3 block(threads);
        dim3 grid((unsigned)((spatial + threads - 1) / threads), (unsigned)Cout, 1);
        conv_transpose3d_forward_generic<<<grid, block>>>(
            xp, wp, yp,
            N, Cin, Di, Hi, Wi,
            Cout, kD, kH, kW,
            (int)stride, (int)padding, (int)dilation,
            Do, Ho, Wo
        );
    }

    return y;
}
"""

cpp_source = r"""
torch::Tensor conv_transpose3d_square_input_square_kernel_padded_dilated_strided_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose3d_sq_opt2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_square_input_square_kernel_padded_dilated_strided_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    ConvTranspose3d via a custom CUDA kernel for cubic kernels and uniform stride/padding/dilation.
    Optimized fast-path for (k=3, stride=2, padding=1, dilation=2). Bias is not supported.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if bias:
            raise ValueError("This custom kernel version does not support bias; set bias=False.")
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

        w = torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = nn.Parameter(w)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew only supports CUDA tensors.")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda:
            w = w.to(device=x.device)
        if not w.is_contiguous():
            w = w.contiguous()

        return self.custom_ops_lib.conv_transpose3d_square_input_square_kernel_padded_dilated_strided_cuda(
            x, w, self.stride, self.padding, self.dilation
        )