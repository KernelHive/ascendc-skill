import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static inline __host__ __device__ int div_up_int(int a, int b) { return (a + b - 1) / b; }

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

__device__ __forceinline__ float ro_ld(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// 64KB constant memory => 16384 floats
__constant__ float W_CONST[16384];

static bool try_copy_w_to_const(const float* w_dev, int64_t total_floats) {
    if (total_floats <= 0 || total_floats > 16384) return false;
    cudaError_t err = cudaMemcpyToSymbol(W_CONST, w_dev, (size_t)total_floats * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    return err == cudaSuccess;
}

// ------------------------
// General gather kernel (kept for correctness)
// ------------------------
__global__ __launch_bounds__(256, 2)
void conv_transpose2d_forward_gather_kernel(
    const float* __restrict__ x,        // [N, Cin, Hin, Win]
    const float* __restrict__ w,        // [Cin, CoutPerG, k, k]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int CoutPerG, int k, int stride, int padding,
    int groups,
    int Hout, int Wout,
    int use_const_w
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Hout * (int64_t)Wout;
    if ((int64_t)tid >= total) return;

    int t = tid;
    int ow = t % Wout; t /= Wout;
    int oh = t % Hout; t /= Hout;
    int oc = t % Cout; t /= Cout;
    int n  = t;

    int g = oc / CoutPerG;
    int ocg = oc - g * CoutPerG;

    int CinPerG = Cin / groups;
    int ic_start = g * CinPerG;
    int ic_end   = ic_start + CinPerG;

    float acc = 0.0f;
    if (bias) acc = ro_ld(bias + oc);

    const int HW = Hin * Win;
    const int kk = k * k;

#pragma unroll
    for (int kh = 0; kh < 7; ++kh) {
        if (kh >= k) break;
        int ih_num = oh + padding - kh;
        if (ih_num % stride) continue;
        int ih = ih_num / stride;
        if ((unsigned)ih >= (unsigned)Hin) continue;

#pragma unroll
        for (int kw = 0; kw < 7; ++kw) {
            if (kw >= k) break;
            int iw_num = ow + padding - kw;
            if (iw_num % stride) continue;
            int iw = iw_num / stride;
            if ((unsigned)iw >= (unsigned)Win) continue;

            int x_sp = ih * Win + iw;
            int x_base = (n * Cin) * HW + x_sp;
            int w_k = kh * k + kw;

            for (int ic = ic_start; ic < ic_end; ++ic) {
                float xv = ro_ld(x + x_base + ic * HW);
                int w_off = ((ic * CoutPerG + ocg) * kk + w_k);
                float wv = use_const_w ? W_CONST[w_off] : ro_ld(w + w_off);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    y[tid] = acc;
}

// ------------------------
// Improved fast path: stride=1, padding=0, output_padding=0, groups=1, k=3
// - Stage BOTH x tile and weights slab (for OC_BLOCK output channels) into shared memory.
// - Double-buffer (ping-pong) shared memory for x+weights to reduce barrier overhead.
// - OC blocking increased to improve reuse.
// ------------------------
template<int OC_BLOCK>
__global__ __launch_bounds__(256, 2)
void conv_t2d_s1p0_k3_tiled_oc_sharedw_pingpong_kernel(
    const float* __restrict__ x,    // [N,Cin,H,W]
    const float* __restrict__ w,    // [Cin,Cout,3,3] (ignored if use_const_w)
    const float* __restrict__ bias, // [Cout] or nullptr
    float* __restrict__ out,        // [N,Cout,Hout,Wout]
    int N, int Cin, int H, int W,
    int Cout, int Hout, int Wout,
    int use_const_w
) {
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int K = 3;
    constexpr int KAREA = 9;
    constexpr int SH_W = TILE_W + (K - 1); // 18
    constexpr int SH_H = TILE_H + (K - 1); // 18
    constexpr int SH_X_SIZE = SH_W * SH_H; // 324
    constexpr int SH_W_SIZE = OC_BLOCK * KAREA;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int lane = ty * TILE_W + tx; // 0..255

    int ow0 = (int)blockIdx.x * TILE_W;
    int oh0 = (int)blockIdx.y * TILE_H;

    int oc_blocks = div_up_int(Cout, OC_BLOCK);
    int z = (int)blockIdx.z;
    int n = z / oc_blocks;
    int ocb = z - n * oc_blocks;
    int oc0 = ocb * OC_BLOCK;
    if (n >= N) return;

    extern __shared__ float smem[];
    // layout: [2][x_tile + w_slab]
    float* buf0 = smem;
    float* buf1 = smem + (SH_X_SIZE + SH_W_SIZE);

    const int ih_base = oh0 - (K - 1);
    const int iw_base = ow0 - (K - 1);

    const int HW = H * W;
    const int outHW = Hout * Wout;
    const int x_n_base = n * Cin * HW;

    float acc[OC_BLOCK];
#pragma unroll
    for (int i = 0; i < OC_BLOCK; ++i) acc[i] = 0.f;

    // Helper lambda-ish via macro (no captures)
#define LOAD_TO_BUF(BUF_PTR, IC) do { \
        float* sx = (BUF_PTR); \
        float* sw = (BUF_PTR) + SH_X_SIZE; \
        const int x_base = x_n_base + (IC) * HW; \
        /* x tile */ \
        for (int idx = lane; idx < SH_X_SIZE; idx += 256) { \
            int shy = idx / SH_W; \
            int shx = idx - shy * SH_W; \
            int ih = ih_base + shy; \
            int iw = iw_base + shx; \
            float v = 0.f; \
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) { \
                v = ro_ld(x + x_base + ih * W + iw); \
            } \
            sx[idx] = v; \
        } \
        /* weights slab for this ic and oc-block */ \
        for (int idx = lane; idx < SH_W_SIZE; idx += 256) { \
            int oci = idx / KAREA; \
            int kidx = idx - oci * KAREA; \
            int oc = oc0 + oci; \
            float v = 0.f; \
            if (oc < Cout) { \
                int w_off = ((IC) * Cout + oc) * KAREA + kidx; \
                v = use_const_w ? W_CONST[w_off] : ro_ld(w + w_off); \
            } \
            sw[idx] = v; \
        } \
    } while(0)

    // Preload ic=0 into buf0
    if (Cin > 0) {
        LOAD_TO_BUF(buf0, 0);
    }
    __syncthreads();

    for (int ic = 0; ic < Cin; ++ic) {
        float* cur = (ic & 1) ? buf1 : buf0;
        float* nxt = (ic & 1) ? buf0 : buf1;

        // Preload next ic in parallel with compute (but must not overwrite "cur")
        if (ic + 1 < Cin) {
            LOAD_TO_BUF(nxt, ic + 1);
        }

        // Compute on current buffer
        int oh = oh0 + ty;
        int ow = ow0 + tx;
        if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
            float* sx = cur;
            float* sw = cur + SH_X_SIZE;

            int sh_oy = (K - 1) + ty; // 2 + ty
            int sh_ox = (K - 1) + tx; // 2 + tx

#pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int sh_y = sh_oy - kh;
                int sh_row = sh_y * SH_W;

                float xv0 = sx[sh_row + (sh_ox - 0)];
                float xv1 = sx[sh_row + (sh_ox - 1)];
                float xv2 = sx[sh_row + (sh_ox - 2)];

                int kbase = kh * 3;

#pragma unroll
                for (int oci = 0; oci < OC_BLOCK; ++oci) {
                    const float* wptr = sw + oci * KAREA + kbase;
                    float sum = 0.f;
                    sum = fmaf(xv0, wptr[0], sum);
                    sum = fmaf(xv1, wptr[1], sum);
                    sum = fmaf(xv2, wptr[2], sum);
                    acc[oci] += sum;
                }
            }
        }

        __syncthreads(); // ensures nxt loads complete before it becomes cur
    }

#undef LOAD_TO_BUF

    // Store
    int oh = oh0 + ty;
    int ow = ow0 + tx;
    if ((unsigned)oh < (unsigned)Hout && (unsigned)ow < (unsigned)Wout) {
        int out_sp = oh * Wout + ow;
        int out_n_base = n * Cout * outHW;
#pragma unroll
        for (int oci = 0; oci < OC_BLOCK; ++oci) {
            int oc = oc0 + oci;
            if (oc >= Cout) continue;
            float v = acc[oci];
            if (bias) v += ro_ld(bias + oc);
            out[out_n_base + oc * outHW + out_sp] = v;
        }
    }
}

torch::Tensor conv_transpose2d_square_input_square_kernel_cuda(
    torch::Tensor x,      // [N,Cin,H,W]
    torch::Tensor w,      // [Cin, Cout/groups, k, k]
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be [Cin, Cout/groups, k, k]");
    TORCH_CHECK(stride >= 1, "stride must be >=1");
    TORCH_CHECK(groups >= 1, "groups must be >=1");

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    const int N   = (int)x_c.size(0);
    const int Cin = (int)x_c.size(1);
    const int Hin = (int)x_c.size(2);
    const int Win = (int)x_c.size(3);
    TORCH_CHECK(Hin == Win, "square input expected (H==W)");

    const int kH = (int)w_c.size(2);
    const int kW = (int)w_c.size(3);
    TORCH_CHECK(kH == kW, "square kernel expected");
    const int k = kH;
    TORCH_CHECK(k >= 1 && k <= 7, "supported k in [1,7]");

    TORCH_CHECK((int)w_c.size(0) == Cin, "w.shape[0] must equal Cin");
    TORCH_CHECK(Cin % (int)groups == 0, "Cin must be divisible by groups");

    const int CoutPerG = (int)w_c.size(1);
    const int Cout = CoutPerG * (int)groups;

    const int Hout = (int)((Hin - 1) * (int)stride - 2 * (int)padding + k + (int)output_padding);
    const int Wout = (int)((Win - 1) * (int)stride - 2 * (int)padding + k + (int)output_padding);
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");
    TORCH_CHECK(Hout == Wout, "square output expected");

    const float* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        auto b = bias_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK((int)b.numel() == Cout, "bias must be [Cout]");
        bias_c = b.contiguous();
        bias_ptr = bias_c.data_ptr<float>();
    }

    auto y = torch::empty({N, Cout, Hout, Wout}, x_c.options());

    // Specialized fast path: k=3, stride=1, padding=0, outpad=0, groups=1
    if (stride == 1 && padding == 0 && output_padding == 0 && groups == 1 && k == 3) {
        int use_const_w = 0;
        int64_t total_w_floats = (int64_t)Cin * (int64_t)Cout * 9;
        if (try_copy_w_to_const(w_c.data_ptr<float>(), total_w_floats)) use_const_w = 1;

        constexpr int TILE_W = 16;
        constexpr int TILE_H = 16;
        constexpr int OC_BLOCK = 8;

        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid(div_up_int(Wout, TILE_W),
                  div_up_int(Hout, TILE_H),
                  (unsigned int)(N * div_up_int(Cout, OC_BLOCK)));

        constexpr int SH_X_SIZE = (TILE_W + 2) * (TILE_H + 2); // 18*18=324
        constexpr int SH_W_SIZE = OC_BLOCK * 9;
        size_t shmem = (size_t)2 * (size_t)(SH_X_SIZE + SH_W_SIZE) * sizeof(float);

        conv_t2d_s1p0_k3_tiled_oc_sharedw_pingpong_kernel<OC_BLOCK><<<grid, block, shmem>>>(
            x_c.data_ptr<float>(),
            w_c.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Hout, Wout,
            use_const_w
        );
        return y;
    }

    // General path (optionally const weights)
    int use_const_w = 0;
    {
        int64_t kk = (int64_t)k * (int64_t)k;
        int64_t total_w_floats = (int64_t)Cin * (int64_t)CoutPerG * kk;
        if (try_copy_w_to_const(w_c.data_ptr<float>(), total_w_floats)) use_const_w = 1;
    }

    const int threads = 256;
    int64_t total = (int64_t)N * (int64_t)Cout * (int64_t)Hout * (int64_t)Wout;
    const int blocks = (int)((total + threads - 1) / threads);

    conv_transpose2d_forward_gather_kernel<<<blocks, threads>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, CoutPerG, k, (int)stride, (int)padding,
        (int)groups,
        Hout, Wout,
        use_const_w
    );
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_transpose2d_square_input_square_kernel_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv_transpose2d_square_input_square_kernel_fastk3_sharedw_pp_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose2d_square_input_square_kernel_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Drop-in replacement for nn.ConvTranspose2d using an optimized CUDA forward kernel.

    Fast path:
      - float32 CUDA contiguous NCHW
      - square input/kernel
      - stride=1, padding=0, output_padding=0, groups=1, kernel_size=3

    Other configurations fall back to a general gather kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.custom_ops = custom_ops_lib
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.groups = int(groups)

        w = torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.zeros(out_channels, dtype=torch.float32)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_ops.conv_transpose2d_square_input_square_kernel_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )