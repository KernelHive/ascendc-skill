import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    __shared__ float shared[8]; // up to 256 threads -> 8 warps
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

// ------------------------------------------------------------
// ConvTranspose3d gather specialized for stride=2, pad=1, K=3
// Computes y(n,co,od,oh,ow) without atomics.
// ------------------------------------------------------------
static __forceinline__ __device__ float convT3d_gather_s2p1k3_one(
    const float* __restrict__ x, // [N,Cin,Din,Hin,Win]
    const float* __restrict__ w, // [Cin,Cout,3,3,3]
    int n, int co, int od, int oh, int ow,
    int Cin, int Cout,
    int Din, int Hin, int Win
){
    int64_t in_hw  = (int64_t)Hin * Win;
    int64_t in_dhw = (int64_t)Din * in_hw;
    int64_t x_n_base = (int64_t)n * (int64_t)Cin * in_dhw;

    int64_t w_ci_stride = (int64_t)Cout * 27;
    int64_t w_co_stride = 27;

    float acc = 0.0f;

    #pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        int td = od + 1 - kd;
        if (td & 1) continue;
        int id = td >> 1;
        if ((unsigned)id >= (unsigned)Din) continue;

        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int th = oh + 1 - kh;
            if (th & 1) continue;
            int ih = th >> 1;
            if ((unsigned)ih >= (unsigned)Hin) continue;

            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int tw = ow + 1 - kw;
                if (tw & 1) continue;
                int iw = tw >> 1;
                if ((unsigned)iw >= (unsigned)Win) continue;

                int64_t x_sp = (int64_t)id * in_hw + (int64_t)ih * Win + iw;
                int64_t kidx = ((int64_t)kd * 3 + kh) * 3 + kw;

                #pragma unroll 1
                for (int ci = 0; ci < Cin; ++ci) {
                    float xv = __ldg(x + x_n_base + (int64_t)ci * in_dhw + x_sp);
                    float wv = __ldg(w + (int64_t)ci * w_ci_stride + (int64_t)co * w_co_stride + kidx);
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }
    return acc;
}

// ------------------------------------------------------------
// Kernel A: Compute BN sums and sumsqs over (y + bias) without writing y.
// Produces global arrays sum[C], sumsq[C].
// Grid: (C, N, tiles over D*H*W)
// Each block accumulates a chunk and atomically adds once per block.
// ------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void convT_bn_sums_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ sum,    // [Cout]
    float* __restrict__ sumsq,  // [Cout]
    int N, int Cin, int Cout,
    int Din, int Hin, int Win,
    int Dout, int Hout, int Wout,
    int64_t HWout
){
    int co = (int)blockIdx.x;
    int n  = (int)blockIdx.y;
    int tile = (int)blockIdx.z;
    if (co >= Cout || n >= N) return;

    // Tile over spatial linear index s in [0, HWout*Dout)
    int64_t spatial = (int64_t)Dout * HWout;
    int64_t tile_size = (int64_t)blockDim.x;
    int64_t s0 = (int64_t)tile * tile_size;

    float lsum = 0.0f;
    float lsumsq = 0.0f;
    float b = __ldg(bias + co);

    int64_t s = s0 + (int64_t)threadIdx.x;
    if (s < spatial) {
        int od = (int)(s / HWout);
        int64_t rem = s - (int64_t)od * HWout;
        int oh = (int)(rem / Wout);
        int ow = (int)(rem - (int64_t)oh * Wout);

        float yv = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, ow, Cin, Cout, Din, Hin, Win);
        yv += b;
        lsum += yv;
        lsumsq = fmaf(yv, yv, lsumsq);
    }

    float bsum = block_reduce_sum(lsum);
    float bsumsq = block_reduce_sum(lsumsq);

    if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0) {
        atomicAdd(sum + co, bsum);
        atomicAdd(sumsq + co, bsumsq);
    }
}

// ------------------------------------------------------------
// Kernel B: finalize mean/var from sum/sumsq
// mean = sum/M; var = sumsq/M - mean^2
// ------------------------------------------------------------
__global__ void finalize_mean_var_kernel(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float* __restrict__ mean,
    float* __restrict__ var,
    int C, float invM
){
    int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = sum[c];
    float ss = sumsq[c];
    float m = s * invM;
    float v = ss * invM - m * m;
    mean[c] = m;
    var[c] = v;
}

// ------------------------------------------------------------
// Kernel C: Fused (recompute convT) + BN affine + AvgPool(4,4,4) stride 4 (valid).
// Vectorize over ow2 by 4 when W2%4==0 and ow2 aligned.
// ------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void convT_bn_affine_avgpool4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,  // [N,C,D2,H2,W2]
    int N, int Cin, int Cout,
    int Din, int Hin, int Win,
    int Dout, int Hout, int Wout,
    int D2, int H2, int W2,
    float eps,
    int vec4 // 1 if vectorized path enabled
){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * Cout * (int64_t)D2 * H2 * W2;
    if (idx >= total) return;

    int64_t t = idx;
    int ow2 = (int)(t % W2); t /= W2;
    int oh2 = (int)(t % H2); t /= H2;
    int od2 = (int)(t % D2); t /= D2;
    int co  = (int)(t % Cout); t /= Cout;
    int n   = (int)t;

    float b  = __ldg(bias + co);
    float m  = __ldg(mean + co);
    float vv = __ldg(var + co);
    float inv = rsqrtf(vv + eps);
    float g  = __ldg(gamma + co);
    float be = __ldg(beta + co);

    int id0 = od2 * 4;
    int ih0 = oh2 * 4;
    int iw0 = ow2 * 4;

    // Compute average over 64 points, each point is convT gather + bias then BN affine
    float acc = 0.0f;

    // Unroll small loops; keep kw inner-most.
    #pragma unroll 1
    for (int kd = 0; kd < 4; ++kd) {
        int od = id0 + kd;
        #pragma unroll 1
        for (int kh = 0; kh < 4; ++kh) {
            int oh = ih0 + kh;

            // Vectorize over the 4 ow within the pooling window if desired.
            if (vec4) {
                // ow are contiguous; compute 4 outputs for ow=iw0..iw0+3
                float v0 = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, iw0 + 0, Cin, Cout, Din, Hin, Win) + b;
                float v1 = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, iw0 + 1, Cin, Cout, Din, Hin, Win) + b;
                float v2 = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, iw0 + 2, Cin, Cout, Din, Hin, Win) + b;
                float v3 = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, iw0 + 3, Cin, Cout, Din, Hin, Win) + b;

                float y0 = (v0 - m) * inv;
                float y1 = (v1 - m) * inv;
                float y2 = (v2 - m) * inv;
                float y3 = (v3 - m) * inv;

                acc += fmaf(y0, g, be);
                acc += fmaf(y1, g, be);
                acc += fmaf(y2, g, be);
                acc += fmaf(y3, g, be);
            } else {
                #pragma unroll
                for (int kw = 0; kw < 4; ++kw) {
                    int ow = iw0 + kw;
                    float v = convT3d_gather_s2p1k3_one(x, w, n, co, od, oh, ow, Cin, Cout, Din, Hin, Win) + b;
                    float yn = (v - m) * inv;
                    acc += fmaf(yn, g, be);
                }
            }
        }
    }

    out[idx] = acc * (1.0f / 64.0f);
}

torch::Tensor conv_transpose3d_batch_norm_avg_pool_avg_pool_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
){
    CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(b);
    CHECK_CUDA(bn_weight); CHECK_CUDA(bn_bias);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(b);
    CHECK_CONTIGUOUS(bn_weight); CHECK_CONTIGUOUS(bn_bias);
    CHECK_F32(x); CHECK_F32(w); CHECK_F32(b);
    CHECK_F32(bn_weight); CHECK_F32(bn_bias);

    TORCH_CHECK(x.dim() == 5, "x must be [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be [Cin,Cout,3,3,3]");
    TORCH_CHECK(b.dim() == 1, "b must be [Cout]");
    TORCH_CHECK(bn_weight.dim() == 1 && bn_bias.dim() == 1, "bn params must be [Cout]");

    int64_t N   = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Din = x.size(2);
    int64_t Hin = x.size(3);
    int64_t Win = x.size(4);

    TORCH_CHECK(w.size(0) == Cin, "w.size(0) must match Cin");
    int64_t Cout = w.size(1);
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3 && w.size(4) == 3, "optimized path supports K=3 only");
    TORCH_CHECK(b.numel() == Cout, "bias must have Cout elements");
    TORCH_CHECK(bn_weight.numel() == Cout && bn_bias.numel() == Cout, "bn params must have Cout elements");

    // stride=2, pad=1, K=3:
    int64_t Dout = 2 * Din - 1;
    int64_t Hout = 2 * Hin - 1;
    int64_t Wout = 2 * Win - 1;

    TORCH_CHECK(Dout >= 4 && Hout >= 4 && Wout >= 4, "convt output too small");

    int64_t D2 = (Dout - 4) / 4 + 1;
    int64_t H2 = (Hout - 4) / 4 + 1;
    int64_t W2 = (Wout - 4) / 4 + 1;
    TORCH_CHECK(D2 > 0 && H2 > 0 && W2 > 0, "invalid pooled output size");

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

    // A) BN sums/sumsq (from y+bias) without materializing y
    auto sum   = torch::zeros({Cout}, x.options());
    auto sumsq = torch::zeros({Cout}, x.options());

    int64_t HWout = Hout * Wout;
    int64_t spatial = Dout * HWout;
    const int threads = 256;
    int tiles = (int)((spatial + threads - 1) / threads);

    {
        dim3 grid((unsigned)Cout, (unsigned)N, (unsigned)tiles);
        convT_bn_sums_kernel<<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            sum.data_ptr<float>(),
            sumsq.data_ptr<float>(),
            (int)N, (int)Cin, (int)Cout,
            (int)Din, (int)Hin, (int)Win,
            (int)Dout, (int)Hout, (int)Wout,
            (int64_t)HWout
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto mean = torch::empty({Cout}, x.options());
    auto var  = torch::empty({Cout}, x.options());
    {
        float invM = 1.0f / (float)((double)N * (double)Dout * (double)Hout * (double)Wout);
        int blk = 256;
        int grd = (int)((Cout + blk - 1) / blk);
        finalize_mean_var_kernel<<<grd, blk, 0, stream>>>(
            sum.data_ptr<float>(),
            sumsq.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            (int)Cout, invM
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // B) Final fused convT(recompute) + BN affine + pool4
    auto out = torch::empty({N, Cout, D2, H2, W2}, x.options());
    {
        int64_t total = out.numel();
        int blocks = (int)((total + threads - 1) / threads);

        // Enable simple vec4 mode when W2 is multiple of 4 and aligned mapping helps.
        // (Our baseline shapes yield W2=15 -> scalar path; still correct.)
        int vec4 = ((W2 % 4) == 0) ? 1 : 0;

        convT_bn_affine_avgpool4_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)Cout,
            (int)Din, (int)Hin, (int)Win,
            (int)Dout, (int)Hout, (int)Wout,
            (int)D2, (int)H2, (int)W2,
            (float)eps,
            vec4
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose3d_batch_norm_avg_pool_avg_pool_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_convt3d_bn_poolpool_gather_fused_s2p1k3_v2_noy",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_batch_norm_avg_pool_avg_pool_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Custom CUDA replacement for:
      ConvTranspose3d(stride=2,padding=1,kernel_size=3) -> BatchNorm3d(training stats) -> AvgPool3d(2) -> AvgPool3d(2)

    Fast-path constraints:
      - CUDA float32, contiguous
      - kernel_size == 3 (cubic), stride == 2, padding == 1
      - groups=1, dilation=1, output_padding=0
      - BN uses batch statistics; running stats are not updated.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.bias_shape = bias_shape  # signature compatibility

        if self.stride != 2 or self.padding != 1 or self.kernel_size != 3:
            raise ValueError("ModelNew supports stride=2, padding=1, kernel_size=3 only")

        w = torch.empty(self.in_channels, self.out_channels, 3, 3, 3, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))

        self.bn_weight = nn.Parameter(torch.ones(self.out_channels, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        self.eps = 1e-5

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew supports CUDA tensors only")
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        dev = x.device
        w = self.weight if self.weight.device == dev else self.weight.to(dev)
        b = self.bias if self.bias.device == dev else self.bias.to(dev)
        gw = self.bn_weight if self.bn_weight.device == dev else self.bn_weight.to(dev)
        gb = self.bn_bias if self.bn_bias.device == dev else self.bn_bias.to(dev)

        return self.custom_ops.conv_transpose3d_batch_norm_avg_pool_avg_pool_cuda(
            x,
            w.contiguous(),
            b.contiguous(),
            gw.contiguous(),
            gb.contiguous(),
            float(self.eps),
        )