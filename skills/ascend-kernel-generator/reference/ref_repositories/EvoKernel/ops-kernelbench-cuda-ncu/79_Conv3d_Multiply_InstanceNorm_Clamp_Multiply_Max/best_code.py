import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

#define COUT 16
#define CIN 3
#define K 3
#define K3 27

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    return v;
}

__device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

__global__ __launch_bounds__(128, 4)
void conv3d_y1_sum_sumsq_atomic_k3_cin3_cout16_nocnt(
    const float* __restrict__ x,   // [N,3,D,H,W]
    const float* __restrict__ w,   // [16,3,3,3,3]
    const float* __restrict__ b,   // [16] or nullptr
    const float* __restrict__ m,   // [16]
    float* __restrict__ sum,       // [N,16]
    float* __restrict__ sumsq,     // [N,16]
    int N, int D, int H, int W,
    int D0, int H0, int W0
){
    int n = (int)blockIdx.y;
    if ((unsigned)n >= (unsigned)N) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;

    int HW = H * W;
    int DHW = D * HW;
    int outDHW0 = D0 * H0 * W0;

    float lsum[COUT];
    float lsq[COUT];
    #pragma unroll
    for (int co = 0; co < COUT; ++co) { lsum[co] = 0.0f; lsq[co] = 0.0f; }

    int linear0 = (int)(blockIdx.x * blockDim.x + tid);
    int stride = (int)(gridDim.x * blockDim.x);

    const int x_n_base = n * CIN * DHW;

    for (int t = linear0; t < outDHW0; t += stride) {
        int tmp = t;
        int ow0 = tmp % W0; tmp /= W0;
        int oh0 = tmp % H0; tmp /= H0;
        int od0 = tmp;

        int id0 = od0, ih0 = oh0, iw0 = ow0;

        float acc[COUT];
        #pragma unroll
        for (int co = 0; co < COUT; ++co) acc[co] = (b != nullptr) ? ldgf(b + co) : 0.0f;

        #pragma unroll
        for (int ci = 0; ci < CIN; ++ci) {
            const float* __restrict__ x_ci = x + x_n_base + ci * DHW;

            const float* x_d0 = x_ci + (id0 + 0) * HW + (ih0 + 0) * W + iw0;
            const float* x_d1 = x_ci + (id0 + 1) * HW + (ih0 + 0) * W + iw0;
            const float* x_d2 = x_ci + (id0 + 2) * HW + (ih0 + 0) * W + iw0;

            float xv[K3];
            // kd0
            xv[0] = ldgf(x_d0 + 0); xv[1] = ldgf(x_d0 + 1); xv[2] = ldgf(x_d0 + 2);
            const float* x_h1 = x_d0 + W;
            xv[3] = ldgf(x_h1 + 0); xv[4] = ldgf(x_h1 + 1); xv[5] = ldgf(x_h1 + 2);
            const float* x_h2 = x_h1 + W;
            xv[6] = ldgf(x_h2 + 0); xv[7] = ldgf(x_h2 + 1); xv[8] = ldgf(x_h2 + 2);
            // kd1
            xv[9] = ldgf(x_d1 + 0); xv[10] = ldgf(x_d1 + 1); xv[11] = ldgf(x_d1 + 2);
            x_h1 = x_d1 + W;
            xv[12] = ldgf(x_h1 + 0); xv[13] = ldgf(x_h1 + 1); xv[14] = ldgf(x_h1 + 2);
            x_h2 = x_h1 + W;
            xv[15] = ldgf(x_h2 + 0); xv[16] = ldgf(x_h2 + 1); xv[17] = ldgf(x_h2 + 2);
            // kd2
            xv[18] = ldgf(x_d2 + 0); xv[19] = ldgf(x_d2 + 1); xv[20] = ldgf(x_d2 + 2);
            x_h1 = x_d2 + W;
            xv[21] = ldgf(x_h1 + 0); xv[22] = ldgf(x_h1 + 1); xv[23] = ldgf(x_h1 + 2);
            x_h2 = x_h1 + W;
            xv[24] = ldgf(x_h2 + 0); xv[25] = ldgf(x_h2 + 1); xv[26] = ldgf(x_h2 + 2);

            #pragma unroll
            for (int co = 0; co < COUT; ++co) {
                const float* __restrict__ w_ci_co = w + co * (CIN * K3) + ci * K3;
                float a = acc[co];
                #pragma unroll
                for (int k = 0; k < K3; ++k) {
                    a = fmaf(xv[k], ldgf(w_ci_co + k), a);
                }
                acc[co] = a;
            }
        }

        #pragma unroll
        for (int co = 0; co < COUT; ++co) {
            float mc = ldgf(m + co);
            float y1 = acc[co] * mc;
            lsum[co] += y1;
            lsq[co] = fmaf(y1, y1, lsq[co]);
        }
    }

    #pragma unroll
    for (int co = 0; co < COUT; ++co) {
        float s = warp_reduce_sum(lsum[co]);
        float q = warp_reduce_sum(lsq[co]);
        if (lane == 0) {
            int idx = n * COUT + co;
            atomicAdd(sum + idx, s);
            atomicAdd(sumsq + idx, q);
        }
    }
}

__global__ void finalize_mean_invstd(
    const float* __restrict__ sum,     // [N,16]
    const float* __restrict__ sumsq,   // [N,16]
    float* __restrict__ mean,          // [N,16]
    float* __restrict__ invstd,        // [N,16]
    float eps,
    int N,
    int count // D0*H0*W0
){
    int idx = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int total = N * COUT;
    if (idx >= total) return;
    float invc = 1.0f / (float)count;
    float mu = sum[idx] * invc;
    float var = sumsq[idx] * invc - mu * mu;
    var = fmaxf(var, 0.0f);
    mean[idx] = mu;
    invstd[idx] = rsqrtf(var + eps);
}

__global__ __launch_bounds__(128, 4)
void conv3d_instnorm_clamp_mul_max_k3_cin3_cout16_group4(
    const float* __restrict__ x,     // [N,3,D,H,W]
    const float* __restrict__ w,     // [16,3,3,3,3]
    const float* __restrict__ b,     // [16] or nullptr
    const float* __restrict__ m,     // [16]
    const float* __restrict__ mean,  // [N,16]
    const float* __restrict__ invstd,// [N,16]
    float* __restrict__ out,         // [N,D0,H0,W0]
    float clamp_min, float clamp_max,
    int N, int D, int H, int W,
    int D0, int H0, int W0
){
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * D0 * H0 * W0;
    if (idx >= total) return;

    int tmp = (int)idx;
    int ow0 = tmp % W0; tmp /= W0;
    int oh0 = tmp % H0; tmp /= H0;
    int od0 = tmp % D0; tmp /= D0;
    int n = tmp;

    int HW = H * W;
    int DHW = D * HW;
    int id0 = od0, ih0 = oh0, iw0 = ow0;
    const int x_n_base = n * CIN * DHW;

    // Load input patch into registers once (27 values per ci) to reuse across 4-channel groups.
    // This is still heavy but avoids reloading x for each channel.
    float xv0[K3], xv1[K3], xv2[K3];
    {
        const float* __restrict__ x0 = x + x_n_base + 0 * DHW;
        const float* __restrict__ x1 = x + x_n_base + 1 * DHW;
        const float* __restrict__ x2 = x + x_n_base + 2 * DHW;

        const float* x0d0 = x0 + (id0 + 0) * HW + (ih0 + 0) * W + iw0;
        const float* x0d1 = x0 + (id0 + 1) * HW + (ih0 + 0) * W + iw0;
        const float* x0d2 = x0 + (id0 + 2) * HW + (ih0 + 0) * W + iw0;

        const float* x1d0 = x1 + (id0 + 0) * HW + (ih0 + 0) * W + iw0;
        const float* x1d1 = x1 + (id0 + 1) * HW + (ih0 + 0) * W + iw0;
        const float* x1d2 = x1 + (id0 + 2) * HW + (ih0 + 0) * W + iw0;

        const float* x2d0 = x2 + (id0 + 0) * HW + (ih0 + 0) * W + iw0;
        const float* x2d1 = x2 + (id0 + 1) * HW + (ih0 + 0) * W + iw0;
        const float* x2d2 = x2 + (id0 + 2) * HW + (ih0 + 0) * W + iw0;

        auto load27 = [&](const float* d0, const float* d1, const float* d2, float* xv) {
            xv[0]=ldgf(d0+0); xv[1]=ldgf(d0+1); xv[2]=ldgf(d0+2);
            const float* h1=d0+W;
            xv[3]=ldgf(h1+0); xv[4]=ldgf(h1+1); xv[5]=ldgf(h1+2);
            const float* h2=h1+W;
            xv[6]=ldgf(h2+0); xv[7]=ldgf(h2+1); xv[8]=ldgf(h2+2);

            xv[9]=ldgf(d1+0); xv[10]=ldgf(d1+1); xv[11]=ldgf(d1+2);
            h1=d1+W;
            xv[12]=ldgf(h1+0); xv[13]=ldgf(h1+1); xv[14]=ldgf(h1+2);
            h2=h1+W;
            xv[15]=ldgf(h2+0); xv[16]=ldgf(h2+1); xv[17]=ldgf(h2+2);

            xv[18]=ldgf(d2+0); xv[19]=ldgf(d2+1); xv[20]=ldgf(d2+2);
            h1=d2+W;
            xv[21]=ldgf(h1+0); xv[22]=ldgf(h1+1); xv[23]=ldgf(h1+2);
            h2=h1+W;
            xv[24]=ldgf(h2+0); xv[25]=ldgf(h2+1); xv[26]=ldgf(h2+2);
        };

        load27(x0d0, x0d1, x0d2, xv0);
        load27(x1d0, x1d1, x1d2, xv1);
        load27(x2d0, x2d1, x2d2, xv2);
    }

    float best = -INFINITY;
    int base_nc = n * COUT;

    // process output channels in groups of 4 to reduce register pressure
    #pragma unroll
    for (int co_base = 0; co_base < COUT; co_base += 4) {
        float acc0 = (b != nullptr) ? ldgf(b + co_base + 0) : 0.0f;
        float acc1 = (b != nullptr) ? ldgf(b + co_base + 1) : 0.0f;
        float acc2 = (b != nullptr) ? ldgf(b + co_base + 2) : 0.0f;
        float acc3 = (b != nullptr) ? ldgf(b + co_base + 3) : 0.0f;

        // ci = 0
        {
            const float* __restrict__ w0 = w + (co_base + 0) * (CIN * K3) + 0 * K3;
            const float* __restrict__ w1 = w + (co_base + 1) * (CIN * K3) + 0 * K3;
            const float* __restrict__ w2 = w + (co_base + 2) * (CIN * K3) + 0 * K3;
            const float* __restrict__ w3 = w + (co_base + 3) * (CIN * K3) + 0 * K3;
            #pragma unroll
            for (int k = 0; k < K3; ++k) {
                float xvk = xv0[k];
                acc0 = fmaf(xvk, ldgf(w0 + k), acc0);
                acc1 = fmaf(xvk, ldgf(w1 + k), acc1);
                acc2 = fmaf(xvk, ldgf(w2 + k), acc2);
                acc3 = fmaf(xvk, ldgf(w3 + k), acc3);
            }
        }
        // ci = 1
        {
            const float* __restrict__ w0 = w + (co_base + 0) * (CIN * K3) + 1 * K3;
            const float* __restrict__ w1 = w + (co_base + 1) * (CIN * K3) + 1 * K3;
            const float* __restrict__ w2 = w + (co_base + 2) * (CIN * K3) + 1 * K3;
            const float* __restrict__ w3 = w + (co_base + 3) * (CIN * K3) + 1 * K3;
            #pragma unroll
            for (int k = 0; k < K3; ++k) {
                float xvk = xv1[k];
                acc0 = fmaf(xvk, ldgf(w0 + k), acc0);
                acc1 = fmaf(xvk, ldgf(w1 + k), acc1);
                acc2 = fmaf(xvk, ldgf(w2 + k), acc2);
                acc3 = fmaf(xvk, ldgf(w3 + k), acc3);
            }
        }
        // ci = 2
        {
            const float* __restrict__ w0 = w + (co_base + 0) * (CIN * K3) + 2 * K3;
            const float* __restrict__ w1 = w + (co_base + 1) * (CIN * K3) + 2 * K3;
            const float* __restrict__ w2 = w + (co_base + 2) * (CIN * K3) + 2 * K3;
            const float* __restrict__ w3 = w + (co_base + 3) * (CIN * K3) + 2 * K3;
            #pragma unroll
            for (int k = 0; k < K3; ++k) {
                float xvk = xv2[k];
                acc0 = fmaf(xvk, ldgf(w0 + k), acc0);
                acc1 = fmaf(xvk, ldgf(w1 + k), acc1);
                acc2 = fmaf(xvk, ldgf(w2 + k), acc2);
                acc3 = fmaf(xvk, ldgf(w3 + k), acc3);
            }
        }

        float m0 = ldgf(m + co_base + 0);
        float m1 = ldgf(m + co_base + 1);
        float m2 = ldgf(m + co_base + 2);
        float m3 = ldgf(m + co_base + 3);

        float y10 = acc0 * m0;
        float y11 = acc1 * m1;
        float y12 = acc2 * m2;
        float y13 = acc3 * m3;

        float mu0 = ldgf(mean + base_nc + co_base + 0);
        float mu1 = ldgf(mean + base_nc + co_base + 1);
        float mu2 = ldgf(mean + base_nc + co_base + 2);
        float mu3 = ldgf(mean + base_nc + co_base + 3);

        float is0 = ldgf(invstd + base_nc + co_base + 0);
        float is1 = ldgf(invstd + base_nc + co_base + 1);
        float is2 = ldgf(invstd + base_nc + co_base + 2);
        float is3 = ldgf(invstd + base_nc + co_base + 3);

        float y20 = (y10 - mu0) * is0;
        float y21 = (y11 - mu1) * is1;
        float y22 = (y12 - mu2) * is2;
        float y23 = (y13 - mu3) * is3;

        float y30 = clampf(y20, clamp_min, clamp_max);
        float y31 = clampf(y21, clamp_min, clamp_max);
        float y32 = clampf(y22, clamp_min, clamp_max);
        float y33 = clampf(y23, clamp_min, clamp_max);

        float y40 = y30 * m0;
        float y41 = y31 * m1;
        float y42 = y32 * m2;
        float y43 = y33 * m3;

        best = fmaxf(best, y40);
        best = fmaxf(best, y41);
        best = fmaxf(best, y42);
        best = fmaxf(best, y43);
    }

    out[idx] = best;
}

torch::Tensor conv3d_multiply_instance_norm_clamp_multiply_max_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor m,
    double clamp_min,
    double clamp_max,
    double eps
){
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && m.is_cuda(), "x,w,m must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32 && m.dtype() == torch::kFloat32, "float32 only");
    TORCH_CHECK(x.dim() == 5, "x must be [N,Cin,D,H,W]");
    TORCH_CHECK(w.dim() == 5, "w must be [Cout,Cin,K,K,K]");
    TORCH_CHECK(w.size(1) == CIN, "specialized path requires Cin=3");
    TORCH_CHECK(w.size(0) == COUT, "specialized path requires Cout=16");
    TORCH_CHECK(w.size(2) == K && w.size(3) == K && w.size(4) == K, "specialized path requires K=3");
    TORCH_CHECK(m.dim() == 1 && m.numel() == COUT, "m must be [16]");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto mc = m.contiguous();

    const int N = (int)xc.size(0);
    const int D = (int)xc.size(2);
    const int H = (int)xc.size(3);
    const int W = (int)xc.size(4);

    const int D0 = D - K + 1;
    const int H0 = H - K + 1;
    const int W0 = W - K + 1;
    TORCH_CHECK(D0 > 0 && H0 > 0 && W0 > 0, "invalid output shape");

    torch::Tensor bc;
    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        TORCH_CHECK(b.is_cuda() && b.dtype() == torch::kFloat32, "bias must be CUDA float32");
        TORCH_CHECK(b.dim() == 1 && b.numel() == COUT, "bias must be [16]");
        bc = b.contiguous();
        bptr = bc.data_ptr<float>();
    }

    auto sum = torch::zeros({N, COUT}, xc.options());
    auto sumsq = torch::zeros({N, COUT}, xc.options());

    int outDHW0 = D0 * H0 * W0;

    // stats kernel: 2D grid (x over output positions, y over batch)
    int threads1 = 128;
    int blocks1x = (outDHW0 + threads1 - 1) / threads1;
    dim3 grid1((unsigned)blocks1x, (unsigned)N, 1);
    conv3d_y1_sum_sumsq_atomic_k3_cin3_cout16_nocnt<<<grid1, threads1, 0, stream>>>(
        xc.data_ptr<float>(),
        wc.data_ptr<float>(),
        bptr,
        mc.data_ptr<float>(),
        sum.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        N, D, H, W, D0, H0, W0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // finalize mean/invstd
    auto mean = torch::empty({N, COUT}, xc.options());
    auto invstd = torch::empty({N, COUT}, xc.options());
    int threadsF = 128;
    int totalF = N * COUT;
    int blocksF = (totalF + threadsF - 1) / threadsF;
    finalize_mean_invstd<<<blocksF, threadsF, 0, stream>>>(
        sum.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        (float)eps,
        N,
        outDHW0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // output kernel
    auto out = torch::empty({N, D0, H0, W0}, xc.options());
    int64_t total = (int64_t)N * D0 * H0 * W0;
    int threads2 = 128;
    int blocks2 = (int)((total + threads2 - 1) / threads2);
    conv3d_instnorm_clamp_mul_max_k3_cin3_cout16_group4<<<blocks2, threads2, 0, stream>>>(
        xc.data_ptr<float>(),
        wc.data_ptr<float>(),
        bptr,
        mc.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        out.data_ptr<float>(),
        (float)clamp_min, (float)clamp_max,
        N, D, H, W, D0, H0, W0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor conv3d_multiply_instance_norm_clamp_multiply_max_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor m,
    double clamp_min,
    double clamp_max,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_conv3d_mul_instnorm_clamp_mul_max_opt7",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3d_multiply_instance_norm_clamp_multiply_max_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized specialized implementation for:
      Cin=3, Cout=16, K=3, stride=1, padding=0
      multiplier_shape=(16,1,1,1)
      InstanceNorm3d affine=False, track_running_stats=False

    Expects to be used as a drop-in module with its own parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        if int(in_channels) != 3 or int(out_channels) != 16 or int(kernel_size) != 3:
            raise ValueError("This optimized kernel is specialized for in_channels=3,out_channels=16,kernel_size=3.")
        if tuple(multiplier_shape) != (int(out_channels), 1, 1, 1):
            raise ValueError("multiplier_shape must be (out_channels,1,1,1).")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.eps = 1e-5

        w = torch.empty(self.out_channels, self.in_channels, 3, 3, 3, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        b = torch.empty(self.out_channels, dtype=torch.float32)
        fan_in = self.in_channels * 27
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(b, -bound, bound)
        self.bias = nn.Parameter(b)

        self.multiplier = nn.Parameter(torch.randn(multiplier_shape, dtype=torch.float32))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        w = self.weight
        b = self.bias
        m4 = self.multiplier

        if not w.is_cuda:
            w = w.cuda()
        if not b.is_cuda:
            b = b.cuda()
        if not m4.is_cuda:
            m4 = m4.cuda()

        w = w.contiguous()
        b = b.contiguous()
        m = m4.view(self.out_channels).contiguous()

        return self.custom_ops_lib.conv3d_multiply_instance_norm_clamp_multiply_max_forward_cuda(
            x, w, b, m, self.clamp_min, self.clamp_max, self.eps
        )