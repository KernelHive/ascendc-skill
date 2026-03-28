import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- Custom CUDA/C++ extension: BN(inference)+tanh+maxpool2d(2,2,s2)+GroupNorm ----

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static inline __device__ float tanh_approx(float x) {
    return tanhf(x);
}

__global__ void bntanh_maxpool_kernel(
    const float* __restrict__ x,   // [N,C,H,W]
    float* __restrict__ y,         // [N,C,Hp,Wp] where Hp=H/2, Wp=W/2
    const float* __restrict__ bn_gamma,
    const float* __restrict__ bn_beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int N, int C, int H, int W,
    float eps)
{
    int Hp = H >> 1;
    int Wp = W >> 1;
    int64_t total = (int64_t)N * C * Hp * Wp;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total; idx += stride) {
        int64_t t = idx;
        int wp = (int)(t % Wp); t /= Wp;
        int hp = (int)(t % Hp); t /= Hp;
        int c  = (int)(t % C);  t /= C;
        int n  = (int)t;

        float mu = running_mean[c];
        float var = running_var[c];
        float inv = rsqrtf(var + eps);
        float g = bn_gamma[c];
        float b = bn_beta[c];

        int h0 = hp * 2;
        int w0 = wp * 2;

        // read four points from input
        int64_t base = ((int64_t)n * C + c) * (int64_t)H * W;

        float v00 = x[base + (int64_t)(h0 + 0) * W + (w0 + 0)];
        float v01 = x[base + (int64_t)(h0 + 0) * W + (w0 + 1)];
        float v10 = x[base + (int64_t)(h0 + 1) * W + (w0 + 0)];
        float v11 = x[base + (int64_t)(h0 + 1) * W + (w0 + 1)];

        // BN affine -> tanh
        v00 = tanh_approx((v00 - mu) * inv * g + b);
        v01 = tanh_approx((v01 - mu) * inv * g + b);
        v10 = tanh_approx((v10 - mu) * inv * g + b);
        v11 = tanh_approx((v11 - mu) * inv * g + b);

        float m0 = fmaxf(v00, v01);
        float m1 = fmaxf(v10, v11);
        float mp = fmaxf(m0, m1);

        y[idx] = mp;
    }
}

__global__ void groupnorm_stats_kernel(
    const float* __restrict__ x,   // [N,C,Hp,Wp] contiguous
    float* __restrict__ mean,      // [N*G]
    float* __restrict__ rstd,      // [N*G]
    int N, int C, int HxW, int G, float eps)
{
    // one block per (n,g)
    int idx = blockIdx.x;
    int n = idx / G;
    int g = idx - n * G;
    int Cg = C / G;
    int D = Cg * HxW;

    const float* x_ptr = x + ((n * C + g * Cg) * HxW);

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x_ptr[i];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float sh_sum[256];
    __shared__ float sh_sumsq[256];
    int t = threadIdx.x;
    sh_sum[t] = sum;
    sh_sumsq[t] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_sum[t] += sh_sum[t + stride];
            sh_sumsq[t] += sh_sumsq[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        float m = sh_sum[0] / (float)D;
        float var = sh_sumsq[0] / (float)D - m * m;
        float inv = rsqrtf(var + eps);
        mean[idx] = m;
        rstd[idx] = inv;
    }
}

__global__ void groupnorm_affine_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int HxW, int G)
{
    int idx = blockIdx.x;
    int n = idx / G;
    int g = idx - n * G;
    int Cg = C / G;
    int D = Cg * HxW;

    float m = mean[idx];
    float inv = rstd[idx];

    const float* x_ptr = x + ((n * C + g * Cg) * HxW);
    float* y_ptr = y + ((n * C + g * Cg) * HxW);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        int c_in_group = i / HxW;
        int c = g * Cg + c_in_group;
        float v = x_ptr[i];
        float nv = (v - m) * inv;
        float out = nv * gamma[c] + beta[c];
        y_ptr[i] = out;
    }
}

torch::Tensor bntanh_maxpool_groupnorm_forward_cuda(
    torch::Tensor x,               // [N,C,H,W]
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double bn_eps,
    torch::Tensor gn_gamma,
    torch::Tensor gn_beta,
    int64_t num_groups,
    double gn_eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");

    TORCH_CHECK(bn_gamma.is_cuda() && bn_beta.is_cuda(), "bn gamma/beta must be CUDA");
    TORCH_CHECK(running_mean.is_cuda() && running_var.is_cuda(), "running stats must be CUDA");
    TORCH_CHECK(gn_gamma.is_cuda() && gn_beta.is_cuda(), "gn gamma/beta must be CUDA");

    TORCH_CHECK(bn_gamma.dtype() == torch::kFloat32 && bn_beta.dtype() == torch::kFloat32, "bn params must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32 && running_var.dtype() == torch::kFloat32, "running stats must be float32");
    TORCH_CHECK(gn_gamma.dtype() == torch::kFloat32 && gn_beta.dtype() == torch::kFloat32, "gn params must be float32");

    TORCH_CHECK(bn_gamma.is_contiguous() && bn_beta.is_contiguous(), "bn params must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous() && running_var.is_contiguous(), "running stats must be contiguous");
    TORCH_CHECK(gn_gamma.is_contiguous() && gn_beta.is_contiguous(), "gn params must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "H and W must be divisible by 2 for 2x2 s2 maxpool");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    int64_t Hp = H / 2;
    int64_t Wp = W / 2;

    auto pooled = torch::empty({N, C, Hp, Wp}, x.options());

    // BN+tanh+maxpool
    {
        int threads = 256;
        int64_t total = N * C * Hp * Wp;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = blocks > 65535 ? 65535 : blocks;

        bntanh_maxpool_kernel<<<blocks, threads>>>(
            (const float*)x.data_ptr<float>(),
            (float*)pooled.data_ptr<float>(),
            (const float*)bn_gamma.data_ptr<float>(),
            (const float*)bn_beta.data_ptr<float>(),
            (const float*)running_mean.data_ptr<float>(),
            (const float*)running_var.data_ptr<float>(),
            (int)N, (int)C, (int)H, (int)W,
            (float)bn_eps
        );
    }

    // GroupNorm (2-pass)
    auto out = torch::empty_like(pooled);
    auto mean = torch::empty({N * num_groups}, x.options());
    auto rstd = torch::empty({N * num_groups}, x.options());

    int HxW = (int)(Hp * Wp);
    int blocks_g = (int)(N * num_groups);
    int threads_g = 256;

    groupnorm_stats_kernel<<<blocks_g, threads_g>>>(
        (const float*)pooled.data_ptr<float>(),
        (float*)mean.data_ptr<float>(),
        (float*)rstd.data_ptr<float>(),
        (int)N, (int)C, (int)HxW, (int)num_groups, (float)gn_eps
    );

    groupnorm_affine_kernel<<<blocks_g, threads_g>>>(
        (const float*)pooled.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (const float*)mean.data_ptr<float>(),
        (const float*)rstd.data_ptr<float>(),
        (const float*)gn_gamma.data_ptr<float>(),
        (const float*)gn_beta.data_ptr<float>(),
        (int)N, (int)C, (int)HxW, (int)num_groups
    );

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>

torch::Tensor bntanh_maxpool_groupnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double bn_eps,
    torch::Tensor gn_gamma,
    torch::Tensor gn_beta,
    int64_t num_groups,
    double gn_eps);
"""

custom_ops_lib = load_inline(
    name="custom_bntanh_maxpool_gn_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["bntanh_maxpool_groupnorm_forward_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    ConvTranspose2d + fused (BatchNorm2d inference + tanh + MaxPool2d(2,2) + GroupNorm) via custom CUDA.
    Notes:
      - Fused op matches BatchNorm in eval/inference mode using running stats.
      - Supports float32, contiguous NCHW.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        # Keep modules so weights/state_dict behavior remains standard
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=int(num_groups), num_channels=out_channels)

        self.num_groups = int(num_groups)
        self.bn_eps = float(self.batch_norm.eps)
        self.gn_eps = float(self.group_norm.eps)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)

        # Fused path assumes BN uses running stats (eval behavior).
        # If model is in training, fall back to PyTorch ops for correctness.
        if self.training:
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.group_norm(x)
            return x

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        # Ensure parameters are on same device and contiguous
        bn_w = self.batch_norm.weight
        bn_b = self.batch_norm.bias
        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var

        gn_w = self.group_norm.weight
        gn_b = self.group_norm.bias

        dev = x.device
        if bn_w.device != dev:
            bn_w = bn_w.to(dev)
            bn_b = bn_b.to(dev)
            rm = rm.to(dev)
            rv = rv.to(dev)
        if gn_w.device != dev:
            gn_w = gn_w.to(dev)
            gn_b = gn_b.to(dev)

        return self.custom_ops_lib.bntanh_maxpool_groupnorm_forward_cuda(
            x,
            bn_w.contiguous(),
            bn_b.contiguous(),
            rm.contiguous(),
            rv.contiguous(),
            float(self.bn_eps),
            gn_w.contiguous(),
            gn_b.contiguous(),
            int(self.num_groups),
            float(self.gn_eps),
        )