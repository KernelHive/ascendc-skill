import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA extension: fused stem Conv3x3(s2,p1,bias=False) + BN(eval) + ReLU6
# -----------------------------------------------------------------------------

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

#define CUDA_KERNEL_CHECK() do { \
  cudaError_t err = cudaGetLastError(); \
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err)); \
} while(0)

__device__ __forceinline__ float relu6(float x) {
    x = x > 0.0f ? x : 0.0f;
    return x < 6.0f ? x : 6.0f;
}

// Fused: conv2d 3x3 stride=2 pad=1 (bias-free) + BN(inference) + ReLU6, NCHW float32.
// x: [N,Cin,H,W]
// w: [Cout,Cin,3,3] (OIHW)
// bn params: [Cout] gamma,beta,mean,var
__global__ void conv3x3s2p1_bn_relu6_nchw(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    float* __restrict__ y,
    int N, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t1 = idx / Wout;
    int oh = t1 % Hout;
    int t2 = t1 / Hout;
    int oc = t2 % Cout;
    int n  = t2 / Cout;

    // stride=2 pad=1
    int in_y0 = oh * 2 - 1;
    int in_x0 = ow * 2 - 1;

    float acc = 0.0f;
    int w_oc_base = oc * Cin * 9; // 3*3

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

    // BN inference: (acc - mean) * rsqrt(var + eps) * gamma + beta
    float inv_std = rsqrtf(var[oc] + eps);
    float v = (acc - mean[oc]) * inv_std * gamma[oc] + beta[oc];

    // ReLU6
    v = relu6(v);

    y[(((n * Cout + oc) * Hout + oh) * Wout) + ow] = v;
}

torch::Tensor stem_conv_bn_relu6_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps
) {
    CHECK_CUDA(x); CHECK_CUDA(w);
    CHECK_CUDA(gamma); CHECK_CUDA(beta); CHECK_CUDA(mean); CHECK_CUDA(var);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w);
    CHECK_CONTIGUOUS(gamma); CHECK_CONTIGUOUS(beta); CHECK_CONTIGUOUS(mean); CHECK_CONTIGUOUS(var);
    CHECK_FLOAT(x); CHECK_FLOAT(w);
    CHECK_FLOAT(gamma); CHECK_FLOAT(beta); CHECK_FLOAT(mean); CHECK_FLOAT(var);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(w.dim() == 4, "w must be OIHW");
    TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "w must be 3x3");

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);

    int64_t Cout = w.size(0);
    TORCH_CHECK(w.size(1) == Cin, "weight Cin mismatch");

    TORCH_CHECK(gamma.numel() == Cout, "gamma must be [Cout]");
    TORCH_CHECK(beta.numel() == Cout, "beta must be [Cout]");
    TORCH_CHECK(mean.numel() == Cout, "mean must be [Cout]");
    TORCH_CHECK(var.numel() == Cout, "var must be [Cout]");

    // stride=2 pad=1 k=3
    int64_t Hout = (Hin + 2 - 3) / 2 + 1;
    int64_t Wout = (Win + 2 - 3) / 2 + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    int total = (int)(N * Cout * Hout * Wout);
    const int block = 256;
    const int grid = (total + block - 1) / block;

    conv3x3s2p1_bn_relu6_nchw<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        (float)eps,
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Hin, (int)Win,
        (int)Cout, (int)Hout, (int)Wout
    );
    CUDA_KERNEL_CHECK();
    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor stem_conv_bn_relu6_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mobilenetv2_stem",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["stem_conv_bn_relu6_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)


# -----------------------------------------------------------------------------
# ModelNew: preserves module structure; replaces only stem execution in forward
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            layers = []
            if expand_ratio != 1:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            return nn.Sequential(*layers)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Fused stem only in eval mode to match BN semantics (running stats).
        use_fast = (x.is_cuda and x.dtype == torch.float32 and (not self.training))
        if use_fast:
            x = x.contiguous()
            conv0 = self.features[0]
            bn0 = self.features[1]
            use_fast = (
                conv0.weight.is_cuda and
                bn0.weight.is_cuda and bn0.bias.is_cuda and
                bn0.running_mean.is_cuda and bn0.running_var.is_cuda
            )

        if use_fast:
            x = custom_ops_lib.stem_conv_bn_relu6_forward_cuda(
                x,
                self.features[0].weight.contiguous(),
                self.features[1].weight.contiguous(),
                self.features[1].bias.contiguous(),
                self.features[1].running_mean.contiguous(),
                self.features[1].running_var.contiguous(),
                float(self.features[1].eps),
            )
            # Skip the original stem (0..2) since we already did it fused.
            x = self.features[3:](x)
        else:
            x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x