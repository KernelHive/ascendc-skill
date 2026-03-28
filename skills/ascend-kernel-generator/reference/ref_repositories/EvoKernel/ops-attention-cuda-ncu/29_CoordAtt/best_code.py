import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# ---- Custom CUDA extension: fused CoordAtt gating (identity * a_w * a_h) ----
# Supports float32 CUDA tensors (common for inference/benchmark). Falls back to PyTorch if not CUDA/float32.
coord_att_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coord_att_fused_mul_kernel(
    const float* __restrict__ x,    // [N,C,H,W]
    const float* __restrict__ aw,   // [N,C,1,W]
    const float* __restrict__ ah,   // [N,C,H,1]
    float* __restrict__ out,
    int N, int C, int H, int W
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if ((long long)idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int c = tmp % C;
    int n = tmp / C;

    long long x_off  = ((long long)((n * C + c) * H + h) * W + w);
    long long aw_off = ((long long)((n * C + c) * 1 + 0) * W + w); // [N,C,1,W]
    long long ah_off = ((long long)((n * C + c) * H + h) * 1 + 0); // [N,C,H,1]

    out[x_off] = x[x_off] * aw[aw_off] * ah[ah_off];
}

torch::Tensor coord_att_fused_mul_cuda(torch::Tensor x, torch::Tensor aw, torch::Tensor ah) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(aw.is_cuda(), "aw must be a CUDA tensor");
    TORCH_CHECK(ah.is_cuda(), "ah must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(aw.dtype() == torch::kFloat32, "aw must be float32");
    TORCH_CHECK(ah.dtype() == torch::kFloat32, "ah must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(aw.is_contiguous(), "aw must be contiguous");
    TORCH_CHECK(ah.is_contiguous(), "ah must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    TORCH_CHECK(aw.dim() == 4, "aw must be [N,C,1,W]");
    TORCH_CHECK(ah.dim() == 4, "ah must be [N,C,H,1]");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    TORCH_CHECK(aw.size(0) == N && aw.size(1) == C && aw.size(2) == 1 && aw.size(3) == W,
                "aw must have shape [N,C,1,W]");
    TORCH_CHECK(ah.size(0) == N && ah.size(1) == C && ah.size(2) == H && ah.size(3) == 1,
                "ah must have shape [N,C,H,1]");

    auto out = torch::empty_like(x);

    long long total = (long long)N * C * H * W;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    coord_att_fused_mul_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        aw.data_ptr<float>(),
        ah.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}
"""

coord_att_cpp_src = r"""
torch::Tensor coord_att_fused_mul_cuda(torch::Tensor x, torch::Tensor aw, torch::Tensor ah);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_coord_att",
    cpp_sources=coord_att_cpp_src,
    cuda_sources=coord_att_cuda_src,
    functions=["coord_att_fused_mul_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
)


# ---- Original modules (kept) ----
class h_sigmoid(nn.Module):
    """Hard sigmoid activation function."""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard swish activation function."""
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# ---- Optimized model using custom CUDA op for coord_att gating ----
class ModelNew(nn.Module):
    """
    CoordAtt with fused final gating: out = identity * a_w * a_h (single CUDA kernel).
    """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)                          # [N,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # [N,C,W,1] after permute -> [N,C,W,1], but original intends [N,C,W,1] then cat on dim=2

        y = torch.cat([x_h, x_w], dim=2)              # [N,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)      # x_h: [N,mip,H,1], x_w: [N,mip,W,1]
        x_w = x_w.permute(0, 1, 3, 2)                 # [N,mip,1,W]

        a_h = torch.sigmoid(self.conv_h(x_h))         # [N,oup,H,1]
        a_w = torch.sigmoid(self.conv_w(x_w))         # [N,oup,1,W]

        # Ensure contiguous for the CUDA kernel
        if x.is_cuda and x.dtype == torch.float32:
            out = self.custom_ops_lib.coord_att_fused_mul_cuda(
                identity.contiguous(),
                a_w.contiguous(),
                a_h.contiguous(),
            )
        else:
            out = identity * a_w * a_h

        return out