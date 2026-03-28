import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Custom CUDA: concat along channel dim for 4 NCHW tensors with same N,H,W
# -------------------------
concat_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__global__ void concat4_nchw_kernel(
    const float* __restrict__ a, int Ca,
    const float* __restrict__ b, int Cb,
    const float* __restrict__ c, int Cc,
    const float* __restrict__ d, int Cd,
    float* __restrict__ out,
    int N, int H, int W
) {
    // Flattened over N * (Ca+Cb+Cc+Cd) * H * W
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Ctot = Ca + Cb + Cc + Cd;
    int total = N * Ctot * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int t1 = idx / W;
    int h = t1 % H;
    int t2 = t1 / H;
    int ch = t2 % Ctot;
    int n  = t2 / Ctot;

    int hw = h * W + w;
    int out_off = ((n * Ctot + ch) * H * W) + hw;

    const float* src = nullptr;
    int src_c = ch;
    int Csrc = 0;

    if (ch < Ca) {
        src = a; Csrc = Ca;
    } else if (ch < Ca + Cb) {
        src = b; src_c = ch - Ca; Csrc = Cb;
    } else if (ch < Ca + Cb + Cc) {
        src = c; src_c = ch - (Ca + Cb); Csrc = Cc;
    } else {
        src = d; src_c = ch - (Ca + Cb + Cc); Csrc = Cd;
    }

    int src_off = ((n * Csrc + src_c) * H * W) + hw;
    out[out_off] = src[src_off];
}

torch::Tensor concat4_nchw_forward_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d
) {
    CHECK_CUDA(a); CHECK_CUDA(b); CHECK_CUDA(c); CHECK_CUDA(d);
    CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b); CHECK_CONTIGUOUS(c); CHECK_CONTIGUOUS(d);
    CHECK_FLOAT(a); CHECK_FLOAT(b); CHECK_FLOAT(c); CHECK_FLOAT(d);

    TORCH_CHECK(a.dim() == 4 && b.dim() == 4 && c.dim() == 4 && d.dim() == 4, "all inputs must be NCHW");
    int64_t N = a.size(0);
    int64_t H = a.size(2);
    int64_t W = a.size(3);

    TORCH_CHECK(b.size(0) == N && c.size(0) == N && d.size(0) == N, "N mismatch");
    TORCH_CHECK(b.size(2) == H && c.size(2) == H && d.size(2) == H, "H mismatch");
    TORCH_CHECK(b.size(3) == W && c.size(3) == W && d.size(3) == W, "W mismatch");

    int64_t Ca = a.size(1);
    int64_t Cb = b.size(1);
    int64_t Cc = c.size(1);
    int64_t Cd = d.size(1);
    int64_t Ctot = Ca + Cb + Cc + Cd;

    auto out = torch::empty({N, Ctot, H, W}, a.options());

    int total = (int)(N * Ctot * H * W);
    const int block = 256;
    const int grid = (total + block - 1) / block;

    concat4_nchw_kernel<<<grid, block>>>(
        a.data_ptr<float>(), (int)Ca,
        b.data_ptr<float>(), (int)Cb,
        c.data_ptr<float>(), (int)Cc,
        d.data_ptr<float>(), (int)Cd,
        out.data_ptr<float>(),
        (int)N, (int)H, (int)W
    );
    return out;
}
"""

concat_cpp_source = r"""
torch::Tensor concat4_nchw_forward_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_inception_concat4",
    cpp_sources=concat_cpp_source,
    cuda_sources=concat_cuda_source,
    functions=["concat4_nchw_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)

# -------------------------
# Modules
# -------------------------
class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        # Keep convs/pool in PyTorch for correctness; fuse only concat.
        b1 = self.branch1x1(x)

        b3 = self.branch3x3_1(x)
        b3 = self.branch3x3_2(b3)

        b5 = self.branch5x5_1(x)
        b5 = self.branch5x5_2(b5)

        bp = self.pool(x)
        bp = self.pool_proj(bp)

        # Custom concat along channels
        if x.is_cuda and x.dtype == torch.float32:
            return custom_ops_lib.concat4_nchw_forward_cuda(
                b1.contiguous(), b3.contiguous(), b5.contiguous(), bp.contiguous()
            )
        else:
            return torch.cat([b1, b3, b5, bp], dim=1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x