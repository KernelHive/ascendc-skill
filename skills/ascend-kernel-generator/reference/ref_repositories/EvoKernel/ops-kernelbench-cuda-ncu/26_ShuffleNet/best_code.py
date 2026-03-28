import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------
# Custom CUDA: channel shuffle
# ----------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_nchw_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int groups)
{
    // Flattened over N*C*H*W
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H * W;
    if (idx >= total) return;

    int w = (int)(idx % W);
    int tmp = (int)(idx / W);
    int h = (int)(tmp % H);
    tmp /= H;
    int c = (int)(tmp % C);
    int n = (int)(tmp / C);

    int channels_per_group = C / groups;
    // Channel shuffle mapping:
    // input view: [N, groups, cpg, H, W] then transpose(1,2) => [N, cpg, groups, H, W]
    // So output channel c_out corresponds to input channel:
    // c_out = c_in_group * groups + g
    // c_in  = g * cpg + c_in_group
    int g = c / channels_per_group;       // group index
    int c_in_group = c - g * channels_per_group;

    int c_out = c_in_group * groups + g;

    int64_t in_offset  = (((int64_t)n * C + c) * H + h) * W + w;
    int64_t out_offset = (((int64_t)n * C + c_out) * H + h) * W + w;
    y[out_offset] = x[in_offset];
}

torch::Tensor channel_shuffle_forward_cuda(torch::Tensor x, int64_t groups) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous (NCHW)");
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(groups > 0, "groups must be > 0");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(C % groups == 0, "channels must be divisible by groups");

    auto y = torch::empty_like(x);

    int threads = 256;
    int64_t total = N * C * H * W;
    int blocks = (int)((total + threads - 1) / threads);

    channel_shuffle_nchw_kernel<<<blocks, threads>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)groups
    );

    return y;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor channel_shuffle_forward_cuda(torch::Tensor x, int64_t groups);
"""

custom_ops_lib = load_inline(
    name="custom_shufflenet_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["channel_shuffle_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
)

# ----------------------------
# Modules (with custom shuffle)
# ----------------------------
class ChannelShuffleNew(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = int(groups)
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep explicit guards similar to the successful example:
        # enforce float32 + contiguous for the custom CUDA op.
        if not x.is_cuda:
            # Fall back to PyTorch implementation on CPU
            n, c, h, w = x.size()
            g = self.groups
            cpg = c // g
            x = x.view(n, g, cpg, h, w).transpose(1, 2).contiguous().view(n, c, h, w)
            return x

        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        return self.custom_ops_lib.channel_shuffle_forward_cuda(x, self.groups)


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffleNew(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out + self.shortcut(x)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super().__init__()
        self.custom_ops_lib = custom_ops_lib  # keep reference alive

        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = [ShuffleNetUnitNew(in_channels, out_channels, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x