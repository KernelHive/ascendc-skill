import torch
import torch.nn as nn
import torch.nn.functional as F

# Source:
# - Repo: Chenhao-Guan/deepseek-mhc | Path: mhc/block.py | URL: https://github.com/Chenhao-Guan/deepseek-mhc/blob/main/mhc/block.py
# - Repo: Chenhao-Guan/deepseek-mhc | Path: mhc/sinkhorn.py | URL: https://github.com/Chenhao-Guan/deepseek-mhc/blob/main/mhc/sinkhorn.py
# - Repo: Chenhao-Guan/deepseek-mhc | Path: mhc/utils.py | URL: https://github.com/Chenhao-Guan/deepseek-mhc/blob/main/mhc/utils.py


class Model(nn.Module):
    """mHC ResNet bottleneck block with 2D stream mixing."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_streams: int = 4,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        sinkhorn_iter: int = 20,
        sinkhorn_eps: float = 1e-8,
        sinkhorn_temperature: float = 1.0,
        use_dynamic_mapping: bool = True,
    ):
        super().__init__()
        self.num_streams = int(num_streams)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.use_dynamic_mapping = bool(use_dynamic_mapping)

        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if (out_channels * self.expansion) % self.num_streams != 0:
            raise ValueError("expanded out_channels must be divisible by num_streams")

        if use_dynamic_mapping:
            self.mapping_projection = nn.Linear(in_channels, num_streams * num_streams)
            self.mapping_bias = nn.Parameter(torch.zeros(num_streams * num_streams))
        else:
            self.static_matrix = nn.Parameter(torch.randn(num_streams, num_streams))

    def _sinkhorn_knopp(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=-50.0, max=50.0)
        x = torch.exp(x / self.sinkhorn_temperature)
        for _ in range(self.sinkhorn_iter):
            x = x / (x.sum(dim=-1, keepdim=True) + self.sinkhorn_eps)
            x = x / (x.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
        return x

    def _apply_mapping(self, streams: torch.Tensor, mapping_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_streams, channels, height, width = streams.shape
        streams_flat = streams.view(batch_size, num_streams, -1)
        mixed = torch.bmm(mapping_matrix, streams_flat)
        return mixed.view(batch_size, num_streams, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        batch_size, channels, height, width = out.shape
        streams = out.view(
            batch_size, self.num_streams, channels // self.num_streams, height, width
        )

        if self.use_dynamic_mapping:
            x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
            mapping_flat = self.mapping_projection(x_pooled) + self.mapping_bias
            mapping_matrix = mapping_flat.view(
                batch_size, self.num_streams, self.num_streams
            )
        else:
            mapping_matrix = self.static_matrix.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        mapping_matrix = self._sinkhorn_knopp(mapping_matrix)
        streams = self._apply_mapping(streams, mapping_matrix)
        out = streams.view(batch_size, channels, height, width)

        if identity.shape != out.shape:
            raise ValueError("identity shape mismatch; use matching in/out and stride")

        out = out + identity
        out = self.relu(out)
        return out


# Configuration
batch_size = 8
in_channels = 256
out_channels = 64
height = 32
width = 32
num_streams = 4


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, num_streams, 1, 1, 64, 20, 1e-8, 1.0, True]
