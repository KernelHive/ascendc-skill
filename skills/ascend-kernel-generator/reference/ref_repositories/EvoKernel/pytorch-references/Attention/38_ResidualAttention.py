import torch
import torch.nn as nn


class Model(nn.Module):
    """Residual Attention Module.
    Uses a 1x1 convolution to project features to class logits, then combines
    average and max pooling across spatial dimensions with a weighting factor (la)
    to produce class scores. Output shape is (batch_size, num_class).
    """
    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y_raw = self.fc(x).flatten(2)  # b,num_class,hxw
        y_avg = torch.mean(y_raw, dim=2)  # b,num_class
        y_max = torch.max(y_raw, dim=2)[0]  # b,num_class
        score = y_avg + self.la * y_max
        return score


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512, 1000, 0.2]
