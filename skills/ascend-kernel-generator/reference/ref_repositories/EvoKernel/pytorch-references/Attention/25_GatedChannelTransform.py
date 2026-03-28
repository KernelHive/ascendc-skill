import torch
import torch.nn as nn


class Model(nn.Module):
    """Gated Channel Transformation (GCT) for Visual Recognition.

    Introduces a channel normalization layer with L2 (or L1) normalization to
    reduce parameters and computational complexity. A lightweight gating
    mechanism modulates channel responses via learned alpha, gamma, and beta
    parameters.
    """

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


batch_size = 128
in_channels = 512
height = 7
width = 7
epsilon = 1e-5
mode = 'l2'


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, epsilon, mode]
