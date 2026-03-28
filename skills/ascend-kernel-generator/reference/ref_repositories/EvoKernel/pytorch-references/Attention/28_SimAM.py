import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks.

    Applies a spatial attention mechanism based on energy function minimization,
    using no additional learnable parameters beyond a small lambda for numerical stability.
    """

    def __init__(self, e_lambda=1e-4):
        super(Model, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * self.activation(y)


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [1e-4]
