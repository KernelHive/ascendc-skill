import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=0.0):
        """
        Args:
            d_model: Output dimensionality of the model.
            d_k: Dimensionality of queries and keys.
            d_v: Dimensionality of values.
            h: Number of heads.
        """
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    """
    Simplified scaled dot-product attention (no linear projections for Q, K, V).
    """

    def __init__(self, d_model, h, dropout=0.0):
        """
        Args:
            d_model: Output dimensionality of the model.
            h: Number of heads.
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class PositionAttentionModule(nn.Module):
    """
    Position attention module for DANet.
    Applies spatial self-attention over feature map positions.
    """

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)  # bs, h*w, c
        y = self.pa(y, y, y)  # bs, h*w, c
        return y


class ChannelAttentionModule(nn.Module):
    """
    Channel attention module for DANet.
    Applies self-attention across channels.
    """

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # bs, c, h*w
        y = self.pa(y, y, y)  # bs, c, h*w
        return y


class Model(nn.Module):
    """
    Dual Attention Network (DANet) module.

    Combines Position Attention Module and Channel Attention Module
    to capture long-range contextual information from both spatial
    and channel dimensions.

    Reference: Dual Attention Network for Scene Segmentation (CVPR 2019)
    """

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        """
        Args:
            d_model: Number of input channels.
            kernel_size: Kernel size for the convolutional layers.
            H: Height of the input feature map.
            W: Width of the input feature map.
        """
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)

    def forward(self, x):
        bs, c, h, w = x.shape
        p_out = self.position_attention_module(x)
        c_out = self.channel_attention_module(x)
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        c_out = c_out.view(bs, c, h, w)
        return p_out + c_out


batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512, 3, 7, 7]
