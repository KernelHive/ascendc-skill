import torch
import torch.nn as nn
import math


class Depth_Pointwise_Conv1d(nn.Module):
    """
    Depthwise separable 1D convolution: depthwise conv followed by pointwise conv.
    """
    def __init__(self, in_ch, out_ch, k):
        super(Depth_Pointwise_Conv1d, self).__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2
            )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class Model(nn.Module):
    """
    MUSE (Multi-Scale) Attention: combines standard multi-head scaled dot-product
    self-attention with a multi-scale convolutional branch using depthwise separable
    convolutions of kernel sizes 1, 3, and 5, weighted by learned dynamic parameters.
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(Model, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values):
        """
        Computes MUSE attention combining self-attention and multi-scale convolutions.
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return: Output tensor (b_s, nq, d_model)
        """
        # Self Attention
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

        # Multi-scale convolutional branch
        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs, dim, n
        dy_paras = self.softmax(self.dy_paras)
        out2 = dy_paras[0] * self.conv1(v2) + dy_paras[1] * self.conv3(v2) + dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs, n, dim

        out = out + out2
        return out


batch_size = 32
seq_len = 49
d_model = 512
d_k = 64
d_v = 64
h = 8


def get_inputs():
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    return [queries, keys, values]


def get_init_inputs():
    return [d_model, d_k, d_v, h]
