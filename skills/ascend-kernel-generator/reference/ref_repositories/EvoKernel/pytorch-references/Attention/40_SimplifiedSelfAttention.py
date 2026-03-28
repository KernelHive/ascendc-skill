import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Simplified Scaled Dot-Product Attention without separate Q, K, V projections.
    Directly reshapes input queries, keys, values into multi-head format and applies
    scaled dot-product attention followed by an output linear projection.
    """

    def __init__(self, d_model, h):
        """
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        """
        super(Model, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, queries, keys, values):
        """
        Computes simplified multi-head scaled dot-product attention.
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return: Output tensor (b_s, nq, d_model)
        """
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


batch_size = 32
seq_len = 49
d_model = 512
h = 8


def get_inputs():
    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)
    return [queries, keys, values]


def get_init_inputs():
    return [d_model, h]
