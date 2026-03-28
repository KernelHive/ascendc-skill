import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Scaled Dot-Product Attention with modular linear projections for queries, keys, and values.
    Multi-head attention mechanism with separate FC layers for Q, K, V projections and output.
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

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values):
        """
        Computes multi-head scaled dot-product attention.
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return: Output tensor (b_s, nq, d_model)
        """
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
