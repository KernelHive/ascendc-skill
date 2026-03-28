import torch
import torch.nn as nn


class Model(nn.Module):
    """
    External Attention mechanism using two linear layers (memory keys and memory values)
    with double normalization (softmax + L1 normalization).
    """

    def __init__(self, d_model, S=64):
        """
        :param d_model: Dimensionality of the model
        :param S: Size of the external memory (number of memory keys)
        """
        super(Model, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        """
        :param queries: Input tensor (b_s, n, d_model)
        :return: Output tensor (b_s, n, d_model)
        """
        attn = self.mk(queries)  # bs, n, S
        attn = self.softmax(attn)  # bs, n, S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs, n, S
        out = self.mv(attn)  # bs, n, d_model
        return out


batch_size = 32
seq_len = 49
d_model = 512
S = 64


def get_inputs():
    queries = torch.randn(batch_size, seq_len, d_model)
    return [queries]


def get_init_inputs():
    return [d_model, S]
