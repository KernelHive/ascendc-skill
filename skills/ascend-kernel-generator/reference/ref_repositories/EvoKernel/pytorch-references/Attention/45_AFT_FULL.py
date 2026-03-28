import torch
import torch.nn as nn


class Model(nn.Module):
    """
    AFT-Full (Attention Free Transformer - Full variant): computes attention-free
    token mixing using element-wise operations with learned position biases.
    Uses sigmoid gating on queries and exponential weighting on keys with position biases.
    """

    def __init__(self, d_model, n=49):
        """
        :param d_model: Dimensionality of the model
        :param n: Sequence length (number of tokens)
        """
        super(Model, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        :param input: Input tensor (bs, n, dim)
        :return: Output tensor (bs, n, dim)
        """
        bs, n, dim = input.shape

        q = self.fc_q(input)  # bs, n, dim
        k = self.fc_k(input).view(1, bs, n, dim)  # 1, bs, n, dim
        v = self.fc_v(input).view(1, bs, n, dim)  # 1, bs, n, dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n, bs, dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n, bs, dim

        out = (numerator / denominator)  # n, bs, dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs, n, dim

        return out


batch_size = 32
seq_len = 49
d_model = 512


def get_inputs():
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    return [input_tensor]


def get_init_inputs():
    return [d_model, seq_len]
