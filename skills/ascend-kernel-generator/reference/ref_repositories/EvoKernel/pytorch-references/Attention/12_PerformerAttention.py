import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Performer Attention mechanism using FAVOR+ (Fast Attention Via positive Orthogonal Random features).
    Approximates softmax kernel using random feature maps to achieve linear complexity.
    Paper: Rethinking Attention with Performers
    """

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nb_features = self.d_k
        self.eps = 1e-6

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Create random projection matrix
        projection = torch.randn(self.nb_features, self.d_k)
        self.register_buffer('projection_matrix', projection)

    def softmax_kernel_transformation(self, data, is_query, projection_matrix):
        """
        FAVOR+ core: random feature approximation.
        Approximates the softmax kernel as inner products of random features.
        """
        # data: [batch, heads, seq_len, d_k]
        # projection: [nb_features, d_k]

        # Compute data normalizer for scaling
        data_normalizer = 1.0 / math.sqrt(math.sqrt(self.d_k))
        data = data * data_normalizer

        # Compute random features: [batch, heads, seq_len, nb_features]
        data_proj = torch.matmul(data, projection_matrix.T)

        # Apply non-linearity (ReLU for positive-definiteness)
        data_proj = F.relu(data_proj) + 1e-6

        return data_proj

    def forward(self, x):
        """
        Forward pass using FAVOR+ approximation for linear attention.
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply FAVOR+ kernel transformation
        Q = self.softmax_kernel_transformation(Q, True, self.projection_matrix)
        K = self.softmax_kernel_transformation(K, False, self.projection_matrix)

        # Linear attention computation
        # KV: [batch, heads, nb_features, d_k]
        KV = torch.matmul(K.transpose(-2, -1), V)

        # QKV: [batch, heads, seq_len, d_k]
        QKV = torch.matmul(Q, KV)

        # Normalization
        Z = 1.0 / (torch.einsum('bhnf,bhf->bhn', Q, K.sum(dim=2)).unsqueeze(-1) + self.eps)
        output = QKV * Z

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads]
