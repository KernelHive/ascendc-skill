import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Divides query heads into G groups, each group sharing one key-value pair.
    Used as a building block for AdaptiveAttention.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class Model(nn.Module):
    """
    Adaptive Attention.
    Dynamically selects between MHA, GQA, or MQA based on the input
    using a learned routing network.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads_options: list = None,
            dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads_options: List of KV head count options (e.g. [8, 4, 1] for MHA, GQA, MQA)
            dropout: Dropout probability
        """
        super().__init__()

        if n_kv_heads_options is None:
            n_kv_heads_options = [8, 4, 1]

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads_options = n_kv_heads_options

        # Create attention layers for each KV head configuration
        self.attention_layers = nn.ModuleList([
            GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
            for n_kv_heads in n_kv_heads_options
        ])

        # Router network (decides which attention to use)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(n_kv_heads_options))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # Use mean pooling of input as routing signal
        routing_input = x.mean(dim=1)  # [batch_size, d_model]
        routing_logits = self.router(routing_input)  # [batch_size, num_options]
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Select the most likely attention configuration
        selected_idx = routing_probs.argmax(dim=-1)

        # Dynamic batching: group samples with same selection
        outputs = torch.zeros_like(x)
        for idx in range(len(self.n_kv_heads_options)):
            mask = (selected_idx == idx)
            if mask.any():
                batch_input = x[mask]
                batch_output = self.attention_layers[idx](batch_input)
                outputs[mask] = batch_output

        return outputs


# Configuration variables
batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
n_kv_heads_options = [8, 4, 1]


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads]
