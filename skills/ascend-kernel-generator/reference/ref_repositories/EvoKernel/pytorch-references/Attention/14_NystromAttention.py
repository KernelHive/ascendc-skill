import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Nystrom Attention mechanism.
    Uses the Nystrom method to approximate the full attention matrix with landmark points.
    Achieves efficient attention by sampling landmark points and computing a low-rank approximation.
    Paper: Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention
    """

    def __init__(self, d_model, n_heads, num_landmarks=32):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_landmarks: Number of landmark points for Nystrom approximation
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.num_landmarks = num_landmarks
        self.eps = 1e-6

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Forward pass using Nystrom approximation.
        Selects landmark points via uniform sampling, then approximates
        the full attention matrix as A @ B^(-1) @ A^T.
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Select landmark points
        if seq_len <= self.num_landmarks:
            # Sequence too short, use standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
        else:
            # Nystrom approximation
            # 1. Select landmark indices (uniform sampling)
            landmark_indices = torch.linspace(
                0, seq_len - 1, self.num_landmarks, dtype=torch.long, device=x.device
            )

            # 2. Get landmark Q, K
            Q_landmarks = Q[:, :, landmark_indices]  # [batch, heads, m, d_k]
            K_landmarks = K[:, :, landmark_indices]  # [batch, heads, m, d_k]

            # 3. Compute three key matrices
            # A: Q with landmark K attention [batch, heads, n, m]
            A = torch.matmul(Q, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
            A = F.softmax(A, dim=-1)

            # B: landmark Q with landmark K attention [batch, heads, m, m]
            B = torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) / math.sqrt(self.d_k)
            B = F.softmax(B, dim=-1)

            # 4. Compute pseudo-inverse
            B_inv = torch.pinverse(B + self.eps * torch.eye(
                self.num_landmarks, device=B.device
            ).unsqueeze(0).unsqueeze(0))

            # 5. Nystrom approximation: A @ B^(-1) @ A^T @ V
            attention_matrix_approx = torch.matmul(torch.matmul(A, B_inv), A.transpose(-2, -1))

            # Normalization
            attention_matrix_approx = attention_matrix_approx / (
                    attention_matrix_approx.sum(dim=-1, keepdim=True) + self.eps
            )

            # Apply to V
            output = torch.matmul(attention_matrix_approx, V)

        # Merge heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


batch_size = 32
seq_len = 512
d_model = 512
n_heads = 8
num_landmarks = 32


def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]


def get_init_inputs():
    return [d_model, n_heads, num_landmarks]
