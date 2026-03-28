import torch
from torch import nn
import torch.nn.functional as F
import math


def to(x):
    return {'device': x.device, 'dtype': x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = x.reshape(b, l * (m + 1))
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = torch.einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = logits.reshape(b * h, w, -1)
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = q.reshape(q.shape[0], block, block, q.shape[-1])
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.reshape(q.shape[0], block, block, rel_logits_w.shape[-2], rel_logits_w.shape[-1])
        rel_logits_w = rel_logits_w.reshape(q.shape[0], block * block, rel_logits_w.shape[-2] * rel_logits_w.shape[-1])

        q = q.permute(0, 2, 1, 3)
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.reshape(q.shape[0], block, block, rel_logits_h.shape[-2], rel_logits_h.shape[-1])
        rel_logits_h = rel_logits_h.reshape(q.shape[0], block * block, rel_logits_h.shape[-1] * rel_logits_h.shape[-2])
        return rel_logits_w + rel_logits_h


class Model(nn.Module):
    """
    Halo Attention module that extracts local block neighborhoods with halo
    regions and applies multi-head self-attention with relative position embeddings.
    """
    def __init__(self, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        block = self.block_size
        halo = self.halo_size
        heads = self.heads
        device = x.device

        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version for deriving key values
        q_inp = x.reshape(b, c, h // block, block, w // block, block).permute(0, 2, 4, 3, 5, 1).reshape(b * (h // block) * (w // block), block * block, c)

        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = kv_inp.reshape(b, c, -1, kv_inp.shape[-1]).permute(0, 3, 2, 1).reshape(b * kv_inp.shape[-1], -1, c)

        # derive queries, keys, values
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # split heads
        q = q.reshape(q.shape[0], q.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(q.shape[0] * heads, q.shape[1], -1)
        k = k.reshape(k.shape[0], k.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(k.shape[0] * heads, k.shape[1], -1)
        v = v.reshape(v.shape[0], v.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(v.shape[0] * heads, v.shape[1], -1)

        # scale
        q = q * self.scale

        # attention
        sim = torch.einsum('b i d, b j d -> b i j', q, k)

        # mask out padding
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + (halo * 2), stride=block, padding=halo)
        mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
        num_windows = mask.shape[-1]
        mask = mask.permute(0, 3, 1, 2).reshape(b * num_windows, 1, -1).expand(-1, heads, -1).reshape(b * num_windows * heads, 1, -1)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)

        # aggregate
        out = torch.einsum('b i j, b j d -> b i d', attn, v)

        # merge and combine heads
        out = out.reshape(-1, heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3).reshape(-1, out.shape[1], heads * out.shape[2])
        out = self.to_out(out)

        # merge blocks back to original feature map
        out = out.reshape(b, h // block, w // block, block, block, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return out


batch_size = 128
dim = 512
block_size = 2
halo_size = 1
dim_head = 64
heads = 8

def get_inputs():
    return [torch.randn(batch_size, dim, 8, 8)]

def get_init_inputs():
    return [dim, block_size, halo_size, dim_head, heads]
