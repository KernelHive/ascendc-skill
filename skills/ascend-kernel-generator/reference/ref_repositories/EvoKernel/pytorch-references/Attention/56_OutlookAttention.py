import torch
from torch import nn
import math
from torch.nn import functional as F


class Model(nn.Module):
    """
    Outlook Attention module that generates attention weights from pooled features
    and applies them to unfolded local patches to produce the output.
    """
    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size, padding, stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        # map to new feature v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                    self.kernel_size * self.kernel_size,
                                    h * w).permute(0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # generate Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads,
                                        self.kernel_size * self.kernel_size,
                                        self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,num_head,H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # get weighted features
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out)

        return out


batch_size = 128
dim = 512
num_heads = 1
kernel_size = 3
padding = 1
stride = 1

def get_inputs():
    return [torch.randn(batch_size, 7, 7, dim)]

def get_init_inputs():
    return [dim, num_heads, kernel_size, padding, stride]
