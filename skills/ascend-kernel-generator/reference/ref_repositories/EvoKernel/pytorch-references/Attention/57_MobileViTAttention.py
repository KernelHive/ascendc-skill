import torch
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        b, p, n, hd = qkv[0].shape[0], qkv[0].shape[1], qkv[0].shape[2], self.heads
        q, k, v = [t.reshape(b, p, n, hd, -1).permute(0, 1, 3, 2, 4) for t in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, p, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class Model(nn.Module):
    """
    MobileViT Attention module that combines local convolution with global
    transformer attention through patch-based unfolding and folding.
    """
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()  # bs,c,h,w

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        ph, pw = self.ph, self.pw
        # rearrange: bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim
        nh, nw = h // ph, w // pw
        y = y.reshape(y.shape[0], y.shape[1], nh, ph, nw, pw)
        y = y.permute(0, 3, 5, 2, 4, 1)  # bs, ph, pw, nh, nw, dim
        y = y.reshape(y.shape[0], ph * pw, nh * nw, -1)  # bs, (ph pw), (nh nw), dim

        y = self.trans(y)

        # rearrange back: bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)
        y = y.reshape(y.shape[0], ph, pw, nh, nw, -1)
        y = y.permute(0, 5, 3, 1, 4, 2)  # bs, dim, nh, ph, nw, pw
        y = y.reshape(y.shape[0], -1, nh * ph, nw * pw)  # bs, dim, h, w

        ## Fusion
        y = self.conv3(y)  # bs,c,h,w
        y = torch.cat([x, y], 1)  # bs,2*c,h,w
        y = self.conv4(y)  # bs,c,h,w

        return y


batch_size = 32
in_channel = 3
dim = 512
kernel_size = 3
patch_size = 7

def get_inputs():
    return [torch.randn(batch_size, in_channel, 49, 49)]

def get_init_inputs():
    return [in_channel, dim, kernel_size, patch_size]
