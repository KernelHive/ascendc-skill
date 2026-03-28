import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ---------------------------
# Original submodules (kept)
# ---------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.0):
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class PositionAttentionModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)  # (bs, hw, c)
        y = self.pa(y, y, y)
        return y


class ChannelAttentionModule(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # (bs, c, hw)
        y = self.pa(y, y, y)
        return y


# ---------------------------
# Custom CUDA: fused permute(p_out)+add(c_out)
#   p_out: [bs, hw, c] contiguous
#   c_out: [bs, c, hw] contiguous
#   out:   [bs, c, hw] contiguous
# ---------------------------

da_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__)
#define LDG(p) __ldg(p)
#else
#define LDG(p) (*(p))
#endif

// Generic 1D kernel over (b,c): out[b,c,hw] = p[b,hw,c] + c[b,c,hw]
__global__ __launch_bounds__(128, 4)
void fused_permute_add_generic_1d(
    const float* __restrict__ p, // [bs, hw, ch]
    const float* __restrict__ c, // [bs, ch, hw]
    float* __restrict__ out,     // [bs, ch, hw]
    int bs, int hw, int ch
){
    int idx = (int)blockIdx.x; // 0..bs*ch-1
    int b = idx / ch;
    int ci = idx - b * ch;
    if (b >= bs) return;

    int t = (int)threadIdx.x;
    int stride = (int)blockDim.x;

    int64_t base_c = ((int64_t)b * ch + ci) * (int64_t)hw;

    const float* cptr = c + base_c;
    float* optr = out + base_c;

    for (int hwi = t; hwi < hw; hwi += stride) {
        float cv = LDG(cptr + hwi);
        float pv = LDG(p + (((int64_t)b * hw + hwi) * (int64_t)ch + ci));
        optr[hwi] = pv + cv;
    }
}

// HW=49 specialized: warp per (b, c4) where c4 is a group of 4 channels.
// Correct vectorization is along CHANNELS at fixed hwi for p_out (contiguous).
// For c/out ([bs,ch,HW]) channels are strided by HW, so we load/store 4 scalars (still coalesced across lanes in HW).
__global__ __launch_bounds__(128, 3)
void fused_permute_add_hw49_c4_warp(
    const float* __restrict__ p, // [bs, 49, ch]
    const float* __restrict__ c, // [bs, ch, 49]
    float* __restrict__ out,     // [bs, ch, 49]
    int bs, int ch
){
    constexpr int HW = 49;
    constexpr int WARPS_PER_BLOCK = 4;

    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int ch4 = ch >> 2; // ch must be divisible by 4 for this kernel

    int64_t warp_global = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int64_t total_warps = (int64_t)bs * (int64_t)ch4;
    if (warp_global >= total_warps) return;

    int b = (int)(warp_global / ch4);
    int c4i = (int)(warp_global - (int64_t)b * ch4);
    int c_base = c4i * 4;

    // process spatial positions with lane-stride
    // two iterations cover HW=49: lane (0..31), lane+32 (32..48)
    int h0 = lane;
    if (h0 < 32) {
        // p vector load: p[b,h0,c_base..c_base+3] contiguous
        const float4* p4 = reinterpret_cast<const float4*>(p + ((int64_t)b * HW + h0) * (int64_t)ch + c_base);
        float4 pv = LDG(p4);

        // c scalar loads: c[b,c_base+k,h0] where channels are strided by HW
        int64_t cbase0 = ((int64_t)b * ch + (int64_t)c_base) * (int64_t)HW + h0;
        float c0 = LDG(c + cbase0 + 0LL * HW);
        float c1 = LDG(c + cbase0 + 1LL * HW);
        float c2 = LDG(c + cbase0 + 2LL * HW);
        float c3 = LDG(c + cbase0 + 3LL * HW);

        // store out similarly
        int64_t obase0 = ((int64_t)b * ch + (int64_t)c_base) * (int64_t)HW + h0;
        out[obase0 + 0LL * HW] = pv.x + c0;
        out[obase0 + 1LL * HW] = pv.y + c1;
        out[obase0 + 2LL * HW] = pv.z + c2;
        out[obase0 + 3LL * HW] = pv.w + c3;
    }

    int h1 = lane + 32;
    if (h1 < HW) {
        const float4* p4 = reinterpret_cast<const float4*>(p + ((int64_t)b * HW + h1) * (int64_t)ch + c_base);
        float4 pv = LDG(p4);

        int64_t cbase0 = ((int64_t)b * ch + (int64_t)c_base) * (int64_t)HW + h1;
        float c0 = LDG(c + cbase0 + 0LL * HW);
        float c1 = LDG(c + cbase0 + 1LL * HW);
        float c2 = LDG(c + cbase0 + 2LL * HW);
        float c3 = LDG(c + cbase0 + 3LL * HW);

        int64_t obase0 = ((int64_t)b * ch + (int64_t)c_base) * (int64_t)HW + h1;
        out[obase0 + 0LL * HW] = pv.x + c0;
        out[obase0 + 1LL * HW] = pv.y + c1;
        out[obase0 + 2LL * HW] = pv.z + c2;
        out[obase0 + 3LL * HW] = pv.w + c3;
    }
}

torch::Tensor da_fused_permute_add_cuda(torch::Tensor p_out, torch::Tensor c_out, int64_t H, int64_t W) {
    TORCH_CHECK(p_out.is_cuda() && c_out.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(p_out.dtype() == torch::kFloat32 && c_out.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(p_out.is_contiguous(), "p_out must be contiguous");
    TORCH_CHECK(c_out.is_contiguous(), "c_out must be contiguous");
    TORCH_CHECK(p_out.dim() == 3, "p_out must be [bs, hw, c]");
    TORCH_CHECK(c_out.dim() == 3, "c_out must be [bs, c, hw]");

    int bs = (int)p_out.size(0);
    int hw = (int)p_out.size(1);
    int ch = (int)p_out.size(2);

    TORCH_CHECK((int)(H * W) == hw, "H*W must equal p_out.size(1)");
    TORCH_CHECK((int)c_out.size(0) == bs && (int)c_out.size(1) == ch && (int)c_out.size(2) == hw,
                "c_out shape must be [bs, c, hw]");

    auto out = torch::empty({bs, ch, hw}, p_out.options());

    if (hw == 49 && (ch % 4 == 0)) {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;

        int ch4 = ch >> 2;
        int64_t total_warps = (int64_t)bs * (int64_t)ch4;
        int64_t blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        fused_permute_add_hw49_c4_warp<<<(unsigned int)blocks, THREADS>>>(
            p_out.data_ptr<float>(),
            c_out.data_ptr<float>(),
            out.data_ptr<float>(),
            bs, ch
        );
    } else {
        int64_t blocks = (int64_t)bs * (int64_t)ch;
        fused_permute_add_generic_1d<<<(unsigned int)blocks, 128>>>(
            p_out.data_ptr<float>(),
            c_out.data_ptr<float>(),
            out.data_ptr<float>(),
            bs, hw, ch
        );
    }

    return out;
}
"""

da_cpp_src = r"""
torch::Tensor da_fused_permute_add_cuda(torch::Tensor p_out, torch::Tensor c_out, int64_t H, int64_t W);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_da_module_fused_v9_hw49_c4_correct",
    cpp_sources=da_cpp_src,
    cuda_sources=da_cuda_src,
    functions=["da_fused_permute_add_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


# ---------------------------
# New model using fused custom op
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.channel_attention_module = ChannelAttentionModule(d_model=d_model, kernel_size=kernel_size, H=H, W=W)
        self.H = H
        self.W = W
        self.custom_ops = custom_ops_lib

    def forward(self, x):
        bs, c, h, w = x.shape
        p_out = self.position_attention_module(x)  # (bs, hw, c)
        c_out = self.channel_attention_module(x)   # (bs, c, hw)

        if x.is_cuda and x.dtype == torch.float32:
            out_flat = self.custom_ops.da_fused_permute_add_cuda(
                p_out.contiguous(),
                c_out.contiguous(),
                h, w
            )  # [bs, c, hw] where hw=h*w
            return out_flat.view(bs, c, h, w)

        p_hw = p_out.permute(0, 2, 1).contiguous().view(bs, c, h, w)
        c_hw = c_out.contiguous().view(bs, c, h, w)
        return p_hw + c_hw


# ---------------------------
# Input helpers (kept compatible)
# ---------------------------

batch_size = 128
in_channels = 512
height = 7
width = 7


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [512, 3, 7, 7]