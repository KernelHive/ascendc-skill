import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ----------------------------
# CUDA extension: S2Attention
# - s2_attention_descriptor_cuda(t3) -> a [B,C]
# - s2_attention_apply_cuda(t3, gates) -> out [B,W,H,C]
# where t3 is contiguous [B,W,H,3C] float32 CUDA.
# Shifts match the reference in-place assignment semantics by performing
# sequential scatter updates (with bounds) in the same order as Python.
# ----------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// index in [W,H,C] contiguous
__device__ __forceinline__ int64_t idx_whc(int w, int h, int c, int H, int C) {
    return ((int64_t)w * (int64_t)H + (int64_t)h) * (int64_t)C + (int64_t)c;
}

// Descriptor kernel: compute a[b,c] = sum_{w,h} (x1_shifted + x2_shifted + x3)
// BUT x1_shifted and x2_shifted must match in-place semantics.
// We implement this by simulating the sequential assignments on-the-fly
// for each queried destination element (w,h,c), by resolving what value
// ends up in that location after the 4 ordered assignment steps.
__device__ __forceinline__ float resolve_shift1_inplace(
    const float* __restrict__ x1, int w, int h, int c, int W, int H, int C)
{
    // Start with original value
    int w_src = w, h_src = h;
    float v = ldg_f(x1 + idx_whc(w_src, h_src, c, H, C));

    int cq = C >> 2;
    int ch = C >> 1;
    int c3q = (3 * C) >> 2;

    // Apply assignments in order, considering overwrites.

    // 1) x[:, 1:, :, :cq] = x[:, :W-1, :, :cq]
    if (c < cq && w >= 1) {
        // destination (w,h,c) overwritten by value from (w-1,h,c) from "current" tensor state,
        // but source indices are < W-1 and can themselves have been modified by earlier steps?
        // There are no earlier steps, so it's original.
        v = ldg_f(x1 + idx_whc(w - 1, h, c, H, C));
    }

    // 2) x[:, :W-1, :, cq:ch] = x[:, 1:, :, cq:ch]
    if (c >= cq && c < ch && w <= W - 2) {
        // source is (w+1,h,c) after step1 has possibly modified it if it lies in first quarter.
        // But c in [cq,ch) so step1 does not affect this channel range, hence original.
        v = ldg_f(x1 + idx_whc(w + 1, h, c, H, C));
    }

    // 3) x[:, :, 1:, ch:c3q] = x[:, :, :H-1, ch:c3q]
    if (c >= ch && c < c3q && h >= 1) {
        // channels in [ch,c3q), prior steps 1-2 do not affect these channels.
        v = ldg_f(x1 + idx_whc(w, h - 1, c, H, C));
    }

    // 4) x[:, :, :H-1, c3q:] = x[:, :, 1:, c3q:]
    if (c >= c3q && h <= H - 2) {
        // channels in [c3q,C), prior steps 1-3 do not affect these channels.
        v = ldg_f(x1 + idx_whc(w, h + 1, c, H, C));
    }

    return v;
}

__device__ __forceinline__ float resolve_shift2_inplace(
    const float* __restrict__ x2, int w, int h, int c, int W, int H, int C)
{
    float v = ldg_f(x2 + idx_whc(w, h, c, H, C));

    int cq = C >> 2;
    int ch = C >> 1;
    int c3q = (3 * C) >> 2;

    // 1) x[:, :, 1:, :cq] = x[:, :, :H-1, :cq]
    if (c < cq && h >= 1) {
        v = ldg_f(x2 + idx_whc(w, h - 1, c, H, C));
    }

    // 2) x[:, :, :H-1, cq:ch] = x[:, :, 1:, cq:ch]
    if (c >= cq && c < ch && h <= H - 2) {
        v = ldg_f(x2 + idx_whc(w, h + 1, c, H, C));
    }

    // 3) x[:, 1:, :, ch:c3q] = x[:, :W-1, :, ch:c3q]
    if (c >= ch && c < c3q && w >= 1) {
        v = ldg_f(x2 + idx_whc(w - 1, h, c, H, C));
    }

    // 4) x[:, :W-1, :, c3q:] = x[:, 1:, :, c3q:]
    if (c >= c3q && w <= W - 2) {
        v = ldg_f(x2 + idx_whc(w + 1, h, c, H, C));
    }

    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// a[b,c] = sum_{w,h} (shift1(x1) + shift2(x2) + x3)
__global__ void s2_descriptor_kernel(
    const float* __restrict__ t3, // [B,W,H,3C]
    float* __restrict__ a,        // [B,C]
    int B, int W, int H, int C)
{
    int b = (int)blockIdx.x;
    int c = (int)blockIdx.y;
    if (b >= B || c >= C) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int HW = W * H;
    int64_t base = ((int64_t)b * (int64_t)W * (int64_t)H) * (int64_t)(3 * C);

    // pointers to per-(w,h) packed layout: last dim is 3C, so branch pointers are offset in last dim
    // We address as: t3[ ((w*H + h) * 3C + (branch*C + c)) ]
    const float* tb = t3 + base;

    float sum = 0.f;
    for (int i = tid; i < HW; i += (int)blockDim.x) {
        int w = i / H;
        int h = i - w * H;
        const float* p = tb + ((int64_t)w * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
        // For resolving shifts, we need access to branch tensors as [W,H,C] contiguous.
        // We can treat x1/x2 as separate [W,H,C] views with stride (3C) between consecutive c blocks.
        // To keep loads simple, build temporary pointers to a "virtual" contiguous [W,H,C]:
        // We implement resolve_* expecting x* laid out contiguous in C; but here C is strided by 3C.
        // Therefore we cannot pass p directly. Instead, implement direct loads using packed stride.

        // Directly read original x1/x2/x3 at (w,h,c):
        float x1_orig = ldg_f(p + (int64_t)(0 * C + c));
        float x2_orig = ldg_f(p + (int64_t)(1 * C + c));
        float x3      = ldg_f(p + (int64_t)(2 * C + c));

        // Emulate resolve_shift1/2 but with packed (3C) stride:
        int cq = C >> 2;
        int ch = C >> 1;
        int c3q = (3 * C) >> 2;

        float x1s = x1_orig;
        if (c < cq && w >= 1) {
            const float* psrc = tb + ((int64_t)(w - 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
            x1s = ldg_f(psrc + (int64_t)(0 * C + c));
        }
        if (c >= cq && c < ch && w <= W - 2) {
            const float* psrc = tb + ((int64_t)(w + 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
            x1s = ldg_f(psrc + (int64_t)(0 * C + c));
        }
        if (c >= ch && c < c3q && h >= 1) {
            const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h - 1)) * (int64_t)(3 * C);
            x1s = ldg_f(psrc + (int64_t)(0 * C + c));
        }
        if (c >= c3q && h <= H - 2) {
            const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h + 1)) * (int64_t)(3 * C);
            x1s = ldg_f(psrc + (int64_t)(0 * C + c));
        }

        float x2s = x2_orig;
        if (c < cq && h >= 1) {
            const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h - 1)) * (int64_t)(3 * C);
            x2s = ldg_f(psrc + (int64_t)(1 * C + c));
        }
        if (c >= cq && c < ch && h <= H - 2) {
            const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h + 1)) * (int64_t)(3 * C);
            x2s = ldg_f(psrc + (int64_t)(1 * C + c));
        }
        if (c >= ch && c < c3q && w >= 1) {
            const float* psrc = tb + ((int64_t)(w - 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
            x2s = ldg_f(psrc + (int64_t)(1 * C + c));
        }
        if (c >= c3q && w <= W - 2) {
            const float* psrc = tb + ((int64_t)(w + 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
            x2s = ldg_f(psrc + (int64_t)(1 * C + c));
        }

        sum += (x1s + x2s + x3);
    }

    float wsum = warp_sum(sum);
    __shared__ float smem[8]; // up to 256 threads
    if (lane == 0) smem[warp] = wsum;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (((int)blockDim.x) + 31) >> 5;
        float v = (lane < nwarps) ? smem[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) a[(int64_t)b * (int64_t)C + (int64_t)c] = v;
    }
}

// Apply gates: out[b,w,h,c] = g0*shift1(x1)+g1*shift2(x2)+g2*x3
__global__ void s2_apply_kernel(
    const float* __restrict__ t3,    // [B,W,H,3C]
    const float* __restrict__ gates, // [B,3,C]
    float* __restrict__ out,         // [B,W,H,C]
    int B, int W, int H, int C)
{
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t N = (int64_t)B * (int64_t)W * (int64_t)H * (int64_t)C;
    if (idx >= N) return;

    int c = (int)(idx % (int64_t)C);
    int64_t t = idx / (int64_t)C;
    int h = (int)(t % (int64_t)H);
    t /= (int64_t)H;
    int w = (int)(t % (int64_t)W);
    int b = (int)(t / (int64_t)W);

    int64_t base = ((int64_t)b * (int64_t)W * (int64_t)H) * (int64_t)(3 * C);
    const float* tb = t3 + base;

    const float* p = tb + ((int64_t)w * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
    float x3 = ldg_f(p + (int64_t)(2 * C + c));

    int cq = C >> 2;
    int ch = C >> 1;
    int c3q = (3 * C) >> 2;

    // shift1 for x1 packed
    float x1s = ldg_f(p + (int64_t)(0 * C + c));
    if (c < cq && w >= 1) {
        const float* psrc = tb + ((int64_t)(w - 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
        x1s = ldg_f(psrc + (int64_t)(0 * C + c));
    }
    if (c >= cq && c < ch && w <= W - 2) {
        const float* psrc = tb + ((int64_t)(w + 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
        x1s = ldg_f(psrc + (int64_t)(0 * C + c));
    }
    if (c >= ch && c < c3q && h >= 1) {
        const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h - 1)) * (int64_t)(3 * C);
        x1s = ldg_f(psrc + (int64_t)(0 * C + c));
    }
    if (c >= c3q && h <= H - 2) {
        const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h + 1)) * (int64_t)(3 * C);
        x1s = ldg_f(psrc + (int64_t)(0 * C + c));
    }

    // shift2 for x2 packed
    float x2s = ldg_f(p + (int64_t)(1 * C + c));
    if (c < cq && h >= 1) {
        const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h - 1)) * (int64_t)(3 * C);
        x2s = ldg_f(psrc + (int64_t)(1 * C + c));
    }
    if (c >= cq && c < ch && h <= H - 2) {
        const float* psrc = tb + ((int64_t)w * (int64_t)H + (int64_t)(h + 1)) * (int64_t)(3 * C);
        x2s = ldg_f(psrc + (int64_t)(1 * C + c));
    }
    if (c >= ch && c < c3q && w >= 1) {
        const float* psrc = tb + ((int64_t)(w - 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
        x2s = ldg_f(psrc + (int64_t)(1 * C + c));
    }
    if (c >= c3q && w <= W - 2) {
        const float* psrc = tb + ((int64_t)(w + 1) * (int64_t)H + (int64_t)h) * (int64_t)(3 * C);
        x2s = ldg_f(psrc + (int64_t)(1 * C + c));
    }

    int64_t gbase = ((int64_t)b * (int64_t)3 * (int64_t)C) + (int64_t)c;
    float g0 = ldg_f(gates + gbase + (int64_t)0 * (int64_t)C);
    float g1 = ldg_f(gates + gbase + (int64_t)1 * (int64_t)C);
    float g2 = ldg_f(gates + gbase + (int64_t)2 * (int64_t)C);

    out[idx] = g0 * x1s + g1 * x2s + g2 * x3;
}

torch::Tensor s2_attention_descriptor_cuda(torch::Tensor t3) {
    CHECK_INPUT(t3);
    TORCH_CHECK(t3.dim() == 4, "t3 must be [B,W,H,3C]");
    int B = (int)t3.size(0);
    int W = (int)t3.size(1);
    int H = (int)t3.size(2);
    int CC3 = (int)t3.size(3);
    TORCH_CHECK((CC3 % 3) == 0, "last dim must be 3C");
    int C = CC3 / 3;
    TORCH_CHECK((C % 4) == 0, "C must be divisible by 4");

    t3 = t3.contiguous();
    auto a = torch::empty({B, C}, t3.options());

    auto stream = at::cuda::getDefaultCUDAStream();
    dim3 grid((unsigned)B, (unsigned)C, 1);
    s2_descriptor_kernel<<<grid, 256, 0, stream>>>(
        (const float*)t3.data_ptr<float>(),
        (float*)a.data_ptr<float>(),
        B, W, H, C
    );
    return a;
}

torch::Tensor s2_attention_apply_cuda(torch::Tensor t3, torch::Tensor gates) {
    CHECK_INPUT(t3);
    CHECK_INPUT(gates);
    TORCH_CHECK(t3.dim() == 4, "t3 must be [B,W,H,3C]");
    TORCH_CHECK(gates.dim() == 3, "gates must be [B,3,C]");

    int B = (int)t3.size(0);
    int W = (int)t3.size(1);
    int H = (int)t3.size(2);
    int CC3 = (int)t3.size(3);
    TORCH_CHECK((CC3 % 3) == 0, "last dim must be 3C");
    int C = CC3 / 3;
    TORCH_CHECK((C % 4) == 0, "C must be divisible by 4");
    TORCH_CHECK((int)gates.size(0) == B, "gates B mismatch");
    TORCH_CHECK((int)gates.size(1) == 3, "gates dim1 must be 3");
    TORCH_CHECK((int)gates.size(2) == C, "gates C mismatch");

    t3 = t3.contiguous();
    gates = gates.contiguous();
    auto out = torch::empty({B, W, H, C}, t3.options());

    int64_t N = (int64_t)B * (int64_t)W * (int64_t)H * (int64_t)C;
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);

    auto stream = at::cuda::getDefaultCUDAStream();
    s2_apply_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)t3.data_ptr<float>(),
        (const float*)gates.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, W, H, C
    );
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor s2_attention_descriptor_cuda(torch::Tensor t3);
torch::Tensor s2_attention_apply_cuda(torch::Tensor t3, torch::Tensor gates);
"""


custom_ops_lib = load_inline(
    name="custom_s2_attention_ops_v6",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["s2_attention_descriptor_cuda", "s2_attention_apply_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = int(channel)
        self.k = int(k)
        self.mlp1 = nn.Linear(self.channel, self.channel, bias=False)
        self.gelu = nn.GELU()  # keep PyTorch exact default
        self.mlp2 = nn.Linear(self.channel, self.channel * self.k, bias=False)
        self.softmax = nn.Softmax(1)


class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.channels = int(channels)
        self.mlp1 = nn.Linear(self.channels, self.channels * 3)
        self.mlp2 = nn.Linear(self.channels, self.channels)
        self.split_attention = SplitAttention(channel=self.channels, k=3)


class ModelNew(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.s2_attention = S2Attention(channels=int(channels))
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,W,H] as in provided model
        b, c, w, h = x.size()

        t = x.permute(0, 2, 3, 1).contiguous()          # [B,W,H,C]
        t3 = self.s2_attention.mlp1(t).contiguous()     # [B,W,H,3C]

        if t3.is_cuda and t3.dtype == torch.float32:
            # Descriptor a matches reference: sum over (k,w,h) of shifted branches + identity
            a = self.custom_ops_lib.s2_attention_descriptor_cuda(t3)  # [B,C]

            sa = self.s2_attention.split_attention
            hat_a = sa.mlp2(sa.gelu(sa.mlp1(a)))  # [B,3C]
            gates = sa.softmax(hat_a.view(b, 3, c)).contiguous()  # [B,3,C]

            out = self.custom_ops_lib.s2_attention_apply_cuda(t3, gates)  # [B,W,H,C]
        else:
            # Fallback reference (CPU/other dtype)
            x_proj = t3
            x1 = x_proj[:, :, :, :c].clone()
            x2 = x_proj[:, :, :, c:2 * c].clone()
            x3 = x_proj[:, :, :, 2 * c:].clone()

            # spatial_shift1
            x1[:, 1:, :, : c // 4] = x1[:, : w - 1, :, : c // 4]
            x1[:, : w - 1, :, c // 4 : c // 2] = x1[:, 1:, :, c // 4 : c // 2]
            x1[:, :, 1:, c // 2 : c * 3 // 4] = x1[:, :, : h - 1, c // 2 : c * 3 // 4]
            x1[:, :, : h - 1, 3 * c // 4 :] = x1[:, :, 1:, 3 * c // 4 :]

            # spatial_shift2
            x2[:, :, 1:, : c // 4] = x2[:, :, : h - 1, : c // 4]
            x2[:, :, : h - 1, c // 4 : c // 2] = x2[:, :, 1:, c // 4 : c // 2]
            x2[:, 1:, :, c // 2 : c * 3 // 4] = x2[:, : w - 1, :, c // 2 : c * 3 // 4]
            x2[:, : w - 1, :, 3 * c // 4 :] = x2[:, 1:, :, 3 * c // 4 :]

            x_all = torch.stack([x1, x2, x3], 1)  # [B,3,W,H,C]
            # SplitAttention forward (inline)
            b0, k, ww, hh, cc = x_all.shape
            xa = x_all.reshape(b0, k, -1, cc)
            a = torch.sum(torch.sum(xa, 1), 1)
            sa = self.s2_attention.split_attention
            hat_a = sa.mlp2(sa.gelu(sa.mlp1(a))).reshape(b0, k, cc)
            bar_a = sa.softmax(hat_a)
            out = (bar_a.unsqueeze(-2) * xa).sum(1).reshape(b0, ww, hh, cc)

        y = self.s2_attention.mlp2(out).permute(0, 3, 1, 2).contiguous()
        return y