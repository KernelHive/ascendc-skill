import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ----------------------------
# Reference helpers (fallback)
# ----------------------------
def position(H, W, is_cuda=True, device=None, dtype=torch.float32):
    """Generate 2D positional encoding grid with coordinates in [-1, 1]."""
    if device is None:
        device = "cuda" if is_cuda else "cpu"
    loc_w = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)
    loc_h = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride_op(x, stride):
    """Downsample feature map by taking every stride-th element."""
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    """Initialize tensor values to 0.5."""
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    """Initialize tensor values to 0."""
    if tensor is not None:
        tensor.data.fill_(0.0)


# ----------------------------
# Custom CUDA extension
# ----------------------------
acmix_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_F32(x)

__device__ __forceinline__ int64_t idx4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t C, int64_t H, int64_t W) {
    return ((a * C + b) * H + c) * W + d;
}

__device__ __forceinline__ int reflect_int(int x, int size) {
    // PyTorch ReflectionPad2d behavior (no repeating border).
    if (size <= 1) return 0;
    while (x < 0 || x >= size) {
        if (x < 0) x = -x;
        if (x >= size) x = 2 * size - 2 - x;
    }
    return x;
}

__device__ __forceinline__ float warp_allreduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return __shfl_sync(0xffffffff, v, 0);
}

// Compute out_att only:
// Inputs:
//  x:      [B, Cin, H, W] (only used to build pe via conv_p; conv_p takes 2-channel pos grid, so x not needed here)
//  q,k,v:  [B, out_planes, H, W] contiguous float32
//  convp_w:[head_dim, 2, 1, 1] contiguous float32
//  convp_b:[head_dim] contiguous float32
// Output:
//  out_att:[B, out_planes, H, W] contiguous float32
//
// Supports: kernel_att=7, stride=1, dilation=1, H=W=7, head=4, out_planes divisible by head.
__global__ __launch_bounds__(128, 2)
void acmix_out_att_kernel_7x7(
    const float* __restrict__ q,      // [B, C, 7, 7]
    const float* __restrict__ k,      // [B, C, 7, 7]
    const float* __restrict__ v,      // [B, C, 7, 7]
    const float* __restrict__ convp_w,// [head_dim, 2] flattened
    const float* __restrict__ convp_b,// [head_dim]
    float* __restrict__ out_att,      // [B, C, 7, 7]
    int B, int C, int head, int head_dim
){
    // Map: one warp computes one (b, h0, y, x) query position.
    // Each lane covers a subset of head_dim channels (cd) and accumulates logits for its cd subset, then
    // cooperatively reduces over cd subsets by doing a second pass that computes logits exactly (still looping cd),
    // but using warp-parallel cd chunks.
    //
    // Given head_dim=128 for C=512, head=4. K=7 => KK=49.
    // This kernel is tuned for the benchmark; it is correct but not maximal-performance.

    constexpr int H = 7;
    constexpr int W = 7;
    constexpr int K = 7;
    constexpr int KK = 49;
    constexpr int PAD = 3;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    int qpos = (int)(blockIdx.x * warps_per_block + warp); // 0..B*head*H*W-1
    int total = B * head * H * W;
    if (qpos >= total) return;

    int t = qpos;
    int xq = t % W; t /= W;
    int yq = t % H; t /= H;
    int h0 = t % head; t /= head;
    int b0 = t;

    // Compute q_pe for each cd on the fly using conv_p over normalized pos grid [-1,1].
    // pos_w = -1 + 2*x/(W-1), pos_h = -1 + 2*y/(H-1)
    float posx = (W == 1) ? 0.f : (-1.f + 2.f * (float)xq / (float)(W - 1));
    float posy = (H == 1) ? 0.f : (-1.f + 2.f * (float)yq / (float)(H - 1));

    // We will compute logits[p] for p in 0..48.
    float logits[KK];
    #pragma unroll
    for (int p = 0; p < KK; ++p) logits[p] = 0.f;

    float scaling = rsqrtf((float)head_dim);

    // Parallelize reduction over cd: each lane handles cd = lane, lane+32, ...
    for (int cd = lane; cd < head_dim; cd += 32) {
        int oc = h0 * head_dim + cd;
        float qv = q[idx4(b0, oc, yq, xq, C, H, W)] * scaling;

        // q_pe[cd] = convp_w[cd,0]*posx + convp_w[cd,1]*posy + b
        const float* wrow = convp_w + (int64_t)cd * 2;
        float qpe = fmaf(wrow[0], posx, fmaf(wrow[1], posy, convp_b[cd]));

        // For each patch position p (ky,kx), sample k and rpe (which is pe at reflected coords).
        // logits[p] += qv * (k_sample + qpe - rpe_sample)
        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            int iy = yq + ky - PAD;
            int ry = reflect_int(iy, H);
            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                int ix = xq + kx - PAD;
                int rx = reflect_int(ix, W);

                int p = ky * K + kx;
                float kval = k[idx4(b0, oc, ry, rx, C, H, W)];

                // rpe is pe at reflected location (same conv_p, but at (ry,rx))
                float posx_r = (W == 1) ? 0.f : (-1.f + 2.f * (float)rx / (float)(W - 1));
                float posy_r = (H == 1) ? 0.f : (-1.f + 2.f * (float)ry / (float)(H - 1));
                float rpe = fmaf(wrow[0], posx_r, fmaf(wrow[1], posy_r, convp_b[cd]));

                logits[p] = fmaf(qv, (kval + qpe - rpe), logits[p]);
            }
        }
    }

    // Now we need logits summed over all cd lanes. Do warp reduction for each p.
    #pragma unroll
    for (int p = 0; p < KK; ++p) {
        float v = logits[p];
        // tree reduce across warp for each p
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        logits[p] = __shfl_sync(0xffffffff, v, 0);
    }

    // Lane 0 computes softmax and broadcasts normalization; also stores att weights in registers for later.
    float attw[KK];
    float maxv = -1.0e30f;
    float sum = 0.f;

    if (lane == 0) {
        #pragma unroll
        for (int p = 0; p < KK; ++p) maxv = fmaxf(maxv, logits[p]);
        #pragma unroll
        for (int p = 0; p < KK; ++p) {
            float e = __expf(logits[p] - maxv);
            attw[p] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        #pragma unroll
        for (int p = 0; p < KK; ++p) attw[p] *= inv;
    }

    // Broadcast attw[p] from lane0 to all lanes via shuffles (one by one).
    // (KK=49, acceptable for this benchmark)
    float att[KK];
    #pragma unroll
    for (int p = 0; p < KK; ++p) {
        float v0 = 0.f;
        if (lane == 0) v0 = attw[p];
        att[p] = __shfl_sync(0xffffffff, v0, 0);
    }

    // Compute out_att for each cd in parallel: out_cd = sum_p att[p] * v_sample(cd,p)
    for (int cd = lane; cd < head_dim; cd += 32) {
        int oc = h0 * head_dim + cd;
        float outv = 0.f;

        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            int iy = yq + ky - PAD;
            int ry = reflect_int(iy, H);
            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                int ix = xq + kx - PAD;
                int rx = reflect_int(ix, W);
                int p = ky * K + kx;
                float vv = v[idx4(b0, oc, ry, rx, C, H, W)];
                outv = fmaf(att[p], vv, outv);
            }
        }

        out_att[idx4(b0, oc, yq, xq, C, H, W)] = outv;
    }
}

torch::Tensor acmix_out_att_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor convp_w,
    torch::Tensor convp_b,
    int head
){
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(convp_w);
    CHECK_INPUT(convp_b);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,C,H,W]");
    int B = (int)q.size(0);
    int C = (int)q.size(1);
    int H = (int)q.size(2);
    int W = (int)q.size(3);

    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "k and v must match q shape");
    TORCH_CHECK(H == 7 && W == 7, "fast path supports H=W=7 only");
    TORCH_CHECK(C % head == 0, "C must be divisible by head");
    int head_dim = C / head;

    TORCH_CHECK(convp_w.dim() == 4 && convp_w.size(0) == head_dim && convp_w.size(1) == 2, "convp_w must be [head_dim,2,1,1]");
    TORCH_CHECK(convp_w.size(2) == 1 && convp_w.size(3) == 1, "convp_w must be [head_dim,2,1,1]");
    TORCH_CHECK(convp_b.dim() == 1 && convp_b.size(0) == head_dim, "convp_b must be [head_dim]");

    auto out = torch::empty_like(q);

    const int threads = 128;
    const int warps_per_block = threads / 32;
    int total = B * head * H * W;
    int blocks = (total + warps_per_block - 1) / warps_per_block;

    acmix_out_att_kernel_7x7<<<blocks, threads>>>(
        (const float*)q.data_ptr<float>(),
        (const float*)k.data_ptr<float>(),
        (const float*)v.data_ptr<float>(),
        (const float*)convp_w.data_ptr<float>(),
        (const float*)convp_b.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, C, head, head_dim
    );

    return out;
}
"""

acmix_cpp_src = r"""
#include <torch/extension.h>
torch::Tensor acmix_out_att_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor convp_w,
    torch::Tensor convp_b,
    int head
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_acmix_outatt_v1",
    cpp_sources=acmix_cpp_src,
    cuda_sources=acmix_cuda_src,
    functions=["acmix_out_att_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


# ----------------------------
# Original module
# ----------------------------
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(
            self.kernel_conv * self.kernel_conv * self.head_dim,
            out_planes,
            kernel_size=self.kernel_conv,
            bias=True,
            groups=self.head_dim,
            padding=1,
            stride=stride,
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.0
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(position(h, w, x.is_cuda, device=x.device, dtype=x.dtype))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride_op(q_att, self.stride)
            q_pe = stride_op(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(
            torch.cat(
                [
                    q.view(b, self.head, self.head_dim, h * w),
                    k.view(b, self.head, self.head_dim, h * w),
                    v.view(b, self.head, self.head_dim, h * w),
                ],
                1,
            )
        )
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


# ----------------------------
# Optimized Model wrapper
# ----------------------------
class ModelNew(nn.Module):
    """ACmix optimized: custom CUDA for attention branch (out_att) for H=W=7, k_att=7, stride=1, head=4."""
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super().__init__()
        self.acmix = ACmix(
            in_planes=in_planes,
            out_planes=out_planes,
            kernel_att=kernel_att,
            head=head,
            kernel_conv=kernel_conv,
            stride=stride,
            dilation=dilation,
        )
        self.custom_ops_lib = custom_ops_lib

    def forward(self, x):
        # Always compute q/k/v with PyTorch convs (cuDNN fast).
        q = self.acmix.conv1(x)
        k = self.acmix.conv2(x)
        v = self.acmix.conv3(x)

        # Convolution branch (leave to PyTorch).
        b, c, h, w = q.shape
        f_all = self.acmix.fc(
            torch.cat(
                [
                    q.view(b, self.acmix.head, self.acmix.head_dim, h * w),
                    k.view(b, self.acmix.head, self.acmix.head_dim, h * w),
                    v.view(b, self.acmix.head, self.acmix.head_dim, h * w),
                ],
                1,
            )
        )
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.acmix.dep_conv(f_conv)

        # Attention branch: custom CUDA fast path.
        use_cuda_path = (
            x.is_cuda
            and x.dtype == torch.float32
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            and self.acmix.kernel_att == 7
            and self.acmix.stride == 1
            and self.acmix.dilation == 1
            and h == 7 and w == 7
            and self.acmix.head == 4
            and self.acmix.conv_p.weight.is_cuda
        )

        if use_cuda_path:
            convp_w = self.acmix.conv_p.weight.contiguous()
            convp_b = self.acmix.conv_p.bias.contiguous()
            out_att = self.custom_ops_lib.acmix_out_att_forward_cuda(q, k, v, convp_w, convp_b, int(self.acmix.head))
        else:
            # Fallback to original attention computation.
            scaling = float(self.acmix.head_dim) ** -0.5
            h_out, w_out = h // self.acmix.stride, w // self.acmix.stride
            pe = self.acmix.conv_p(position(h, w, x.is_cuda, device=x.device, dtype=x.dtype))
            q_att = q.view(b * self.acmix.head, self.acmix.head_dim, h, w) * scaling
            k_att = k.view(b * self.acmix.head, self.acmix.head_dim, h, w)
            v_att = v.view(b * self.acmix.head, self.acmix.head_dim, h, w)
            q_pe = pe

            unfold_k = self.acmix.unfold(self.acmix.pad_att(k_att)).view(
                b * self.acmix.head, self.acmix.head_dim, self.acmix.kernel_att * self.acmix.kernel_att, h_out, w_out
            )
            unfold_rpe = self.acmix.unfold(self.acmix.pad_att(pe)).view(
                1, self.acmix.head_dim, self.acmix.kernel_att * self.acmix.kernel_att, h_out, w_out
            )
            att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
            att = self.acmix.softmax(att)
            out_att_u = self.acmix.unfold(self.acmix.pad_att(v_att)).view(
                b * self.acmix.head, self.acmix.head_dim, self.acmix.kernel_att * self.acmix.kernel_att, h_out, w_out
            )
            out_att = (att.unsqueeze(1) * out_att_u).sum(2).view(b, self.acmix.out_planes, h_out, w_out)

        # Mix (extract scalars on Python side; do not call .item() in CUDA).
        return self.acmix.rate1 * out_att + self.acmix.rate2 * out_conv