import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------
# Custom CUDA/C++ extension
# ----------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline __device__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
static inline __device__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
static inline __device__ float warp_bcast0(float v) { return __shfl_sync(0xffffffff, v, 0); }

// 2 positions per warp: for fixed (n,h) process (w0=2*warp_in_row, w1=w0+1) if in-range.
// Grid: blocks = N * H * ceil(W/2) = N*7*4 = N*28
// Block: 1 warp (32 threads).
__global__ __launch_bounds__(32, 8)
void cca7_row2_warp_kernel(
    const float* __restrict__ qH, // [N*W, 7, Cq]
    const float* __restrict__ kH, // [N*W, Cq, 7]
    const float* __restrict__ vH, // [N*W, Cv, 7]
    const float* __restrict__ qW, // [N*H, 7, Cq]
    const float* __restrict__ kW, // [N*H, Cq, 7]
    const float* __restrict__ vW, // [N*H, Cv, 7]
    float* __restrict__ out,      // [N, Cv, 7, 7]
    int N, int Cq, int Cv
){
    int lane = (int)(threadIdx.x & 31);
    int block = (int)blockIdx.x; // 0..N*28-1

    int n = block / 28;
    int rem = block - n * 28;
    int h = rem / 4;         // 0..6
    int wp = rem - h * 4;    // 0..3
    int w0 = wp * 2;
    int w1 = w0 + 1;
    bool valid1 = (w1 < 7);

    // Indices into folded batches
    int bw0 = n * 7 + w0;
    int bw1 = n * 7 + w1;
    int bh  = n * 7 + h;

    // Pointers for both positions
    const float* qH0 = qH + ((long long)bw0 * 7 + h) * Cq; // [Cq]
    const float* kH0 = kH + (long long)bw0 * Cq * 7;      // [Cq,7]
    const float* vH0 = vH + (long long)bw0 * Cv * 7;      // [Cv,7]

    const float* qH1 = valid1 ? (qH + ((long long)bw1 * 7 + h) * Cq) : nullptr;
    const float* kH1 = valid1 ? (kH + (long long)bw1 * Cq * 7) : nullptr;
    const float* vH1 = valid1 ? (vH + (long long)bw1 * Cv * 7) : nullptr;

    const float* qW0 = qW + ((long long)bh * 7 + w0) * Cq; // [Cq]
    const float* kW0 = kW + (long long)bh * Cq * 7;        // [Cq,7]
    const float* vW0 = vW + (long long)bh * Cv * 7;        // [Cv,7]

    const float* qW1 = valid1 ? (qW + ((long long)bh * 7 + w1) * Cq) : nullptr;

    // Shared staging for V lines to remove repeated global loads in Cv loop.
    // Per position we need vH[pos][Cv,7] and vW[bh][Cv,7]; vW is same bh for both.
    // We'll stage for each c handled by lane-stride into shared.
    extern __shared__ float smem[]; // size: (2*7 + 2*7) * Cv? too big. Instead stage per-c chunk:
    // We'll stage only 7+7 values for the current channel 'c' on the fly in registers, but that still loads global.
    // Better: stage vW for all lanes for current c in registers using vectorized loads? Can't vectorize due to stride 7.
    // Compromise: stage vW and vH for current c into shared once per c-chunk across lanes (each lane loads some y's).
    // Layout in shared for one c at a time:
    float* sh_vH0 = smem;          // 7 floats
    float* sh_vH1 = sh_vH0 + 7;    // 7 floats
    float* sh_vW0 = sh_vH1 + 7;    // 7 floats (same for both positions)
    float* sh_vW1 = sh_vW0 + 7;    // 7 floats (same as sh_vW0, but keep to avoid conditionals)

    // Compute logits for both positions; lanes 0..6 compute one j/y each.
    float logitH0 = -1.0e30f, logitW0 = -1.0e30f;
    float logitH1 = -1.0e30f, logitW1 = -1.0e30f;

    if (lane < 7) {
        int j = lane;
        // H logits for pos0 and pos1
        float s0 = 0.f;
        float s1 = 0.f;
        #pragma unroll 4
        for (int c = 0; c < 1024; c += 1) { // will break dynamically
            if (c >= Cq) break;
            float q0v = __ldg(qH0 + c);
            float k0v = __ldg(kH0 + c * 7 + j);
            s0 = fmaf(q0v, k0v, s0);
            if (valid1) {
                float q1v = __ldg(qH1 + c);
                float k1v = __ldg(kH1 + c * 7 + j);
                s1 = fmaf(q1v, k1v, s1);
            }
        }
        if (j == h) s0 = -1.0e20f;
        logitH0 = s0;
        if (valid1) {
            if (j == h) s1 = -1.0e20f;
            logitH1 = s1;
        }

        // W logits (same kW0 for both, different qW)
        int y = lane;
        s0 = 0.f; s1 = 0.f;
        #pragma unroll 4
        for (int c = 0; c < 1024; c += 1) {
            if (c >= Cq) break;
            float k = __ldg(kW0 + c * 7 + y);
            s0 = fmaf(__ldg(qW0 + c), k, s0);
            if (valid1) s1 = fmaf(__ldg(qW1 + c), k, s1);
        }
        logitW0 = s0;
        if (valid1) logitW1 = s1;
    }

    // Softmax normalization for both positions.
    float max0 = -1.0e30f, max1 = -1.0e30f;
    if (lane < 7) {
        max0 = fmaxf(logitH0, logitW0);
        if (valid1) max1 = fmaxf(logitH1, logitW1);
    }
    float m0 = warp_bcast0(warp_max(max0));
    float m1 = valid1 ? warp_bcast0(warp_max(max1)) : 0.f;

    float sum0 = 0.f, sum1 = 0.f;
    if (lane < 7) {
        sum0 = __expf(logitH0 - m0) + __expf(logitW0 - m0);
        if (valid1) sum1 = __expf(logitH1 - m1) + __expf(logitW1 - m1);
    }
    float d0 = warp_bcast0(warp_sum(sum0));
    float invd0 = 1.0f / d0;
    float d1 = valid1 ? warp_bcast0(warp_sum(sum1)) : 1.f;
    float invd1 = valid1 ? (1.0f / d1) : 0.f;

    // Precompute probabilities per lane for both positions (for its j/y); broadcast later.
    float pH0 = 0.f, pW0 = 0.f, pH1 = 0.f, pW1 = 0.f;
    if (lane < 7) {
        pH0 = __expf(logitH0 - m0) * invd0;
        pW0 = __expf(logitW0 - m0) * invd0;
        if (valid1) {
            pH1 = __expf(logitH1 - m1) * invd1;
            pW1 = __expf(logitW1 - m1) * invd1;
        }
    }

    // Accumulate over Cv channels.
    for (int c = lane; c < Cv; c += 32) {
        // Stage 7+7 V values for this channel into shared (lanes 0..6 write).
        if (lane < 7) {
            sh_vH0[lane] = __ldg(vH0 + (long long)c * 7 + lane);
            sh_vW0[lane] = __ldg(vW0 + (long long)c * 7 + lane);
            if (valid1) sh_vH1[lane] = __ldg(vH1 + (long long)c * 7 + lane);
            else sh_vH1[lane] = 0.f;
            sh_vW1[lane] = sh_vW0[lane];
        }
        // warp-synchronous; no __syncthreads needed

        float acc0 = 0.f;
        float acc1 = 0.f;

        #pragma unroll
        for (int j = 0; j < 7; ++j) {
            float ph0 = __shfl_sync(0xffffffff, pH0, j);
            acc0 = fmaf(ph0, sh_vH0[j], acc0);
            if (valid1) {
                float ph1 = __shfl_sync(0xffffffff, pH1, j);
                acc1 = fmaf(ph1, sh_vH1[j], acc1);
            }
        }
        #pragma unroll
        for (int y = 0; y < 7; ++y) {
            float pw0 = __shfl_sync(0xffffffff, pW0, y);
            acc0 = fmaf(pw0, sh_vW0[y], acc0);
            if (valid1) {
                float pw1 = __shfl_sync(0xffffffff, pW1, y);
                acc1 = fmaf(pw1, sh_vW1[y], acc1);
            }
        }

        long long out0 = ((long long)n * Cv + c) * 49 + (long long)h * 7 + w0;
        out[out0] = acc0;
        if (valid1) {
            long long out1 = ((long long)n * Cv + c) * 49 + (long long)h * 7 + w1;
            out[out1] = acc1;
        }
    }
}

torch::Tensor criss_cross_attention_fused7_cuda(
    torch::Tensor qH, torch::Tensor kH, torch::Tensor vH,
    torch::Tensor qW, torch::Tensor kW, torch::Tensor vW,
    int64_t N
){
    CHECK_INPUT(qH); CHECK_INPUT(kH); CHECK_INPUT(vH);
    CHECK_INPUT(qW); CHECK_INPUT(kW); CHECK_INPUT(vW);

    TORCH_CHECK(qH.dim() == 3 && kH.dim() == 3 && vH.dim() == 3, "qH/kH/vH must be 3D");
    TORCH_CHECK(qW.dim() == 3 && kW.dim() == 3 && vW.dim() == 3, "qW/kW/vW must be 3D");

    int BW = (int)qH.size(0);
    int H  = (int)qH.size(1);
    int Cq = (int)qH.size(2);
    TORCH_CHECK(H == 7, "Only H=7 supported");
    TORCH_CHECK((int)kH.size(0) == BW && (int)kH.size(1) == Cq && (int)kH.size(2) == 7, "kH must be [BW,Cq,7]");
    int Cv = (int)vH.size(1);
    TORCH_CHECK((int)vH.size(0) == BW && (int)vH.size(2) == 7, "vH must be [BW,Cv,7]");

    int BH = (int)qW.size(0);
    int W  = (int)qW.size(1);
    TORCH_CHECK(W == 7, "Only W=7 supported");
    TORCH_CHECK((int)qW.size(2) == Cq, "qW Cq must match");
    TORCH_CHECK((int)kW.size(0) == BH && (int)kW.size(1) == Cq && (int)kW.size(2) == 7, "kW must be [BH,Cq,7]");
    TORCH_CHECK((int)vW.size(0) == BH && (int)vW.size(1) == Cv && (int)vW.size(2) == 7, "vW must be [BH,Cv,7]");

    TORCH_CHECK((int64_t)BW == N * 7, "qH BW must be N*7");
    TORCH_CHECK((int64_t)BH == N * 7, "qW BH must be N*7");

    c10::cuda::CUDAGuard device_guard(qH.device());
    auto stream = at::cuda::getDefaultCUDAStream(qH.device().index());

    auto out = torch::empty({N, Cv, 7, 7}, qH.options());

    int blocks = (int)(N * 28); // N * H * ceil(W/2)
    int threads = 32;
    size_t shmem = (size_t)(28 * sizeof(float)); // 4 arrays * 7 floats
    cca7_row2_warp_kernel<<<blocks, threads, shmem, stream>>>(
        (const float*)qH.data_ptr<float>(),
        (const float*)kH.data_ptr<float>(),
        (const float*)vH.data_ptr<float>(),
        (const float*)qW.data_ptr<float>(),
        (const float*)kW.data_ptr<float>(),
        (const float*)vW.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)N, Cq, Cv
    );

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor criss_cross_attention_fused7_cuda(
    torch::Tensor qH, torch::Tensor kH, torch::Tensor vH,
    torch::Tensor qW, torch::Tensor kW, torch::Tensor vW,
    int64_t N
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_criss_cross_attention_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["criss_cross_attention_fused7_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-lineinfo"],
    verbose=False,
)

# ----------------------------
# Reference helpers
# ----------------------------

def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class Model(nn.Module):
    def __init__(self, in_dim):
        super(Model, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width, x.device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x

# ----------------------------
# Optimized Model
# ----------------------------

class ModelNew(nn.Module):
    """Criss-Cross Attention with optimized fused CUDA kernel for H=W=7 attention core."""
    def __init__(self, in_dim):
        super(ModelNew, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)  # fallback
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ops = custom_ops_lib

    def forward(self, x):
        N, _, H, W = x.size()

        if (not x.is_cuda) or (x.dtype != torch.float32) or (not x.is_contiguous()) or (H != 7) or (W != 7):
            return Model.forward(self, x)

        proj_query = self.query_conv(x).contiguous()
        proj_key = self.key_conv(x).contiguous()
        proj_value = self.value_conv(x).contiguous()
        Cq = proj_query.size(1)
        Cv = proj_value.size(1)

        # Layouts matching the CUDA kernel contract
        qH = proj_query.permute(0, 3, 1, 2).contiguous().view(N * W, Cq, H).permute(0, 2, 1).contiguous()  # [N*7,7,Cq]
        qW = proj_query.permute(0, 2, 1, 3).contiguous().view(N * H, Cq, W).permute(0, 2, 1).contiguous()  # [N*7,7,Cq]

        kH = proj_key.permute(0, 3, 1, 2).contiguous().view(N * W, Cq, H).contiguous()  # [N*7,Cq,7]
        kW = proj_key.permute(0, 2, 1, 3).contiguous().view(N * H, Cq, W).contiguous()  # [N*7,Cq,7]

        vH = proj_value.permute(0, 3, 1, 2).contiguous().view(N * W, Cv, H).contiguous()  # [N*7,Cv,7]
        vW = proj_value.permute(0, 2, 1, 3).contiguous().view(N * H, Cv, W).contiguous()  # [N*7,Cv,7]

        if not (qH.is_contiguous() and kH.is_contiguous() and vH.is_contiguous() and
                qW.is_contiguous() and kW.is_contiguous() and vW.is_contiguous()):
            return Model.forward(self, x)

        out_sum = self.ops.criss_cross_attention_fused7_cuda(qH, kH, vH, qW, kW, vW, N)
        return self.gamma * out_sum + x