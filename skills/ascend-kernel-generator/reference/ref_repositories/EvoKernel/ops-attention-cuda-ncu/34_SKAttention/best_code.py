import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.cpp_extension import load_inline

# Fused CUDA ops:
# 1) sk_u_mean_from_xs: compute S = mean_{h,w}(sum_k xk) => [B,C]
# 2) sk_attention_fused_x_cuda: softmax(logits over K) and weighted sum over branches => out [B,C,H,W]
sk_attention_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float ldgf(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// -----------------------
// Op (1): S = mean(U)
// -----------------------
template<int K_SPEC>
__global__ void sk_u_mean_from_xs_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ x3,
    const float* __restrict__ x4,
    const float* __restrict__ x5,
    const float* __restrict__ x6,
    const float* __restrict__ x7,
    float* __restrict__ s_out,  // [B,C]
    int K, int B, int C, int H, int W
){
    int bc = (int)blockIdx.x; // 0..B*C-1
    int b = bc / C;
    int c = bc - b * C;
    int HW = H * W;
    int base = (b * C + c) * HW;

    const float* xs[8] = {x0,x1,x2,x3,x4,x5,x6,x7};

    float local = 0.f;
    for (int idx = (int)threadIdx.x; idx < HW; idx += (int)blockDim.x) {
        float u = 0.f;
        if constexpr (K_SPEC > 0) {
            #pragma unroll
            for (int k = 0; k < K_SPEC; ++k) u += ldgf(xs[k] + base + idx);
        } else {
            for (int k = 0; k < K; ++k) u += ldgf(xs[k] + base + idx);
        }
        local += u;
    }

    // block reduction via warp sums
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float v = warp_reduce_sum(local);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    float outsum = 0.f;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        outsum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        outsum = warp_reduce_sum(outsum);
    }
    if (threadIdx.x == 0) s_out[b * C + c] = outsum * (1.0f / (float)HW);
}

// Special warp fast path for K=4, HW=49 -> directly produce mean(U) for each (b,c)
__global__ void sk_u_mean_k4_hw49_warp_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ x3,
    float* __restrict__ s_out, // [B,C]
    int B, int C
){
    int bc = (int)blockIdx.x;
    int b = bc / C;
    int c = bc - b * C;
    constexpr int HW = 49;
    int lane = threadIdx.x & 31;

    int base = (b * C + c) * HW;
    float local = 0.f;

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int idx = lane + (it << 5);
        if (idx < HW) {
            float a0 = ldgf(x0 + base + idx);
            float a1 = ldgf(x1 + base + idx);
            float a2 = ldgf(x2 + base + idx);
            float a3 = ldgf(x3 + base + idx);
            local += (a0 + a1 + a2 + a3);
        }
    }

    float sum = warp_reduce_sum(local);
    if (lane == 0) s_out[b * C + c] = sum * (1.0f / 49.0f);
}

// -----------------------
// Op (2): attention fuse
// -----------------------
template<int K_SPEC>
__global__ void sk_attention_fused_generic_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ x3,
    const float* __restrict__ x4,
    const float* __restrict__ x5,
    const float* __restrict__ x6,
    const float* __restrict__ x7,
    const float* __restrict__ attn_logits, // [K,B,C]
    float* __restrict__ out,               // [B,C,H,W]
    float* __restrict__ u_sum,             // [B,C] optional
    int K, int B, int C, int H, int W, int write_usum
) {
    int bc = (int)blockIdx.x; // 0..B*C-1
    int b = bc / C;
    int c = bc - b * C;
    int HW = H * W;

    // compute weights in thread0, store in shared (small), sync once
    __shared__ float w_sh[8];
    if (threadIdx.x == 0) {
        float maxv = -1e20f;
        if constexpr (K_SPEC > 0) {
            #pragma unroll
            for (int k = 0; k < K_SPEC; ++k) {
                float v = ldgf(attn_logits + (k * B * C + b * C + c));
                maxv = v > maxv ? v : maxv;
            }
            float denom = 0.f;
            #pragma unroll
            for (int k = 0; k < K_SPEC; ++k) {
                float v = ldgf(attn_logits + (k * B * C + b * C + c));
                float e = __expf(v - maxv);
                w_sh[k] = e;
                denom += e;
            }
            float inv = 1.f / denom;
            #pragma unroll
            for (int k = 0; k < K_SPEC; ++k) w_sh[k] *= inv;
        } else {
            for (int k = 0; k < K; ++k) {
                float v = ldgf(attn_logits + (k * B * C + b * C + c));
                maxv = v > maxv ? v : maxv;
            }
            float denom = 0.f;
            for (int k = 0; k < K; ++k) {
                float v = ldgf(attn_logits + (k * B * C + b * C + c));
                float e = __expf(v - maxv);
                w_sh[k] = e;
                denom += e;
            }
            float inv = 1.f / denom;
            for (int k = 0; k < K; ++k) w_sh[k] *= inv;
        }
    }
    __syncthreads();

    const float* xs[8] = {x0,x1,x2,x3,x4,x5,x6,x7};
    int base = (b * C + c) * HW;

    float local_usum = 0.f;

    for (int idx = (int)threadIdx.x; idx < HW; idx += (int)blockDim.x) {
        float v = 0.f;
        float u = 0.f;

        if constexpr (K_SPEC > 0) {
            #pragma unroll
            for (int k = 0; k < K_SPEC; ++k) {
                float a = ldgf(xs[k] + base + idx);
                u += a;
                v = fmaf(w_sh[k], a, v);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                float a = ldgf(xs[k] + base + idx);
                u += a;
                v = fmaf(w_sh[k], a, v);
            }
        }

        out[base + idx] = v;
        local_usum += u;
    }

    if (!write_usum) return;

    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float vv = warp_reduce_sum(local_usum);
    if (lane == 0) warp_sums[warp] = vv;
    __syncthreads();
    float outsum = 0.f;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        outsum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        outsum = warp_reduce_sum(outsum);
    }
    if (threadIdx.x == 0) u_sum[b * C + c] = outsum;
}

// warp fast path for K=4, HW=49
__global__ void sk_attention_fused_k4_hw49_warp_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    const float* __restrict__ x3,
    const float* __restrict__ attn_logits, // [4,B,C]
    float* __restrict__ out,               // [B,C,7,7]
    float* __restrict__ u_sum,             // [B,C] optional
    int B, int C, int write_usum
) {
    int bc = (int)blockIdx.x;
    int b = bc / C;
    int c = bc - b * C;
    constexpr int HW = 49;

    int lane = threadIdx.x & 31;
    unsigned mask = 0xffffffffu;

    float w0=0.f, w1=0.f, w2=0.f, w3=0.f;
    if (lane == 0) {
        float l0 = ldgf(attn_logits + (0 * B * C + b * C + c));
        float l1 = ldgf(attn_logits + (1 * B * C + b * C + c));
        float l2 = ldgf(attn_logits + (2 * B * C + b * C + c));
        float l3 = ldgf(attn_logits + (3 * B * C + b * C + c));
        float maxv = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
        float e0 = __expf(l0 - maxv);
        float e1 = __expf(l1 - maxv);
        float e2 = __expf(l2 - maxv);
        float e3 = __expf(l3 - maxv);
        float inv = 1.f / (e0 + e1 + e2 + e3);
        w0 = e0 * inv; w1 = e1 * inv; w2 = e2 * inv; w3 = e3 * inv;
    }
    w0 = __shfl_sync(mask, w0, 0);
    w1 = __shfl_sync(mask, w1, 0);
    w2 = __shfl_sync(mask, w2, 0);
    w3 = __shfl_sync(mask, w3, 0);

    int base = (b * C + c) * HW;

    float local_usum = 0.f;

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int idx = lane + (it << 5);
        if (idx < HW) {
            float a0 = ldgf(x0 + base + idx);
            float a1 = ldgf(x1 + base + idx);
            float a2 = ldgf(x2 + base + idx);
            float a3 = ldgf(x3 + base + idx);
            float u = a0 + a1 + a2 + a3;
            float v = fmaf(w0, a0, fmaf(w1, a1, fmaf(w2, a2, w3 * a3)));
            out[base + idx] = v;
            local_usum += u;
        }
    }

    if (!write_usum) return;
    float sum = warp_reduce_sum(local_usum);
    if (lane == 0) u_sum[b * C + c] = sum;
}

// -------- C++/CUDA bindings --------
std::vector<torch::Tensor> sk_u_mean_from_xs_cuda(std::vector<torch::Tensor> xs) {
    TORCH_CHECK(xs.size() >= 1 && xs.size() <= 8, "xs size must be in [1,8]");
    auto x0 = xs[0];
    TORCH_CHECK(x0.is_cuda(), "xs[0] must be CUDA");
    TORCH_CHECK(x0.dtype() == torch::kFloat32, "xs[0] must be float32");
    TORCH_CHECK(x0.is_contiguous(), "xs[0] must be contiguous");
    TORCH_CHECK(x0.dim() == 4, "xs[0] must be [B,C,H,W]");

    int64_t B64 = x0.size(0), C64 = x0.size(1), H64 = x0.size(2), W64 = x0.size(3);
    for (int i = 1; i < (int)xs.size(); ++i) {
        TORCH_CHECK(xs[i].is_cuda(), "xi must be CUDA");
        TORCH_CHECK(xs[i].dtype() == torch::kFloat32, "xi must be float32");
        TORCH_CHECK(xs[i].is_contiguous(), "xi must be contiguous");
        TORCH_CHECK(xs[i].dim() == 4, "xi must be [B,C,H,W]");
        TORCH_CHECK(xs[i].sizes() == x0.sizes(), "xi shape mismatch");
    }

    int K = (int)xs.size();
    int B = (int)B64, C = (int)C64, H = (int)H64, W = (int)W64;

    auto s = torch::empty({B64, C64}, x0.options());

    const float* xptrs[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    for (int i = 0; i < K; ++i) xptrs[i] = xs[i].data_ptr<float>();
    float* s_p = s.data_ptr<float>();

    int blocks = B * C;

    if (K == 4 && H == 7 && W == 7) {
        sk_u_mean_k4_hw49_warp_kernel<<<blocks, 32>>>(
            xptrs[0], xptrs[1], xptrs[2], xptrs[3], s_p, B, C
        );
        return {s};
    }

    int threads = 128; // reduction-friendly
    if (K == 4) {
        sk_u_mean_from_xs_kernel<4><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            s_p, K,B,C,H,W
        );
    } else if (K == 3) {
        sk_u_mean_from_xs_kernel<3><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            s_p, K,B,C,H,W
        );
    } else if (K == 2) {
        sk_u_mean_from_xs_kernel<2><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            s_p, K,B,C,H,W
        );
    } else if (K == 1) {
        sk_u_mean_from_xs_kernel<1><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            s_p, K,B,C,H,W
        );
    } else {
        sk_u_mean_from_xs_kernel<0><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            s_p, K,B,C,H,W
        );
    }
    return {s};
}

std::vector<torch::Tensor> sk_attention_fused_x_cuda(std::vector<torch::Tensor> xs, torch::Tensor attn_logits, bool write_usum) {
    TORCH_CHECK(attn_logits.is_cuda(), "attn_logits must be CUDA");
    TORCH_CHECK(attn_logits.dtype() == torch::kFloat32, "attn_logits must be float32");
    TORCH_CHECK(attn_logits.is_contiguous(), "attn_logits must be contiguous");
    TORCH_CHECK(attn_logits.dim() == 3, "attn_logits must be [K,B,C]");

    int64_t K64 = attn_logits.size(0);
    int64_t B64 = attn_logits.size(1);
    int64_t C64 = attn_logits.size(2);
    TORCH_CHECK(K64 >= 1 && K64 <= 8, "K must be in [1,8]");
    TORCH_CHECK((int64_t)xs.size() == K64, "xs.size() must equal K");

    auto x0 = xs[0];
    TORCH_CHECK(x0.is_cuda(), "x0 must be CUDA");
    TORCH_CHECK(x0.dtype() == torch::kFloat32, "x0 must be float32");
    TORCH_CHECK(x0.is_contiguous(), "x0 must be contiguous");
    TORCH_CHECK(x0.dim() == 4, "x0 must be [B,C,H,W]");
    TORCH_CHECK(x0.size(0) == B64 && x0.size(1) == C64, "x0 shape mismatch");

    int64_t H64 = x0.size(2);
    int64_t W64 = x0.size(3);

    for (int i = 1; i < (int)K64; ++i) {
        TORCH_CHECK(xs[i].is_cuda(), "xi must be CUDA");
        TORCH_CHECK(xs[i].dtype() == torch::kFloat32, "xi must be float32");
        TORCH_CHECK(xs[i].is_contiguous(), "xi must be contiguous");
        TORCH_CHECK(xs[i].dim() == 4, "xi must be [B,C,H,W]");
        TORCH_CHECK(xs[i].sizes() == x0.sizes(), "xi shape mismatch");
    }

    auto out = torch::empty({B64, C64, H64, W64}, x0.options());
    auto u_sum = torch::empty({B64, C64}, x0.options());

    int K = (int)K64, B = (int)B64, C = (int)C64, H = (int)H64, W = (int)W64;

    const float* xptrs[8] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    for (int i = 0; i < K; ++i) xptrs[i] = xs[i].data_ptr<float>();

    int blocks = B * C;
    const float* logits_p = attn_logits.data_ptr<float>();
    float* out_p = out.data_ptr<float>();
    float* us_p = u_sum.data_ptr<float>();
    int write_flag = write_usum ? 1 : 0;

    if (K == 4 && H == 7 && W == 7) {
        sk_attention_fused_k4_hw49_warp_kernel<<<blocks, 32>>>(
            xptrs[0], xptrs[1], xptrs[2], xptrs[3],
            logits_p, out_p, us_p, B, C, write_flag
        );
        return {out, u_sum};
    }

    int threads = 64;
    if (K == 4) {
        sk_attention_fused_generic_kernel<4><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            logits_p, out_p, us_p, K,B,C,H,W, write_flag
        );
    } else if (K == 3) {
        sk_attention_fused_generic_kernel<3><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            logits_p, out_p, us_p, K,B,C,H,W, write_flag
        );
    } else if (K == 2) {
        sk_attention_fused_generic_kernel<2><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            logits_p, out_p, us_p, K,B,C,H,W, write_flag
        );
    } else if (K == 1) {
        sk_attention_fused_generic_kernel<1><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            logits_p, out_p, us_p, K,B,C,H,W, write_flag
        );
    } else {
        sk_attention_fused_generic_kernel<0><<<blocks, threads>>>(
            xptrs[0],xptrs[1],xptrs[2],xptrs[3],xptrs[4],xptrs[5],xptrs[6],xptrs[7],
            logits_p, out_p, us_p, K,B,C,H,W, write_flag
        );
    }

    return {out, u_sum};
}
"""

sk_attention_cpp_source = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> sk_u_mean_from_xs_cuda(std::vector<torch::Tensor> xs);
std::vector<torch::Tensor> sk_attention_fused_x_cuda(std::vector<torch::Tensor> xs, torch::Tensor attn_logits, bool write_usum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sk_u_mean_from_xs_cuda", &sk_u_mean_from_xs_cuda, "sk_u_mean_from_xs_cuda");
  m.def("sk_attention_fused_x_cuda", &sk_attention_fused_x_cuda, "sk_attention_fused_x_cuda (out, u_sum)");
}
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_sk_attention_opt_fused_u_mean_v5",
    cpp_sources=sk_attention_cpp_source,
    cuda_sources=sk_attention_cuda_source,
    functions=None,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Selective Kernel Attention (SK Attention) with improved fusion:
    - compute S = mean(sum_k conv_outs[k]) in a custom CUDA kernel to avoid U materialization
    - keep fused softmax+weighted-sum kernel for final V
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.kernels = list(kernels)

        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in range(len(kernels))])

        self.custom_ops = custom_ops_lib

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]  # list([B,C,H,W]) float32

        # Fused: S = mean(U) without materializing U
        (S,) = self.custom_ops.sk_u_mean_from_xs_cuda(conv_outs)  # [B,C]

        Z = self.fc(S)  # [B,d]
        logits = [fc(Z) for fc in self.fcs]  # list([B,C])
        attn_logits = torch.stack(logits, 0).contiguous()  # [K,B,C] float32

        out, _ = self.custom_ops.sk_attention_fused_x_cuda(conv_outs, attn_logits, False)
        return out