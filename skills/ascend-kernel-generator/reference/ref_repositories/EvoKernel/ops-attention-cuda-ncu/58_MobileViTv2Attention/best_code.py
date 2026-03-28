import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline int div_up_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float ldg_f(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ float dot_wi_x_vec4(const float* __restrict__ x, const float* __restrict__ wi, int D) {
    float acc = 0.f;
    int k = 0;
    uintptr_t ax = (uintptr_t)(x);
    uintptr_t aw = (uintptr_t)(wi);
    bool ok = ((ax & 15u) == 0u) && ((aw & 15u) == 0u);
    if (ok) {
        for (; k + 3 < D; k += 4) {
            float4 xv = *reinterpret_cast<const float4*>(x + k);
            float4 wv = *reinterpret_cast<const float4*>(wi + k);
            acc = fmaf(xv.x, wv.x, acc);
            acc = fmaf(xv.y, wv.y, acc);
            acc = fmaf(xv.z, wv.z, acc);
            acc = fmaf(xv.w, wv.w, acc);
        }
    }
    for (; k < D; ++k) acc = fmaf(x[k], ldg_f(wi + k), acc);
    return acc;
}

// Kernel A: compute ctx[b,j] (B,D). One warp per (b,j). N<=64.
__global__ void fused_softmax_context_kernel(
    const float* __restrict__ x,   // (B,N,D)
    const float* __restrict__ wi,  // (D)
    const float* __restrict__ bi,  // (1)
    const float* __restrict__ wk,  // (D,D) row-major [j][k]
    const float* __restrict__ bk,  // (D)
    float* __restrict__ ctx,       // (B,D)
    int B, int N, int D
) {
    int b = (int)blockIdx.x;
    int j = (int)blockIdx.y; // one warp per j
    int lane = (int)threadIdx.x; // 0..31
    if (j >= D) return;

    const float* wk_row = wk + (size_t)j * (size_t)D;

    // Pass 1: max i_t over t
    float local_max = -INFINITY;
    for (int t = lane; t < N; t += 32) {
        const float* x_bt = x + ((size_t)b * (size_t)N + (size_t)t) * (size_t)D;
        float it = ldg_f(bi) + dot_wi_x_vec4(x_bt, wi, D);
        local_max = fmaxf(local_max, it);
    }
    float m = warp_reduce_max(local_max);
    m = __shfl_sync(0xffffffff, m, 0);

    // Pass 2: denom and ctx accumulation
    float denom_local = 0.f;
    float ctx_acc = 0.f;

    for (int t = 0; t < N; ++t) {
        const float* x_bt = x + ((size_t)b * (size_t)N + (size_t)t) * (size_t)D;

        float part_i = 0.f;
        for (int k = lane; k < D; k += 32) part_i = fmaf(x_bt[k], ldg_f(wi + k), part_i);
        float it = warp_reduce_sum(part_i);
        it = __shfl_sync(0xffffffff, it, 0) + ldg_f(bi);

        float e = __expf(it - m);
        denom_local += e;

        float part_k = 0.f;
        for (int k = lane; k < D; k += 32) part_k = fmaf(x_bt[k], ldg_f(wk_row + k), part_k);
        float ksum = warp_reduce_sum(part_k);
        ksum = __shfl_sync(0xffffffff, ksum, 0) + ldg_f(bk + j);

        ctx_acc = fmaf(e, ksum, ctx_acc);
    }

    float denom = warp_reduce_sum(denom_local);
    denom = __shfl_sync(0xffffffff, denom, 0);
    denom = fmaxf(denom, 1e-9f);

    if (lane == 0) ctx[(size_t)b * (size_t)D + (size_t)j] = ctx_acc / denom;
}

// Baseline output kernel (generic fallback): one warp per (b,t,dblk128), each lane 4 outputs.
__global__ void output_warp4_kernel(
    const float* __restrict__ x,    // (B,N,D)
    const float* __restrict__ ctx,  // (B,D)
    const float* __restrict__ wv,   // (D,D) row-major [j][k]
    const float* __restrict__ bv,   // (D)
    const float* __restrict__ wo,   // (D,D) row-major [d][j]
    const float* __restrict__ bo,   // (D)
    float* __restrict__ y,          // (B,N,D)
    int B, int N, int D
) {
    int b = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    int dblk = (int)blockIdx.z;
    int lane = (int)threadIdx.x; // 0..31

    int d0 = dblk * 128 + lane * 4;
    if (d0 >= D) return;

    const float* x_bt = x + ((size_t)b * (size_t)N + (size_t)t) * (size_t)D;
    const float* ctx_b = ctx + (size_t)b * (size_t)D;

    float acc0 = (d0 + 0 < D) ? ldg_f(bo + d0 + 0) : 0.f;
    float acc1 = (d0 + 1 < D) ? ldg_f(bo + d0 + 1) : 0.f;
    float acc2 = (d0 + 2 < D) ? ldg_f(bo + d0 + 2) : 0.f;
    float acc3 = (d0 + 3 < D) ? ldg_f(bo + d0 + 3) : 0.f;

    for (int j = 0; j < D; ++j) {
        const float* wv_row = wv + (size_t)j * (size_t)D;

        float part = 0.f;
        for (int k = lane; k < D; k += 32) part = fmaf(x_bt[k], ldg_f(wv_row + k), part);
        float sum = warp_reduce_sum(part);
        sum = __shfl_sync(0xffffffff, sum, 0);

        float vj = (sum + ldg_f(bv + j)) * ldg_f(ctx_b + j);

        if (d0 + 0 < D) acc0 = fmaf(vj, ldg_f(wo + (size_t)(d0 + 0) * (size_t)D + (size_t)j), acc0);
        if (d0 + 1 < D) acc1 = fmaf(vj, ldg_f(wo + (size_t)(d0 + 1) * (size_t)D + (size_t)j), acc1);
        if (d0 + 2 < D) acc2 = fmaf(vj, ldg_f(wo + (size_t)(d0 + 2) * (size_t)D + (size_t)j), acc2);
        if (d0 + 3 < D) acc3 = fmaf(vj, ldg_f(wo + (size_t)(d0 + 3) * (size_t)D + (size_t)j), acc3);
    }

    float* y_bt = y + ((size_t)b * (size_t)N + (size_t)t) * (size_t)D;
    if (d0 + 0 < D) y_bt[d0 + 0] = acc0;
    if (d0 + 1 < D) y_bt[d0 + 1] = acc1;
    if (d0 + 2 < D) y_bt[d0 + 2] = acc2;
    if (d0 + 3 < D) y_bt[d0 + 3] = acc3;
}

// Fast output for D=512: one CTA per (b,t) with 4 warps covering 4*dblk128.
// For each j, compute vj once (warp0), stage in shared, all warps update their outputs.
// This removes redundant (Wv*x) computation across dblk.
__global__ void output_group4_d512_kernel(
    const float* __restrict__ x,    // (B,N,512)
    const float* __restrict__ ctx,  // (B,512)
    const float* __restrict__ wv,   // (512,512) row-major [j][k]
    const float* __restrict__ bv,   // (512)
    const float* __restrict__ wo,   // (512,512) row-major [d][j]
    const float* __restrict__ bo,   // (512)
    float* __restrict__ y,          // (B,N,512)
    int B, int N
) {
    int b = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    int group = (int)blockIdx.z; // group of 4 dblk128 tiles; for D=512, only group=0

    int tid = (int)threadIdx.x;       // 0..127
    int warp_id = tid >> 5;           // 0..3
    int lane = tid & 31;              // 0..31

    // each warp owns one dblk128 within this group
    int dblk = group * 4 + warp_id;   // 0..3
    int d0 = dblk * 128 + lane * 4;
    if (d0 >= 512) return;

    const float* x_bt = x + ((size_t)b * (size_t)N + (size_t)t) * 512ull;
    const float* ctx_b = ctx + (size_t)b * 512ull;

    float acc0 = ldg_f(bo + d0 + 0);
    float acc1 = ldg_f(bo + d0 + 1);
    float acc2 = ldg_f(bo + d0 + 2);
    float acc3 = ldg_f(bo + d0 + 3);

    __shared__ float vsh; // scalar per j

    // Iterate j
#pragma unroll 1
    for (int j = 0; j < 512; ++j) {
        if (warp_id == 0) {
            // lane-strided dot over k for vj
            const float* wv_row = wv + (size_t)j * 512ull;
            float part = 0.f;
#pragma unroll
            for (int k = lane; k < 512; k += 32) {
                part = fmaf(x_bt[k], ldg_f(wv_row + k), part);
            }
            float sum = warp_reduce_sum(part);
            sum = __shfl_sync(0xffffffff, sum, 0);
            float vj = (sum + ldg_f(bv + j)) * ldg_f(ctx_b + j);
            if (lane == 0) vsh = vj;
        }
        __syncthreads();
        float vj_all = vsh;

        // update this warp's 4 outputs
        const float* wo_row0 = wo + (size_t)(d0 + 0) * 512ull;
        const float* wo_row1 = wo + (size_t)(d0 + 1) * 512ull;
        const float* wo_row2 = wo + (size_t)(d0 + 2) * 512ull;
        const float* wo_row3 = wo + (size_t)(d0 + 3) * 512ull;

        acc0 = fmaf(vj_all, ldg_f(wo_row0 + j), acc0);
        acc1 = fmaf(vj_all, ldg_f(wo_row1 + j), acc1);
        acc2 = fmaf(vj_all, ldg_f(wo_row2 + j), acc2);
        acc3 = fmaf(vj_all, ldg_f(wo_row3 + j), acc3);

        __syncthreads();
    }

    float* y_bt = y + ((size_t)b * (size_t)N + (size_t)t) * 512ull;
    y_bt[d0 + 0] = acc0;
    y_bt[d0 + 1] = acc1;
    y_bt[d0 + 2] = acc2;
    y_bt[d0 + 3] = acc3;
}

torch::Tensor mobile_vi_tv2_attention_cuda(
    torch::Tensor x,
    torch::Tensor wi, torch::Tensor bi,
    torch::Tensor wk, torch::Tensor bk,
    torch::Tensor wv, torch::Tensor bv,
    torch::Tensor wo, torch::Tensor bo
) {
    CHECK_INPUT(x);
    CHECK_INPUT(wi); CHECK_INPUT(bi);
    CHECK_INPUT(wk); CHECK_INPUT(bk);
    CHECK_INPUT(wv); CHECK_INPUT(bv);
    CHECK_INPUT(wo); CHECK_INPUT(bo);

    TORCH_CHECK(x.dim() == 3, "x must be (B,N,D)");
    int B = (int)x.size(0);
    int N = (int)x.size(1);
    int D = (int)x.size(2);

    TORCH_CHECK(N <= 64, "This optimized CUDA path supports N<=64 (got N=", N, ")");

    TORCH_CHECK(wi.numel() == D, "wi must have D elements");
    TORCH_CHECK(bi.numel() == 1, "bi must have 1 element");
    TORCH_CHECK(wk.sizes() == std::vector<int64_t>({D, D}), "wk must be (D,D)");
    TORCH_CHECK(bk.numel() == D, "bk must have D elements");
    TORCH_CHECK(wv.sizes() == std::vector<int64_t>({D, D}), "wv must be (D,D)");
    TORCH_CHECK(bv.numel() == D, "bv must have D elements");
    TORCH_CHECK(wo.sizes() == std::vector<int64_t>({D, D}), "wo must be (D,D)");
    TORCH_CHECK(bo.numel() == D, "bo must have D elements");

    auto y = torch::empty({B, N, D}, x.options());
    auto ctx = torch::empty({B, D}, x.options());

    dim3 grid_ctx(B, D);
    fused_softmax_context_kernel<<<grid_ctx, 32>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)wi.data_ptr<float>(),
        (const float*)bi.data_ptr<float>(),
        (const float*)wk.data_ptr<float>(),
        (const float*)bk.data_ptr<float>(),
        (float*)ctx.data_ptr<float>(),
        B, N, D
    );

    if (D == 512) {
        // one group for 512 (4 dblk128)
        dim3 grid_out(B, N, 1);
        output_group4_d512_kernel<<<grid_out, 128>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)ctx.data_ptr<float>(),
            (const float*)wv.data_ptr<float>(),
            (const float*)bv.data_ptr<float>(),
            (const float*)wo.data_ptr<float>(),
            (const float*)bo.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N
        );
    } else {
        int dblocks = div_up_int(D, 128);
        dim3 grid_out(B, N, dblocks);
        output_warp4_kernel<<<grid_out, 32>>>(
            (const float*)x.data_ptr<float>(),
            (const float*)ctx.data_ptr<float>(),
            (const float*)wv.data_ptr<float>(),
            (const float*)bv.data_ptr<float>(),
            (const float*)wo.data_ptr<float>(),
            (const float*)bo.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            B, N, D
        );
    }

    return y;
}
"""

cpp_src = r"""
torch::Tensor mobile_vi_tv2_attention_cuda(
    torch::Tensor x,
    torch::Tensor wi, torch::Tensor bi,
    torch::Tensor wk, torch::Tensor bk,
    torch::Tensor wv, torch::Tensor bv,
    torch::Tensor wo, torch::Tensor bo
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_mobile_vi_tv2_attention_group4_d512_v1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["mobile_vi_tv2_attention_cuda"],
    extra_cuda_cflags=["--use_fast_math", "-lineinfo"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    MobileViTv2 Attention with optimized custom CUDA forward (fp32 CUDA only, N<=64).
    Falls back to PyTorch reference otherwise.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.init_weights()
        self.custom_ops_lib = custom_ops_lib

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (not input.is_cuda) or (input.dtype != torch.float32) or (input.size(1) > 64):
            i = self.fc_i(input)  # (B,N,1)
            weight_i = torch.softmax(i, dim=1)  # (B,N,1)
            context_score = weight_i * self.fc_k(input)  # (B,N,D)
            context_vector = torch.sum(context_score, dim=1, keepdim=True)  # (B,1,D)
            v = self.fc_v(input) * context_vector  # (B,N,D)
            out = self.fc_o(v)  # (B,N,D)
            return out

        x = input.contiguous()

        wi = self.fc_i.weight.contiguous().view(-1)
        bi = self.fc_i.bias.contiguous().view(1)

        wk = self.fc_k.weight.contiguous()
        bk = self.fc_k.bias.contiguous()

        wv = self.fc_v.weight.contiguous()
        bv = self.fc_v.bias.contiguous()

        wo = self.fc_o.weight.contiguous()
        bo = self.fc_o.bias.contiguous()

        return self.custom_ops_lib.mobile_vi_tv2_attention_cuda(x, wi, bi, wk, bk, wv, bv, wo, bo)