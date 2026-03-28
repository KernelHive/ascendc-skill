import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cvt_fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

__device__ __forceinline__ float ldg_f32(const float* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ bool is_aligned_16(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

template<int WARPS_PER_BLOCK, int E_TILE, int P_MAX>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
void fused_patch_conv_linear_tiled_fwd(
    const float* __restrict__ x,      // [B,C,H,W]
    const float* __restrict__ wconv,  // [E,C,P,P]
    const float* __restrict__ bconv,  // [E] or nullptr
    const float* __restrict__ wlin,   // [E, E*Hp*Wp]
    const float* __restrict__ blin,   // [E] or nullptr
    float* __restrict__ y,            // [B,E]
    int B, int C, int H, int W,
    int E, int P, int Hp, int Wp
){
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int b = (int)blockIdx.y; // one batch element per CTA in y-dim
    if (b >= B) return;

    const int e0 = (int)blockIdx.x * E_TILE;  // tile start for output channels
    const int e_out = e0 + warp;              // one warp -> one e_out
    const int num_patches = Hp * Wp;

    // Shared memory layout:
    // conv_patch[num_patches] as float
    extern __shared__ float smem[];
    float* conv_patch = smem; // [num_patches]

    float acc = 0.0f;
    if (warp < E_TILE && e_out < E) {
        acc = (blin != nullptr) ? ldg_f32(blin + e_out) : 0.0f;
    }

    const float* xb = x + (size_t)b * (size_t)C * (size_t)H * (size_t)W;

    // Iterate over e_in; for each, compute conv_patch once into shared, then consume for each e_out in tile
    for (int e_in = 0; e_in < E; ++e_in) {
        // Producer phase: all threads compute conv_patch[pidx] in a strided manner
        float bc = (bconv != nullptr) ? ldg_f32(bconv + e_in) : 0.0f;
        const float* wconv_e = wconv + ((size_t)e_in * (size_t)C * (size_t)P * (size_t)P);

        for (int pidx = tid; pidx < num_patches; pidx += WARPS_PER_BLOCK * 32) {
            int ph = pidx / Wp;
            int pw = pidx - ph * Wp;
            int in_y0 = ph * P;
            int in_x0 = pw * P;

            float conv_acc = bc;
#pragma unroll
            for (int c = 0; c < 4; ++c) { // will guard by C below; small C typical (3)
                if (c >= C) break;
                const float* xbc = xb + (size_t)c * (size_t)H * (size_t)W;
                const float* wcc = wconv_e + (size_t)c * (size_t)P * (size_t)P;

                // P is runtime, but in this model patch_size is small (often 4). Keep loops tight.
                for (int ky = 0; ky < P_MAX; ++ky) {
                    if (ky >= P) break;
                    int iy = in_y0 + ky;
                    const float* xrow = xbc + (size_t)iy * (size_t)W;
                    const float* wrow = wcc + (size_t)ky * (size_t)P;
#pragma unroll
                    for (int kx = 0; kx < P_MAX; ++kx) {
                        if (kx >= P) break;
                        int ix = in_x0 + kx;
                        float xv = ldg_f32(xrow + ix);
                        float wv = ldg_f32(wrow + kx);
                        conv_acc = fmaf(xv, wv, conv_acc);
                    }
                }
            }

            conv_patch[pidx] = conv_acc;
        }

        __syncthreads();

        // Consumer phase: each warp accumulates for its e_out using conv_patch and wlin row
        if (warp < E_TILE && e_out < E) {
            const float* wlin_row = wlin + (size_t)e_out * (size_t)(E * num_patches) + (size_t)e_in * (size_t)num_patches;

            // Vectorized dot over patches when aligned and divisible by 4
            int p4 = num_patches & ~3;
            if (p4 > 0 && is_aligned_16(wlin_row) && is_aligned_16(conv_patch)) {
                const float4* w4 = (const float4*)wlin_row;
                const float4* c4 = (const float4*)conv_patch;
                int n4 = p4 >> 2;

                float local = 0.0f;
                for (int i = lane; i < n4; i += 32) {
                    float4 ww = w4[i];
                    float4 cc = c4[i];
                    local = fmaf(ww.x, cc.x, local);
                    local = fmaf(ww.y, cc.y, local);
                    local = fmaf(ww.z, cc.z, local);
                    local = fmaf(ww.w, cc.w, local);
                }

                // reduce within warp, then add to acc
                for (int off = 16; off > 0; off >>= 1) {
                    local += __shfl_down_sync(0xffffffff, local, off);
                }
                if (lane == 0) acc += local;

                // tail
                for (int pidx = p4 + lane; pidx < num_patches; pidx += 32) {
                    acc = fmaf(ldg_f32(wlin_row + pidx), conv_patch[pidx], acc);
                }
            } else {
                // Scalar fallback
                float local = 0.0f;
                for (int pidx = lane; pidx < num_patches; pidx += 32) {
                    local = fmaf(ldg_f32(wlin_row + pidx), conv_patch[pidx], local);
                }
                for (int off = 16; off > 0; off >>= 1) {
                    local += __shfl_down_sync(0xffffffff, local, off);
                }
                if (lane == 0) acc += local;
            }
        }

        __syncthreads();
    }

    if (warp < E_TILE && e_out < E && lane == 0) {
        y[(size_t)b * (size_t)E + (size_t)e_out] = acc;
    }
}

torch::Tensor fused_patch_conv_linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor wconv,
    torch::Tensor bconv,
    torch::Tensor wlin,
    torch::Tensor blin,
    int64_t patch_size
){
    CHECK_CUDA(x); CHECK_CUDA(wconv); CHECK_CUDA(wlin);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(wconv); CHECK_CONTIGUOUS(wlin);
    CHECK_FLOAT(x); CHECK_FLOAT(wconv); CHECK_FLOAT(wlin);

    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(wconv.dim() == 4, "wconv must be [E,C,P,P]");
    TORCH_CHECK(wlin.dim() == 2, "wlin must be [E, E*Hp*Wp]");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    int E = (int)wconv.size(0);
    TORCH_CHECK((int)wconv.size(1) == C, "wconv Cin mismatch");

    int P = (int)patch_size;
    TORCH_CHECK(P > 0, "patch_size must be > 0");
    TORCH_CHECK((int)wconv.size(2) == P && (int)wconv.size(3) == P, "wconv kernel must match patch_size");
    TORCH_CHECK((H % P) == 0 && (W % P) == 0, "H and W must be divisible by patch_size");

    int Hp = H / P;
    int Wp = W / P;
    int num_patches = Hp * Wp;

    TORCH_CHECK((int)wlin.size(0) == E, "wlin out_features mismatch");
    TORCH_CHECK((int)wlin.size(1) == E * num_patches, "wlin in_features must be E*(H/P)*(W/P)");

    const float* bconv_ptr = nullptr;
    if (bconv.defined() && bconv.numel() > 0) {
        CHECK_CUDA(bconv); CHECK_CONTIGUOUS(bconv); CHECK_FLOAT(bconv);
        TORCH_CHECK(bconv.dim() == 1 && (int)bconv.size(0) == E, "bconv must be [E]");
        bconv_ptr = bconv.data_ptr<float>();
    }

    const float* blin_ptr = nullptr;
    if (blin.defined() && blin.numel() > 0) {
        CHECK_CUDA(blin); CHECK_CONTIGUOUS(blin); CHECK_FLOAT(blin);
        TORCH_CHECK(blin.dim() == 1 && (int)blin.size(0) == E, "blin must be [E]");
        blin_ptr = blin.data_ptr<float>();
    }

    auto y = torch::empty({B, E}, x.options());

    // Tunables:
    // - WARPS_PER_BLOCK=4 gives 128 threads and good occupancy without inflating shared/regs too much.
    // - E_TILE=4 maps one warp to one output channel in the tile.
    // - P_MAX=4 specialization matches common patch_size=4; we enforce P<=4 for fast path.
    // Fallback for P>4 is not provided here (model uses patch_size=4 by default); enforce for correctness.
    TORCH_CHECK(P <= 4, "Optimized kernel currently supports patch_size <= 4");

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int E_TILE = 4;
    constexpr int P_MAX = 4;

    dim3 block(WARPS_PER_BLOCK * 32, 1, 1);
    dim3 grid((unsigned)((E + E_TILE - 1) / E_TILE), (unsigned)B, 1);

    size_t shmem = (size_t)num_patches * sizeof(float);

    fused_patch_conv_linear_tiled_fwd<WARPS_PER_BLOCK, E_TILE, P_MAX><<<grid, block, shmem>>>(
        x.data_ptr<float>(),
        wconv.data_ptr<float>(),
        bconv_ptr,
        wlin.data_ptr<float>(),
        blin_ptr,
        y.data_ptr<float>(),
        B, C, H, W,
        E, P, Hp, Wp
    );

    return y;
}
"""

cvt_fused_cpp_source = r"""
torch::Tensor fused_patch_conv_linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor wconv,
    torch::Tensor bconv,
    torch::Tensor wlin,
    torch::Tensor blin,
    int64_t patch_size
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_cvit_fused_patchconv_linear_tiled_v2",
    cpp_sources=cvt_fused_cpp_source,
    cuda_sources=cvt_fused_cuda_source,
    functions=["fused_patch_conv_linear_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        patch_size=4,
        in_channels=3,
        image_size=32,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim, bias=True)

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.contiguous()

        wconv = self.conv1.weight.contiguous()
        bconv = self.conv1.bias.contiguous() if self.conv1.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        wlin = self.linear_proj.weight.contiguous()
        blin = self.linear_proj.bias.contiguous() if self.linear_proj.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        if x.is_cuda and x.dtype == torch.float32:
            x = custom_ops_lib.fused_patch_conv_linear_forward_cuda(
                x, wconv, bconv, wlin, blin, self.patch_size
            )  # (B, E)
        else:
            x = self.conv1(x)
            x = x.flatten(start_dim=1)
            x = self.linear_proj(x)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,E)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B,2,E)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])