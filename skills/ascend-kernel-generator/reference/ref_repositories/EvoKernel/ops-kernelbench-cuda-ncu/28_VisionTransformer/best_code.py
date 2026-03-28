import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# v4: Persistent fused patchify + linear
# - Shared-memory patch staging (K values) once per (b,s)
# - Warp-per-output (one warp computes one embedding dimension e)
# - Multiple warps per block compute E_TILE outputs for same patch (reuse patch)
# - Persistent work-queue loop to reduce launch overhead and improve residency
# - Specialized fast kernel for common ViT: C=3, p=16, K=768, E=512
# -----------------------------------------------------------------------------

vit_fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int WARPS_PER_BLOCK, int E_TILE>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 3)
void vit_patchify_linear_persistent_vit(
    const float* __restrict__ img,  // [B,C,H,W] contiguous NCHW
    const float* __restrict__ w,    // [E,K] contiguous row-major
    const float* __restrict__ b,    // [E] or nullptr
    float* __restrict__ out,        // [B,S,E]
    int B, int H, int W,
    int p, int Hp, int Wp,
    int S, int E,
    int* __restrict__ work_counter
){
    // Specialized for C=3, p=16, K=768, but we still pass w/b/out pointers
    constexpr int C = 3;
    constexpr int K = 768; // 3*16*16

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    extern __shared__ float smem[];
    float* spatch = smem; // size K floats

    // 2D grid over (e_tiles) in x and "some blocks" in y; y just provides more CTAs
    int e_tile_idx = (int)blockIdx.x;
    int e0 = e_tile_idx * E_TILE;

    // persistent loop over patches (bs in [0, B*S))
    while (true) {
        int bs = atomicAdd(work_counter, 1);
        if (bs >= B * S) break;

        int bch = bs / S;
        int s   = bs - bch * S;

        // Patch origin (py,px)
        int py = s / Wp;
        int px = s - py * Wp;
        int y0 = py * p;
        int x0 = px * p;

        const float* img_b = img + (size_t)bch * (size_t)(C * H * W);

        // Stage patch into shared memory with low index arithmetic:
        // Layout k = c*(p*p) + ky*p + kx
        // For each c, row-major p x p. Use contiguous loads per row.
        // Each thread loads multiple k positions.
        for (int k = tid; k < K; k += WARPS_PER_BLOCK * 32) {
            int c = k >> 8;            // /256 since p*p=256
            int r = k & 255;           // %256
            int ky = r >> 4;           // /16
            int kx = r & 15;           // %16
            const float* img_c = img_b + (size_t)c * (size_t)(H * W);
            spatch[k] = img_c[(size_t)(y0 + ky) * (size_t)W + (size_t)(x0 + kx)];
        }
        __syncthreads();

        // Each warp computes one output embedding e within the tile.
        int oi = warp; // 0..WARPS_PER_BLOCK-1
        if (oi < E_TILE) {
            int e = e0 + oi;
            if (e < E) {
                float acc = (b != nullptr) ? ldg_f32(b + e) : 0.0f;
                const float* wrow = w + (size_t)e * (size_t)K;

                // Lane-strided dot product over K
#pragma unroll 1
                for (int k = lane; k < K; k += 32) {
                    acc = fmaf(spatch[k], ldg_f32(wrow + k), acc);
                }
                acc = warp_reduce_sum(acc);
                if (lane == 0) {
                    out[((size_t)bch * (size_t)S + (size_t)s) * (size_t)E + (size_t)e] = acc;
                }
            }
        }
        __syncthreads();
    }
}

template<int WARPS_PER_BLOCK, int E_TILE, int K_TILE>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void vit_patchify_linear_persistent_generic(
    const float* __restrict__ img,  // [B,C,H,W] contiguous NCHW
    const float* __restrict__ w,    // [E,K] contiguous row-major
    const float* __restrict__ b,    // [E] or nullptr
    float* __restrict__ out,        // [B,S,E]
    int B, int C, int H, int W,
    int p, int Hp, int Wp,
    int S, int E, int K,
    int* __restrict__ work_counter
){
    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    // Shared patch only (kept small by not generalizing to huge K)
    extern __shared__ float smem[];
    float* spatch = smem; // size K floats

    int e_tile_idx = (int)blockIdx.x;
    int e0 = e_tile_idx * E_TILE;

    while (true) {
        int bs = atomicAdd(work_counter, 1);
        if (bs >= B * S) break;

        int bch = bs / S;
        int s   = bs - bch * S;

        int py = s / Wp;
        int px = s - py * Wp;
        int y0 = py * p;
        int x0 = px * p;

        const float* img_b = img + (size_t)bch * (size_t)(C * H * W);

        // Stage patch
        for (int k = tid; k < K; k += WARPS_PER_BLOCK * 32) {
            int kk = k;
            int kx = kk % p; kk /= p;
            int ky = kk % p; kk /= p;
            int c  = kk;
            const float* img_c = img_b + (size_t)c * (size_t)(H * W);
            spatch[k] = img_c[(size_t)(y0 + ky) * (size_t)W + (size_t)(x0 + kx)];
        }
        __syncthreads();

        int oi = warp;
        if (oi < E_TILE) {
            int e = e0 + oi;
            if (e < E) {
                float acc = (b != nullptr) ? ldg_f32(b + e) : 0.0f;
                const float* wrow = w + (size_t)e * (size_t)K;

#pragma unroll 1
                for (int k = lane; k < K; k += 32) {
                    acc = fmaf(spatch[k], ldg_f32(wrow + k), acc);
                }
                acc = warp_reduce_sum(acc);
                if (lane == 0) {
                    out[((size_t)bch * (size_t)S + (size_t)s) * (size_t)E + (size_t)e] = acc;
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor vit_patchify_linear_forward_cuda(
    torch::Tensor img,   // [B,C,H,W] float32 contiguous
    torch::Tensor w,     // [E,K] float32 contiguous
    torch::Tensor b,     // [E] float32 contiguous or empty
    int64_t patch_size
){
    CHECK_CUDA(img); CHECK_CUDA(w);
    CHECK_CONTIGUOUS(img); CHECK_CONTIGUOUS(w);
    CHECK_FLOAT(img); CHECK_FLOAT(w);

    TORCH_CHECK(img.dim() == 4, "img must be [B,C,H,W]");
    TORCH_CHECK(w.dim() == 2, "w must be [E,K]");
    TORCH_CHECK(patch_size > 0, "patch_size must be > 0");

    int B = (int)img.size(0);
    int C = (int)img.size(1);
    int H = (int)img.size(2);
    int W_ = (int)img.size(3);
    int p = (int)patch_size;

    TORCH_CHECK(H % p == 0 && W_ % p == 0, "H and W must be divisible by patch_size");

    int Hp = H / p;
    int Wp = W_ / p;
    int S  = Hp * Wp;
    int E  = (int)w.size(0);
    int K  = C * p * p;

    TORCH_CHECK((int)w.size(1) == K, "w in_features must be C*p*p");

    const float* bptr = nullptr;
    if (b.defined() && b.numel() > 0) {
        CHECK_CUDA(b); CHECK_CONTIGUOUS(b); CHECK_FLOAT(b);
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == E, "b must be [E]");
        bptr = b.data_ptr<float>();
    }

    auto out = torch::empty({B, S, E}, img.options());

    // Work counter (device scalar int)
    auto counter = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(img.device()));
    int* counter_ptr = counter.data_ptr<int>();

    // Tuning: E_TILE == WARPS_PER_BLOCK so each warp computes one e.
    // Keep WARPS moderate to reduce shared-sync overhead; still reuse patch across E_TILE outputs.
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads
    constexpr int E_TILE = 8;

    dim3 block(WARPS_PER_BLOCK * 32, 1, 1);

    // More CTAs than SMs helps hide latency for persistent loop; y-dim adds additional blocks.
    // x-dim tiles over E, y-dim provides multiplicity.
    int grid_x = (E + E_TILE - 1) / E_TILE;

    // Heuristic multiplicity: 4 * SM count (capped) without querying device properties (keep simple).
    // Use a moderate constant; persistent loop will balance work.
    int grid_y = 32;

    dim3 grid((unsigned)grid_x, (unsigned)grid_y, 1);

    bool vit_special = (C == 3 && p == 16 && K == 768 && E == 512);

    if (vit_special) {
        size_t shmem = (size_t)768 * sizeof(float);
        vit_patchify_linear_persistent_vit<WARPS_PER_BLOCK, E_TILE><<<grid, block, shmem>>>(
            img.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            out.data_ptr<float>(),
            B, H, W_,
            p, Hp, Wp,
            S, E,
            counter_ptr
        );
    } else {
        // Generic path stages full patch, so shared memory size is K floats.
        // Keep it as-is; for extreme K this may be large, but ViT patch K is small.
        size_t shmem = (size_t)K * sizeof(float);
        vit_patchify_linear_persistent_generic<WARPS_PER_BLOCK, E_TILE, 0><<<grid, block, shmem>>>(
            img.data_ptr<float>(),
            w.data_ptr<float>(),
            bptr,
            out.data_ptr<float>(),
            B, C, H, W_,
            p, Hp, Wp,
            S, E, K,
            counter_ptr
        );
    }

    return out;
}
"""

vit_fused_cpp_source = r"""
torch::Tensor vit_patchify_linear_forward_cuda(
    torch::Tensor img,
    torch::Tensor w,
    torch::Tensor b,
    int64_t patch_size
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_vit_patchify_linear_persistent_v4",
    cpp_sources=vit_fused_cpp_source,
    cuda_sources=vit_fused_cuda_source,
    functions=["vit_patchify_linear_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = int(patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
            ),
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        p = self.patch_size

        img_c = img.contiguous()
        w = self.patch_to_embedding.weight.contiguous()
        b = (
            self.patch_to_embedding.bias.contiguous()
            if self.patch_to_embedding.bias is not None
            else torch.empty(0, device=img.device, dtype=img.dtype)
        )

        if img_c.is_cuda and img_c.dtype == torch.float32:
            x = custom_ops_lib.vit_patchify_linear_forward_cuda(img_c, w, b, p)  # [B,S,dim]
        else:
            x = img_c.unfold(2, p, p).unfold(3, p, p).reshape(img_c.shape[0], -1, p * p * img_c.shape[1])
            x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img_c.shape[0], -1, -1)  # [B,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B,S+1,dim]
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)