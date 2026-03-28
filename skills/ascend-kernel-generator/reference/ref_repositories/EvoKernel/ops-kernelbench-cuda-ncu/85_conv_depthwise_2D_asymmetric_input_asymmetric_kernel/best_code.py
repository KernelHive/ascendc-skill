import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- CUDA/C++ extension: depthwise conv2d (specialized hot path for 3x7 s1 p0 d1) ----
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ float ro_load_f32(const float* p) { return __ldg(p); }
#else
__device__ __forceinline__ float ro_load_f32(const float* p) { return *p; }
#endif

// ---------------- Generic kernel (baseline-like) ----------------
__global__ void dwconv2d_generic_forward_kernel(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ w,   // [C,1,kH,kW]
    const float* __restrict__ b,   // [C] or nullptr
    float* __restrict__ y,         // [N,C,outH,outW]
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW,
    bool has_bias
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    long long total = (long long)N * C * outH * outW;
    if ((long long)idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int n  = idx / (outW * outH * C);

    int ih0 = oh * sH - pH;
    int iw0 = ow * sW - pW;

    const int x_base = ((n * C + c) * H) * W;
    const int w_base = c * (kH * kW);

    float acc = has_bias ? ro_load_f32(b + c) : 0.0f;

    for (int kh = 0; kh < kH; ++kh) {
        int ih = ih0 + kh * dH;
        if ((unsigned)ih >= (unsigned)H) continue;
        int x_row = x_base + ih * W;
        int w_row = w_base + kh * kW;

        #pragma unroll 8
        for (int kw = 0; kw < 8; ++kw) {
            if (kw >= kW) break;
            int iw = iw0 + kw * dW;
            if ((unsigned)iw >= (unsigned)W) continue;
            acc = fmaf(ro_load_f32(x + x_row + iw), ro_load_f32(w + w_row + kw), acc);
        }
        for (int kw = 8; kw < kW; ++kw) {
            int iw = iw0 + kw * dW;
            if ((unsigned)iw >= (unsigned)W) continue;
            acc = fmaf(ro_load_f32(x + x_row + iw), ro_load_f32(w + w_row + kw), acc);
        }
    }

    y[((n * C + c) * outH + oh) * outW + ow] = acc;
}

// ---------------- Hot kernel: kH=3,kW=7,s=1,p=0,d=1 ----------------
//
// Tile geometry:
// blockDim = (BLOCK_X, BLOCK_Y, 1)
// Each thread computes WPT adjacent output columns in the same output row.
// Output tile width = BLOCK_X * WPT, height = BLOCK_Y.
// Shared patch: (BLOCK_Y+2) x (BLOCK_X*WPT + 6).
//
// We store shared patch as 1D array to make vector store alignment predictable:
// sh[idx] where idx is linear; idx*4 bytes is always 4-byte aligned.
// For float2 vector stores, require idx % 2 == 0 (8-byte aligned).
//
#ifndef HOT_BLOCK_X
#define HOT_BLOCK_X 32
#endif
#ifndef HOT_BLOCK_Y
#define HOT_BLOCK_Y 4
#endif
#ifndef HOT_WPT
#define HOT_WPT 4
#endif

static __device__ __forceinline__ void load_w3x7_regs(const float* __restrict__ w, float wreg[21]) {
    // w is [C,1,3,7] contiguous flattened per-c: 21 floats
    #pragma unroll
    for (int i = 0; i < 21; ++i) wreg[i] = ro_load_f32(w + i);
}

template <bool HAS_BIAS, bool INTERIOR>
__global__ __launch_bounds__(HOT_BLOCK_X * HOT_BLOCK_Y, 2)
void dwconv2d_k3x7_s1_p0_d1_sm_wpt4(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ w,   // [C,1,3,7]
    const float* __restrict__ b,   // [C] or nullptr
    float* __restrict__ y,         // [N,C,outH,outW]
    int N, int C, int H, int W,
    int outH, int outW
) {
    int tx = (int)threadIdx.x; // 0..31
    int ty = (int)threadIdx.y; // 0..3

    int tile_oh0 = (int)blockIdx.y * HOT_BLOCK_Y;
    int tile_ow0 = (int)blockIdx.x * (HOT_BLOCK_X * HOT_WPT);

    int nc = (int)blockIdx.z;     // 0..N*C-1
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N) return;

    // Shared patch dims
    constexpr int SH_H = HOT_BLOCK_Y + 2;                 // 6
    constexpr int SH_W = HOT_BLOCK_X * HOT_WPT + 6;       // 32*4+6 = 134
    __shared__ float sh[SH_H * SH_W];

    const float* x_plane = x + ((n * C + c) * H * W);

    // Determine if we can do interior (no bounds) math & staging
    // Requirements for interior:
    // tile_oh0 + (HOT_BLOCK_Y-1) < outH, tile_ow0 + (HOT_BLOCK_X*WPT-1) < outW
    // input patch uses ih in [tile_oh0 .. tile_oh0+HOT_BLOCK_Y+1] <= outH-1+2 = H-1
    // and iw in [tile_ow0 .. tile_ow0+HOT_BLOCK_X*WPT+5] <= outW-1+6 = W-1
    if constexpr (INTERIOR) {
        // no runtime check; caller guarantees
    } else {
        // boundary path still stages safely with checks
    }

    // Cooperative load of shared patch
    int linear_tid = ty * HOT_BLOCK_X + tx;
    int num_threads = HOT_BLOCK_X * HOT_BLOCK_Y;
    int patch_elems = SH_H * SH_W;

    // Prefer float2 loads when:
    // - global address is 8B aligned (index even)
    // - shared index is even
    // - we can read 2 contiguous floats within the row (col+1 < SH_W)
    // For boundary we must also bounds-check (ih<H && iw+1 < W).
    for (int idx = linear_tid * 2; idx < patch_elems; idx += num_threads * 2) {
        // idx is even by construction
        int r = idx / SH_W;
        int col = idx - r * SH_W;
        if (col + 1 >= SH_W) {
            // tail: handle last element with scalar by the thread that hits it
            int idx1 = idx;
            if (idx1 < patch_elems) {
                int r1 = idx1 / SH_W;
                int c1 = idx1 - r1 * SH_W;
                int ih1 = tile_oh0 + r1;
                int iw1 = tile_ow0 + c1;
                float v = 0.0f;
                if constexpr (INTERIOR) {
                    v = x_plane[ih1 * W + iw1];
                } else {
                    if ((unsigned)ih1 < (unsigned)H && (unsigned)iw1 < (unsigned)W) v = x_plane[ih1 * W + iw1];
                }
                sh[idx1] = v;
            }
            continue;
        }

        int ih = tile_oh0 + r;
        int iw = tile_ow0 + col;

        // vectorize if global index aligned and within bounds
        // global offset = ih*W + iw
        int goff = ih * W + iw;

        bool can_vec = ((goff & 1) == 0); // 8B alignment for float2
        if constexpr (!INTERIOR) {
            can_vec = can_vec && ((unsigned)ih < (unsigned)H) && (iw + 1 < W);
        }
        if (can_vec) {
            float2 v2 = *reinterpret_cast<const float2*>(x_plane + goff);
            // shared idx is even => 8B aligned, safe
            *reinterpret_cast<float2*>(sh + idx) = v2;
        } else {
            // scalar
            float v0 = 0.0f, v1 = 0.0f;
            if constexpr (INTERIOR) {
                // If not aligned, still in bounds
                v0 = x_plane[goff];
                v1 = x_plane[goff + 1];
            } else {
                if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) v0 = x_plane[goff];
                if ((unsigned)ih < (unsigned)H && (unsigned)(iw + 1) < (unsigned)W) v1 = x_plane[goff + 1];
            }
            sh[idx] = v0;
            sh[idx + 1] = v1;
        }
    }

    // Handle possible single leftover element when patch_elems is odd
    if ((patch_elems & 1) && linear_tid == 0) {
        int idx = patch_elems - 1;
        int r = idx / SH_W;
        int col = idx - r * SH_W;
        int ih = tile_oh0 + r;
        int iw = tile_ow0 + col;
        float v = 0.0f;
        if constexpr (INTERIOR) v = x_plane[ih * W + iw];
        else {
            if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) v = x_plane[ih * W + iw];
        }
        sh[idx] = v;
    }

    __syncthreads();

    int oh = tile_oh0 + ty;
    int ow = tile_ow0 + tx * HOT_WPT;
    if (oh >= outH) return;

    // Load weights (per-thread) from global, but it's tiny (21 floats) and cached.
    float wreg[21];
    const float* w_c = w + c * 21;
    load_w3x7_regs(w_c, wreg);

    float biasv = 0.0f;
    if constexpr (HAS_BIAS) biasv = ro_load_f32(b + c);

    // Compute 4 outputs: ow+0..ow+3
    // For INTERIOR tiles, all outputs are valid.
    bool v0 = true, v1 = true, v2 = true, v3 = true;
    if constexpr (!INTERIOR) {
        v0 = (ow + 0) < outW;
        v1 = (ow + 1) < outW;
        v2 = (ow + 2) < outW;
        v3 = (ow + 3) < outW;
        if (!(v0 || v1 || v2 || v3)) return;
    }

    float acc0 = biasv, acc1 = biasv, acc2 = biasv, acc3 = biasv;

    // Shared base index for this thread's first output in this row
    int sy = ty;
    int sx = tx * HOT_WPT;
    int base0 = (sy + 0) * SH_W + (sx + 0);
    int base1 = (sy + 1) * SH_W + (sx + 0);
    int base2 = (sy + 2) * SH_W + (sx + 0);

    // Fully unrolled 3x7, reused across 4 shifted outputs
    // Row 0
    float r00 = sh[base0 + 0]; float r01 = sh[base0 + 1]; float r02 = sh[base0 + 2]; float r03 = sh[base0 + 3];
    float r04 = sh[base0 + 4]; float r05 = sh[base0 + 5]; float r06 = sh[base0 + 6]; float r07 = sh[base0 + 7];
    float r08 = sh[base0 + 8]; float r09 = sh[base0 + 9];
    // Row 1
    float r10 = sh[base1 + 0]; float r11 = sh[base1 + 1]; float r12 = sh[base1 + 2]; float r13 = sh[base1 + 3];
    float r14 = sh[base1 + 4]; float r15 = sh[base1 + 5]; float r16 = sh[base1 + 6]; float r17 = sh[base1 + 7];
    float r18 = sh[base1 + 8]; float r19 = sh[base1 + 9];
    // Row 2
    float r20 = sh[base2 + 0]; float r21 = sh[base2 + 1]; float r22 = sh[base2 + 2]; float r23 = sh[base2 + 3];
    float r24 = sh[base2 + 4]; float r25 = sh[base2 + 5]; float r26 = sh[base2 + 6]; float r27 = sh[base2 + 7];
    float r28 = sh[base2 + 8]; float r29 = sh[base2 + 9];

    // Output shifts by +1, +2, +3 reuse same loaded values:
    // out0 uses cols 0..6, out1 uses 1..7, out2 uses 2..8, out3 uses 3..9.

    // Row0 weights wreg[0..6]
    acc0 = fmaf(r00, wreg[0], acc0); acc0 = fmaf(r01, wreg[1], acc0); acc0 = fmaf(r02, wreg[2], acc0);
    acc0 = fmaf(r03, wreg[3], acc0); acc0 = fmaf(r04, wreg[4], acc0); acc0 = fmaf(r05, wreg[5], acc0); acc0 = fmaf(r06, wreg[6], acc0);

    acc1 = fmaf(r01, wreg[0], acc1); acc1 = fmaf(r02, wreg[1], acc1); acc1 = fmaf(r03, wreg[2], acc1);
    acc1 = fmaf(r04, wreg[3], acc1); acc1 = fmaf(r05, wreg[4], acc1); acc1 = fmaf(r06, wreg[5], acc1); acc1 = fmaf(r07, wreg[6], acc1);

    acc2 = fmaf(r02, wreg[0], acc2); acc2 = fmaf(r03, wreg[1], acc2); acc2 = fmaf(r04, wreg[2], acc2);
    acc2 = fmaf(r05, wreg[3], acc2); acc2 = fmaf(r06, wreg[4], acc2); acc2 = fmaf(r07, wreg[5], acc2); acc2 = fmaf(r08, wreg[6], acc2);

    acc3 = fmaf(r03, wreg[0], acc3); acc3 = fmaf(r04, wreg[1], acc3); acc3 = fmaf(r05, wreg[2], acc3);
    acc3 = fmaf(r06, wreg[3], acc3); acc3 = fmaf(r07, wreg[4], acc3); acc3 = fmaf(r08, wreg[5], acc3); acc3 = fmaf(r09, wreg[6], acc3);

    // Row1 weights wreg[7..13]
    acc0 = fmaf(r10, wreg[7], acc0); acc0 = fmaf(r11, wreg[8], acc0); acc0 = fmaf(r12, wreg[9], acc0);
    acc0 = fmaf(r13, wreg[10], acc0); acc0 = fmaf(r14, wreg[11], acc0); acc0 = fmaf(r15, wreg[12], acc0); acc0 = fmaf(r16, wreg[13], acc0);

    acc1 = fmaf(r11, wreg[7], acc1); acc1 = fmaf(r12, wreg[8], acc1); acc1 = fmaf(r13, wreg[9], acc1);
    acc1 = fmaf(r14, wreg[10], acc1); acc1 = fmaf(r15, wreg[11], acc1); acc1 = fmaf(r16, wreg[12], acc1); acc1 = fmaf(r17, wreg[13], acc1);

    acc2 = fmaf(r12, wreg[7], acc2); acc2 = fmaf(r13, wreg[8], acc2); acc2 = fmaf(r14, wreg[9], acc2);
    acc2 = fmaf(r15, wreg[10], acc2); acc2 = fmaf(r16, wreg[11], acc2); acc2 = fmaf(r17, wreg[12], acc2); acc2 = fmaf(r18, wreg[13], acc2);

    acc3 = fmaf(r13, wreg[7], acc3); acc3 = fmaf(r14, wreg[8], acc3); acc3 = fmaf(r15, wreg[9], acc3);
    acc3 = fmaf(r16, wreg[10], acc3); acc3 = fmaf(r17, wreg[11], acc3); acc3 = fmaf(r18, wreg[12], acc3); acc3 = fmaf(r19, wreg[13], acc3);

    // Row2 weights wreg[14..20]
    acc0 = fmaf(r20, wreg[14], acc0); acc0 = fmaf(r21, wreg[15], acc0); acc0 = fmaf(r22, wreg[16], acc0);
    acc0 = fmaf(r23, wreg[17], acc0); acc0 = fmaf(r24, wreg[18], acc0); acc0 = fmaf(r25, wreg[19], acc0); acc0 = fmaf(r26, wreg[20], acc0);

    acc1 = fmaf(r21, wreg[14], acc1); acc1 = fmaf(r22, wreg[15], acc1); acc1 = fmaf(r23, wreg[16], acc1);
    acc1 = fmaf(r24, wreg[17], acc1); acc1 = fmaf(r25, wreg[18], acc1); acc1 = fmaf(r26, wreg[19], acc1); acc1 = fmaf(r27, wreg[20], acc1);

    acc2 = fmaf(r22, wreg[14], acc2); acc2 = fmaf(r23, wreg[15], acc2); acc2 = fmaf(r24, wreg[16], acc2);
    acc2 = fmaf(r25, wreg[17], acc2); acc2 = fmaf(r26, wreg[18], acc2); acc2 = fmaf(r27, wreg[19], acc2); acc2 = fmaf(r28, wreg[20], acc2);

    acc3 = fmaf(r23, wreg[14], acc3); acc3 = fmaf(r24, wreg[15], acc3); acc3 = fmaf(r25, wreg[16], acc3);
    acc3 = fmaf(r26, wreg[17], acc3); acc3 = fmaf(r27, wreg[18], acc3); acc3 = fmaf(r28, wreg[19], acc3); acc3 = fmaf(r29, wreg[20], acc3);

    // Store
    int out_base = ((n * C + c) * outH + oh) * outW + ow;
    if constexpr (INTERIOR) {
        y[out_base + 0] = acc0;
        y[out_base + 1] = acc1;
        y[out_base + 2] = acc2;
        y[out_base + 3] = acc3;
    } else {
        if (v0) y[out_base + 0] = acc0;
        if (v1) y[out_base + 1] = acc1;
        if (v2) y[out_base + 2] = acc2;
        if (v3) y[out_base + 3] = acc3;
    }
}

torch::Tensor conv_depthwise2d_asymmetric_input_asymmetric_kernel_forward_cuda(
    torch::Tensor x,        // [N,C,H,W]
    torch::Tensor w,        // [C,1,kH,kW]
    c10::optional<torch::Tensor> bias_opt, // [C] optional
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 4, "x must be NCHW (4D)");
    TORCH_CHECK(w.dim() == 4, "w must be [C,1,kH,kW]");
    TORCH_CHECK(w.size(1) == 1, "w second dim must be 1 for depthwise");
    TORCH_CHECK(sH > 0 && sW > 0, "stride must be > 0");
    TORCH_CHECK(dH > 0 && dW > 0, "dilation must be > 0");

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    TORCH_CHECK((int)w.size(0) == C, "w.size(0) must equal input channels");
    const int kH = (int)w.size(2);
    const int kW = (int)w.size(3);

    const int outH = (int)((H + 2 * (int)pH - (int)dH * (kH - 1) - 1) / (int)sH + 1);
    const int outW = (int)((W + 2 * (int)pW - (int)dW * (kW - 1) - 1) / (int)sW + 1);
    TORCH_CHECK(outH > 0 && outW > 0, "computed output size is non-positive");

    const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    const float* b_ptr = nullptr;
    if (has_bias) {
        auto b = bias_opt.value();
        CHECK_INPUT(b);
        TORCH_CHECK(b.dim() == 1 && (int)b.size(0) == C, "bias must be [C]");
        b_ptr = b.data_ptr<float>();
    }

    auto y = torch::empty({N, C, outH, outW}, x.options());

    // Hot path for (3x7, stride=1, pad=0, dilation=1) and typical contiguous layout.
    if (kH == 3 && kW == 7 && sH == 1 && sW == 1 && pH == 0 && pW == 0 && dH == 1 && dW == 1) {
        dim3 block(HOT_BLOCK_X, HOT_BLOCK_Y, 1);
        dim3 grid(
            (outW + (HOT_BLOCK_X * HOT_WPT) - 1) / (HOT_BLOCK_X * HOT_WPT),
            (outH + HOT_BLOCK_Y - 1) / HOT_BLOCK_Y,
            N * C
        );

        // Interior region in tile coordinates: tiles fully inside output.
        // A tile is interior if tile_oh0 + HOT_BLOCK_Y <= outH and tile_ow0 + HOT_BLOCK_X*WPT <= outW.
        // We'll launch the same grid, but choose kernel based on whether there exist boundary tiles.
        const int tiles_x = (outW + (HOT_BLOCK_X * HOT_WPT) - 1) / (HOT_BLOCK_X * HOT_WPT);
        const int tiles_y = (outH + HOT_BLOCK_Y - 1) / HOT_BLOCK_Y;
        const int interior_tiles_x = outW / (HOT_BLOCK_X * HOT_WPT);
        const int interior_tiles_y = outH / HOT_BLOCK_Y;

        // If all tiles are interior (exact multiple), use interior kernel everywhere.
        bool all_interior = (interior_tiles_x == tiles_x) && (interior_tiles_y == tiles_y);

        if (has_bias) {
            if (all_interior) {
                dwconv2d_k3x7_s1_p0_d1_sm_wpt4<true, true><<<grid, block>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                    N, C, H, W, outH, outW
                );
            } else {
                // For simplicity, use boundary-safe kernel for all tiles when not perfectly divisible.
                dwconv2d_k3x7_s1_p0_d1_sm_wpt4<true, false><<<grid, block>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), b_ptr, y.data_ptr<float>(),
                    N, C, H, W, outH, outW
                );
            }
        } else {
            if (all_interior) {
                dwconv2d_k3x7_s1_p0_d1_sm_wpt4<false, true><<<grid, block>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), nullptr, y.data_ptr<float>(),
                    N, C, H, W, outH, outW
                );
            } else {
                dwconv2d_k3x7_s1_p0_d1_sm_wpt4<false, false><<<grid, block>>>(
                    x.data_ptr<float>(), w.data_ptr<float>(), nullptr, y.data_ptr<float>(),
                    N, C, H, W, outH, outW
                );
            }
        }
        return y;
    }

    // Generic fallback
    const int threads = 256;
    const long long total = (long long)N * C * outH * outW;
    const int blocks = (int)((total + threads - 1) / threads);

    dwconv2d_generic_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b_ptr,
        y.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kH, kW,
        (int)sH, (int)sW,
        (int)pH, (int)pW,
        (int)dH, (int)dW,
        has_bias
    );

    return y;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv_depthwise2d_asymmetric_input_asymmetric_kernel_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> bias_opt,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW
);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_dwconv_k3x7_sm_wpt4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_depthwise2d_asymmetric_input_asymmetric_kernel_forward_cuda"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Depthwise Conv2d (groups=in_channels) implemented with an optimized custom CUDA kernel (forward-only).
    Hot path: kH=3,kW=7,stride=1,padding=0,dilation=1.
    Generic fallback otherwise.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if int(out_channels) != int(in_channels):
            raise ValueError("ModelNew implements depthwise conv: out_channels must equal in_channels.")
        if int(groups) != int(in_channels):
            raise ValueError("ModelNew implements depthwise conv: groups must equal in_channels.")

        self.in_channels = int(in_channels)
        self.kH = int(kernel_size_h)
        self.kW = int(kernel_size_w)
        self.sH = int(stride_h)
        self.sW = int(stride_w)
        self.pH = int(padding_h)
        self.pW = int(padding_w)
        self.dH = int(dilation_h)
        self.dW = int(dilation_w)
        self.bias_enabled = bool(bias)

        w = torch.empty(self.in_channels, 1, self.kH, self.kW, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)

        if self.bias_enabled:
            self.bias = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.custom_ops = custom_ops_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensor input")
        if x.dtype != torch.float32:
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        w = self.weight
        if not w.is_cuda:
            w = w.to(device=x.device)
        if w.dtype != torch.float32:
            w = w.float()
        if not w.is_contiguous():
            w = w.contiguous()

        b = None
        if self.bias is not None:
            b = self.bias
            if not b.is_cuda:
                b = b.to(device=x.device)
            if b.dtype != torch.float32:
                b = b.float()
            if not b.is_contiguous():
                b = b.contiguous()

        return self.custom_ops.conv_depthwise2d_asymmetric_input_asymmetric_kernel_forward_cuda(
            x, w, b,
            self.sH, self.sW,
            self.pH, self.pW,
            self.dH, self.dW
        )