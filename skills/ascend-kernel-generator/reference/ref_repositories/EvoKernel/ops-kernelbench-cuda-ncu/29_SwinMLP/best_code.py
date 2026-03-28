import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Optimized fused window movement + head packing/unpacking for SwinMLP Conv1d path
# Improvements over baseline:
#   - Add specialized fast-path kernels for window_size=7 and shift_size in {0,3}
#   - Require head_dim%4==0 for fast path and use float4 vectorized IO
#   - Unroll heads for common num_heads (3/6/12/24) to reduce loop overhead
#   - Reduce index math and improve cache locality
#   - Avoid redundant .contiguous() where possible
# -----------------------------------------------------------------------------

swin_mlp_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_FLOAT
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#endif

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float ldg_f32(const float* p) { return __ldg(p); }
#else
__host__ __forceinline__ float ldg_f32(const float* p) { return *p; }
#endif

__device__ __forceinline__ bool is_aligned_16(const void* p) { return (((uintptr_t)p) & 15) == 0; }

__device__ __forceinline__ float4 load_f4_ro(const float* p) {
    float4 v;
    v.x = ldg_f32(p + 0);
    v.y = ldg_f32(p + 1);
    v.z = ldg_f32(p + 2);
    v.w = ldg_f32(p + 3);
    return v;
}
__device__ __forceinline__ void store_f4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

// Generic decode (kept for fallback)
__device__ __forceinline__ void decode_row_partition_generic(
    int64_t row, int Ws, int nWh, int nWw,
    int &b, int &win_h, int &win_w, int &token, int &wy, int &wx
) {
    int Ws2 = Ws * Ws;
    token = (int)(row % Ws2);
    int64_t t = row / Ws2;

    win_w = (int)(t % nWw);
    t /= nWw;
    win_h = (int)(t % nWh);
    b = (int)(t / nWh);

    wy = token / Ws;
    wx = token - wy * Ws;
}

// ------------------------------
// Fast-path specialized kernels
// window_size=7 -> Ws2=49
// shift_size either 0 or 3 (Swin alternation when window_size=7)
// head_dim multiple of 4
// ------------------------------

template<int SHIFT, int NH> // SHIFT 0 or 3, NH in {3,6,12,24}
__global__ __launch_bounds__(256, 2)
void fused_partition_pack_bhwc_ws7_f4(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int H, int W, int C,
    int P_l, int P_t,
    int nWh, int nWw, int head_dim
) {
    // blockIdx.y enumerates rows = B*nW*49, each row is one (window, token)
    int64_t row = (int64_t)blockIdx.y;
    int lane4 = (int)blockIdx.x * blockDim.x + threadIdx.x; // in float4 units
    int hd4 = head_dim >> 2;
    if (lane4 >= hd4) return;

    // decode row for Ws=7
    // token in [0,48]
    int token = (int)(row % 49);
    int64_t t = row / 49;

    int win_w = (int)(t % nWw);
    t /= nWw;
    int win_h = (int)(t % nWh);
    int b = (int)(t / nWh);

    int wy = token / 7;
    int wx = token - wy * 7;

    int py = win_h * 7 + wy;
    int px = win_w * 7 + wx;
    int oy = (SHIFT == 0) ? py : (py - P_t);
    int ox = (SHIFT == 0) ? px : (px - P_l);

    int64_t widx = (row / 49); // == ((b*nWh + win_h)*nWw + win_w)
    int c_lane = lane4 << 2;

    // base output for each head:
    // out[widx, h*49 + token, c_lane:c_lane+4]
    // contiguous in head_dim
    if ((unsigned)oy >= (unsigned)H || (unsigned)ox >= (unsigned)W) {
        #pragma unroll
        for (int h = 0; h < NH; ++h) {
            int cin = h * 49 + token;
            int64_t o_off = ((widx * (int64_t)(NH * 49) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
            float* op = out + o_off;
            // out is from torch::empty => aligned enough typically, but guard anyway
            if (is_aligned_16(op)) store_f4(op, make_float4(0.f,0.f,0.f,0.f));
            else { op[0]=0.f; op[1]=0.f; op[2]=0.f; op[3]=0.f; }
        }
        return;
    }

    int64_t base_x = (((int64_t)b * H + oy) * W + ox) * (int64_t)C;

    #pragma unroll
    for (int h = 0; h < NH; ++h) {
        const float* xp = x + base_x + (int64_t)h * (int64_t)head_dim + c_lane;
        int cin = h * 49 + token;
        int64_t o_off = ((widx * (int64_t)(NH * 49) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
        float* op = out + o_off;

        float4 v = load_f4_ro(xp);
        if (is_aligned_16(op)) store_f4(op, v);
        else { op[0]=v.x; op[1]=v.y; op[2]=v.z; op[3]=v.w; }
    }
}

template<int SHIFT, int NH>
__global__ __launch_bounds__(256, 2)
void fused_unpack_reverse_bhwc_ws7_f4(
    const float* __restrict__ in,
    float* __restrict__ xout,
    int B, int H, int W, int C,
    int P_l, int P_t,
    int nWh, int nWw, int head_dim
) {
    // blockIdx.y enumerates pixels in output: B*H*W
    int64_t p = (int64_t)blockIdx.y;
    int lane4 = (int)blockIdx.x * blockDim.x + threadIdx.x; // float4 units in head_dim
    int hd4 = head_dim >> 2;
    if (lane4 >= hd4) return;

    int ox = (int)(p % W);
    int64_t t = p / W;
    int oy = (int)(t % H);
    int b = (int)(t / H);

    int py = (SHIFT == 0) ? oy : (oy + P_t);
    int px = (SHIFT == 0) ? ox : (ox + P_l);

    int win_h = py / 7;
    int win_w = px / 7;
    int wy = py - win_h * 7;
    int wx = px - win_w * 7;
    int token = wy * 7 + wx; // 0..48

    int64_t widx = ((int64_t)b * (int64_t)nWh + (int64_t)win_h) * (int64_t)nWw + (int64_t)win_w;

    int c_lane = lane4 << 2;
    int64_t base_x = (((int64_t)b * H + oy) * W + ox) * (int64_t)C;

    #pragma unroll
    for (int h = 0; h < NH; ++h) {
        int cin = h * 49 + token;
        int64_t in_off = ((widx * (int64_t)(NH * 49) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
        const float* ip = in + in_off;

        float* xp = xout + base_x + (int64_t)h * (int64_t)head_dim + c_lane;

        float4 v = load_f4_ro(ip);
        if (is_aligned_16(xp)) store_f4(xp, v);
        else { xp[0]=v.x; xp[1]=v.y; xp[2]=v.z; xp[3]=v.w; }
    }
}

// ------------------------------
// Generic fallback kernels (kept from baseline, lightly cleaned up)
// ------------------------------

template<int VEC>
__global__ __launch_bounds__(128, 3)
void fused_partition_pack_bhwc_vec_generic(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int H, int W, int C,
    int Ws, int shift_size, int P_l, int P_t,
    int nWh, int nWw, int nH, int head_dim,
    int hdvec_per_tile
) {
    int64_t row = (int64_t)blockIdx.y;
    int lane = (int)blockIdx.x * hdvec_per_tile + threadIdx.x;
    int hdvec = head_dim / VEC;
    if (lane >= hdvec) return;

    int b, win_h, win_w, token, wy, wx;
    decode_row_partition_generic(row, Ws, nWh, nWw, b, win_h, win_w, token, wy, wx);

    int Ws2 = Ws * Ws;
    int py = win_h * Ws + wy;
    int px = win_w * Ws + wx;
    int oy = (shift_size == 0) ? py : (py - P_t);
    int ox = (shift_size == 0) ? px : (px - P_l);

    int64_t widx = (row / (Ws2));
    int c_lane = lane * VEC;

    if ((unsigned)oy >= (unsigned)H || (unsigned)ox >= (unsigned)W) {
        for (int h = 0; h < nH; ++h) {
            int cin = h * Ws2 + token;
            int64_t o_off = ((widx * (int64_t)(nH * Ws2) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
            float* op = out + o_off;
            #pragma unroll
            for (int i = 0; i < VEC; ++i) op[i] = 0.f;
        }
        return;
    }

    int64_t base_x = (((int64_t)b * H + oy) * W + ox) * (int64_t)C;

    for (int h = 0; h < nH; ++h) {
        const float* xp = x + base_x + (int64_t)h * (int64_t)head_dim + c_lane;
        int Ws2 = Ws * Ws;
        int cin = h * Ws2 + token;
        int64_t o_off = ((widx * (int64_t)(nH * Ws2) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
        float* op = out + o_off;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) op[i] = ldg_f32(xp + i);
    }
}

__global__ __launch_bounds__(256, 2)
void fused_partition_pack_bhwc_scalar_generic(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int H, int W, int C,
    int Ws, int shift_size, int P_l, int P_t,
    int nWh, int nWw, int nH, int head_dim
) {
    int64_t total = (int64_t)B * (int64_t)nWh * (int64_t)nWw * (int64_t)(Ws * Ws) * (int64_t)nH * (int64_t)head_dim;
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    int c_in_head = (int)(idx % head_dim);
    int64_t t = idx / head_dim;

    int h = (int)(t % nH);
    t /= nH;

    int Ws2 = Ws * Ws;
    int token = (int)(t % Ws2);
    t /= Ws2;

    int win_w = (int)(t % nWw);
    t /= nWw;
    int win_h = (int)(t % nWh);
    int b = (int)(t / nWh);

    int wy = token / Ws;
    int wx = token - wy * Ws;

    int py = win_h * Ws + wy;
    int px = win_w * Ws + wx;
    int oy = (shift_size == 0) ? py : (py - P_t);
    int ox = (shift_size == 0) ? px : (px - P_l);

    float v = 0.f;
    if ((unsigned)oy < (unsigned)H && (unsigned)ox < (unsigned)W) {
        int c = h * head_dim + c_in_head;
        int64_t x_off = (((int64_t)b * H + oy) * W + ox) * (int64_t)C + c;
        v = ldg_f32(x + x_off);
    }

    int64_t widx = ((int64_t)b * (int64_t)nWh + win_h) * (int64_t)nWw + win_w;
    int cin = h * (Ws2) + token;
    int64_t o_off = ((widx * (int64_t)(nH * Ws2) + (int64_t)cin) * (int64_t)head_dim) + c_in_head;
    out[o_off] = v;
}

template<int VEC>
__global__ __launch_bounds__(128, 3)
void fused_unpack_reverse_bhwc_vec_generic(
    const float* __restrict__ in,
    float* __restrict__ xout,
    int B, int H, int W, int C,
    int Ws, int shift_size, int P_l, int P_t,
    int nWh, int nWw, int nH, int head_dim,
    int hdvec_per_tile
) {
    int64_t p = (int64_t)blockIdx.y;
    int lane = (int)blockIdx.x * hdvec_per_tile + threadIdx.x;
    int hdvec = head_dim / VEC;
    if (lane >= hdvec) return;

    int ox = (int)(p % W);
    int64_t t = p / W;
    int oy = (int)(t % H);
    int b = (int)(t / H);

    int py = (shift_size == 0) ? oy : (oy + P_t);
    int px = (shift_size == 0) ? ox : (ox + P_l);

    int win_h = py / Ws;
    int win_w = px / Ws;
    int wy = py - win_h * Ws;
    int wx = px - win_w * Ws;
    int token = wy * Ws + wx;

    int Ws2 = Ws * Ws;
    int64_t widx = ((int64_t)b * (int64_t)nWh + win_h) * (int64_t)nWw + win_w;
    int c_lane = lane * VEC;
    int64_t base_x = (((int64_t)b * H + oy) * W + ox) * (int64_t)C;

    for (int h = 0; h < nH; ++h) {
        int cin = h * Ws2 + token;
        int64_t in_off = ((widx * (int64_t)(nH * Ws2) + (int64_t)cin) * (int64_t)head_dim) + c_lane;
        const float* ip = in + in_off;

        float* xp = xout + base_x + (int64_t)h * (int64_t)head_dim + c_lane;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) xp[i] = ldg_f32(ip + i);
    }
}

__global__ __launch_bounds__(256, 2)
void fused_unpack_reverse_bhwc_scalar_generic(
    const float* __restrict__ in,
    float* __restrict__ xout,
    int B, int H, int W, int C,
    int Ws, int shift_size, int P_l, int P_t,
    int nWh, int nWw, int nH, int head_dim
) {
    int64_t total = (int64_t)B * (int64_t)H * (int64_t)W * (int64_t)C;
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    int c = (int)(idx % C);
    int64_t t = idx / C;

    int ox = (int)(t % W);
    t /= W;
    int oy = (int)(t % H);
    int b = (int)(t / H);

    int h = c / head_dim;
    int c_in_head = c - h * head_dim;

    int py = (shift_size == 0) ? oy : (oy + P_t);
    int px = (shift_size == 0) ? ox : (ox + P_l);

    int win_h = py / Ws;
    int win_w = px / Ws;
    int wy = py - win_h * Ws;
    int wx = px - win_w * Ws;
    int token = wy * Ws + wx;

    int Ws2 = Ws * Ws;
    int64_t widx = ((int64_t)b * (int64_t)nWh + win_h) * (int64_t)nWw + win_w;
    int cin = h * Ws2 + token;

    int64_t in_off = ((widx * (int64_t)(nH * Ws2) + (int64_t)cin) * (int64_t)head_dim) + c_in_head;
    int64_t x_off = (((int64_t)b * H + oy) * W + ox) * (int64_t)C + c;
    xout[x_off] = ldg_f32(in + in_off);
}

// ------------------------------
// C++ exposed wrappers with fast-path selection
// ------------------------------

torch::Tensor fused_pad_partition_pack_forward_cuda(torch::Tensor x_bhwc, int64_t window_size, int64_t shift_size, int64_t num_heads) {
    CHECK_CUDA(x_bhwc);
    CHECK_CONTIGUOUS(x_bhwc);
    CHECK_FLOAT(x_bhwc);
    TORCH_CHECK(x_bhwc.dim() == 4, "x must be BHWC");

    int64_t B = x_bhwc.size(0);
    int64_t H = x_bhwc.size(1);
    int64_t W = x_bhwc.size(2);
    int64_t C = x_bhwc.size(3);

    TORCH_CHECK(window_size > 0, "window_size must be > 0");
    TORCH_CHECK(shift_size >= 0 && shift_size < window_size, "shift_size must be in [0, window_size)");
    TORCH_CHECK(num_heads > 0, "num_heads must be > 0");
    TORCH_CHECK(C % num_heads == 0, "C must be divisible by num_heads");

    int64_t head_dim = C / num_heads;

    int64_t _H = H + (shift_size > 0 ? window_size : 0);
    int64_t _W = W + (shift_size > 0 ? window_size : 0);
    TORCH_CHECK(_H % window_size == 0 && _W % window_size == 0, "padded H/W must be divisible by window_size");
    if (shift_size == 0) {
        TORCH_CHECK(H % window_size == 0 && W % window_size == 0, "H/W must be divisible by window_size when shift_size=0");
    }

    int64_t P_l = (shift_size > 0) ? (window_size - shift_size) : 0;
    int64_t P_t = (shift_size > 0) ? (window_size - shift_size) : 0;

    int64_t nWh = _H / window_size;
    int64_t nWw = _W / window_size;
    int64_t nW = nWh * nWw;
    int64_t Ws2 = window_size * window_size;

    auto out = torch::empty({B * nW, num_heads * Ws2, head_dim}, x_bhwc.options());

    // Fast path: Ws=7, shift=0 or 3, head_dim %4==0, num_heads in {3,6,12,24}
    if (window_size == 7 && (shift_size == 0 || shift_size == 3) && (head_dim % 4 == 0) &&
        (num_heads == 3 || num_heads == 6 || num_heads == 12 || num_heads == 24)) {

        const int threads = 256;
        int hd4 = (int)(head_dim / 4);
        int grid_x = (hd4 + threads - 1) / threads;
        int64_t rows = (int64_t)B * (int64_t)nW * 49;
        dim3 block(threads, 1, 1);
        dim3 grid(grid_x, (unsigned int)rows, 1);

        const float* xptr = x_bhwc.data_ptr<float>();
        float* optr = out.data_ptr<float>();

        int Pli = (int)P_l;
        int Pti = (int)P_t;

        if (shift_size == 0) {
            if (num_heads == 3) fused_partition_pack_bhwc_ws7_f4<0,3><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 6) fused_partition_pack_bhwc_ws7_f4<0,6><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 12) fused_partition_pack_bhwc_ws7_f4<0,12><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else fused_partition_pack_bhwc_ws7_f4<0,24><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
        } else {
            if (num_heads == 3) fused_partition_pack_bhwc_ws7_f4<3,3><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 6) fused_partition_pack_bhwc_ws7_f4<3,6><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 12) fused_partition_pack_bhwc_ws7_f4<3,12><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else fused_partition_pack_bhwc_ws7_f4<3,24><<<grid, block>>>(xptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
        }
        return out;
    }

    // Generic vector fallback
    bool vec4 = (head_dim % 4 == 0);
    bool vec2 = (!vec4) && (head_dim % 2 == 0);

    if (vec4 || vec2) {
        const int VEC = vec4 ? 4 : 2;
        int hdvec = (int)(head_dim / VEC);

        const int threads = 128;
        int hdvec_per_tile = threads;
        int grid_x = (hdvec + hdvec_per_tile - 1) / hdvec_per_tile;

        int64_t rows = (int64_t)B * (int64_t)nW * (int64_t)Ws2;
        dim3 block(threads, 1, 1);
        dim3 grid(grid_x, (unsigned int)rows, 1);

        if (vec4) {
            fused_partition_pack_bhwc_vec_generic<4><<<grid, block>>>(
                x_bhwc.data_ptr<float>(), out.data_ptr<float>(),
                (int)B,(int)H,(int)W,(int)C,
                (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
                (int)nWh,(int)nWw,(int)num_heads,(int)head_dim,
                hdvec_per_tile
            );
        } else {
            fused_partition_pack_bhwc_vec_generic<2><<<grid, block>>>(
                x_bhwc.data_ptr<float>(), out.data_ptr<float>(),
                (int)B,(int)H,(int)W,(int)C,
                (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
                (int)nWh,(int)nWw,(int)num_heads,(int)head_dim,
                hdvec_per_tile
            );
        }
    } else {
        const int threads = 256;
        int64_t total = (int64_t)B * (int64_t)nWh * (int64_t)nWw * (int64_t)Ws2 * (int64_t)num_heads * (int64_t)head_dim;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = blocks > 65535 ? 65535 : blocks;
        fused_partition_pack_bhwc_scalar_generic<<<blocks, threads>>>(
            x_bhwc.data_ptr<float>(), out.data_ptr<float>(),
            (int)B,(int)H,(int)W,(int)C,
            (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
            (int)nWh,(int)nWw,(int)num_heads,(int)head_dim
        );
    }

    return out;
}

torch::Tensor fused_unpack_reverse_crop_forward_cuda(torch::Tensor packed, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size, int64_t shift_size, int64_t num_heads) {
    CHECK_CUDA(packed);
    CHECK_CONTIGUOUS(packed);
    CHECK_FLOAT(packed);
    TORCH_CHECK(packed.dim() == 3, "packed must be [B*nW, nH*Ws2, head_dim]");
    TORCH_CHECK(num_heads > 0, "num_heads must be > 0");
    TORCH_CHECK(C % num_heads == 0, "C must be divisible by num_heads");

    int64_t head_dim = C / num_heads;

    TORCH_CHECK(window_size > 0, "window_size must be > 0");
    TORCH_CHECK(shift_size >= 0 && shift_size < window_size, "shift_size must be in [0, window_size)");

    int64_t _H = H + (shift_size > 0 ? window_size : 0);
    int64_t _W = W + (shift_size > 0 ? window_size : 0);
    TORCH_CHECK(_H % window_size == 0 && _W % window_size == 0, "padded H/W must be divisible by window_size");
    if (shift_size == 0) {
        TORCH_CHECK(H % window_size == 0 && W % window_size == 0, "H/W must be divisible by window_size when shift_size=0");
    }

    int64_t P_l = (shift_size > 0) ? (window_size - shift_size) : 0;
    int64_t P_t = (shift_size > 0) ? (window_size - shift_size) : 0;

    int64_t nWh = _H / window_size;
    int64_t nWw = _W / window_size;
    int64_t nW = nWh * nWw;
    int64_t Ws2 = window_size * window_size;

    TORCH_CHECK(packed.size(0) == B * nW, "packed dim0 must be B*nW (padded grid)");
    TORCH_CHECK(packed.size(1) == num_heads * Ws2, "packed dim1 must be nH*Ws2");
    TORCH_CHECK(packed.size(2) == head_dim, "packed dim2 must be head_dim");

    auto out = torch::empty({B, H, W, C}, packed.options());

    // Fast path
    if (window_size == 7 && (shift_size == 0 || shift_size == 3) && (head_dim % 4 == 0) &&
        (num_heads == 3 || num_heads == 6 || num_heads == 12 || num_heads == 24)) {

        const int threads = 256;
        int hd4 = (int)(head_dim / 4);
        int grid_x = (hd4 + threads - 1) / threads;
        int64_t pixels = (int64_t)B * (int64_t)H * (int64_t)W;
        dim3 block(threads, 1, 1);
        dim3 grid(grid_x, (unsigned int)pixels, 1);

        const float* iptr = packed.data_ptr<float>();
        float* optr = out.data_ptr<float>();

        int Pli = (int)P_l;
        int Pti = (int)P_t;

        if (shift_size == 0) {
            if (num_heads == 3) fused_unpack_reverse_bhwc_ws7_f4<0,3><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 6) fused_unpack_reverse_bhwc_ws7_f4<0,6><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 12) fused_unpack_reverse_bhwc_ws7_f4<0,12><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else fused_unpack_reverse_bhwc_ws7_f4<0,24><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
        } else {
            if (num_heads == 3) fused_unpack_reverse_bhwc_ws7_f4<3,3><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 6) fused_unpack_reverse_bhwc_ws7_f4<3,6><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else if (num_heads == 12) fused_unpack_reverse_bhwc_ws7_f4<3,12><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
            else fused_unpack_reverse_bhwc_ws7_f4<3,24><<<grid, block>>>(iptr, optr, (int)B,(int)H,(int)W,(int)C, Pli,Pti, (int)nWh,(int)nWw, (int)head_dim);
        }
        return out;
    }

    // Generic fallback
    bool vec4 = (head_dim % 4 == 0);
    bool vec2 = (!vec4) && (head_dim % 2 == 0);

    if (vec4 || vec2) {
        const int VEC = vec4 ? 4 : 2;
        int hdvec = (int)(head_dim / VEC);

        const int threads = 128;
        int hdvec_per_tile = threads;
        int grid_x = (hdvec + hdvec_per_tile - 1) / hdvec_per_tile;

        int64_t pixels = (int64_t)B * (int64_t)H * (int64_t)W;
        dim3 block(threads, 1, 1);
        dim3 grid(grid_x, (unsigned int)pixels, 1);

        if (vec4) {
            fused_unpack_reverse_bhwc_vec_generic<4><<<grid, block>>>(
                packed.data_ptr<float>(), out.data_ptr<float>(),
                (int)B,(int)H,(int)W,(int)C,
                (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
                (int)nWh,(int)nWw,(int)num_heads,(int)head_dim,
                hdvec_per_tile
            );
        } else {
            fused_unpack_reverse_bhwc_vec_generic<2><<<grid, block>>>(
                packed.data_ptr<float>(), out.data_ptr<float>(),
                (int)B,(int)H,(int)W,(int)C,
                (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
                (int)nWh,(int)nWw,(int)num_heads,(int)head_dim,
                hdvec_per_tile
            );
        }
    } else {
        const int threads = 256;
        int64_t total = (int64_t)B * (int64_t)H * (int64_t)W * (int64_t)C;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = blocks > 65535 ? 65535 : blocks;
        fused_unpack_reverse_bhwc_scalar_generic<<<blocks, threads>>>(
            packed.data_ptr<float>(), out.data_ptr<float>(),
            (int)B,(int)H,(int)W,(int)C,
            (int)window_size,(int)shift_size,(int)P_l,(int)P_t,
            (int)nWh,(int)nWw,(int)num_heads,(int)head_dim
        );
    }

    return out;
}
"""

swin_mlp_cpp_source = r"""
torch::Tensor fused_pad_partition_pack_forward_cuda(torch::Tensor x_bhwc, int64_t window_size, int64_t shift_size, int64_t num_heads);
torch::Tensor fused_unpack_reverse_crop_forward_cuda(torch::Tensor packed, int64_t B, int64_t H, int64_t W, int64_t C, int64_t window_size, int64_t shift_size, int64_t num_heads);
"""

custom_ops_lib = load_inline(
    name="custom_ops_lib_swin_mlp_windows_packfuse_v2",
    cpp_sources=swin_mlp_cpp_source,
    cuda_sources=swin_mlp_cuda_source,
    functions=[
        "fused_pad_partition_pack_forward_cuda",
        "fused_unpack_reverse_crop_forward_cuda",
    ],
    with_cuda=True,
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinMLPBlockNew(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)

        # Conv1d expects (N, Cin, L). Here Cin = nH*Ws2, L = head_dim.
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size**2,
            self.num_heads * self.window_size**2,
            kernel_size=1,
            groups=self.num_heads,
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and C == self.dim

        shortcut = x
        x = self.norm1(x)

        # Keep BHWC contiguous for our kernels
        x_bhwc = x.view(B, H, W, C).contiguous()

        packed = custom_ops_lib.fused_pad_partition_pack_forward_cuda(
            x_bhwc, int(self.window_size), int(self.shift_size), int(self.num_heads)
        )
        # Conv1d requires contiguous
        if not packed.is_contiguous():
            packed = packed.contiguous()

        packed_out = self.spatial_mlp(packed)

        if not packed_out.is_contiguous():
            packed_out = packed_out.contiguous()

        x_bhwc_out = custom_ops_lib.fused_unpack_reverse_crop_forward_cuda(
            packed_out,
            int(B),
            int(H),
            int(W),
            int(C),
            int(self.window_size),
            int(self.shift_size),
            int(self.num_heads),
        )

        x = x_bhwc_out.view(B, H * W, C).contiguous()
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayerNew(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        dps = drop_path if isinstance(drop_path, list) else [drop_path for _ in range(depth)]
        self.blocks = nn.ModuleList(
            [
                SwinMLPBlockNew(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dps[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ModelNew(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerNew(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x