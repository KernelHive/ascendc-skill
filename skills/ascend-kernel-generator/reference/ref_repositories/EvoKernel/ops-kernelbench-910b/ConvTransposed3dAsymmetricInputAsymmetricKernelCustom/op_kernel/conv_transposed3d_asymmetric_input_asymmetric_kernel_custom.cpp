
#include "kernel_operator.h"

// Specialized ConvTranspose3d (PyTorch layout):
// x: [N,Cin,Din,Hin,Win] = [16,32,16,32,64]
// w: [Cin,Cout,KD,KH,KW] = [32,16,3,5,7]
// y: [16,16,18,36,70]
//
// This round:
// - Block mapping over flattened output elements (not rows) to reduce div/mod pressure.
//   Each block walks a contiguous [start,end) range and advances indices with carry logic.
// - Keep TILE_OW=8 and safe-interior sliding-window load (tile+KW-1) per (ci,kd,kh).
// - Cache weights (7 floats) once per (ci,kd,kh) for both safe and border paths.

class KernelConvTranspose3dAsymAsymFlatOutTileOw8 {
public:
    __aicore__ inline KernelConvTranspose3dAsymAsymFlatOutTileOw8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t totalY, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalY_ = totalY;
        blockDim_ = (blockDim == 0 ? 1u : blockDim);
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 16;
        constexpr int CIN = 32;
        constexpr int DIN = 16;
        constexpr int HIN = 32;
        constexpr int WIN = 64;

        constexpr int COUT = 16;
        constexpr int KD = 3;
        constexpr int KH = 5;
        constexpr int KW = 7;

        constexpr int DOUT = 18;
        constexpr int HOUT = 36;
        constexpr int WOUT = 70;

        // Strides (NCDHW contiguous)
        constexpr int64_t X_STRIDE_NC = (int64_t)DIN * HIN * WIN;
        constexpr int64_t X_STRIDE_D  = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_H  = (int64_t)WIN;

        constexpr int64_t Y_STRIDE_CO = (int64_t)DOUT * HOUT * WOUT;
        constexpr int64_t Y_STRIDE_D  = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_H  = (int64_t)WOUT;

        // Weight: [ci,co,kd,kh,kw] contiguous
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KD * KH * KW;
        constexpr int64_t W_STRIDE_CO = (int64_t)KD * KH * KW;
        constexpr int64_t W_STRIDE_KD = (int64_t)KH * KW;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;

        constexpr int OW_SAFE_BEG = 6;
        constexpr int OW_SAFE_END = 63;

        constexpr int TILE_OW = 8;
        constexpr int XWIN = TILE_OW + KW - 1; // 14

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;

        const uint32_t elemsPerBlock = (totalY_ + bdim - 1) / bdim;
        uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (start > totalY_) start = totalY_;
        if (end > totalY_) end = totalY_;
        if (start >= end) return;

        // Decode start index once (scalar div/mod once per block), then carry-forward.
        uint32_t idx = start;
        int ow = (int)(idx % WOUT); idx /= WOUT;
        int oh = (int)(idx % HOUT); idx /= HOUT;
        int od = (int)(idx % DOUT); idx /= DOUT;
        int co = (int)(idx % COUT); idx /= COUT;
        int n  = (int)idx;

        // Precompute per-(n,co) bases; update with carry.
        int64_t yBaseNC = ((int64_t)n * COUT + (int64_t)co) * Y_STRIDE_CO;
        int64_t xBaseN  = (int64_t)n * CIN * X_STRIDE_NC;
        int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

        uint32_t cur = start;
        while (cur < end) {
            // Process a width tile if we are aligned and can stay on same (n,co,od,oh).
            int tile = TILE_OW;
            int remainW = WOUT - ow;
            if (tile > remainW) tile = remainW;
            // Also clamp to end to avoid crossing block boundary.
            uint32_t remainElems = end - cur;
            if ((uint32_t)tile > remainElems) tile = (int)remainElems;

            // If tile would cross row boundary in flattened order (it doesn't as long as ow+tile<=WOUT).
            // So we are safe.

            const int64_t yRowBase = yBaseNC + (int64_t)od * Y_STRIDE_D + (int64_t)oh * Y_STRIDE_H;

            // kd range s.t. id=od-kd in [0..DIN-1]
            int kdBeg, kdEnd;
            {
                int kb = od - (DIN - 1);
                int ke = od;
                if (kb < 0) kb = 0;
                if (ke > KD - 1) ke = KD - 1;
                kdBeg = kb;
                kdEnd = ke + 1;
            }

            // kh range s.t. ih=oh-kh in [0..HIN-1]
            int khBeg, khEnd;
            {
                int hb = oh - (HIN - 1);
                int he = oh;
                if (hb < 0) hb = 0;
                if (he > KH - 1) he = KH - 1;
                khBeg = hb;
                khEnd = he + 1;
            }

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            if (kdBeg < kdEnd && khBeg < khEnd) {
                const bool tileSafe = (ow >= OW_SAFE_BEG) && (ow + tile - 1 <= OW_SAFE_END);

                if (tileSafe) {
                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_NC;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int kd = kdBeg; kd < kdEnd; ++kd) {
                            const int id = od - kd;
                            const int64_t xBaseD = xBaseCi + (int64_t)id * X_STRIDE_D;
                            const int64_t wBaseKd = wBaseCiCo + (int64_t)kd * W_STRIDE_KD;

                            for (int kh = khBeg; kh < khEnd; ++kh) {
                                const int ih = oh - kh;
                                const int64_t xBaseH = xBaseD + (int64_t)ih * X_STRIDE_H;
                                const int64_t wBaseKh = wBaseKd + (int64_t)kh * W_STRIDE_KH;

                                // cache weights
                                const float wv0 = wGm.GetValue(wBaseKh + 0);
                                const float wv1 = wGm.GetValue(wBaseKh + 1);
                                const float wv2 = wGm.GetValue(wBaseKh + 2);
                                const float wv3 = wGm.GetValue(wBaseKh + 3);
                                const float wv4 = wGm.GetValue(wBaseKh + 4);
                                const float wv5 = wGm.GetValue(wBaseKh + 5);
                                const float wv6 = wGm.GetValue(wBaseKh + 6);

                                // shared input window: [ow-6 .. ow+(tile-1)] but load full XWIN (14) for simplicity
                                float xv[XWIN];
#pragma unroll
                                for (int j = 0; j < XWIN; ++j) {
                                    xv[j] = xGm.GetValue(xBaseH + (int64_t)(ow - 6 + j));
                                }

                                if (tile >= 1) acc0 += xv[6] * wv0 + xv[5] * wv1 + xv[4] * wv2 + xv[3] * wv3 + xv[2] * wv4 + xv[1] * wv5 + xv[0] * wv6;
                                if (tile >= 2) acc1 += xv[7] * wv0 + xv[6] * wv1 + xv[5] * wv2 + xv[4] * wv3 + xv[3] * wv4 + xv[2] * wv5 + xv[1] * wv6;
                                if (tile >= 3) acc2 += xv[8] * wv0 + xv[7] * wv1 + xv[6] * wv2 + xv[5] * wv3 + xv[4] * wv4 + xv[3] * wv5 + xv[2] * wv6;
                                if (tile >= 4) acc3 += xv[9] * wv0 + xv[8] * wv1 + xv[7] * wv2 + xv[6] * wv3 + xv[5] * wv4 + xv[4] * wv5 + xv[3] * wv6;
                                if (tile >= 5) acc4 += xv[10] * wv0 + xv[9] * wv1 + xv[8] * wv2 + xv[7] * wv3 + xv[6] * wv4 + xv[5] * wv5 + xv[4] * wv6;
                                if (tile >= 6) acc5 += xv[11] * wv0 + xv[10] * wv1 + xv[9] * wv2 + xv[8] * wv3 + xv[7] * wv4 + xv[6] * wv5 + xv[5] * wv6;
                                if (tile >= 7) acc6 += xv[12] * wv0 + xv[11] * wv1 + xv[10] * wv2 + xv[9] * wv3 + xv[8] * wv4 + xv[7] * wv5 + xv[6] * wv6;
                                if (tile >= 8) acc7 += xv[13] * wv0 + xv[12] * wv1 + xv[11] * wv2 + xv[10] * wv3 + xv[9] * wv4 + xv[8] * wv5 + xv[7] * wv6;
                            }
                        }
                    }
                } else {
                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_NC;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int kd = kdBeg; kd < kdEnd; ++kd) {
                            const int id = od - kd;
                            const int64_t xBaseD = xBaseCi + (int64_t)id * X_STRIDE_D;
                            const int64_t wBaseKd = wBaseCiCo + (int64_t)kd * W_STRIDE_KD;

                            for (int kh = khBeg; kh < khEnd; ++kh) {
                                const int ih = oh - kh;
                                const int64_t xBaseH = xBaseD + (int64_t)ih * X_STRIDE_H;
                                const int64_t wBaseKh = wBaseKd + (int64_t)kh * W_STRIDE_KH;

                                // cache weights once per (ci,kd,kh)
                                float wv[KW];
#pragma unroll
                                for (int i = 0; i < KW; ++i) {
                                    wv[i] = wGm.GetValue(wBaseKh + (int64_t)i);
                                }

#pragma unroll
                                for (int kw = 0; kw < KW; ++kw) {
                                    const float wval = wv[kw];
                                    if (tile >= 1) { int iw = (ow + 0) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 2) { int iw = (ow + 1) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 3) { int iw = (ow + 2) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 4) { int iw = (ow + 3) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 5) { int iw = (ow + 4) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 6) { int iw = (ow + 5) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 7) { int iw = (ow + 6) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                    if (tile >= 8) { int iw = (ow + 7) - kw; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue(xBaseH + (int64_t)iw) * wval; }
                                }
                            }
                        }
                    }
                }
            }

            if (tile >= 1) yGm.SetValue(yRowBase + (int64_t)(ow + 0), acc0);
            if (tile >= 2) yGm.SetValue(yRowBase + (int64_t)(ow + 1), acc1);
            if (tile >= 3) yGm.SetValue(yRowBase + (int64_t)(ow + 2), acc2);
            if (tile >= 4) yGm.SetValue(yRowBase + (int64_t)(ow + 3), acc3);
            if (tile >= 5) yGm.SetValue(yRowBase + (int64_t)(ow + 4), acc4);
            if (tile >= 6) yGm.SetValue(yRowBase + (int64_t)(ow + 5), acc5);
            if (tile >= 7) yGm.SetValue(yRowBase + (int64_t)(ow + 6), acc6);
            if (tile >= 8) yGm.SetValue(yRowBase + (int64_t)(ow + 7), acc7);

            // Advance flattened index by tile with carry updates (reduce div/mod in hot path).
            cur += (uint32_t)tile;
            ow += tile;

            if (ow >= WOUT) {
                ow = 0;
                oh += 1;
                if (oh >= HOUT) {
                    oh = 0;
                    od += 1;
                    if (od >= DOUT) {
                        od = 0;
                        co += 1;
                        if (co >= COUT) {
                            co = 0;
                            n += 1;
                        }
                        wBaseCo = (int64_t)co * W_STRIDE_CO;
                        yBaseNC = ((int64_t)n * COUT + (int64_t)co) * Y_STRIDE_CO;
                        xBaseN  = (int64_t)n * CIN * X_STRIDE_NC;
                    }
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalY_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed3d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dAsymAsymFlatOutTileOw8 op;
    op.Init(x, weight, y, tiling_data.totalY, tiling_data.blockDim);
    op.Process();
}
