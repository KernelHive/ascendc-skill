
#include "kernel_operator.h"

// Specialized ConvTranspose2d (PyTorch layout):
// x: [64,64,128,256], w: [64,128,3,5], y: [64,128,130,260], fp32
// stride=1, pad=0, dil=1, outpad=0.
//
// This version increases launch parallelism by mapping blocks to width-tiles:
// each block computes one (n,co,ho,woTile) of TILE_WO=8.
// This reduces long per-row scalar loops and helps hide scalar/control latency.
//
// Safe interior tiles reuse a contiguous x segment per (ci,kh) across 8 lanes and fully unroll kw=5.

class KernelConvTranspose2dAsymInK3x5S1P0_Fp32_TiledOw8 {
public:
    __aicore__ inline KernelConvTranspose2dAsymInK3x5S1P0_Fp32_TiledOw8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalTiles, uint32_t tilesPerRow,
                               uint32_t wout, uint32_t tileWo, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalTiles_ = totalTiles;
        tilesPerRow_ = tilesPerRow;
        wout_ = wout;
        tileWo_ = tileWo;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 64;
        constexpr int CIN = 64;
        constexpr int COUT = 128;
        constexpr int HIN = 128;
        constexpr int WIN = 256;
        constexpr int KH = 3;
        constexpr int KW = 5;
        constexpr int HOUT = 130;
        constexpr int WOUT = 260;
        constexpr int TILE_WO = 8;

        constexpr int64_t ROWS_TOTAL = (int64_t)N * COUT * HOUT;
        constexpr int64_t TILES_PER_ROW = (WOUT + TILE_WO - 1) / TILE_WO; // 33
        constexpr int64_t TILES_TOTAL = ROWS_TOTAL * TILES_PER_ROW;

        if ((int)wout_ != WOUT) return;
        if ((int)tileWo_ != TILE_WO) return;
        if ((int)tilesPerRow_ != (int)TILES_PER_ROW) return;

        uint32_t totalTiles = totalTiles_;
        if (totalTiles == 0 || totalTiles > (uint32_t)TILES_TOTAL) totalTiles = (uint32_t)TILES_TOTAL;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;
        const uint32_t tilesPerBlock = (totalTiles + bdim - 1) / bdim;

        uint32_t tileStart = bid * tilesPerBlock;
        uint32_t tileEnd = tileStart + tilesPerBlock;
        if (tileStart > totalTiles) tileStart = totalTiles;
        if (tileEnd > totalTiles) tileEnd = totalTiles;

        // Strides (NCHW contiguous)
        constexpr int64_t X_STRIDE_H = (int64_t)WIN;
        constexpr int64_t X_STRIDE_C = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * X_STRIDE_C;

        constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_C = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * Y_STRIDE_C;

        // Weight: [ci,co,kh,kw]
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;                 // 5
        constexpr int64_t W_STRIDE_CO = (int64_t)KH * KW;            // 15
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * W_STRIDE_CO; // 1920

        // Safe wo range: iw = wo-kw in [0..WIN-1] for kw=0..4 => wo in [4..255]
        constexpr int WO_SAFE_BEG = KW - 1;   // 4
        constexpr int WO_SAFE_END = WIN - 1;  // 255

        float xseg[12]; // tile + (KW-1) = 8+4

        for (uint32_t gtid = tileStart; gtid < tileEnd; ++gtid) {
            uint32_t t = gtid;

            const int woTile = (int)(t % (uint32_t)TILES_PER_ROW);
            t /= (uint32_t)TILES_PER_ROW;

            const int ho = (int)(t % HOUT); t /= HOUT;
            const int co = (int)(t % COUT); t /= COUT;
            const int n  = (int)t;

            const int wo0 = woTile * TILE_WO;
            const int tile = (wo0 + TILE_WO <= WOUT) ? TILE_WO : (WOUT - wo0);

            const int64_t yRowBase = (int64_t)n * Y_STRIDE_N + (int64_t)co * Y_STRIDE_C + (int64_t)ho * Y_STRIDE_H;

            // kh range where ih=ho-kh in [0..HIN-1]
            int khBeg, khEnd;
            {
                int kb = ho - (HIN - 1);
                int ke = ho;
                if (kb < 0) kb = 0;
                if (ke > KH - 1) ke = KH - 1;
                khBeg = kb;
                khEnd = ke + 1;
            }

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            if (khBeg < khEnd) {
                const int64_t xBaseN = (int64_t)n * X_STRIDE_N;
                const int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

                const bool tileSafe = (wo0 >= WO_SAFE_BEG) && (wo0 + tile - 1 <= WO_SAFE_END);

                if (tileSafe) {
                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_C;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int kh = khBeg; kh < khEnd; ++kh) {
                            const int ih = ho - kh;
                            const int64_t xBaseH = xBaseCi + (int64_t)ih * X_STRIDE_H;
                            const int64_t wBaseKh = wBaseCiCo + (int64_t)kh * W_STRIDE_KH;

                            const float w0 = wGm.GetValue((uint64_t)(wBaseKh + 0));
                            const float w1 = wGm.GetValue((uint64_t)(wBaseKh + 1));
                            const float w2 = wGm.GetValue((uint64_t)(wBaseKh + 2));
                            const float w3 = wGm.GetValue((uint64_t)(wBaseKh + 3));
                            const float w4 = wGm.GetValue((uint64_t)(wBaseKh + 4));

                            const int64_t xSegBase = xBaseH + (int64_t)(wo0 - (KW - 1));
                            const int segLen = tile + (KW - 1); // <= 12
                            #pragma unroll
                            for (int i = 0; i < 12; ++i) {
                                if (i < segLen) xseg[i] = xGm.GetValue((uint64_t)(xSegBase + (int64_t)i));
                            }

                            if (tile >= 1) { const int b = 4;  acc0 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 2) { const int b = 5;  acc1 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 3) { const int b = 6;  acc2 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 4) { const int b = 7;  acc3 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 5) { const int b = 8;  acc4 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 6) { const int b = 9;  acc5 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 7) { const int b = 10; acc6 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                            if (tile >= 8) { const int b = 11; acc7 += xseg[b]  * w0 + xseg[b-1]*w1 + xseg[b-2]*w2 + xseg[b-3]*w3 + xseg[b-4]*w4; }
                        }
                    }
                } else {
                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_C;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int kh = khBeg; kh < khEnd; ++kh) {
                            const int ih = ho - kh;
                            const int64_t xBaseH = xBaseCi + (int64_t)ih * X_STRIDE_H;
                            const int64_t wBaseKh = wBaseCiCo + (int64_t)kh * W_STRIDE_KH;

                            const float w0 = wGm.GetValue((uint64_t)(wBaseKh + 0));
                            const float w1 = wGm.GetValue((uint64_t)(wBaseKh + 1));
                            const float w2 = wGm.GetValue((uint64_t)(wBaseKh + 2));
                            const float w3 = wGm.GetValue((uint64_t)(wBaseKh + 3));
                            const float w4 = wGm.GetValue((uint64_t)(wBaseKh + 4));

                            if (tile >= 1) { const int woL = wo0 + 0;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 2) { const int woL = wo0 + 1;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 3) { const int woL = wo0 + 2;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 4) { const int woL = wo0 + 3;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 5) { const int woL = wo0 + 4;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 6) { const int woL = wo0 + 5;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 7) { const int woL = wo0 + 6;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                            if (tile >= 8) { const int woL = wo0 + 7;
                                int iw = woL - 0; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0;
                                iw = woL - 1;     if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1;
                                iw = woL - 2;     if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2;
                                iw = woL - 3;     if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3;
                                iw = woL - 4;     if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4;
                            }
                        }
                    }
                }
            } // else: no valid kh => keep acc=0

            if (tile >= 1) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 0)), acc0);
            if (tile >= 2) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 1)), acc1);
            if (tile >= 3) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 2)), acc2);
            if (tile >= 4) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 3)), acc3);
            if (tile >= 5) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 4)), acc4);
            if (tile >= 6) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 5)), acc5);
            if (tile >= 7) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 6)), acc6);
            if (tile >= 8) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 7)), acc7);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalTiles_{0};
    uint32_t tilesPerRow_{0};
    uint32_t wout_{0};
    uint32_t tileWo_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose2dAsymInK3x5S1P0_Fp32_TiledOw8 op;
    op.Init(x, weight, y,
            tiling_data.totalTiles, tiling_data.tilesPerRow,
            tiling_data.wout, tiling_data.tileWo, tiling_data.blockDim);
    op.Process();
}
