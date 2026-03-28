
#include "kernel_operator.h"

// Specialized ConvTranspose2d (PyTorch layout):
// x: [N,Cin,Hin,Win] = [8,64,512,512]
// w: [Cin,Cout,Kh,Kw] = [64,64,3,7]
// stride=(1,1), pad=(0,0), outpad=(0,0), dil=(1,1), groups=1, bias=False
// y: [8,64,Hout=514,Wout=518]
//
// Optimizations in this round:
// - Wider WO tile (TILE_WO=8) to amortize scalar/control overhead.
// - Safe interior: for each (ci,kh) load 7 weights once and one contiguous x segment (tile+KW-1) once,
//   then update 8 accumulators from that shared segment (reuse), reducing GM loads and scalar address ops.
// - Border: keep bounds checks but hoist weight loads and explicitly unroll kw to reduce loop/control cost.

class KernelConvTranspose2dSqInAsymK3x7S1P0_Fp32_ReuseOw8 {
public:
    __aicore__ inline KernelConvTranspose2dSqInAsymK3x7S1P0_Fp32_ReuseOw8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalRows, uint32_t wout, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalRows_ = totalRows;
        wout_ = wout;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 8;
        constexpr int CIN = 64;
        constexpr int COUT = 64;
        constexpr int HIN = 512;
        constexpr int WIN = 512;
        constexpr int KH = 3;
        constexpr int KW = 7;
        constexpr int HOUT = 514;
        constexpr int WOUT = 518;

        constexpr int64_t ROWS_TOTAL = (int64_t)N * COUT * HOUT;

        uint32_t totalRows = totalRows_;
        if (totalRows == 0 || totalRows > (uint32_t)ROWS_TOTAL) totalRows = (uint32_t)ROWS_TOTAL;
        if ((int)wout_ != WOUT) return;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;

        const uint32_t rowsPerBlock = (totalRows + bdim - 1) / bdim;
        uint32_t rowStart = bid * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > totalRows) rowStart = totalRows;
        if (rowEnd > totalRows) rowEnd = totalRows;

        // Strides (NCHW contiguous)
        constexpr int64_t X_STRIDE_W = 1;
        constexpr int64_t X_STRIDE_H = (int64_t)WIN;
        constexpr int64_t X_STRIDE_C = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * X_STRIDE_C;

        constexpr int64_t Y_STRIDE_W = 1;
        constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_C = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * Y_STRIDE_C;

        // Weight layout: [ci,co,kh,kw] contiguous
        constexpr int64_t W_STRIDE_KW = 1;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;
        constexpr int64_t W_STRIDE_CO = (int64_t)KH * KW;            // 21
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * W_STRIDE_CO; // 1344

        // Safe wo range: require iw=wo-kw in [0..WIN-1] for kw=0..6 => wo in [6..511]
        constexpr int WO_SAFE_BEG = 6;
        constexpr int WO_SAFE_END = WIN - 1;

        constexpr int TILE_WO = 8;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            uint32_t t = row;
            const int ho = (int)(t % HOUT); t /= HOUT;
            const int co = (int)(t % COUT); t /= COUT;
            const int n  = (int)t;

            const int64_t yRowBase = (int64_t)n * Y_STRIDE_N + (int64_t)co * Y_STRIDE_C + (int64_t)ho * Y_STRIDE_H;

            // kh range s.t. ih=ho-kh in [0..HIN-1]
            int khBeg, khEnd;
            {
                int kb = ho - (HIN - 1);
                int ke = ho;
                if (kb < 0) kb = 0;
                if (ke > KH - 1) ke = KH - 1;
                khBeg = kb;
                khEnd = ke + 1;
            }

            if (khBeg >= khEnd) {
                for (int wo = 0; wo < WOUT; ++wo) {
                    yGm.SetValue((uint64_t)(yRowBase + (int64_t)wo), 0.0f);
                }
                continue;
            }

            const int64_t xBaseN = (int64_t)n * X_STRIDE_N;
            const int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

            int wo = 0;
            while (wo < WOUT) {
                const int tile = (wo + TILE_WO <= WOUT) ? TILE_WO : (WOUT - wo);
                const int wo0 = wo;

                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

                const bool tileSafe = (wo0 >= WO_SAFE_BEG) && (wo0 + tile - 1 <= WO_SAFE_END);

                if (tileSafe) {
                    // x segment indices: [wo0-6 .. wo0+tile-1] => length tile+6 (<=14)
                    float xseg[14];

                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_C;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int kh = khBeg; kh < khEnd; ++kh) {
                            const int ih = ho - kh;
                            const int64_t xBaseH = xBaseCi + (int64_t)ih * X_STRIDE_H;
                            const int64_t wBaseKh = wBaseCiCo + (int64_t)kh * W_STRIDE_KH;

                            // Load weights once per (ci,kh)
                            const float w0 = wGm.GetValue((uint64_t)(wBaseKh + 0));
                            const float w1 = wGm.GetValue((uint64_t)(wBaseKh + 1));
                            const float w2 = wGm.GetValue((uint64_t)(wBaseKh + 2));
                            const float w3 = wGm.GetValue((uint64_t)(wBaseKh + 3));
                            const float w4 = wGm.GetValue((uint64_t)(wBaseKh + 4));
                            const float w5 = wGm.GetValue((uint64_t)(wBaseKh + 5));
                            const float w6 = wGm.GetValue((uint64_t)(wBaseKh + 6));

                            const int64_t xSegBase = xBaseH + (int64_t)(wo0 - 6);
                            const int segLen = tile + (KW - 1); // tile+6, <=14
                            #pragma unroll
                            for (int i = 0; i < 14; ++i) {
                                if (i < segLen) xseg[i] = xGm.GetValue((uint64_t)(xSegBase + (int64_t)i));
                            }

                            // lane l uses xseg[6+l-kw] with w[kw]
                            if (tile >= 1) acc0 += xseg[6] * w0 + xseg[5] * w1 + xseg[4] * w2 + xseg[3] * w3 + xseg[2] * w4 + xseg[1] * w5 + xseg[0] * w6;
                            if (tile >= 2) acc1 += xseg[7] * w0 + xseg[6] * w1 + xseg[5] * w2 + xseg[4] * w3 + xseg[3] * w4 + xseg[2] * w5 + xseg[1] * w6;
                            if (tile >= 3) acc2 += xseg[8] * w0 + xseg[7] * w1 + xseg[6] * w2 + xseg[5] * w3 + xseg[4] * w4 + xseg[3] * w5 + xseg[2] * w6;
                            if (tile >= 4) acc3 += xseg[9] * w0 + xseg[8] * w1 + xseg[7] * w2 + xseg[6] * w3 + xseg[5] * w4 + xseg[4] * w5 + xseg[3] * w6;
                            if (tile >= 5) acc4 += xseg[10] * w0 + xseg[9] * w1 + xseg[8] * w2 + xseg[7] * w3 + xseg[6] * w4 + xseg[5] * w5 + xseg[4] * w6;
                            if (tile >= 6) acc5 += xseg[11] * w0 + xseg[10] * w1 + xseg[9] * w2 + xseg[8] * w3 + xseg[7] * w4 + xseg[6] * w5 + xseg[5] * w6;
                            if (tile >= 7) acc6 += xseg[12] * w0 + xseg[11] * w1 + xseg[10] * w2 + xseg[9] * w3 + xseg[8] * w4 + xseg[7] * w5 + xseg[6] * w6;
                            if (tile >= 8) acc7 += xseg[13] * w0 + xseg[12] * w1 + xseg[11] * w2 + xseg[10] * w3 + xseg[9] * w4 + xseg[8] * w5 + xseg[7] * w6;
                        }
                    }
                } else {
                    // Border: bounds checks, but still load weights once per (ci,kh) and explicitly unroll kw.
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
                            const float w5 = wGm.GetValue((uint64_t)(wBaseKh + 5));
                            const float w6 = wGm.GetValue((uint64_t)(wBaseKh + 6));

                            // kw=0
                            if (tile >= 1) { int iw = (wo0 + 0) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 0; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w0; }

                            // kw=1
                            if (tile >= 1) { int iw = (wo0 + 0) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w1; }

                            // kw=2
                            if (tile >= 1) { int iw = (wo0 + 0) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w2; }

                            // kw=3
                            if (tile >= 1) { int iw = (wo0 + 0) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 3; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w3; }

                            // kw=4
                            if (tile >= 1) { int iw = (wo0 + 0) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 4; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w4; }

                            // kw=5
                            if (tile >= 1) { int iw = (wo0 + 0) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 5; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w5; }

                            // kw=6
                            if (tile >= 1) { int iw = (wo0 + 0) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 2) { int iw = (wo0 + 1) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 3) { int iw = (wo0 + 2) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 4) { int iw = (wo0 + 3) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 5) { int iw = (wo0 + 4) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 6) { int iw = (wo0 + 5) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 7) { int iw = (wo0 + 6) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                            if (tile >= 8) { int iw = (wo0 + 7) - 6; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw)) * w6; }
                        }
                    }
                }

                if (tile >= 1) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 0)), acc0);
                if (tile >= 2) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 1)), acc1);
                if (tile >= 3) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 2)), acc2);
                if (tile >= 4) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 3)), acc3);
                if (tile >= 5) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 4)), acc4);
                if (tile >= 6) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 5)), acc5);
                if (tile >= 7) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 6)), acc6);
                if (tile >= 8) yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo0 + 7)), acc7);

                wo += tile;
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalRows_{0};
    uint32_t wout_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed2d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose2dSqInAsymK3x7S1P0_Fp32_ReuseOw8 op;
    op.Init(x, weight, y, tiling_data.totalRows, tiling_data.wout, tiling_data.blockDim);
    op.Process();
}
