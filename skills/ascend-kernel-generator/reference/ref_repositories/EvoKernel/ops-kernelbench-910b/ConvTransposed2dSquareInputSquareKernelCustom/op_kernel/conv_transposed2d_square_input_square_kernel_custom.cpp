
#include "kernel_operator.h"

// Specialized ConvTranspose2d (stride=1,pad=0,dil=1,groups=1,bias=0) for:
// x: [8,64,1024,1024], w: [64,64,3,3] (ci,co,kh,kw), y: [8,64,1026,1026]
//
// This round: keep row-grouped mapping (min div/mod) but tighten the hot loop.
// For interior ow tiles (dominant), switch to an input-centric micro-kernel:
// load 10 contiguous x values once and reuse across 3 weight taps, reducing
// weight GM reads and scalar address ops versus per-output weight usage.

class KernelConvTranspose2dRowGroupedOw8InputCentric {
public:
    __aicore__ inline KernelConvTranspose2dRowGroupedOw8InputCentric() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        blockDim_ = blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 8;
        constexpr int CIN = 64;
        constexpr int HIN = 1024;
        constexpr int WIN = 1024;

        constexpr int COUT = 64;
        constexpr int KH = 3;
        constexpr int KW = 3;

        constexpr int HOUT = 1026;
        constexpr int WOUT = 1026;

        constexpr int64_t ROWS = (int64_t)N * COUT * HOUT; // 8*64*1026

        // Strides (NCHW)
        constexpr int64_t X_STRIDE_NC = (int64_t)HIN * WIN; // per (n,ci)
        constexpr int64_t X_STRIDE_H  = (int64_t)WIN;

        constexpr int64_t Y_STRIDE_NC = (int64_t)COUT * (int64_t)HOUT * (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_CO = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_H  = (int64_t)WOUT;

        // Weight: [ci,co,kh,kw]
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KH * KW; // 576
        constexpr int64_t W_STRIDE_CO = (int64_t)KH * KW;        // 9
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;             // 3

        // Safe ow range for KW=3 and tile=8: baseIw=ow-2, need baseIw..baseIw+9 within [0..1023]
        // => ow in [2..1016]
        constexpr int OW_SAFE_BEG = 2;
        constexpr int OW_SAFE_END = 1016;

        constexpr int TILE_OW = 8;     // compute 8 ow per step
        constexpr int SEG = TILE_OW + (KW - 1); // 10 contiguous x values

        const int64_t bid = (int64_t)AscendC::GetBlockIdx();
        int64_t bdim = (int64_t)blockDim_;
        if (bdim <= 0) bdim = 1;

        const int64_t rowsPerBlock = (ROWS + bdim - 1) / bdim;
        int64_t rowStart = bid * rowsPerBlock;
        int64_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > ROWS) rowStart = ROWS;
        if (rowEnd > ROWS) rowEnd = ROWS;
        if (rowStart >= rowEnd) return;

        constexpr int64_t ROWS_PER_NC = (int64_t)HOUT;

        int64_t cur = rowStart;

        // Initial decode only once per block
        int64_t nc = cur / ROWS_PER_NC;  // (n*COUT + co)
        int oh = (int)(cur - nc * ROWS_PER_NC);

        int n = (int)(nc / COUT);
        int co = (int)(nc - (int64_t)n * COUT);

        // Bases for current (n,co)
        int64_t xBaseN = (int64_t)n * CIN * X_STRIDE_NC;
        int64_t yBaseNc = (int64_t)n * Y_STRIDE_NC + (int64_t)co * Y_STRIDE_CO;
        int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

        while (cur < rowEnd) {
            int64_t groupEnd = (nc + 1) * ROWS_PER_NC;
            if (groupEnd > rowEnd) groupEnd = rowEnd;

            for (; cur < groupEnd; ++cur, ++oh) {
                const int64_t yRowBase = yBaseNc + (int64_t)oh * Y_STRIDE_H;

                // kh range from ih = oh - kh in [0, HIN-1]
                int khBeg, khEnd;
                {
                    int hb = oh - (HIN - 1);
                    int he = oh;
                    if (hb < 0) hb = 0;
                    if (he > KH - 1) he = KH - 1;
                    khBeg = hb;
                    khEnd = he + 1;
                }

                if (khBeg >= khEnd) {
#pragma unroll 2
                    for (int ow0 = 0; ow0 < WOUT; ++ow0) {
                        yGm.SetValue(yRowBase + (int64_t)ow0, 0.0f);
                    }
                    continue;
                }

                int ow = 0;
                while (ow < WOUT) {
                    const int tile = (ow + TILE_OW <= WOUT) ? TILE_OW : (WOUT - ow);
                    const int ow0 = ow;

                    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

                    const bool tileSafe = (tile == TILE_OW) && (ow0 >= OW_SAFE_BEG) && (ow0 <= OW_SAFE_END);

                    if (tileSafe) {
                        const int baseIw = ow0 - (KW - 1); // ow-2, seg is [baseIw..baseIw+9]
                        for (int ci = 0; ci < CIN; ++ci) {
                            const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_NC;
                            const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                            for (int kh = khBeg; kh < khEnd; ++kh) {
                                const int ih = oh - kh;
                                const int64_t xBaseH = xBaseCi + (int64_t)ih * X_STRIDE_H;
                                const int64_t wBaseKh = wBaseCiCo + (int64_t)kh * W_STRIDE_KH;

                                const float w0 = wGm.GetValue(wBaseKh + 0);
                                const float w1 = wGm.GetValue(wBaseKh + 1);
                                const float w2 = wGm.GetValue(wBaseKh + 2);

                                // Load 10 contiguous x once
                                const float x0 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 0));
                                const float x1 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 1));
                                const float x2 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 2));
                                const float x3 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 3));
                                const float x4 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 4));
                                const float x5 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 5));
                                const float x6 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 6));
                                const float x7 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 7));
                                const float x8 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 8));
                                const float x9 = xGm.GetValue(xBaseH + (int64_t)(baseIw + 9));

                                // Input-centric accumulation: each x contributes to up to 3 outputs with (w2,w1,w0).
                                // o0 uses x0,x1,x2 ; o7 uses x7,x8,x9
                                acc0 += x2 * w0; acc0 += x1 * w1; acc0 += x0 * w2;
                                acc1 += x3 * w0; acc1 += x2 * w1; acc1 += x1 * w2;
                                acc2 += x4 * w0; acc2 += x3 * w1; acc2 += x2 * w2;
                                acc3 += x5 * w0; acc3 += x4 * w1; acc3 += x3 * w2;
                                acc4 += x6 * w0; acc4 += x5 * w1; acc4 += x4 * w2;
                                acc5 += x7 * w0; acc5 += x6 * w1; acc5 += x5 * w2;
                                acc6 += x8 * w0; acc6 += x7 * w1; acc6 += x6 * w2;
                                acc7 += x9 * w0; acc7 += x8 * w1; acc7 += x7 * w2;
                            }
                        }
                    } else {
                        // Generic path (borders and tail tile)
                        for (int ci = 0; ci < CIN; ++ci) {
                            const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_NC;
                            const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                            for (int kh = khBeg; kh < khEnd; ++kh) {
                                const int ih = oh - kh;
                                const int64_t xBaseH = xBaseCi + (int64_t)ih * X_STRIDE_H;
                                const int64_t wBaseKh = wBaseCiCo + (int64_t)kh * W_STRIDE_KH;

                                const float wf0 = wGm.GetValue(wBaseKh + 0);
                                const float wf1 = wGm.GetValue(wBaseKh + 1);
                                const float wf2 = wGm.GetValue(wBaseKh + 2);

                                // kw=0 (iw = ow)
                                if (tile >= 1) { int iw = (ow0 + 0); if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 2) { int iw = (ow0 + 1); if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 3) { int iw = (ow0 + 2); if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 4) { int iw = (ow0 + 3); if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 5) { int iw = (ow0 + 4); if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 6) { int iw = (ow0 + 5); if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 7) { int iw = (ow0 + 6); if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }
                                if (tile >= 8) { int iw = (ow0 + 7); if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue(xBaseH + (int64_t)iw) * wf0; }

                                // kw=1 (iw = ow-1)
                                if (tile >= 1) { int iw = (ow0 + 0) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 2) { int iw = (ow0 + 1) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 3) { int iw = (ow0 + 2) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 4) { int iw = (ow0 + 3) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 5) { int iw = (ow0 + 4) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 6) { int iw = (ow0 + 5) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 7) { int iw = (ow0 + 6) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }
                                if (tile >= 8) { int iw = (ow0 + 7) - 1; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue(xBaseH + (int64_t)iw) * wf1; }

                                // kw=2 (iw = ow-2)
                                if (tile >= 1) { int iw = (ow0 + 0) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc0 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 2) { int iw = (ow0 + 1) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc1 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 3) { int iw = (ow0 + 2) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc2 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 4) { int iw = (ow0 + 3) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc3 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 5) { int iw = (ow0 + 4) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc4 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 6) { int iw = (ow0 + 5) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc5 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 7) { int iw = (ow0 + 6) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc6 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                                if (tile >= 8) { int iw = (ow0 + 7) - 2; if ((uint32_t)iw < (uint32_t)WIN) acc7 += xGm.GetValue(xBaseH + (int64_t)iw) * wf2; }
                            }
                        }
                    }

                    if (tile >= 1) yGm.SetValue(yRowBase + (int64_t)(ow0 + 0), acc0);
                    if (tile >= 2) yGm.SetValue(yRowBase + (int64_t)(ow0 + 1), acc1);
                    if (tile >= 3) yGm.SetValue(yRowBase + (int64_t)(ow0 + 2), acc2);
                    if (tile >= 4) yGm.SetValue(yRowBase + (int64_t)(ow0 + 3), acc3);
                    if (tile >= 5) yGm.SetValue(yRowBase + (int64_t)(ow0 + 4), acc4);
                    if (tile >= 6) yGm.SetValue(yRowBase + (int64_t)(ow0 + 5), acc5);
                    if (tile >= 7) yGm.SetValue(yRowBase + (int64_t)(ow0 + 6), acc6);
                    if (tile >= 8) yGm.SetValue(yRowBase + (int64_t)(ow0 + 7), acc7);

                    ow += tile;
                }
            }

            // Advance to next (n,co) group
            nc += 1;
            oh = 0;
            if (nc >= (int64_t)N * COUT) break;

            n = (int)(nc / COUT);
            co = (int)(nc - (int64_t)n * COUT);

            xBaseN = (int64_t)n * CIN * X_STRIDE_NC;
            yBaseNc = (int64_t)n * Y_STRIDE_NC + (int64_t)co * Y_STRIDE_CO;
            wBaseCo = (int64_t)co * W_STRIDE_CO;
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed2d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dRowGroupedOw8InputCentric op;
    op.Init(x, weight, y, tiling_data.blockDim);
    op.Process();
}
