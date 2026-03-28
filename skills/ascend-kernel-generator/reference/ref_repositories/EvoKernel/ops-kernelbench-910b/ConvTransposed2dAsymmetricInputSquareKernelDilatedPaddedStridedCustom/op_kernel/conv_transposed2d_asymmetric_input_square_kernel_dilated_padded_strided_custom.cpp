
#include "kernel_operator.h"

// Optimized scalar implementation specialized for:
// N=16,CIN=32,COUT=64,HIN=64,WIN=128,KH=KW=3, stride=5, pad=1, dil=2, out_pad=0
// HOUT=318,WOUT=638
//
// Key optimizations vs baseline:
// - Block mapping over "rows" (n,co,ho) and compute full W line per row.
// - One-time div/mod unflatten for rowStart, then increment-with-carry for (ho,co,n).
// - Width loop uses stride-5 residue pattern with straight-line evaluation for active residues r=1,3,4.
// - Hoist kh validity per row; hoist weight base pointers for (co,kh,kw) to reduce inner address arithmetic.

class KernelConvT2dRowWiseIncCarryResidue5Fp32 {
public:
    __aicore__ inline KernelConvT2dRowWiseIncCarryResidue5Fp32() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalRows, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalRows_ = totalRows;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t N = 16;
        constexpr int32_t CIN = 32;
        constexpr int32_t COUT = 64;
        constexpr int32_t HIN = 64;
        constexpr int32_t WIN = 128;

        constexpr int32_t KH = 3;
        constexpr int32_t KW = 3;

        constexpr int32_t SH = 5;
        constexpr int32_t SW = 5;
        constexpr int32_t PH = 1;
        constexpr int32_t PW = 1;
        constexpr int32_t DH = 2;
        constexpr int32_t DW = 2;

        constexpr int32_t HOUT = 318;
        constexpr int32_t WOUT = 638;

        // weight: [CIN, COUT, KH, KW]
        constexpr int64_t W_STRIDE_KW = 1;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;                  // 3
        constexpr int64_t W_STRIDE_CO = (int64_t)KH * KW;             // 9
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KH * KW;      // 576

        // x: [N,CIN,HIN,WIN]
        constexpr int64_t X_STRIDE_W = 1;
        constexpr int64_t X_STRIDE_H = (int64_t)WIN;
        constexpr int64_t X_STRIDE_C = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * HIN * WIN;

        // y: [N,COUT,HOUT,WOUT]
        constexpr int64_t Y_STRIDE_W = 1;
        constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_C = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * HOUT * WOUT;

        constexpr uint32_t ROWS_TOTAL = (uint32_t)(N * COUT * HOUT);
        uint32_t totalRows = totalRows_;
        if (totalRows == 0 || totalRows > ROWS_TOTAL) totalRows = ROWS_TOTAL;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;
        const uint32_t rowsPerBlock = (totalRows + bdim - 1) / bdim;

        const uint32_t rowStart = bid * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > totalRows) rowEnd = totalRows;
        if (rowStart >= rowEnd) return;

        // One-time unflatten for rowStart, then increment-with-carry.
        uint32_t t = rowStart;
        int32_t ho = (int32_t)(t % (uint32_t)HOUT); t /= (uint32_t)HOUT;
        int32_t co = (int32_t)(t % (uint32_t)COUT); t /= (uint32_t)COUT;
        int32_t n  = (int32_t)t;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            const int64_t yRowBase = (int64_t)n * Y_STRIDE_N
                                   + (int64_t)co * Y_STRIDE_C
                                   + (int64_t)ho * Y_STRIDE_H;

            // Precompute valid (kh -> ih) for this ho (0..3 valid, typically 0..1).
            int32_t ih0 = 0, ih1 = 0, ih2 = 0;
            uint8_t vkh0 = 0, vkh1 = 0, vkh2 = 0;
            const int32_t hoPlus = ho + PH;

            // kh=0
            {
                const int32_t num = hoPlus - 0 * DH;
                if (num >= 0 && (num % SH) == 0) {
                    const int32_t v = num / SH;
                    if ((uint32_t)v < (uint32_t)HIN) { ih0 = v; vkh0 = 1; }
                }
            }
            // kh=1
            {
                const int32_t num = hoPlus - 1 * DH;
                if (num >= 0 && (num % SH) == 0) {
                    const int32_t v = num / SH;
                    if ((uint32_t)v < (uint32_t)HIN) { ih1 = v; vkh1 = 1; }
                }
            }
            // kh=2
            {
                const int32_t num = hoPlus - 2 * DH;
                if (num >= 0 && (num % SH) == 0) {
                    const int32_t v = num / SH;
                    if ((uint32_t)v < (uint32_t)HIN) { ih2 = v; vkh2 = 1; }
                }
            }

            const int64_t wCoBase = (int64_t)co * W_STRIDE_CO;

            // Weight base offsets (within a ci slice) for the only kw that can contribute for each residue:
            // r=1 -> kw=1, r=3 -> kw=2, r=4 -> kw=0
            const int64_t w_kw1_kh0 = wCoBase + 0 * W_STRIDE_KH + 1 * W_STRIDE_KW;
            const int64_t w_kw1_kh1 = wCoBase + 1 * W_STRIDE_KH + 1 * W_STRIDE_KW;
            const int64_t w_kw1_kh2 = wCoBase + 2 * W_STRIDE_KH + 1 * W_STRIDE_KW;

            const int64_t w_kw2_kh0 = wCoBase + 0 * W_STRIDE_KH + 2 * W_STRIDE_KW;
            const int64_t w_kw2_kh1 = wCoBase + 1 * W_STRIDE_KH + 2 * W_STRIDE_KW;
            const int64_t w_kw2_kh2 = wCoBase + 2 * W_STRIDE_KH + 2 * W_STRIDE_KW;

            const int64_t w_kw0_kh0 = wCoBase + 0 * W_STRIDE_KH + 0 * W_STRIDE_KW;
            const int64_t w_kw0_kh1 = wCoBase + 1 * W_STRIDE_KH + 0 * W_STRIDE_KW;
            const int64_t w_kw0_kh2 = wCoBase + 2 * W_STRIDE_KH + 0 * W_STRIDE_KW;

            // Walk wo with increment-with-carry over residue cycles of 5.
            // We only compute residues r=1,3,4; r=0,2 are always zero.
            for (int32_t wo0 = 0; wo0 < WOUT; wo0 += 5) {
                // r=0 -> always 0, skip store later
                // r=1 -> wo = wo0+1, kw=1, iw = wo/5
                // r=3 -> wo = wo0+3, kw=2, iw = (wo-3)/5
                // r=4 -> wo = wo0+4, kw=0, iw = (wo+1)/5

                const int32_t wo1 = wo0 + 1;
                const int32_t wo3 = wo0 + 3;
                const int32_t wo4 = wo0 + 4;

                float acc1 = 0.0f;
                float acc3 = 0.0f;
                float acc4 = 0.0f;

                uint8_t v1 = (wo1 < WOUT);
                uint8_t v3 = (wo3 < WOUT);
                uint8_t v4 = (wo4 < WOUT);

                int32_t iw1 = 0, iw3 = 0, iw4 = 0;
                if (v1) { iw1 = wo1 / SW; v1 = ((uint32_t)iw1 < (uint32_t)WIN); }
                if (v3) { iw3 = (wo3 - 3) / SW; v3 = (iw3 >= 0) && ((uint32_t)iw3 < (uint32_t)WIN); }
                if (v4) { iw4 = (wo4 + 1) / SW; v4 = ((uint32_t)iw4 < (uint32_t)WIN); }

                // Base pointers for x for each lane/kh (in elements), excluding ci stride.
                int64_t x1_kh0 = 0, x1_kh1 = 0, x1_kh2 = 0;
                int64_t x3_kh0 = 0, x3_kh1 = 0, x3_kh2 = 0;
                int64_t x4_kh0 = 0, x4_kh1 = 0, x4_kh2 = 0;

                const int64_t xNBase = (int64_t)n * X_STRIDE_N;

                if (v1) {
                    const int64_t baseW = xNBase + (int64_t)iw1;
                    if (vkh0) x1_kh0 = baseW + (int64_t)ih0 * X_STRIDE_H;
                    if (vkh1) x1_kh1 = baseW + (int64_t)ih1 * X_STRIDE_H;
                    if (vkh2) x1_kh2 = baseW + (int64_t)ih2 * X_STRIDE_H;
                }
                if (v3) {
                    const int64_t baseW = xNBase + (int64_t)iw3;
                    if (vkh0) x3_kh0 = baseW + (int64_t)ih0 * X_STRIDE_H;
                    if (vkh1) x3_kh1 = baseW + (int64_t)ih1 * X_STRIDE_H;
                    if (vkh2) x3_kh2 = baseW + (int64_t)ih2 * X_STRIDE_H;
                }
                if (v4) {
                    const int64_t baseW = xNBase + (int64_t)iw4;
                    if (vkh0) x4_kh0 = baseW + (int64_t)ih0 * X_STRIDE_H;
                    if (vkh1) x4_kh1 = baseW + (int64_t)ih1 * X_STRIDE_H;
                    if (vkh2) x4_kh2 = baseW + (int64_t)ih2 * X_STRIDE_H;
                }

#pragma unroll
                for (int32_t ci = 0; ci < CIN; ++ci) {
                    const int64_t xOff = (int64_t)ci * X_STRIDE_C;
                    const int64_t wOff = (int64_t)ci * W_STRIDE_CI;

                    if (v1) {
                        if (vkh0) acc1 += xGm.GetValue(x1_kh0 + xOff) * wGm.GetValue(wOff + w_kw1_kh0);
                        if (vkh1) acc1 += xGm.GetValue(x1_kh1 + xOff) * wGm.GetValue(wOff + w_kw1_kh1);
                        if (vkh2) acc1 += xGm.GetValue(x1_kh2 + xOff) * wGm.GetValue(wOff + w_kw1_kh2);
                    }
                    if (v3) {
                        if (vkh0) acc3 += xGm.GetValue(x3_kh0 + xOff) * wGm.GetValue(wOff + w_kw2_kh0);
                        if (vkh1) acc3 += xGm.GetValue(x3_kh1 + xOff) * wGm.GetValue(wOff + w_kw2_kh1);
                        if (vkh2) acc3 += xGm.GetValue(x3_kh2 + xOff) * wGm.GetValue(wOff + w_kw2_kh2);
                    }
                    if (v4) {
                        if (vkh0) acc4 += xGm.GetValue(x4_kh0 + xOff) * wGm.GetValue(wOff + w_kw0_kh0);
                        if (vkh1) acc4 += xGm.GetValue(x4_kh1 + xOff) * wGm.GetValue(wOff + w_kw0_kh1);
                        if (vkh2) acc4 += xGm.GetValue(x4_kh2 + xOff) * wGm.GetValue(wOff + w_kw0_kh2);
                    }
                }

                const int64_t yBase = yRowBase + (int64_t)wo0;
                // r=0
                if (wo0 < WOUT) yGm.SetValue(yBase + 0, 0.0f);
                // r=1
                if (wo1 < WOUT) yGm.SetValue(yBase + 1, acc1);
                // r=2
                if ((wo0 + 2) < WOUT) yGm.SetValue(yBase + 2, 0.0f);
                // r=3
                if (wo3 < WOUT) yGm.SetValue(yBase + 3, acc3);
                // r=4
                if (wo4 < WOUT) yGm.SetValue(yBase + 4, acc4);
            }

            // increment-with-carry for (ho,co,n)
            ++ho;
            if (ho == HOUT) {
                ho = 0;
                ++co;
                if (co == COUT) {
                    co = 0;
                    ++n;
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalRows_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    AscendC::InitSocState();
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvT2dRowWiseIncCarryResidue5Fp32 op;
    op.Init(x, weight, y, tiling_data.totalRows, tiling_data.blockDim);
    op.Process();
}
