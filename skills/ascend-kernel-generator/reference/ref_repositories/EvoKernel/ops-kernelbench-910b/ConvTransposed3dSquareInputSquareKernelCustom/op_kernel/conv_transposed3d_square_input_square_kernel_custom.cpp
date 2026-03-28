
#include "kernel_operator.h"

// Specialized ConvTranspose3d (PyTorch layout, groups=1):
// x: [8,48,64,64,64] contiguous NCDHW float32
// w: [48,48,3,3,3] contiguous float32 (ConvTranspose3d layout Cin,Cout,KD,KH,KW)
// stride=1, pad=0, outpad=0, dil=1
// y: [8,48,66,66,66]
//
// This round:
// - Tile by full output rows (n,co,od,oh) to remove flattened carry-index overhead.
// - Sliding 3-tap window along ow in the interior [2..63] to cut redundant x loads and scalar address math.
// - Keep only minimal border checks on small ow ranges.
// - Keep weight loads simple (3 taps per kd/kh) to reduce register pressure and instruction overhead.

class KernelConvTranspose3dSquareRowSlidingOw
{
public:
    __aicore__ inline KernelConvTranspose3dSquareRowSlidingOw() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalRows, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalRows_ = totalRows;
        blockDim_ = (blockDim == 0) ? 1u : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 8;
        constexpr int CIN = 48;
        constexpr int DIN = 64;
        constexpr int HIN = 64;
        constexpr int WIN = 64;

        constexpr int COUT = 48;
        constexpr int KD = 3;
        constexpr int KH = 3;
        constexpr int KW = 3;

        constexpr int DOUT = 66;
        constexpr int HOUT = 66;
        constexpr int WOUT = 66;

        // Strides (NCDHW contiguous, elements)
        constexpr int64_t X_STRIDE_NC = (int64_t)DIN * HIN * WIN;
        constexpr int64_t X_STRIDE_D  = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_H  = (int64_t)WIN;

        constexpr int64_t Y_STRIDE_CO = (int64_t)DOUT * HOUT * WOUT;
        constexpr int64_t Y_STRIDE_D  = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_H  = (int64_t)WOUT;

        // Weight [ci,co,kd,kh,kw] contiguous
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KD * KH * KW;
        constexpr int64_t W_STRIDE_CO = (int64_t)KD * KH * KW;
        constexpr int64_t W_STRIDE_KD = (int64_t)KH * KW;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;

        // Interior safe ow range for KW=3: iw=ow-kw in [0..63] for kw=0..2 => ow in [2..63]
        constexpr int OW_SAFE_BEG = KW - 1;   // 2
        constexpr int OW_SAFE_END = WIN - 1; // 63

        constexpr int64_t ROWS_TOTAL = (int64_t)N * COUT * DOUT * HOUT;
        uint32_t totalRows = totalRows_;
        if (totalRows == 0 || (int64_t)totalRows > ROWS_TOTAL) totalRows = (uint32_t)ROWS_TOTAL;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;
        const uint32_t rowsPerBlock = (totalRows + bdim - 1) / bdim;

        uint32_t rowStart = bid * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > totalRows) rowStart = totalRows;
        if (rowEnd > totalRows) rowEnd = totalRows;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            uint32_t t = row;
            const int oh = (int)(t % HOUT); t /= HOUT;
            const int od = (int)(t % DOUT); t /= DOUT;
            const int co = (int)(t % COUT); t /= COUT;
            const int n  = (int)t;

            const int64_t yRowBase = (((int64_t)n * COUT + co) * Y_STRIDE_CO) +
                                     ((int64_t)od * Y_STRIDE_D) +
                                     ((int64_t)oh * Y_STRIDE_H);

            // kd range: id=od-kd in [0..DIN-1]
            int kdBeg, kdEnd;
            {
                int kb = od - (DIN - 1);
                int ke = od;
                if (kb < 0) kb = 0;
                if (ke > KD - 1) ke = KD - 1;
                kdBeg = kb;
                kdEnd = ke + 1;
            }
            // kh range: ih=oh-kh in [0..HIN-1]
            int khBeg, khEnd;
            {
                int hb = oh - (HIN - 1);
                int he = oh;
                if (hb < 0) hb = 0;
                if (he > KH - 1) he = KH - 1;
                khBeg = hb;
                khEnd = he + 1;
            }

            // Accumulate full row in registers/local memory.
            float yAcc[WOUT];
#pragma unroll
            for (int i = 0; i < WOUT; ++i) yAcc[i] = 0.0f;

            if (kdBeg < kdEnd && khBeg < khEnd) {
                const int64_t xBaseN = (int64_t)n * CIN * X_STRIDE_NC;
                const int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

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

                            const float w0 = wGm.GetValue(wBaseKh + 0);
                            const float w1 = wGm.GetValue(wBaseKh + 1);
                            const float w2 = wGm.GetValue(wBaseKh + 2);

                            // Left border ow=0..1 (few points, checks)
                            for (int ow = 0; ow < OW_SAFE_BEG; ++ow) {
                                int iw0 = ow - 0;
                                int iw1 = ow - 1;
                                int iw2 = ow - 2;
                                float sum = 0.0f;
                                if ((uint32_t)iw0 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw0) * w0;
                                if ((uint32_t)iw1 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw1) * w1;
                                if ((uint32_t)iw2 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw2) * w2;
                                yAcc[ow] += sum;
                            }

                            // Interior ow=2..63: sliding window, branch-free.
                            // At ow=2, need x[2],x[1],x[0] for (kw=0,1,2).
                            float xm2 = xGm.GetValue(xBaseH + 0);
                            float xm1 = xGm.GetValue(xBaseH + 1);
                            float x0  = xGm.GetValue(xBaseH + 2);

                            int ow = OW_SAFE_BEG;
                            for (; ow + 7 <= OW_SAFE_END; ow += 8) {
                                yAcc[ow + 0] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 1));
                                yAcc[ow + 1] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 2));
                                yAcc[ow + 2] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 3));
                                yAcc[ow + 3] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 4));
                                yAcc[ow + 4] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 5));
                                yAcc[ow + 5] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 6));
                                yAcc[ow + 6] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 7));
                                yAcc[ow + 7] += x0 * w0 + xm1 * w1 + xm2 * w2;

                                // advance to next chunk start (ow+8)
                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 8));
                            }
                            for (; ow <= OW_SAFE_END; ++ow) {
                                yAcc[ow] += x0 * w0 + xm1 * w1 + xm2 * w2;
                                xm2 = xm1; xm1 = x0; x0 = xGm.GetValue(xBaseH + (int64_t)(ow + 1));
                            }

                            // Right border ow=64..65 (few points, checks)
                            for (int owb = OW_SAFE_END + 1; owb < WOUT; ++owb) {
                                int iw0 = owb - 0;
                                int iw1 = owb - 1;
                                int iw2 = owb - 2;
                                float sum = 0.0f;
                                if ((uint32_t)iw0 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw0) * w0;
                                if ((uint32_t)iw1 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw1) * w1;
                                if ((uint32_t)iw2 < (uint32_t)WIN) sum += xGm.GetValue(xBaseH + (int64_t)iw2) * w2;
                                yAcc[owb] += sum;
                            }
                        }
                    }
                }
            }

            // Store the row (unroll by 8 for lower loop overhead)
            int ow = 0;
            for (; ow + 7 < WOUT; ow += 8) {
                yGm.SetValue(yRowBase + (int64_t)(ow + 0), yAcc[ow + 0]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 1), yAcc[ow + 1]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 2), yAcc[ow + 2]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 3), yAcc[ow + 3]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 4), yAcc[ow + 4]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 5), yAcc[ow + 5]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 6), yAcc[ow + 6]);
                yGm.SetValue(yRowBase + (int64_t)(ow + 7), yAcc[ow + 7]);
            }
            for (; ow < WOUT; ++ow) {
                yGm.SetValue(yRowBase + (int64_t)ow, yAcc[ow]);
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

extern "C" __global__ __aicore__ void conv_transposed3d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose3dSquareRowSlidingOw op;
    op.Init(x, weight, y, tiling_data.totalRows, tiling_data.blockDim);
    op.Process();
}
