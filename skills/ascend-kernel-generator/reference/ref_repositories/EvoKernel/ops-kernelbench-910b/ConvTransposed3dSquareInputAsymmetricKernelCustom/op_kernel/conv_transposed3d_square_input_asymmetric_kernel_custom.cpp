
#include "kernel_operator.h"

// Specialized ConvTranspose3d (PyTorch layout NCDHW):
// x: [16,32,64,64,64]
// w: [32,64,3,5,5] (Cin,Cout,KD,KH,KW)
// y: [16,64,66,68,68]
//
// This round:
// - Replace 4-lane unrolled width compute with a sliding-window across ow for each (ci,kd,kh).
//   This reduces redundant x loads and, critically, reduces scalar address-gen/control overhead.
// - Safe middle uses no bounds checks at all (ow 4..63, so iw=ow-kw always in-range for kw 0..4).
// - Borders use one masked load per ow (for x(ow)), with register ring-buffer shifting.

class KernelConvTranspose3dRowSlidingW5
{
public:
    __aicore__ inline KernelConvTranspose3dRowSlidingW5() {}

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
        constexpr int N = 16;
        constexpr int CIN = 32;
        constexpr int DIN = 64;
        constexpr int HIN = 64;
        constexpr int WIN = 64;

        constexpr int COUT = 64;
        constexpr int KD = 3;
        constexpr int KH = 5;
        constexpr int KW = 5;

        constexpr int DOUT = 66;
        constexpr int HOUT = 68;
        constexpr int WOUT = 68;

        constexpr int64_t ROWS_TOTAL = (int64_t)N * COUT * DOUT * HOUT;

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

        // For kw 0..4, iw = ow-kw in [0..63] => ow in [4..63] is fully safe.
        constexpr int OW_SAFE_BEG = 4;
        constexpr int OW_SAFE_END = 63;

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

            // Zero y row first (keeps accumulation simpler and branch-free)
            for (int ow = 0; ow < WOUT; ++ow) {
                yGm.SetValue(yRowBase + (int64_t)ow, 0.0f);
            }

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
            if (kdBeg >= kdEnd || khBeg >= khEnd) {
                continue;
            }

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
                        const float w3 = wGm.GetValue(wBaseKh + 3);
                        const float w4 = wGm.GetValue(wBaseKh + 4);

                        // ----- Left border: ow=0..3 (masked new load only) -----
                        {
                            float x_m1 = 0.0f; // x(ow-1)
                            float x_m2 = 0.0f; // x(ow-2)
                            float x_m3 = 0.0f; // x(ow-3)
                            float x_m4 = 0.0f; // x(ow-4)

                            for (int ow = 0; ow < OW_SAFE_BEG; ++ow) {
                                float x0 = xGm.GetValue(xBaseH + (int64_t)ow); // ow in [0..3] always in-range for Win=64

                                float acc = yGm.GetValue(yRowBase + (int64_t)ow);
                                acc += x0 * w0 + x_m1 * w1 + x_m2 * w2 + x_m3 * w3 + x_m4 * w4;
                                yGm.SetValue(yRowBase + (int64_t)ow, acc);

                                x_m4 = x_m3;
                                x_m3 = x_m2;
                                x_m2 = x_m1;
                                x_m1 = x0;
                            }
                        }

                        // ----- Safe middle: ow=4..63 (no bounds checks, fully in-range) -----
                        {
                            // initialize ring buffer for ow=4 using x at positions 0..3 already available
                            float x_m1 = xGm.GetValue(xBaseH + 3); // x(3)
                            float x_m2 = xGm.GetValue(xBaseH + 2); // x(2)
                            float x_m3 = xGm.GetValue(xBaseH + 1); // x(1)
                            float x_m4 = xGm.GetValue(xBaseH + 0); // x(0)

                            for (int ow = OW_SAFE_BEG; ow <= OW_SAFE_END; ++ow) {
                                float x0 = xGm.GetValue(xBaseH + (int64_t)ow); // x(ow)

                                float acc = yGm.GetValue(yRowBase + (int64_t)ow);
                                acc += x0 * w0 + x_m1 * w1 + x_m2 * w2 + x_m3 * w3 + x_m4 * w4;
                                yGm.SetValue(yRowBase + (int64_t)ow, acc);

                                x_m4 = x_m3;
                                x_m3 = x_m2;
                                x_m2 = x_m1;
                                x_m1 = x0;
                            }
                        }

                        // ----- Right border: ow=64..67 (masked new load; history always valid) -----
                        {
                            // seed history at ow=64: x(63..60) exist
                            float x_m1 = xGm.GetValue(xBaseH + 63);
                            float x_m2 = xGm.GetValue(xBaseH + 62);
                            float x_m3 = xGm.GetValue(xBaseH + 61);
                            float x_m4 = xGm.GetValue(xBaseH + 60);

                            for (int ow = OW_SAFE_END + 1; ow < WOUT; ++ow) {
                                float x0 = 0.0f;
                                // Only x(ow) may be out-of-range for ow=64..67, Win=64.
                                if ((uint32_t)ow < (uint32_t)WIN) {
                                    x0 = xGm.GetValue(xBaseH + (int64_t)ow);
                                }

                                float acc = yGm.GetValue(yRowBase + (int64_t)ow);
                                acc += x0 * w0 + x_m1 * w1 + x_m2 * w2 + x_m3 * w3 + x_m4 * w4;
                                yGm.SetValue(yRowBase + (int64_t)ow, acc);

                                x_m4 = x_m3;
                                x_m3 = x_m2;
                                x_m2 = x_m1;
                                x_m1 = x0;
                            }
                        }
                    }
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

extern "C" __global__ __aicore__ void conv_transposed3d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose3dRowSlidingW5 op;
    op.Init(x, weight, y, tiling_data.totalRows, tiling_data.blockDim);
    op.Process();
}
