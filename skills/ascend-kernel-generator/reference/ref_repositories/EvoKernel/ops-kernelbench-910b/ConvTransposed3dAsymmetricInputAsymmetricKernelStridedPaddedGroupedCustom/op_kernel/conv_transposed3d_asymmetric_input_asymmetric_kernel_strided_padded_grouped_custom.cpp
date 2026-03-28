
#include "kernel_operator.h"

// Row-wise ConvTranspose3d gather specialized for fixed shapes/params.
// Optimization in this round: avoid GlobalTensor GetValue/SetValue in hot path by using __gm__ pointers,
// and unroll CIN_G=8 with precomputed per-ci offsets to reduce address scalar ops.

class KernelConvTranspose3dRowWiseParity2PtrFp32 {
public:
    __aicore__ inline KernelConvTranspose3dRowWiseParity2PtrFp32() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalRows, uint32_t blockDim)
    {
        xPtr_ = (__gm__ const float*)x;
        wPtr_ = (__gm__ const float*)w;
        yPtr_ = (__gm__ float*)y;
        totalRows_ = totalRows;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t N = 8;
        constexpr int32_t CIN = 32;
        constexpr int32_t DIN = 12;
        constexpr int32_t HIN = 24;
        constexpr int32_t WIN = 48;

        constexpr int32_t GROUPS = 4;
        constexpr int32_t CIN_G = 8;

        constexpr int32_t COUT = 32;
        constexpr int32_t COUT_G = 8;

        constexpr int32_t KD = 3;
        constexpr int32_t KH = 5;
        constexpr int32_t KW = 7;

        constexpr int32_t SD = 2, SH = 2, SW = 2;
        (void)SD; (void)SH; (void)SW;
        constexpr int32_t PD = 1, PH = 2, PW = 3;

        constexpr int32_t DOUT = 24;
        constexpr int32_t HOUT = 48;
        constexpr int32_t WOUT = 96;

        // NCDHW contiguous strides
        constexpr int64_t X_STRIDE_W = 1;
        constexpr int64_t X_STRIDE_H = (int64_t)WIN;
        constexpr int64_t X_STRIDE_D = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_C = (int64_t)DIN * HIN * WIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * DIN * HIN * WIN;

        // weight layout: [CIN, COUT_G, KD, KH, KW]
        constexpr int64_t W_STRIDE_KW = 1;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;
        constexpr int64_t W_STRIDE_KD = (int64_t)KH * KW;
        constexpr int64_t W_STRIDE_CO = (int64_t)KD * KH * KW;
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT_G * KD * KH * KW;

        // output contiguous layout [N,COUT,DOUT,HOUT,WOUT]
        constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_D = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_C = (int64_t)DOUT * HOUT * WOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * DOUT * HOUT * WOUT;

        constexpr int64_t ROWS_TOTAL = (int64_t)N * COUT * DOUT * HOUT;
        uint32_t totalRows = totalRows_;
        if (totalRows == 0 || totalRows > (uint32_t)ROWS_TOTAL) totalRows = (uint32_t)ROWS_TOTAL;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;

        const uint32_t rowsPerBlock = (totalRows + bdim - 1) / bdim;
        uint32_t rowStart = bid * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > totalRows) rowStart = totalRows;
        if (rowEnd > totalRows) rowEnd = totalRows;

        // Precompute per-ci constant offsets to reduce inner-loop multiplications.
        constexpr int64_t xCiOff0 = 0 * X_STRIDE_C;
        constexpr int64_t xCiOff1 = 1 * X_STRIDE_C;
        constexpr int64_t xCiOff2 = 2 * X_STRIDE_C;
        constexpr int64_t xCiOff3 = 3 * X_STRIDE_C;
        constexpr int64_t xCiOff4 = 4 * X_STRIDE_C;
        constexpr int64_t xCiOff5 = 5 * X_STRIDE_C;
        constexpr int64_t xCiOff6 = 6 * X_STRIDE_C;
        constexpr int64_t xCiOff7 = 7 * X_STRIDE_C;

        constexpr int64_t wCiOff0 = 0 * W_STRIDE_CI;
        constexpr int64_t wCiOff1 = 1 * W_STRIDE_CI;
        constexpr int64_t wCiOff2 = 2 * W_STRIDE_CI;
        constexpr int64_t wCiOff3 = 3 * W_STRIDE_CI;
        constexpr int64_t wCiOff4 = 4 * W_STRIDE_CI;
        constexpr int64_t wCiOff5 = 5 * W_STRIDE_CI;
        constexpr int64_t wCiOff6 = 6 * W_STRIDE_CI;
        constexpr int64_t wCiOff7 = 7 * W_STRIDE_CI;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            uint32_t t = row;
            const int32_t oh = (int32_t)(t % HOUT); t /= HOUT;
            const int32_t od = (int32_t)(t % DOUT); t /= DOUT;
            const int32_t co = (int32_t)(t % COUT); t /= COUT;
            const int32_t n  = (int32_t)t;

            const int32_t g = (co >> 3);
            const int32_t co_g = (co & 7);

            const int64_t xBaseN = (int64_t)n * X_STRIDE_N + (int64_t)(g * CIN_G) * X_STRIDE_C;
            const int64_t wBaseCo = (int64_t)(g * CIN_G) * W_STRIDE_CI + (int64_t)co_g * W_STRIDE_CO;
            const int64_t yBaseRow = (int64_t)n * Y_STRIDE_N + (int64_t)co * Y_STRIDE_C +
                                     (int64_t)od * Y_STRIDE_D + (int64_t)oh * Y_STRIDE_H;

            const int32_t td0 = od + PD;
            const int32_t th0 = oh + PH;

            const int32_t kdStart = (td0 & 1);
            const int32_t khStart = (th0 & 1);

            for (int32_t ow = 0; ow < WOUT; ow += 2) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;

                const int32_t tw00 = ow + PW;
                const int32_t tw01 = (ow + 1) + PW;

                const int32_t kwStart0 = (tw00 & 1);
                const int32_t kwStart1 = (tw01 & 1);

                for (int32_t kd = kdStart; kd < KD; kd += 2) {
                    const int32_t id = (td0 - kd) >> 1;
                    if ((uint32_t)id >= (uint32_t)DIN) continue;

                    const int64_t xBaseD = xBaseN + (int64_t)id * X_STRIDE_D;
                    const int64_t wBaseKd = wBaseCo + (int64_t)kd * W_STRIDE_KD;

                    for (int32_t kh = khStart; kh < KH; kh += 2) {
                        const int32_t ih = (th0 - kh) >> 1;
                        if ((uint32_t)ih >= (uint32_t)HIN) continue;

                        const int64_t xBaseH = xBaseD + (int64_t)ih * X_STRIDE_H;
                        const int64_t wBaseKh = wBaseKd + (int64_t)kh * W_STRIDE_KH;

                        // Lane0 taps
                        for (int32_t kw = kwStart0; kw < KW; kw += 2) {
                            const int32_t iw = (tw00 - kw) >> 1;
                            if ((uint32_t)iw >= (uint32_t)WIN) continue;

                            const int64_t xPos = xBaseH + (int64_t)iw;
                            const int64_t wPos = wBaseKh + (int64_t)kw;

                            // Unrolled CIN_G=8 accumulation with precomputed offsets.
                            const float x0 = xPtr_[xPos + xCiOff0];
                            const float x1 = xPtr_[xPos + xCiOff1];
                            const float x2 = xPtr_[xPos + xCiOff2];
                            const float x3 = xPtr_[xPos + xCiOff3];
                            const float x4 = xPtr_[xPos + xCiOff4];
                            const float x5 = xPtr_[xPos + xCiOff5];
                            const float x6 = xPtr_[xPos + xCiOff6];
                            const float x7 = xPtr_[xPos + xCiOff7];

                            const float w0 = wPtr_[wPos + wCiOff0];
                            const float w1 = wPtr_[wPos + wCiOff1];
                            const float w2 = wPtr_[wPos + wCiOff2];
                            const float w3 = wPtr_[wPos + wCiOff3];
                            const float w4 = wPtr_[wPos + wCiOff4];
                            const float w5 = wPtr_[wPos + wCiOff5];
                            const float w6 = wPtr_[wPos + wCiOff6];
                            const float w7 = wPtr_[wPos + wCiOff7];

                            acc0 += x0 * w0;
                            acc0 += x1 * w1;
                            acc0 += x2 * w2;
                            acc0 += x3 * w3;
                            acc0 += x4 * w4;
                            acc0 += x5 * w5;
                            acc0 += x6 * w6;
                            acc0 += x7 * w7;
                        }

                        // Lane1 taps
                        for (int32_t kw = kwStart1; kw < KW; kw += 2) {
                            const int32_t iw = (tw01 - kw) >> 1;
                            if ((uint32_t)iw >= (uint32_t)WIN) continue;

                            const int64_t xPos = xBaseH + (int64_t)iw;
                            const int64_t wPos = wBaseKh + (int64_t)kw;

                            const float x0 = xPtr_[xPos + xCiOff0];
                            const float x1 = xPtr_[xPos + xCiOff1];
                            const float x2 = xPtr_[xPos + xCiOff2];
                            const float x3 = xPtr_[xPos + xCiOff3];
                            const float x4 = xPtr_[xPos + xCiOff4];
                            const float x5 = xPtr_[xPos + xCiOff5];
                            const float x6 = xPtr_[xPos + xCiOff6];
                            const float x7 = xPtr_[xPos + xCiOff7];

                            const float w0 = wPtr_[wPos + wCiOff0];
                            const float w1 = wPtr_[wPos + wCiOff1];
                            const float w2 = wPtr_[wPos + wCiOff2];
                            const float w3 = wPtr_[wPos + wCiOff3];
                            const float w4 = wPtr_[wPos + wCiOff4];
                            const float w5 = wPtr_[wPos + wCiOff5];
                            const float w6 = wPtr_[wPos + wCiOff6];
                            const float w7 = wPtr_[wPos + wCiOff7];

                            acc1 += x0 * w0;
                            acc1 += x1 * w1;
                            acc1 += x2 * w2;
                            acc1 += x3 * w3;
                            acc1 += x4 * w4;
                            acc1 += x5 * w5;
                            acc1 += x6 * w6;
                            acc1 += x7 * w7;
                        }
                    }
                }

                yPtr_[yBaseRow + (int64_t)ow] = acc0;
                yPtr_[yBaseRow + (int64_t)(ow + 1)] = acc1;
            }
        }
    }

private:
    __gm__ const float* xPtr_{nullptr};
    __gm__ const float* wPtr_{nullptr};
    __gm__ float* yPtr_{nullptr};
    uint32_t totalRows_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose3dRowWiseParity2PtrFp32 op;
    op.Init(x, weight, y, tiling_data.totalRows, tiling_data.blockDim);
    op.Process();
}
