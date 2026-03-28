
#include "kernel_operator.h"

// Specialized ConvTranspose2d (PyTorch layout):
// x: [N,Cin,Hin,Win] = [8,32,512,1024]
// w: [Cin,Cout,Kh,Kw] = [32,32,3,3]
// stride=(1,1), pad=(0,0), outpad=(0,0), dil=(1,1), groups=1, bias=False
// y: [8,32,514,1026]
//
// This round:
// - Pass blockDim via tiling to avoid GetBlockNum().
// - Split width loop into left border, long interior, right border.
// - Interior uses TILE_WO=8, with contiguous x-segment reuse but implemented as explicit scalar loads
//   (avoid on-stack local arrays that can trigger toolchain/stack issues).
// - Precompute kh-dependent weight base offsets and hoist base pointers to reduce scalar integer ops.

class KernelConvTranspose2dAsymInSquareK3S1P0_Fp32_Wo8Interior_NoLocalArray {
public:
    __aicore__ inline KernelConvTranspose2dAsymInSquareK3S1P0_Fp32_Wo8Interior_NoLocalArray() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t rows, uint32_t wout, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        rows_ = rows;
        wout_ = wout;
        blockDim_ = blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 8;
        constexpr int CIN = 32;
        constexpr int COUT = 32;
        constexpr int HIN = 512;
        constexpr int WIN = 1024;
        constexpr int KH = 3;
        constexpr int KW = 3;
        constexpr int HOUT = 514;
        constexpr int WOUT = 1026;

        if ((int)rows_ != N * COUT * HOUT || (int)wout_ != WOUT || blockDim_ == 0) return;

        const int64_t coreNum = (int64_t)blockDim_;
        const int64_t coreIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t totalRows = (int64_t)rows_;
        const int64_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
        int64_t rowStart = coreIdx * rowsPerCore;
        int64_t rowEnd = rowStart + rowsPerCore;
        if (rowStart > totalRows) rowStart = totalRows;
        if (rowEnd > totalRows) rowEnd = totalRows;

        // Strides (NCHW contiguous)
        constexpr int64_t X_STRIDE_H = (int64_t)WIN;
        constexpr int64_t X_STRIDE_C = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * X_STRIDE_C;

        constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;
        constexpr int64_t Y_STRIDE_C = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * Y_STRIDE_C;

        // Weight layout: [ci,co,kh,kw]
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;                 // 3
        constexpr int64_t W_STRIDE_CO = (int64_t)KH * KW;            // 9
        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * W_STRIDE_CO; // 288

        constexpr int TILE_WO = 8;
        constexpr int WO_INTERIOR_BEG = KW - 1;     // 2
        constexpr int WO_INTERIOR_END_EXCL = WIN;   // 1024 (wo in [2..1023] valid interior)
        // right border: [1024..1025], left border: [0..1]

        for (int64_t row = rowStart; row < rowEnd; ++row) {
            int64_t t = row;
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
                // Only ho=513 yields empty row; keep simple scalar stores.
                for (int wo = 0; wo < WOUT; ++wo) {
                    yGm.SetValue((uint64_t)(yRowBase + (int64_t)wo), 0.0f);
                }
                continue;
            }

            const int64_t xBaseN = (int64_t)n * X_STRIDE_N;
            const int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

            // Precompute small kh weight base offsets for this (ci,co): wBaseCiCo + kh*3
            // (khBeg/khEnd vary per ho, but KH=3 so this remains tiny and in registers.)

            // Phase 1: left border wo=0..1 (bounds checks, tiny)
            for (int wo = 0; wo < WO_INTERIOR_BEG; ++wo) {
                float acc = 0.0f;
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

                        int iw0 = wo;     if ((uint32_t)iw0 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw0)) * w0;
                        int iw1 = wo - 1; if ((uint32_t)iw1 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw1)) * w1;
                        int iw2 = wo - 2; if ((uint32_t)iw2 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw2)) * w2;
                    }
                }
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)wo), acc);
            }

            // Phase 2: interior wo in [2..1023], TILE=8, no iw bounds checks.
            int wo = WO_INTERIOR_BEG;
            for (; wo + TILE_WO <= WO_INTERIOR_END_EXCL; wo += TILE_WO) {
                float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
                float b0=0.f,b1=0.f,b2=0.f,b3=0.f;

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

                        // Load x segment covering [wo-2 .. wo+7] = 10 floats into registers (no local array).
                        const int64_t xSegBase = xBaseH + (int64_t)(wo - 2);
                        const float x0 = xGm.GetValue((uint64_t)(xSegBase + 0));
                        const float x1 = xGm.GetValue((uint64_t)(xSegBase + 1));
                        const float x2 = xGm.GetValue((uint64_t)(xSegBase + 2));
                        const float x3 = xGm.GetValue((uint64_t)(xSegBase + 3));
                        const float x4 = xGm.GetValue((uint64_t)(xSegBase + 4));
                        const float x5 = xGm.GetValue((uint64_t)(xSegBase + 5));
                        const float x6 = xGm.GetValue((uint64_t)(xSegBase + 6));
                        const float x7 = xGm.GetValue((uint64_t)(xSegBase + 7));
                        const float x8 = xGm.GetValue((uint64_t)(xSegBase + 8));
                        const float x9 = xGm.GetValue((uint64_t)(xSegBase + 9));

                        // lane l uses xseg[l+2], xseg[l+1], xseg[l]
                        a0 += x2 * w0 + x1 * w1 + x0 * w2;
                        a1 += x3 * w0 + x2 * w1 + x1 * w2;
                        a2 += x4 * w0 + x3 * w1 + x2 * w2;
                        a3 += x5 * w0 + x4 * w1 + x3 * w2;

                        b0 += x6 * w0 + x5 * w1 + x4 * w2;
                        b1 += x7 * w0 + x6 * w1 + x5 * w2;
                        b2 += x8 * w0 + x7 * w1 + x6 * w2;
                        b3 += x9 * w0 + x8 * w1 + x7 * w2;
                    }
                }

                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 0)), a0);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 1)), a1);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 2)), a2);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 3)), a3);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 4)), b0);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 5)), b1);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 6)), b2);
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)(wo + 7)), b3);
            }

            // Interior tail (still safe, but small). Keep scalar no-check path.
            for (; wo < WO_INTERIOR_END_EXCL; ++wo) {
                float acc = 0.0f;
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

                        const int64_t xb = xBaseH + (int64_t)wo;
                        acc += xGm.GetValue((uint64_t)(xb - 0)) * w0;
                        acc += xGm.GetValue((uint64_t)(xb - 1)) * w1;
                        acc += xGm.GetValue((uint64_t)(xb - 2)) * w2;
                    }
                }
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)wo), acc);
            }

            // Phase 3: right border wo=1024..1025 (bounds checks, tiny)
            for (int wobr = WO_INTERIOR_END_EXCL; wobr < WOUT; ++wobr) {
                float acc = 0.0f;
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

                        int iw0 = wobr;     if ((uint32_t)iw0 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw0)) * w0;
                        int iw1 = wobr - 1; if ((uint32_t)iw1 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw1)) * w1;
                        int iw2 = wobr - 2; if ((uint32_t)iw2 < (uint32_t)WIN) acc += xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw2)) * w2;
                    }
                }
                yGm.SetValue((uint64_t)(yRowBase + (int64_t)wobr), acc);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t rows_{0};
    uint32_t wout_{0};
    uint32_t blockDim_{0};
};

extern "C" __global__ __aicore__ void conv_transposed2d_asymmetric_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dAsymInSquareK3S1P0_Fp32_Wo8Interior_NoLocalArray op;
    op.Init(x, weight, y, tiling_data.rows, tiling_data.wout, tiling_data.blockDim);
    op.Process();
}
