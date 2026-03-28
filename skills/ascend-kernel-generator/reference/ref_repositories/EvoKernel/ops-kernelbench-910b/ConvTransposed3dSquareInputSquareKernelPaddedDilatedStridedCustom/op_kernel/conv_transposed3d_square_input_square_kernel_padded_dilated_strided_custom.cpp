
#include "kernel_operator.h"

// Specialized ConvTranspose3d gather with reduced inner-loop branching.
// Fixed params:
// x: [16,32,16,32,32], w: [32,64,3,3,3], stride=2, pad=1, dil=2, out: [16,64,33,65,65]
//
// Key optimization: precompute valid (id,kd) and (ih,kh) for each (od,oh) row and
// remove most per-ow parity/bounds branching by iterating only valid kd/kh and using
// simple iw formula per kw.

class KernelConvTranspose3dGatherHoisted {
public:
    __aicore__ inline KernelConvTranspose3dGatherHoisted() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t rows, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        rows_ = rows;
        blockDim_ = blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 16;
        constexpr int CIN = 32;
        constexpr int DIN = 16;
        constexpr int HIN = 32;
        constexpr int WIN = 32;

        constexpr int COUT = 64;
        constexpr int KD = 3;
        constexpr int KH = 3;
        constexpr int KW = 3;

        constexpr int SD = 2;
        constexpr int SH = 2;
        constexpr int SW = 2;

        constexpr int PD = 1;
        constexpr int PH = 1;
        constexpr int PW = 1;

        constexpr int DD = 2;
        constexpr int DH = 2;
        constexpr int DW = 2;

        constexpr int DOUT = (DIN - 1) * SD - 2 * PD + DD * (KD - 1) + 1; // 33
        constexpr int HOUT = (HIN - 1) * SH - 2 * PH + DH * (KH - 1) + 1; // 65
        constexpr int WOUT = (WIN - 1) * SW - 2 * PW + DW * (KW - 1) + 1; // 65

        constexpr int64_t X_STRIDE_N  = (int64_t)CIN * DIN * HIN * WIN;
        constexpr int64_t X_STRIDE_C  = (int64_t)DIN * HIN * WIN;
        constexpr int64_t X_STRIDE_D  = (int64_t)HIN * WIN;
        constexpr int64_t X_STRIDE_H  = (int64_t)WIN;

        constexpr int64_t Y_STRIDE_N  = (int64_t)COUT * DOUT * HOUT * WOUT;
        constexpr int64_t Y_STRIDE_C  = (int64_t)DOUT * HOUT * WOUT;
        constexpr int64_t Y_STRIDE_D  = (int64_t)HOUT * WOUT;
        constexpr int64_t Y_STRIDE_H  = (int64_t)WOUT;

        constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KD * KH * KW;
        constexpr int64_t W_STRIDE_CO = (int64_t)KD * KH * KW;
        constexpr int64_t W_STRIDE_KD = (int64_t)KH * KW;
        constexpr int64_t W_STRIDE_KH = (int64_t)KW;

        int64_t bdim = (int64_t)blockDim_;
        if (bdim <= 0) bdim = 1;
        const int64_t bid = (int64_t)AscendC::GetBlockIdx();
        const int64_t ROWS = (int64_t)rows_;

        const int64_t rowsPerBlock = (ROWS + bdim - 1) / bdim;
        int64_t rowStart = bid * rowsPerBlock;
        int64_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > ROWS) rowStart = ROWS;
        if (rowEnd > ROWS) rowEnd = ROWS;

        for (int64_t row = rowStart; row < rowEnd; ++row) {
            int64_t t = row;
            const int oh = (int)(t % HOUT); t /= HOUT;
            const int od = (int)(t % DOUT); t /= DOUT;
            const int co = (int)(t % COUT); t /= COUT;
            const int n  = (int)t;

            const int64_t yRowBase = (int64_t)n * Y_STRIDE_N +
                                     (int64_t)co * Y_STRIDE_C +
                                     (int64_t)od * Y_STRIDE_D +
                                     (int64_t)oh * Y_STRIDE_H;

            const int64_t xBaseN = (int64_t)n * X_STRIDE_N;
            const int64_t wBaseCo = (int64_t)co * W_STRIDE_CO;

            // Precompute valid depth taps for this od:
            // numD = od + PD - kd*DD = od+1-2*kd must be divisible by 2 -> always true,
            // id = (od+1)/2 - kd, but id must be in [0..DIN-1].
            // However, divisibility by 2 depends on (od+PD) parity because DD=2:
            // numD = (od+PD) - 2*kd. If (od+PD) odd => numD odd for all kd -> no taps.
            const int td = od + PD;
            int idList[3];
            int kdList[3];
            int dCount = 0;
            if ((td & 1) == 0) {
                const int id0 = (td >> 1);
#pragma unroll
                for (int kd = 0; kd < KD; ++kd) {
                    const int id = id0 - kd;
                    if ((uint32_t)id < (uint32_t)DIN) {
                        idList[dCount] = id;
                        kdList[dCount] = kd;
                        ++dCount;
                    }
                }
            }

            // Precompute valid height taps for this oh.
            const int th = oh + PH;
            int ihList[3];
            int khList[3];
            int hCount = 0;
            if ((th & 1) == 0) {
                const int ih0 = (th >> 1);
#pragma unroll
                for (int kh = 0; kh < KH; ++kh) {
                    const int ih = ih0 - kh;
                    if ((uint32_t)ih < (uint32_t)HIN) {
                        ihList[hCount] = ih;
                        khList[hCount] = kh;
                        ++hCount;
                    }
                }
            }

            if (dCount == 0 || hCount == 0) {
                // Entire row is zeros; write zeros (still required).
#pragma unroll
                for (int ow = 0; ow < WOUT; ++ow) {
                    yGm.SetValue(yRowBase + (int64_t)ow, 0.0f);
                }
                continue;
            }

            for (int ow = 0; ow < WOUT; ++ow) {
                float acc = 0.0f;

                // For width, handle parity once per ow and then just compute iw = ((ow+PW)-kw*DW)/SW.
                const int tw = ow + PW;
                const bool wEven = ((tw & 1) == 0);
                // If tw is odd, then (tw - 2*kw) is odd for all kw => no taps => acc stays 0.
                if (wEven) {
                    const int iw0 = (tw >> 1); // candidate for kw=0 => iw = iw0 - 0
                    // iterate ci and valid kd/kh, and only kw where iw in range
#pragma unroll
                    for (int ci = 0; ci < CIN; ++ci) {
                        const int64_t xBaseNC = xBaseN + (int64_t)ci * X_STRIDE_C;
                        const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CI + wBaseCo;

                        for (int di = 0; di < dCount; ++di) {
                            const int id = idList[di];
                            const int kd = kdList[di];
                            const int64_t xBaseD = xBaseNC + (int64_t)id * X_STRIDE_D;
                            const int64_t wBaseKd = wBaseCiCo + (int64_t)kd * W_STRIDE_KD;

                            for (int hi = 0; hi < hCount; ++hi) {
                                const int ih = ihList[hi];
                                const int kh = khList[hi];
                                const int64_t xBaseH = xBaseD + (int64_t)ih * X_STRIDE_H;
                                const int64_t wBaseKh = wBaseKd + (int64_t)kh * W_STRIDE_KH;

                                // kw=0..2 => iw = iw0 - 0/1/2
                                // Load weights as needed; keep addressing simple and branch-light.
                                int iw = iw0;
                                if ((uint32_t)iw < (uint32_t)WIN) {
                                    const float xf = xGm.GetValue(xBaseH + (int64_t)iw);
                                    const float wf = wGm.GetValue(wBaseKh + 0);
                                    acc += xf * wf;
                                }
                                iw = iw0 - 1;
                                if ((uint32_t)iw < (uint32_t)WIN) {
                                    const float xf = xGm.GetValue(xBaseH + (int64_t)iw);
                                    const float wf = wGm.GetValue(wBaseKh + 1);
                                    acc += xf * wf;
                                }
                                iw = iw0 - 2;
                                if ((uint32_t)iw < (uint32_t)WIN) {
                                    const float xf = xGm.GetValue(xBaseH + (int64_t)iw);
                                    const float wf = wGm.GetValue(wBaseKh + 2);
                                    acc += xf * wf;
                                }
                            }
                        }
                    }
                }

                yGm.SetValue(yRowBase + (int64_t)ow, acc);
            }
        }
    }

private:
    uint32_t rows_{0};
    uint32_t blockDim_{1};
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__
void conv_transposed3d_square_input_square_kernel_padded_dilated_strided_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dGatherHoisted op;
    op.Init(x, weight, y, tiling_data.rows, tiling_data.blockDim);
    op.Process();
}
