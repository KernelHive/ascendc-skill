
#include "kernel_operator.h"

// Output-stationary gather ConvTranspose2d specialized for:
// x:      [16,32,128,256]
// weight: [32,16,3,5]  (Cin, Cout/groups, Kh, Kw)
// stride=(2,3), padding=(1,2), dilation=(2,1), groups=4, bias=False
// y:      [16,64,255,766]
//
// This version parallelizes over output elements (no races, no atomics).
// For each output (n,co,ho,wo), it finds valid (hi,wi) for each (kh,kw) by solving:
// ho = hi*SH - PH + kh*DH  => hi = (ho + PH - kh*DH) / SH  if divisible
// wo = wi*SW - PW + kw*DW  => wi = (wo + PW - kw*DW) / SW  if divisible
// and sums over ci within the same group.

class KernelConvTransposed2dOutStationaryFp32 {
public:
    __aicore__ inline KernelConvTransposed2dOutStationaryFp32() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalY, uint32_t totalX, uint32_t totalW,
                               uint32_t blockDim, uint32_t tileElems)
    {
        (void)totalX; (void)totalW;
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        this->totalY = totalY;
        this->blockDim = blockDim;
        this->tileElems = tileElems;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 16;
        constexpr int CIN = 32;
        constexpr int HIN = 128;
        constexpr int WIN = 256;

        constexpr int COUT = 64;
        constexpr int KH = 3;
        constexpr int KW = 5;

        constexpr int SH = 2;
        constexpr int SW = 3;
        constexpr int PH = 1;
        constexpr int PW = 2;
        constexpr int DH = 2;
        constexpr int DW = 1;

        constexpr int GROUPS = 4;
        constexpr int CIN_G = CIN / GROUPS;   // 8
        constexpr int COUT_G = COUT / GROUPS; // 16

        constexpr int HOUT = (HIN - 1) * SH - 2 * PH + DH * (KH - 1) + 1; // 255
        constexpr int WOUT = (WIN - 1) * SW - 2 * PW + DW * (KW - 1) + 1; // 766

        // Guard against mismatched shapes to avoid OOB in specialization context.
        const uint32_t expectedY = (uint32_t)(N * COUT * HOUT * WOUT);
        if (totalY != expectedY) return;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();

        // Linear tiling over y
        const uint32_t stride = blockDim * tileElems;
        uint32_t base = bid * tileElems;

        while (base < totalY) {
            uint32_t end = base + tileElems;
            if (end > totalY) end = totalY;

            for (uint32_t yLinear = base; yLinear < end; ++yLinear) {
                // Decode yLinear -> (n, co, ho, wo)
                uint32_t t = yLinear;
                const int wo = (int)(t % WOUT); t /= WOUT;
                const int ho = (int)(t % HOUT); t /= HOUT;
                const int co = (int)(t % COUT); t /= COUT;
                const int n  = (int)t;

                const int g = co / COUT_G;
                const int coInG = co - g * COUT_G;
                const int ciBase = g * CIN_G;

                float acc = 0.0f;

                // Iterate over kernel taps; only a few will match stride congruence.
                // Precompute hoTerm and woTerm for each tap.
                for (int kh = 0; kh < KH; ++kh) {
                    const int hoTerm = ho + PH - kh * DH;
                    // hoTerm must be divisible by SH and produce hi in [0,HIN)
                    if ((unsigned)hoTerm >= 0x80000000u) {
                        // negative: skip quickly
                    }
                    if ((hoTerm & (SH - 1)) != 0) continue;
                    const int hi = hoTerm / SH;
                    if ((unsigned)hi >= (unsigned)HIN) continue;

                    for (int kw = 0; kw < KW; ++kw) {
                        const int woTerm = wo + PW - kw * DW;
                        // SW=3 not power of two; use modulus.
                        // Fast reject negative first:
                        if (woTerm < 0) continue;
                        const int rem = woTerm - (woTerm / SW) * SW;
                        if (rem != 0) continue;
                        const int wi = woTerm / SW;
                        if ((unsigned)wi >= (unsigned)WIN) continue;

                        // Now sum over Cin in group
                        const int64_t xBase =
                            (((int64_t)n * CIN) * HIN + (int64_t)hi) * WIN + (int64_t)wi;

                        // Weight index uses global ci (as stored), but group correctness comes from restricting ci range
                        for (int ciInG = 0; ciInG < CIN_G; ++ciInG) {
                            const int ci = ciBase + ciInG;
                            const float xf = xGm.GetValue(xBase + (int64_t)ci * (int64_t)HIN * (int64_t)WIN);

                            // wIdx = (((ci * COUT_G + coInG) * KH + kh) * KW + kw)
                            const int64_t wIdx =
                                (((int64_t)ci * COUT_G + (int64_t)coInG) * KH + (int64_t)kh) * KW + (int64_t)kw;
                            const float wf = wGm.GetValue(wIdx);
                            acc += xf * wf;
                        }
                    }
                }

                yGm.SetValue((int64_t)yLinear, acc);
            }

            base += stride;
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalY;
    uint32_t blockDim;
    uint32_t tileElems;
};

extern "C" __global__ __aicore__ void conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed2dOutStationaryFp32 op;
    op.Init(x, weight, y,
            tiling_data.totalY, tiling_data.totalX, tiling_data.totalW,
            tiling_data.blockDim, tiling_data.tileElems);
    op.Process();
}
