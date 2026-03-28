
#include "kernel_operator.h"

// Specialized Conv3d:
// x: [8,3,16,128,128] (N,C,D,H,W) contiguous
// w: [64,3,3,5,7]     (Cout,Cin,KD,KH,KW) contiguous
// stride=1, pad=0, dil=1, groups=1, bias=False
// y: [8,64,14,124,122]
//
// This round (scalar/pipeline focused):
// - Cache all weights for current output channel (315 floats) in registers once per row.
// - OW unroll by 4, but use sliding-window reuse: for each (ci,kd,kh) load x[ow..ow+9] once,
//   then form 4 outputs with shifted 7-tap windows, reducing redundant GM loads and scalar address ops.
// - Deterministic row tiling over (n,co,od,oh) for better occupancy.

class KernelConvStandard3dAsymAsymRowTiledOw4Slide {
public:
    __aicore__ inline KernelConvStandard3dAsymAsymRowTiledOw4Slide() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t blockDim, uint32_t totalRows)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        blockDim_ = (blockDim == 0) ? 1u : blockDim;
        totalRows_ = totalRows;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 8;
        constexpr int CIN = 3;
        constexpr int D = 16;
        constexpr int H = 128;
        constexpr int W = 128;

        constexpr int COUT = 64;
        constexpr int KD = 3;
        constexpr int KH = 5;
        constexpr int KW = 7;

        constexpr int DOUT = 14;
        constexpr int HOUT = 124;
        constexpr int WOUT = 122;

        // Strides in contiguous NCDHW
        constexpr int64_t XsW  = 1;
        constexpr int64_t XsH  = (int64_t)W;
        constexpr int64_t XsD  = (int64_t)H * W;
        constexpr int64_t XsCI = (int64_t)D * H * W;
        constexpr int64_t XsN  = (int64_t)CIN * D * H * W;

        constexpr int64_t YsW  = 1;
        constexpr int64_t YsH  = (int64_t)WOUT;
        constexpr int64_t YsD  = (int64_t)HOUT * WOUT;
        constexpr int64_t YsCO = (int64_t)DOUT * HOUT * WOUT;
        constexpr int64_t YsN  = (int64_t)COUT * DOUT * HOUT * WOUT;

        constexpr int64_t W_PER_OC = (int64_t)CIN * KD * KH * KW; // 315

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;
        const uint32_t totalRows = (totalRows_ == 0) ? (uint32_t)(N * COUT * DOUT * HOUT) : totalRows_;

        const uint32_t rowsPerBlock = (totalRows + bdim - 1) / bdim;
        uint32_t rowStart = bid * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > totalRows) rowStart = totalRows;
        if (rowEnd > totalRows) rowEnd = totalRows;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            // row -> (n, co, od, oh) with oh fastest
            uint32_t t = row;
            const int oh = (int)(t % HOUT); t /= HOUT;
            const int od = (int)(t % DOUT); t /= DOUT;
            const int co = (int)(t % COUT); t /= COUT;
            const int n  = (int)t;

            // Cache weights for this oc into registers.
            // Layout: [ci][kd][kh][kw] in a flat array.
            float wReg[315];
            const int64_t wBaseOc = (int64_t)co * W_PER_OC;
#pragma unroll
            for (int i = 0; i < 315; ++i) {
                wReg[i] = wGm.GetValue(wBaseOc + (int64_t)i);
            }

            const int64_t xBaseN = (int64_t)n * XsN;
            const int64_t yBaseN = (int64_t)n * YsN;

            const int id0 = od;
            const int ih0 = oh;

            const int64_t xBaseCi0 = xBaseN + (int64_t)0 * XsCI + (int64_t)id0 * XsD + (int64_t)ih0 * XsH;
            const int64_t xBaseCi1 = xBaseN + (int64_t)1 * XsCI + (int64_t)id0 * XsD + (int64_t)ih0 * XsH;
            const int64_t xBaseCi2 = xBaseN + (int64_t)2 * XsCI + (int64_t)id0 * XsD + (int64_t)ih0 * XsH;

            const int64_t yRowBase = yBaseN + (int64_t)co * YsCO + (int64_t)od * YsD + (int64_t)oh * YsH;

            int ow = 0;
            constexpr int OW_STEP = 4;
            const int owMainEnd = (WOUT / OW_STEP) * OW_STEP;

            for (; ow < owMainEnd; ow += OW_STEP) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                // For each ci, scan kd/kh and do KW=7 tap.
                // Sliding-window reuse: load x[ow..ow+9] (10 vals) once per (ci,kd,kh),
                // then accumulate 4 outputs with shifted windows.
                int wOff = 0;

                // ci0
                {
                    const int64_t xBaseOw = xBaseCi0 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;

                            const float w0 = wReg[wOff + 0];
                            const float w1 = wReg[wOff + 1];
                            const float w2 = wReg[wOff + 2];
                            const float w3 = wReg[wOff + 3];
                            const float w4 = wReg[wOff + 4];
                            const float w5 = wReg[wOff + 5];
                            const float w6 = wReg[wOff + 6];

                            const float x0 = xGm.GetValue(xBh + 0);
                            const float x1 = xGm.GetValue(xBh + 1);
                            const float x2 = xGm.GetValue(xBh + 2);
                            const float x3 = xGm.GetValue(xBh + 3);
                            const float x4 = xGm.GetValue(xBh + 4);
                            const float x5 = xGm.GetValue(xBh + 5);
                            const float x6 = xGm.GetValue(xBh + 6);
                            const float x7 = xGm.GetValue(xBh + 7);
                            const float x8 = xGm.GetValue(xBh + 8);
                            const float x9 = xGm.GetValue(xBh + 9);

                            acc0 += x0*w0 + x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6;
                            acc1 += x1*w0 + x2*w1 + x3*w2 + x4*w3 + x5*w4 + x6*w5 + x7*w6;
                            acc2 += x2*w0 + x3*w1 + x4*w2 + x5*w3 + x6*w4 + x7*w5 + x8*w6;
                            acc3 += x3*w0 + x4*w1 + x5*w2 + x6*w3 + x7*w4 + x8*w5 + x9*w6;

                            wOff += KW;
                        }
                    }
                }

                // ci1
                {
                    const int64_t xBaseOw = xBaseCi1 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;

                            const float w0 = wReg[wOff + 0];
                            const float w1 = wReg[wOff + 1];
                            const float w2 = wReg[wOff + 2];
                            const float w3 = wReg[wOff + 3];
                            const float w4 = wReg[wOff + 4];
                            const float w5 = wReg[wOff + 5];
                            const float w6 = wReg[wOff + 6];

                            const float x0 = xGm.GetValue(xBh + 0);
                            const float x1 = xGm.GetValue(xBh + 1);
                            const float x2 = xGm.GetValue(xBh + 2);
                            const float x3 = xGm.GetValue(xBh + 3);
                            const float x4 = xGm.GetValue(xBh + 4);
                            const float x5 = xGm.GetValue(xBh + 5);
                            const float x6 = xGm.GetValue(xBh + 6);
                            const float x7 = xGm.GetValue(xBh + 7);
                            const float x8 = xGm.GetValue(xBh + 8);
                            const float x9 = xGm.GetValue(xBh + 9);

                            acc0 += x0*w0 + x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6;
                            acc1 += x1*w0 + x2*w1 + x3*w2 + x4*w3 + x5*w4 + x6*w5 + x7*w6;
                            acc2 += x2*w0 + x3*w1 + x4*w2 + x5*w3 + x6*w4 + x7*w5 + x8*w6;
                            acc3 += x3*w0 + x4*w1 + x5*w2 + x6*w3 + x7*w4 + x8*w5 + x9*w6;

                            wOff += KW;
                        }
                    }
                }

                // ci2
                {
                    const int64_t xBaseOw = xBaseCi2 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;

                            const float w0 = wReg[wOff + 0];
                            const float w1 = wReg[wOff + 1];
                            const float w2 = wReg[wOff + 2];
                            const float w3 = wReg[wOff + 3];
                            const float w4 = wReg[wOff + 4];
                            const float w5 = wReg[wOff + 5];
                            const float w6 = wReg[wOff + 6];

                            const float x0 = xGm.GetValue(xBh + 0);
                            const float x1 = xGm.GetValue(xBh + 1);
                            const float x2 = xGm.GetValue(xBh + 2);
                            const float x3 = xGm.GetValue(xBh + 3);
                            const float x4 = xGm.GetValue(xBh + 4);
                            const float x5 = xGm.GetValue(xBh + 5);
                            const float x6 = xGm.GetValue(xBh + 6);
                            const float x7 = xGm.GetValue(xBh + 7);
                            const float x8 = xGm.GetValue(xBh + 8);
                            const float x9 = xGm.GetValue(xBh + 9);

                            acc0 += x0*w0 + x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6;
                            acc1 += x1*w0 + x2*w1 + x3*w2 + x4*w3 + x5*w4 + x6*w5 + x7*w6;
                            acc2 += x2*w0 + x3*w1 + x4*w2 + x5*w3 + x6*w4 + x7*w5 + x8*w6;
                            acc3 += x3*w0 + x4*w1 + x5*w2 + x6*w3 + x7*w4 + x8*w5 + x9*w6;

                            wOff += KW;
                        }
                    }
                }

                yGm.SetValue(yRowBase + (int64_t)ow + 0, acc0);
                yGm.SetValue(yRowBase + (int64_t)ow + 1, acc1);
                yGm.SetValue(yRowBase + (int64_t)ow + 2, acc2);
                yGm.SetValue(yRowBase + (int64_t)ow + 3, acc3);
            }

            // Tail
            for (; ow < WOUT; ++ow) {
                float acc = 0.0f;
                int wOff = 0;

                // ci0
                {
                    const int64_t xBaseOw = xBaseCi0 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;
                            acc += xGm.GetValue(xBh + 0) * wReg[wOff + 0];
                            acc += xGm.GetValue(xBh + 1) * wReg[wOff + 1];
                            acc += xGm.GetValue(xBh + 2) * wReg[wOff + 2];
                            acc += xGm.GetValue(xBh + 3) * wReg[wOff + 3];
                            acc += xGm.GetValue(xBh + 4) * wReg[wOff + 4];
                            acc += xGm.GetValue(xBh + 5) * wReg[wOff + 5];
                            acc += xGm.GetValue(xBh + 6) * wReg[wOff + 6];
                            wOff += KW;
                        }
                    }
                }
                // ci1
                {
                    const int64_t xBaseOw = xBaseCi1 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;
                            acc += xGm.GetValue(xBh + 0) * wReg[wOff + 0];
                            acc += xGm.GetValue(xBh + 1) * wReg[wOff + 1];
                            acc += xGm.GetValue(xBh + 2) * wReg[wOff + 2];
                            acc += xGm.GetValue(xBh + 3) * wReg[wOff + 3];
                            acc += xGm.GetValue(xBh + 4) * wReg[wOff + 4];
                            acc += xGm.GetValue(xBh + 5) * wReg[wOff + 5];
                            acc += xGm.GetValue(xBh + 6) * wReg[wOff + 6];
                            wOff += KW;
                        }
                    }
                }
                // ci2
                {
                    const int64_t xBaseOw = xBaseCi2 + (int64_t)ow;
#pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int64_t xBd = xBaseOw + (int64_t)kd * XsD;
#pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int64_t xBh = xBd + (int64_t)kh * XsH;
                            acc += xGm.GetValue(xBh + 0) * wReg[wOff + 0];
                            acc += xGm.GetValue(xBh + 1) * wReg[wOff + 1];
                            acc += xGm.GetValue(xBh + 2) * wReg[wOff + 2];
                            acc += xGm.GetValue(xBh + 3) * wReg[wOff + 3];
                            acc += xGm.GetValue(xBh + 4) * wReg[wOff + 4];
                            acc += xGm.GetValue(xBh + 5) * wReg[wOff + 5];
                            acc += xGm.GetValue(xBh + 6) * wReg[wOff + 6];
                            wOff += KW;
                        }
                    }
                }

                yGm.SetValue(yRowBase + (int64_t)ow, acc);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockDim_{1};
    uint32_t totalRows_{0};
};

extern "C" __global__ __aicore__ void conv_standard3d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard3dAsymAsymRowTiledOw4Slide op;
    op.Init(x, weight, y, tiling_data.blockDim, tiling_data.totalRows);
    op.Process();
}
