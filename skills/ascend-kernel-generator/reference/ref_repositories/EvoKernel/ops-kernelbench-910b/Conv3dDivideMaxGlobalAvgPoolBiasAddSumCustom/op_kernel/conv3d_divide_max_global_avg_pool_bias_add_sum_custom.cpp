
#include "kernel_operator.h"

class KernelConv3dDivideMaxGlobalAvgPoolBiasAddSumCustom {
public:
    __aicore__ inline KernelConv3dDivideMaxGlobalAvgPoolBiasAddSumCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR cb, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t,
                               uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->blocks = blocks;

        const uint64_t xSize  = (uint64_t)n * cin * din * hin * win;
        const uint64_t wSize  = (uint64_t)cout * cin * kd * kh * kw;
        const uint64_t cbSize = (uint64_t)cout;
        const uint64_t bSize  = (uint64_t)cout;
        const uint64_t ySize  = (uint64_t)n;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        cbGm.SetGlobalBuffer((__gm__ float*)cb, cbSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void LoadSmallParams(float cbLocal[16], float bLocal[16]) const
    {
#pragma unroll
        for (uint32_t co = 0; co < 16u; ++co) {
            cbLocal[co] = cbGm.GetValue((uint64_t)co);
            bLocal[co]  = bGm.GetValue((uint64_t)co);
        }
    }

    __aicore__ inline void ConvAt2Co_Offsets(uint32_t ni, uint32_t co0,
                                            const uint64_t offs[27],
                                            float cb0, float cb1,
                                            float &out0, float &out1) const
    {
        float acc0 = cb0;
        float acc1 = cb1;

        constexpr uint32_t CIN = 8u;
        const uint32_t co1 = co0 + 1u;

#pragma unroll
        for (uint32_t ci = 0; ci < CIN; ++ci) {
            const uint64_t xBase = (uint64_t)ni * (uint64_t)CIN * 16ULL * 64ULL * 64ULL
                                 + (uint64_t)ci * 16ULL * 64ULL * 64ULL;

            uint64_t wPtr0 = ((uint64_t)co0 * (uint64_t)CIN + (uint64_t)ci) * 27ULL;
            uint64_t wPtr1 = ((uint64_t)co1 * (uint64_t)CIN + (uint64_t)ci) * 27ULL;

#pragma unroll
            for (uint32_t t = 0; t < 27u; ++t) {
                const float xv = xGm.GetValue(xBase + offs[t]);
                acc0 += xv * wGm.GetValue(wPtr0++);
                acc1 += xv * wGm.GetValue(wPtr1++);
            }
        }
        out0 = acc0;
        out1 = acc1;
    }

    __aicore__ inline void MakePatchOffsets(uint32_t od, uint32_t oh, uint32_t ow, uint64_t offs[27]) const
    {
        // Linear offsets within a single (ni,ci) volume base:
        // idx = ((d * H + h) * W + w)
        constexpr uint64_t HIN = 64ULL;
        constexpr uint64_t WIN = 64ULL;
        const uint64_t baseD = (uint64_t)od * HIN * WIN;
        const uint64_t baseH = (uint64_t)oh * WIN;
        const uint64_t baseW = (uint64_t)ow;

        uint32_t t = 0;
#pragma unroll
        for (uint32_t kd0 = 0; kd0 < 3u; ++kd0) {
            const uint64_t dOff = baseD + (uint64_t)kd0 * HIN * WIN;
#pragma unroll
            for (uint32_t kh0 = 0; kh0 < 3u; ++kh0) {
                const uint64_t hOff = dOff + baseH + (uint64_t)kh0 * WIN;
#pragma unroll
                for (uint32_t kw0 = 0; kw0 < 3u; ++kw0) {
                    offs[t++] = hOff + baseW + (uint64_t)kw0;
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        if (bid >= blocks) return;

        if (kd != 3U || kh != 3U || kw != 3U) return;
        if (cin != 8U || cout != 16U) return;
        if (din != 16U || hin != 64U || win != 64U) return;

        constexpr float invDiv = 0.5f;
        constexpr float invPoolElems = 1.0f / 6727.0f; // 7*31*31
        constexpr float negInf = -3.402823466e+38f;

        const uint32_t ni = bid;

        float cbLocal[16];
        float bLocal[16];
        LoadSmallParams(cbLocal, bLocal);

        float ysum = 0.0f;

        // 2 output channels at a time
        for (uint32_t co = 0; co < 16u; co += 2u) {
            float sumMax0 = 0.0f;
            float sumMax1 = 0.0f;

            for (uint32_t pd0 = 0; pd0 < 7u; ++pd0) {
                const uint32_t od0 = pd0 * 2u;
                for (uint32_t ph0 = 0; ph0 < 31u; ++ph0) {
                    const uint32_t oh0 = ph0 * 2u;
                    for (uint32_t pw0 = 0; pw0 < 31u; ++pw0) {
                        const uint32_t ow0 = pw0 * 2u;

                        float maxv0 = negInf;
                        float maxv1 = negInf;

                        // pool 2x2x2: evaluate 8 points; for each, build offsets once, reuse for all Cin and both Co.
#pragma unroll
                        for (uint32_t t8 = 0; t8 < 8u; ++t8) {
                            const uint32_t od = od0 + ((t8 >> 2) & 1u);
                            const uint32_t oh = oh0 + ((t8 >> 1) & 1u);
                            const uint32_t ow = ow0 + ((t8 >> 0) & 1u);

                            uint64_t offs[27];
                            MakePatchOffsets(od, oh, ow, offs);

                            float acc0, acc1;
                            ConvAt2Co_Offsets(ni, co, offs, cbLocal[co], cbLocal[co + 1u], acc0, acc1);

                            const float v0 = acc0 * invDiv;
                            const float v1 = acc1 * invDiv;
                            maxv0 = (v0 > maxv0) ? v0 : maxv0;
                            maxv1 = (v1 > maxv1) ? v1 : maxv1;
                        }

                        sumMax0 += maxv0;
                        sumMax1 += maxv1;
                    }
                }
            }

            ysum += (sumMax0 * invPoolElems + bLocal[co]);
            ysum += (sumMax1 * invPoolElems + bLocal[co + 1u]);
        }

        yGm.SetValue((uint64_t)ni, ysum);
    }

private:
    AscendC::GlobalTensor<float> xGm, wGm, cbGm, bGm, yGm;
    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv3d_divide_max_global_avg_pool_bias_add_sum_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv3dDivideMaxGlobalAvgPoolBiasAddSumCustom op;
    op.Init(x, weight, conv_bias, bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dout, t.hout, t.wout,
            t.dp, t.hp, t.wp,
            t.blocks);
    op.Process();
}
