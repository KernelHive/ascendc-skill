
#include "kernel_operator.h"

// Optimization: keep the robust "one block per batch" mapping, but reduce scalar/control overhead
// by specializing the stride-2 inverse mapping using parity. This removes %2 checks and shrinks
// kernel loops (k in {0,2} for even parity, k=1 for odd parity).

class KernelConvTranspose3dScalingAvgPoolBiasAddScalingCustom {
public:
    __aicore__ inline KernelConvTranspose3dScalingAvgPoolBiasAddScalingCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR bias, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               float scale1, float scale2,
                               uint32_t dp, uint32_t hp, uint32_t wp,
                               uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->scale1 = scale1;
        this->scale2 = scale2;
        this->dp = dp; this->hp = hp; this->wp = wp;
        this->blocks = blocks;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t biasSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * dp * hp * wp;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBiasGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, biasSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t STR = 2;
        constexpr int32_t PAD = 1;

        constexpr int32_t POOL_K = 2;
        constexpr int32_t POOL_S = 2;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;
        const uint32_t ni = bid;

        const int32_t Din = static_cast<int32_t>(din);
        const int32_t Hin = static_cast<int32_t>(hin);
        const int32_t Win = static_cast<int32_t>(win);

        // Fold scale1 and avgpool normalization(8) into one multiplier.
        const float s1_pool = scale1 * 0.125f;
        const float s2 = scale2;

        // Precompute some strides for output indexing
        const uint64_t outNStride = static_cast<uint64_t>(cout) * dp * hp * wp;
        const uint64_t outCStride = static_cast<uint64_t>(dp) * hp * wp;
        const uint64_t outDStride = static_cast<uint64_t>(hp) * wp;
        const uint64_t outHStride = static_cast<uint64_t>(wp);

        for (uint32_t co = 0; co < cout; ++co) {
            // Keep small parameter loads outside inner loops
            const float add_bias = biasGm.GetValue(static_cast<uint64_t>(co));
            const float cb = convBiasGm.GetValue(static_cast<uint64_t>(co));

            for (uint32_t pd = 0; pd < dp; ++pd) {
                const int32_t od0 = static_cast<int32_t>(pd) * POOL_S;
                for (uint32_t ph = 0; ph < hp; ++ph) {
                    const int32_t oh0 = static_cast<int32_t>(ph) * POOL_S;
                    for (uint32_t pw = 0; pw < wp; ++pw) {
                        const int32_t ow0 = static_cast<int32_t>(pw) * POOL_S;

                        float poolAcc = 0.0f;

                        // Pool window md,mh,mw in {0,1}
                        for (int32_t md = 0; md < POOL_K; ++md) {
                            const int32_t od = od0 + md;
                            const int32_t parityD = (od + PAD) & 1;
                            // If parityD==0 -> kD in {0,2}; else -> kD==1.
                            const int32_t kD0 = (parityD == 0) ? 0 : 1;
                            const int32_t kD1 = (parityD == 0) ? 2 : 1;
                            const int32_t kDcount = (parityD == 0) ? 2 : 1;

                            for (int32_t mh = 0; mh < POOL_K; ++mh) {
                                const int32_t oh = oh0 + mh;
                                const int32_t parityH = (oh + PAD) & 1;
                                const int32_t kH0 = (parityH == 0) ? 0 : 1;
                                const int32_t kH1 = (parityH == 0) ? 2 : 1;
                                const int32_t kHcount = (parityH == 0) ? 2 : 1;

                                for (int32_t mw = 0; mw < POOL_K; ++mw) {
                                    const int32_t ow = ow0 + mw;
                                    const int32_t parityW = (ow + PAD) & 1;
                                    const int32_t kW0 = (parityW == 0) ? 0 : 1;
                                    const int32_t kW1 = (parityW == 0) ? 2 : 1;
                                    const int32_t kWcount = (parityW == 0) ? 2 : 1;

                                    float acc = cb;

                                    // Accumulate only valid taps for this output location based on parity.
                                    for (uint32_t ci = 0; ci < cin; ++ci) {
                                        for (int32_t iD = 0; iD < kDcount; ++iD) {
                                            const int32_t kD = (iD == 0) ? kD0 : kD1;
                                            const int32_t id = (od + PAD - kD) >> 1;
                                            if (static_cast<uint32_t>(id) >= din) continue;

                                            for (int32_t iH = 0; iH < kHcount; ++iH) {
                                                const int32_t kH = (iH == 0) ? kH0 : kH1;
                                                const int32_t ih = (oh + PAD - kH) >> 1;
                                                if (static_cast<uint32_t>(ih) >= hin) continue;

                                                // Compute x base once; iw changes in the innermost loop.
                                                const uint64_t xBase =
                                                    ((((static_cast<uint64_t>(ni) * cin + ci) * din + static_cast<uint64_t>(id))
                                                       * hin + static_cast<uint64_t>(ih))
                                                      * win);

                                                for (int32_t iW = 0; iW < kWcount; ++iW) {
                                                    const int32_t kW = (iW == 0) ? kW0 : kW1;
                                                    const int32_t iw = (ow + PAD - kW) >> 1;
                                                    if (static_cast<uint32_t>(iw) >= win) continue;

                                                    const uint64_t xIdx = xBase + static_cast<uint64_t>(iw);

                                                    const uint64_t wIdx =
                                                        ((((static_cast<uint64_t>(ci) * cout + co) * kd + static_cast<uint64_t>(kD))
                                                           * kh + static_cast<uint64_t>(kH))
                                                          * kw + static_cast<uint64_t>(kW));

                                                    acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
                                                }
                                            }
                                        }
                                    }

                                    poolAcc += acc;
                                }
                            }
                        }

                        const float pooled_scaled = poolAcc * s1_pool;
                        const float outv = (pooled_scaled + add_bias) * s2;

                        const uint64_t yIdx =
                            static_cast<uint64_t>(ni) * outNStride +
                            static_cast<uint64_t>(co) * outCStride +
                            static_cast<uint64_t>(pd) * outDStride +
                            static_cast<uint64_t>(ph) * outHStride +
                            static_cast<uint64_t>(pw);

                        yGm.SetValue(yIdx, outv);
                    }
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm, wGm, convBiasGm, biasGm, yGm;
    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    float scale1, scale2;
    uint32_t dp, hp, wp;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dScalingAvgPoolBiasAddScalingCustom op;
    op.Init(x, weight, conv_bias, bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.scale1, t.scale2,
            t.dp, t.hp, t.wp,
            t.blocks);
    op.Process();
}
