
#include "kernel_operator.h"

// Parallelization upgrade: blocks map to (n, co, od). Each block computes one output depth-slice
// [hout,wout] for one batch and one output channel, which increases occupancy and reduces long
// scalar control flow in a single block.
//
// Computes:
//   r = conv_transpose3d(x, weight, conv_bias)
//   y = ((2*r + bias) * r) + r
//
// Specialization: K=3, STR=2, PAD=1, DIL=1, OUT_PAD=1; N=16,Cin=32,Cout=64,D/H/W as in tiling.

class KernelConvTranspose3dSumResidualAddMultiplyResidualAddCustom {
public:
    __aicore__ inline KernelConvTranspose3dSumResidualAddMultiplyResidualAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR bias, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t dout, uint32_t hout, uint32_t wout,
                               uint32_t blocks, uint32_t blocks_per_n, uint32_t blocks_per_co)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->dout = dout; this->hout = hout; this->wout = wout;
        this->blocks = blocks;
        this->blocks_per_n = blocks_per_n;
        this->blocks_per_co = blocks_per_co;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t biasSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * dout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBiasGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, biasSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t STR = 2U;
        constexpr uint32_t PAD = 1U;
        if (kd != 3U || kh != 3U || kw != 3U) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;

        // Decode block -> (ni, co, od)
        const uint32_t ni = bid / blocks_per_n;
        const uint32_t rem = bid - ni * blocks_per_n;
        const uint32_t co = rem / blocks_per_co;
        const uint32_t od = rem - co * blocks_per_co;

        const uint64_t inSpatial  = static_cast<uint64_t>(din) * hin * win;
        const uint64_t outSpatial = static_cast<uint64_t>(dout) * hout * wout;

        const uint64_t xBaseN = static_cast<uint64_t>(ni) * static_cast<uint64_t>(cin)  * inSpatial;
        const uint64_t yBaseN = static_cast<uint64_t>(ni) * static_cast<uint64_t>(cout) * outSpatial;

        const uint64_t xStrideC = inSpatial;
        const uint64_t xStrideD = static_cast<uint64_t>(hin) * win;
        const uint64_t xStrideH = static_cast<uint64_t>(win);

        const uint64_t yStrideC = outSpatial;
        const uint64_t yStrideD = static_cast<uint64_t>(hout) * wout;
        const uint64_t yStrideH = static_cast<uint64_t>(wout);

        // weight: [Cin,Cout,3,3,3]
        const uint64_t wStrideCi = static_cast<uint64_t>(cout) * 27ULL;
        const uint64_t wStrideCo = 27ULL;
        const uint64_t wStrideKd = 9ULL;
        const uint64_t wStrideKh = 3ULL;

        // Cache per-channel constants in registers
        const float cb = convBiasGm.GetValue(static_cast<uint64_t>(co));
        const float b  = biasGm.GetValue(static_cast<uint64_t>(co));

        const uint64_t yBaseC = yBaseN + static_cast<uint64_t>(co) * yStrideC;
        const uint64_t yBaseD = yBaseC + static_cast<uint64_t>(od) * yStrideD;

        // Precompute parity-derived kD set for this od once per block
        const uint32_t pD = (od + PAD) & 1U;
        const uint32_t kD0 = pD ? 1U : 0U;
        const uint32_t kD1 = pD ? 1U : 2U;
        const uint32_t nD  = pD ? 1U : 2U;

        for (uint32_t oh = 0; oh < hout; ++oh) {
            const uint32_t pH = (oh + PAD) & 1U;
            const uint32_t kH0 = pH ? 1U : 0U;
            const uint32_t kH1 = pH ? 1U : 2U;
            const uint32_t nH  = pH ? 1U : 2U;

            const uint64_t yBaseH = yBaseD + static_cast<uint64_t>(oh) * yStrideH;

            // Process W in small unroll to reduce loop/control overhead and reuse (id,ih) computations.
            uint32_t ow = 0;
            for (; ow + 3U < wout; ow += 4U) {
                float acc0 = cb, acc1 = cb, acc2 = cb, acc3 = cb;

                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const uint64_t xBaseC = xBaseN + static_cast<uint64_t>(ci) * xStrideC;
                    const uint64_t wBaseCiCo = static_cast<uint64_t>(ci) * wStrideCi + static_cast<uint64_t>(co) * wStrideCo;

                    // kD loop (1 or 2)
                    {
                        const uint32_t kD = kD0;
                        const int32_t id = static_cast<int32_t>(od + PAD - kD) >> 1;
                        if (static_cast<uint32_t>(id) < din) {
                            const uint64_t xBaseD = xBaseC + static_cast<uint64_t>(id) * xStrideD;
                            const uint64_t wOffD = static_cast<uint64_t>(kD) * wStrideKd;

                            // kH loop (1 or 2)
                            {
                                const uint32_t kH = kH0;
                                const int32_t ih = static_cast<int32_t>(oh + PAD - kH) >> 1;
                                if (static_cast<uint32_t>(ih) < hin) {
                                    const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih) * xStrideH;
                                    const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH) * wStrideKh;
                                    AccumulateW4(xBaseH2, wBaseCiCo + wOffH, ow, acc0, acc1, acc2, acc3);
                                }
                            }
                            if (nH == 2U) {
                                const uint32_t kH = kH1;
                                const int32_t ih = static_cast<int32_t>(oh + PAD - kH) >> 1;
                                if (static_cast<uint32_t>(ih) < hin) {
                                    const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih) * xStrideH;
                                    const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH) * wStrideKh;
                                    AccumulateW4(xBaseH2, wBaseCiCo + wOffH, ow, acc0, acc1, acc2, acc3);
                                }
                            }
                        }
                    }
                    if (nD == 2U) {
                        const uint32_t kD = kD1;
                        const int32_t id = static_cast<int32_t>(od + PAD - kD) >> 1;
                        if (static_cast<uint32_t>(id) < din) {
                            const uint64_t xBaseD = xBaseC + static_cast<uint64_t>(id) * xStrideD;
                            const uint64_t wOffD = static_cast<uint64_t>(kD) * wStrideKd;

                            {
                                const uint32_t kH = kH0;
                                const int32_t ih = static_cast<int32_t>(oh + PAD - kH) >> 1;
                                if (static_cast<uint32_t>(ih) < hin) {
                                    const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih) * xStrideH;
                                    const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH) * wStrideKh;
                                    AccumulateW4(xBaseH2, wBaseCiCo + wOffH, ow, acc0, acc1, acc2, acc3);
                                }
                            }
                            if (nH == 2U) {
                                const uint32_t kH = kH1;
                                const int32_t ih = static_cast<int32_t>(oh + PAD - kH) >> 1;
                                if (static_cast<uint32_t>(ih) < hin) {
                                    const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih) * xStrideH;
                                    const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH) * wStrideKh;
                                    AccumulateW4(xBaseH2, wBaseCiCo + wOffH, ow, acc0, acc1, acc2, acc3);
                                }
                            }
                        }
                    }
                }

                // Fused tail: y = ((2*acc + b)*acc)+acc
                yGm.SetValue(yBaseH + static_cast<uint64_t>(ow + 0U), ((2.0f * acc0 + b) * acc0) + acc0);
                yGm.SetValue(yBaseH + static_cast<uint64_t>(ow + 1U), ((2.0f * acc1 + b) * acc1) + acc1);
                yGm.SetValue(yBaseH + static_cast<uint64_t>(ow + 2U), ((2.0f * acc2 + b) * acc2) + acc2);
                yGm.SetValue(yBaseH + static_cast<uint64_t>(ow + 3U), ((2.0f * acc3 + b) * acc3) + acc3);
            }

            // Remainder
            for (; ow < wout; ++ow) {
                const uint32_t pW = (ow + PAD) & 1U;
                const uint32_t kW0 = pW ? 1U : 0U;
                const uint32_t kW1 = pW ? 1U : 2U;
                const uint32_t nW  = pW ? 1U : 2U;

                float acc = cb;

                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const uint64_t xBaseC = xBaseN + static_cast<uint64_t>(ci) * xStrideC;
                    const uint64_t wBaseCiCo = static_cast<uint64_t>(ci) * wStrideCi + static_cast<uint64_t>(co) * wStrideCo;

                    const uint32_t kD_a = kD0;
                    const int32_t id_a = static_cast<int32_t>(od + PAD - kD_a) >> 1;
                    if (static_cast<uint32_t>(id_a) < din) {
                        const uint64_t xBaseD = xBaseC + static_cast<uint64_t>(id_a) * xStrideD;
                        const uint64_t wOffD = static_cast<uint64_t>(kD_a) * wStrideKd;

                        const uint32_t kH_a = kH0;
                        const int32_t ih_a = static_cast<int32_t>(oh + PAD - kH_a) >> 1;
                        if (static_cast<uint32_t>(ih_a) < hin) {
                            const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih_a) * xStrideH;
                            const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH_a) * wStrideKh;
                            AccumulateW1(xBaseH2, wBaseCiCo + wOffH, ow, kW0, kW1, nW, acc);
                        }
                        if (nH == 2U) {
                            const uint32_t kH_b = kH1;
                            const int32_t ih_b = static_cast<int32_t>(oh + PAD - kH_b) >> 1;
                            if (static_cast<uint32_t>(ih_b) < hin) {
                                const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih_b) * xStrideH;
                                const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH_b) * wStrideKh;
                                AccumulateW1(xBaseH2, wBaseCiCo + wOffH, ow, kW0, kW1, nW, acc);
                            }
                        }
                    }

                    if (nD == 2U) {
                        const uint32_t kD_b = kD1;
                        const int32_t id_b = static_cast<int32_t>(od + PAD - kD_b) >> 1;
                        if (static_cast<uint32_t>(id_b) < din) {
                            const uint64_t xBaseD = xBaseC + static_cast<uint64_t>(id_b) * xStrideD;
                            const uint64_t wOffD = static_cast<uint64_t>(kD_b) * wStrideKd;

                            const uint32_t kH_a2 = kH0;
                            const int32_t ih_a2 = static_cast<int32_t>(oh + PAD - kH_a2) >> 1;
                            if (static_cast<uint32_t>(ih_a2) < hin) {
                                const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih_a2) * xStrideH;
                                const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH_a2) * wStrideKh;
                                AccumulateW1(xBaseH2, wBaseCiCo + wOffH, ow, kW0, kW1, nW, acc);
                            }
                            if (nH == 2U) {
                                const uint32_t kH_b2 = kH1;
                                const int32_t ih_b2 = static_cast<int32_t>(oh + PAD - kH_b2) >> 1;
                                if (static_cast<uint32_t>(ih_b2) < hin) {
                                    const uint64_t xBaseH2 = xBaseD + static_cast<uint64_t>(ih_b2) * xStrideH;
                                    const uint64_t wOffH = wOffD + static_cast<uint64_t>(kH_b2) * wStrideKh;
                                    AccumulateW1(xBaseH2, wBaseCiCo + wOffH, ow, kW0, kW1, nW, acc);
                                }
                            }
                        }
                    }
                }

                yGm.SetValue(yBaseH + static_cast<uint64_t>(ow), ((2.0f * acc + b) * acc) + acc);
            }
        }
    }

private:
    __aicore__ inline void AccumulateW1(uint64_t xBaseH2, uint64_t wBaseOff, uint32_t ow,
                                       uint32_t kW0, uint32_t kW1, uint32_t nW, float &acc)
    {
        constexpr uint32_t PAD = 1U;
        const int32_t ow_i = static_cast<int32_t>(ow);

        if (nW == 1U) {
            const uint32_t kW = kW0;
            const int32_t iw = static_cast<int32_t>(ow_i + static_cast<int32_t>(PAD) - static_cast<int32_t>(kW)) >> 1;
            if (static_cast<uint32_t>(iw) < win) {
                const uint64_t xIdx = xBaseH2 + static_cast<uint64_t>(iw);
                const uint64_t wIdx = wBaseOff + static_cast<uint64_t>(kW);
                acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
            }
        } else {
            const uint32_t kW_a = kW0;
            const int32_t iw_a = static_cast<int32_t>(ow_i + static_cast<int32_t>(PAD) - static_cast<int32_t>(kW_a)) >> 1;
            if (static_cast<uint32_t>(iw_a) < win) {
                const uint64_t xIdx = xBaseH2 + static_cast<uint64_t>(iw_a);
                const uint64_t wIdx = wBaseOff + static_cast<uint64_t>(kW_a);
                acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
            }
            const uint32_t kW_b = kW1;
            const int32_t iw_b = static_cast<int32_t>(ow_i + static_cast<int32_t>(PAD) - static_cast<int32_t>(kW_b)) >> 1;
            if (static_cast<uint32_t>(iw_b) < win) {
                const uint64_t xIdx = xBaseH2 + static_cast<uint64_t>(iw_b);
                const uint64_t wIdx = wBaseOff + static_cast<uint64_t>(kW_b);
                acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
            }
        }
    }

    __aicore__ inline void AccumulateW4(uint64_t xBaseH2, uint64_t wBaseOff, uint32_t ow0,
                                       float &acc0, float &acc1, float &acc2, float &acc3)
    {
        constexpr uint32_t PAD = 1U;

        // For each ow, decide parity-derived kW set and accumulate (1 or 2 taps).
        AccumulateW1(xBaseH2, wBaseOff, ow0 + 0U,
                    ((ow0 + 0U + PAD) & 1U) ? 1U : 0U,
                    ((ow0 + 0U + PAD) & 1U) ? 1U : 2U,
                    ((ow0 + 0U + PAD) & 1U) ? 1U : 2U, acc0);

        AccumulateW1(xBaseH2, wBaseOff, ow0 + 1U,
                    ((ow0 + 1U + PAD) & 1U) ? 1U : 0U,
                    ((ow0 + 1U + PAD) & 1U) ? 1U : 2U,
                    ((ow0 + 1U + PAD) & 1U) ? 1U : 2U, acc1);

        AccumulateW1(xBaseH2, wBaseOff, ow0 + 2U,
                    ((ow0 + 2U + PAD) & 1U) ? 1U : 0U,
                    ((ow0 + 2U + PAD) & 1U) ? 1U : 2U,
                    ((ow0 + 2U + PAD) & 1U) ? 1U : 2U, acc2);

        AccumulateW1(xBaseH2, wBaseOff, ow0 + 3U,
                    ((ow0 + 3U + PAD) & 1U) ? 1U : 0U,
                    ((ow0 + 3U + PAD) & 1U) ? 1U : 2U,
                    ((ow0 + 3U + PAD) & 1U) ? 1U : 2U, acc3);
    }

private:
    AscendC::GlobalTensor<float> xGm, wGm, convBiasGm, biasGm, yGm;
    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    uint32_t dout, hout, wout;
    uint32_t blocks, blocks_per_n, blocks_per_co;
};

extern "C" __global__ __aicore__ void conv_transpose3d_sum_residual_add_multiply_residual_add_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dSumResidualAddMultiplyResidualAddCustom op;
    op.Init(x, weight, conv_bias, bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dout, t.hout, t.wout,
            t.blocks, t.blocks_per_n, t.blocks_per_co);
    op.Process();
}
