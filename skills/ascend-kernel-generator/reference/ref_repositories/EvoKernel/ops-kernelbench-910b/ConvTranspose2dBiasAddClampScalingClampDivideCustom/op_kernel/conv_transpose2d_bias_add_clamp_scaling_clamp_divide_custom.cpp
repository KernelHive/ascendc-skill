
#include "kernel_operator.h"

// Fused operator:
//   y0 = conv_transpose2d(x, w, conv_bias, stride=2, pad=1, out_pad=1, dil=1)   weight layout [Cin,Cout,Kh,Kw]
//   y1 = y0 + bias (bias shape [Cout,1,1] broadcast)
//   y2 = clamp(y1, 0, 1)
//   y3 = y2 * scaling
//   y4 = clamp(y3, 0, 1)
//   y  = y4 / scaling
//
// Specialization targeted:
//   N=128, Cin=Cout=64, Hin=Win=128, Kh=Kw=3, stride=2, pad=1, out_pad=1, dilation=1, scaling=2.0
//
// Optimization:
// - stride=2 parity trick: only 4 taps contribute per output spatial for Kh=Kw=3 (some taps invalid due to k=3).
// - CI-major accumulation with UB-cached wvec[cout] per (ci,kH,kW) tap to reduce scalar co loop GM reads.
// - one block per batch.

class KernelConvTranspose2dBiasAddClampScalingClampDivideCustom {
public:
    __aicore__ inline KernelConvTranspose2dBiasAddClampScalingClampDivideCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR bias, GM_ADDR y,
        uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
        uint32_t cout, uint32_t kh, uint32_t kw,
        uint32_t stride, uint32_t pad, uint32_t out_pad, uint32_t dilation,
        uint32_t hout, uint32_t wout,
        float scaling, float clampMin, float clampMax,
        uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->stride = stride; this->pad = pad; this->out_pad = out_pad; this->dilation = dilation;
        this->hout = hout; this->wout = wout;
        this->scaling = scaling;
        this->clampMin = clampMin;
        this->clampMax = clampMax;
        this->blocks = blocks;

        const uint64_t xSize  = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize  = static_cast<uint64_t>(cin) * cout * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t bSize  = static_cast<uint64_t>(cout);
        const uint64_t ySize  = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe.InitBuffer(qV,    1, cout * sizeof(float));
        pipe.InitBuffer(qTmp,  1, cout * sizeof(float));
        pipe.InitBuffer(qCB,   1, cout * sizeof(float));
        pipe.InitBuffer(qB,    1, cout * sizeof(float));
        pipe.InitBuffer(qWVec, 1, cout * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;
        const uint32_t ni = bid;

        const int32_t PAD = static_cast<int32_t>(pad);
        const int32_t Hin  = static_cast<int32_t>(hin);
        const int32_t Win  = static_cast<int32_t>(win);
        const int32_t Hout = static_cast<int32_t>(hout);
        const int32_t Wout = static_cast<int32_t>(wout);

        const uint64_t outNStride = static_cast<uint64_t>(cout) * hout * wout;
        const uint64_t outCStride = static_cast<uint64_t>(hout) * wout;
        const uint64_t outHStride = static_cast<uint64_t>(wout);

        // Flattened weight indices for layout [ci][co][kh][kw]
        const uint64_t wCoStride = static_cast<uint64_t>(kh) * kw;        // advance when co++
        const uint64_t wCiStride = static_cast<uint64_t>(cout) * kh * kw; // advance when ci++

        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> cb   = qCB.AllocTensor<float>();
        AscendC::LocalTensor<float> b    = qB.AllocTensor<float>();
        AscendC::LocalTensor<float> wvec = qWVec.AllocTensor<float>();

        for (uint32_t co = 0; co < cout; ++co) {
            cb.SetValue(co, convBGm.GetValue(static_cast<uint64_t>(co)));
            b.SetValue(co,  biasGm.GetValue(static_cast<uint64_t>(co))); // bias is [Cout,1,1] flattened
        }

        const float invS = (scaling != 0.0f) ? (1.0f / scaling) : 0.0f;

        for (int32_t oh = 0; oh < Hout; ++oh) {
            const int32_t parityH = (oh + PAD) & 1;
            const int32_t kH0 = parityH;       // 0 or 1
            const int32_t kH1 = parityH + 2;   // 2 or 3; for kh=3, kH1 valid only when parityH==0 (kH1==2)
            const bool kH1Valid = (kH1 < static_cast<int32_t>(kh));

            const int32_t ih0 = (oh + PAD - kH0) >> 1;
            const bool ih0ok = ((uint32_t)ih0 < (uint32_t)Hin);

            int32_t ih1 = 0;
            bool ih1ok = false;
            if (kH1Valid) {
                ih1 = (oh + PAD - kH1) >> 1;
                ih1ok = ((uint32_t)ih1 < (uint32_t)Hin);
            }

            for (int32_t ow = 0; ow < Wout; ++ow) {
                const int32_t parityW = (ow + PAD) & 1;
                const int32_t kW0 = parityW;     // 0 or 1
                const int32_t kW1 = parityW + 2; // 2 or 3; for kw=3, kW1 valid only when parityW==0 (kW1==2)
                const bool kW1Valid = (kW1 < static_cast<int32_t>(kw));

                const int32_t iw0 = (ow + PAD - kW0) >> 1;
                const bool iw0ok = ((uint32_t)iw0 < (uint32_t)Win);

                int32_t iw1 = 0;
                bool iw1ok = false;
                if (kW1Valid) {
                    iw1 = (ow + PAD - kW1) >> 1;
                    iw1ok = ((uint32_t)iw1 < (uint32_t)Win);
                }

                AscendC::DataCopy(v, cb, static_cast<int32_t>(cout));

                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const uint64_t xBase = (static_cast<uint64_t>(ni) * cin + ci) * hin * win;
                    const uint64_t wBaseCi = static_cast<uint64_t>(ci) * wCiStride;

                    // Tap (kH0,kW0) uses x[ih0,iw0]
                    if (ih0ok && iw0ok) {
                        const float xv = xGm.GetValue(xBase + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW0);
                        for (uint32_t co = 0; co < cout; ++co) {
                            wvec.SetValue(co, wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    // Tap (kH0,kW1) uses x[ih0,iw1]
                    if (ih0ok && kW1Valid && iw1ok) {
                        const float xv = xGm.GetValue(xBase + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            wvec.SetValue(co, wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    // Tap (kH1,kW0) uses x[ih1,iw0]
                    if (kH1Valid && ih1ok && iw0ok) {
                        const float xv = xGm.GetValue(xBase + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW0);
                        for (uint32_t co = 0; co < cout; ++co) {
                            wvec.SetValue(co, wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    // Tap (kH1,kW1) uses x[ih1,iw1]
                    if (kH1Valid && ih1ok && kW1Valid && iw1ok) {
                        const float xv = xGm.GetValue(xBase + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            wvec.SetValue(co, wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                }

                // + external bias
                AscendC::Add(v, v, b, static_cast<int32_t>(cout));

                // clamp -> scale -> clamp -> divide
                AscendC::Maxs(v, v, clampMin, static_cast<int32_t>(cout));
                AscendC::Mins(v, v, clampMax, static_cast<int32_t>(cout));
                AscendC::Muls(v, v, scaling,  static_cast<int32_t>(cout));
                AscendC::Maxs(v, v, clampMin, static_cast<int32_t>(cout));
                AscendC::Mins(v, v, clampMax, static_cast<int32_t>(cout));
                if (scaling != 0.0f) {
                    AscendC::Muls(v, v, invS, static_cast<int32_t>(cout));
                }

                const uint64_t base =
                    static_cast<uint64_t>(ni) * outNStride +
                    static_cast<uint64_t>(oh) * outHStride +
                    static_cast<uint64_t>(ow);
                for (uint32_t co = 0; co < cout; ++co) {
                    yGm.SetValue(base + static_cast<uint64_t>(co) * outCStride, v.GetValue(co));
                }
            }
        }

        qWVec.FreeTensor(wvec);
        qB.FreeTensor(b);
        qCB.FreeTensor(cb);
        qTmp.FreeTensor(tmp);
        qV.FreeTensor(v);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECIN,  1> qCB;
    AscendC::TQue<AscendC::TPosition::VECIN,  1> qB;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qWVec;

    AscendC::GlobalTensor<float> xGm, wGm, convBGm, biasGm, yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t stride, pad, out_pad, dilation;
    uint32_t hout, wout;
    float scaling, clampMin, clampMax;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv_transpose2d_bias_add_clamp_scaling_clamp_divide_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dBiasAddClampScalingClampDivideCustom op;
    op.Init(x, weight, conv_bias, bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.stride, t.pad, t.out_pad, t.dilation,
            t.hout, t.wout,
            t.scaling, t.clamp_min, t.clamp_max,
            t.blocks);
    op.Process();
}
