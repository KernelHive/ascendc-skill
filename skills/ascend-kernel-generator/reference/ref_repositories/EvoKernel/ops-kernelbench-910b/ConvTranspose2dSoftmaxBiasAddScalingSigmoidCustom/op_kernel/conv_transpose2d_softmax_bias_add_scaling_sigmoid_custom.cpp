
#include "kernel_operator.h"

// Key change vs current baseline:
// - Reorder ConvTranspose2d accumulation to be CI-major with UB-cached weight vectors.
//   For each output pixel and each valid tap (<=4), load w[co=0..127] for that (ci,kH,kW) into UB once,
//   then do vector v += wvec * xv using Muls+Add. This removes the scalar-heavy co loop and lots of GM reads.
//
// Keeps stable one-block-per-batch launch; softmax/bias/scale/sigmoid remain vectorized (reductions scalar).

class KernelConvTranspose2dSoftmaxBiasAddScalingSigmoidCustom {
public:
    __aicore__ inline KernelConvTranspose2dSoftmaxBiasAddScalingSigmoidCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR bias, GM_ADDR y,
        uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
        uint32_t cout, uint32_t kh, uint32_t kw,
        uint32_t stride, uint32_t pad, uint32_t out_pad, uint32_t dilation,
        uint32_t hout, uint32_t wout,
        float scaling,
        uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->stride = stride; this->pad = pad; this->out_pad = out_pad; this->dilation = dilation;
        this->hout = hout; this->wout = wout;
        this->scaling = scaling;
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

        pipe.InitBuffer(qV,    1, cout * sizeof(float)); // conv result / final
        pipe.InitBuffer(qTmp,  1, cout * sizeof(float)); // scratch
        pipe.InitBuffer(qCB,   1, cout * sizeof(float)); // cached conv_bias
        pipe.InitBuffer(qB,    1, cout * sizeof(float)); // cached bias
        pipe.InitBuffer(qWVec, 1, cout * sizeof(float)); // weight vector cache per tap
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;
        const uint32_t ni = bid;

        constexpr int32_t PAD = 1;

        const int32_t Hin  = static_cast<int32_t>(hin);
        const int32_t Win  = static_cast<int32_t>(win);
        const int32_t Hout = static_cast<int32_t>(hout);
        const int32_t Wout = static_cast<int32_t>(wout);

        const uint64_t outNStride = static_cast<uint64_t>(cout) * hout * wout;
        const uint64_t outCStride = static_cast<uint64_t>(hout) * wout;
        const uint64_t outHStride = static_cast<uint64_t>(wout);

        const uint64_t wCoStride = static_cast<uint64_t>(kh) * kw;          // contiguous in co? no, co is second dim
        const uint64_t wCiStride = static_cast<uint64_t>(cout) * kh * kw;   // stride per ci in flattened [ci][co][kh][kw]

        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> cb   = qCB.AllocTensor<float>();
        AscendC::LocalTensor<float> b    = qB.AllocTensor<float>();
        AscendC::LocalTensor<float> wvec = qWVec.AllocTensor<float>();

        // Cache conv_bias[128] and bias[128] into UB once per block.
        // bias input is [Cout,1,1] but bound as flat [Cout].
        for (uint32_t co = 0; co < cout; ++co) {
            cb.SetValue(co, convBGm.GetValue(static_cast<uint64_t>(co)));
            b.SetValue(co,  biasGm.GetValue(static_cast<uint64_t>(co)));
        }

        for (int32_t oh = 0; oh < Hout; ++oh) {
            const int32_t parityH = (oh + PAD) & 1;
            const int32_t kH0 = parityH;       // 0 or 1
            const int32_t kH1 = parityH + 2;   // 2 or 3

            const int32_t ih0 = (oh + PAD - kH0) >> 1;
            const int32_t ih1 = (oh + PAD - kH1) >> 1;
            const bool ih0ok = ((uint32_t)ih0 < (uint32_t)Hin);
            const bool ih1ok = ((uint32_t)ih1 < (uint32_t)Hin);

            for (int32_t ow = 0; ow < Wout; ++ow) {
                const int32_t parityW = (ow + PAD) & 1;
                const int32_t kW0 = parityW;     // 0 or 1
                const int32_t kW1 = parityW + 2; // 2 or 3

                const int32_t iw0 = (ow + PAD - kW0) >> 1;
                const int32_t iw1 = (ow + PAD - kW1) >> 1;
                const bool iw0ok = ((uint32_t)iw0 < (uint32_t)Win);
                const bool iw1ok = ((uint32_t)iw1 < (uint32_t)Win);

                // v = conv_bias (vector copy)
                AscendC::DataCopy(v, cb, static_cast<int32_t>(cout));

                // CI-major accumulation; at most 4 taps per ci, each tap uses:
                //  wvec[co]=w[ci,co,kH,kW] (loaded once) then v += wvec * xv
                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const uint64_t xBaseN = (static_cast<uint64_t>(ni) * cin + ci) * hin * win;
                    const uint64_t wBaseCi = static_cast<uint64_t>(ci) * wCiStride;

                    if (ih0ok && iw0ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW0);
                        // w[ci,co,kH,kW] layout: ((((ci * cout + co) * kh + kH) * kw) + kW)
                        // For fixed (ci,kH,kW), co varies with stride (kh*kw)
                        for (uint32_t co = 0; co < cout; ++co) {
                            const uint64_t wIdx = wBase + static_cast<uint64_t>(co) * wCoStride;
                            wvec.SetValue(co, wGm.GetValue(wIdx));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    if (ih0ok && iw1ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const uint64_t wIdx = wBase + static_cast<uint64_t>(co) * wCoStride;
                            wvec.SetValue(co, wGm.GetValue(wIdx));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    if (ih1ok && iw0ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW0);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const uint64_t wIdx = wBase + static_cast<uint64_t>(co) * wCoStride;
                            wvec.SetValue(co, wGm.GetValue(wIdx));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                    if (ih1ok && iw1ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const uint64_t wIdx = wBase + static_cast<uint64_t>(co) * wCoStride;
                            wvec.SetValue(co, wGm.GetValue(wIdx));
                        }
                        AscendC::Muls(wvec, wvec, xv, static_cast<int32_t>(cout));
                        AscendC::Add(v, v, wvec, static_cast<int32_t>(cout));
                    }
                }

                // Softmax across cout (stable): tmp = exp(v-max) ; v = tmp/sum
                float maxv = -3.402823466e+38f;
                for (uint32_t co = 0; co < cout; ++co) {
                    const float vv = v.GetValue(co);
                    maxv = (vv > maxv) ? vv : maxv;
                }

                AscendC::DataCopy(tmp, v, static_cast<int32_t>(cout));
                AscendC::Adds(tmp, tmp, -maxv, static_cast<int32_t>(cout));
                AscendC::Exp(tmp, tmp, static_cast<int32_t>(cout));

                float sumExp = 0.0f;
                for (uint32_t co = 0; co < cout; ++co) sumExp += tmp.GetValue(co);
                const float invSum = 1.0f / (sumExp + 1e-20f);
                AscendC::Muls(v, tmp, invSum, static_cast<int32_t>(cout)); // v=softmax

                // z = (v + bias) * scaling (vector)
                AscendC::Add(v, v, b, static_cast<int32_t>(cout));
                AscendC::Muls(v, v, scaling, static_cast<int32_t>(cout));

                // Sigmoid: v = 1 / (1 + exp(-v)) (vector)
                AscendC::DataCopy(tmp, v, static_cast<int32_t>(cout));
                AscendC::Muls(tmp, tmp, -1.0f, static_cast<int32_t>(cout));
                AscendC::Exp(tmp, tmp, static_cast<int32_t>(cout));
                AscendC::Adds(tmp, tmp, 1.0f, static_cast<int32_t>(cout));
                AscendC::Reciprocal(v, tmp, static_cast<int32_t>(cout));

                // Store y[n, :, oh, ow]
                const uint64_t base =
                    static_cast<uint64_t>(ni) * outNStride +
                    static_cast<uint64_t>(oh) * outHStride +
                    static_cast<uint64_t>(ow);
                for (uint32_t co = 0; co < cout; ++co) {
                    const uint64_t yIdx = base + static_cast<uint64_t>(co) * outCStride;
                    yGm.SetValue(yIdx, v.GetValue(co));
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
    float scaling;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv_transpose2d_softmax_bias_add_scaling_sigmoid_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dSoftmaxBiasAddScalingSigmoidCustom op;
    op.Init(x, weight, conv_bias, bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.stride, t.pad, t.out_pad, t.dilation,
            t.hout, t.wout,
            t.scaling,
            t.blocks);
    op.Process();
}
