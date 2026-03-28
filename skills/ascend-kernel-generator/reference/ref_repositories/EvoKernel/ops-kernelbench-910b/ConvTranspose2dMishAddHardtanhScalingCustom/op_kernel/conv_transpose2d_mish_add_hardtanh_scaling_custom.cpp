
#include "kernel_operator.h"

// Fused specialized operator:
//
// x:         [N=128, Cin=64, Hin=128, Win=128] float32
// weight:    [Cin=64, Cout=64, Kh=3, Kw=3] float32
// conv_bias: [Cout=64] float32
//
// ConvTranspose2d params: stride=2, padding=1, output_padding=1, dilation=1
// => y_pre: [N, Cout, 256, 256]
//
// Post-op:
//   y = mish(y_pre)
//   y = y + 0.5
//   y = hardtanh(y, -1, 1)
//   y = y * 2
//
// Notes:
// - Avoid Select/mask APIs (previous failure mode); implement mish via pure vector ops (Exp/Log/Div) and no selection.
// - One block per batch item.
// - Vectorize across Cout=64 (small), scalar accumulate for conv weights as in the known-good reference style.

class KernelConvTranspose2dMishAddHardtanhScalingCustom {
public:
    __aicore__ inline KernelConvTranspose2dMishAddHardtanhScalingCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR y,
        uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
        uint32_t cout, uint32_t kh, uint32_t kw,
        uint32_t stride, uint32_t pad, uint32_t out_pad, uint32_t dilation,
        uint32_t hout, uint32_t wout,
        float add_value, float ht_min, float ht_max, float scale,
        uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->stride = stride; this->pad = pad; this->out_pad = out_pad; this->dilation = dilation;
        this->hout = hout; this->wout = wout;
        this->add_value = add_value;
        this->ht_min = ht_min;
        this->ht_max = ht_max;
        this->scale = scale;
        this->blocks = blocks;

        const uint64_t xSize  = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize  = static_cast<uint64_t>(cin) * cout * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t ySize  = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers
        pipe.InitBuffer(qV,   1, cout * sizeof(float)); // conv result / final
        pipe.InitBuffer(qT1,  1, cout * sizeof(float)); // scratch
        pipe.InitBuffer(qT2,  1, cout * sizeof(float)); // scratch
        pipe.InitBuffer(qCB,  1, cout * sizeof(float)); // cached conv_bias
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;
        const uint32_t ni = bid;

        const int32_t STR = static_cast<int32_t>(stride);
        const int32_t PAD = static_cast<int32_t>(pad);

        const int32_t Hin  = static_cast<int32_t>(hin);
        const int32_t Win  = static_cast<int32_t>(win);
        const int32_t Hout = static_cast<int32_t>(hout);
        const int32_t Wout = static_cast<int32_t>(wout);

        const uint64_t outNStride = static_cast<uint64_t>(cout) * hout * wout;
        const uint64_t outCStride = static_cast<uint64_t>(hout) * wout;
        const uint64_t outHStride = static_cast<uint64_t>(wout);

        AscendC::LocalTensor<float> v  = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> t1 = qT1.AllocTensor<float>();
        AscendC::LocalTensor<float> t2 = qT2.AllocTensor<float>();
        AscendC::LocalTensor<float> cb = qCB.AllocTensor<float>();

        // Cache conv_bias[Cout] into UB once per block.
        for (uint32_t co = 0; co < cout; ++co) {
            cb.SetValue(co, convBGm.GetValue(static_cast<uint64_t>(co)));
        }

        // Precompute strides for weight indexing: w[ci,co,kh,kw]
        const uint64_t wCoStride = static_cast<uint64_t>(kh) * kw;          // step for co
        const uint64_t wCiStride = static_cast<uint64_t>(cout) * kh * kw;   // step for ci

        for (int32_t oh = 0; oh < Hout; ++oh) {
            // For stride=2, pad=1: parity selects which kH yield integer ih
            const int32_t parityH = (oh + PAD) & 1; // 0 or 1
            const int32_t kH0 = parityH;           // 0 or 1
            const int32_t kH1 = parityH + 2;       // 2 or 3; for kh=3, 3 is invalid

            const int32_t ih0 = (oh + PAD - kH0) >> 1;
            const int32_t ih1 = (oh + PAD - kH1) >> 1;

            const bool ih0ok = ((uint32_t)ih0 < (uint32_t)Hin);
            const bool ih1ok = (kH1 < static_cast<int32_t>(kh)) && ((uint32_t)ih1 < (uint32_t)Hin);

            for (int32_t ow = 0; ow < Wout; ++ow) {
                const int32_t parityW = (ow + PAD) & 1;
                const int32_t kW0 = parityW;
                const int32_t kW1 = parityW + 2;   // 2 or 3; for kw=3, 3 invalid

                const int32_t iw0 = (ow + PAD - kW0) >> 1;
                const int32_t iw1 = (ow + PAD - kW1) >> 1;

                const bool iw0ok = ((uint32_t)iw0 < (uint32_t)Win);
                const bool iw1ok = (kW1 < static_cast<int32_t>(kw)) && ((uint32_t)iw1 < (uint32_t)Win);

                // v = conv_bias
                AscendC::DataCopy(v, cb, static_cast<int32_t>(cout));

                // Accumulate convolution transpose (direct, specialized parity => up to 4 taps per ci)
                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const uint64_t xBaseN = (static_cast<uint64_t>(ni) * cin + ci) * hin * win;
                    const uint64_t wBaseCi = static_cast<uint64_t>(ci) * wCiStride;

                    if (ih0ok && iw0ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW0);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const float wv = wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride);
                            v.SetValue(co, v.GetValue(co) + xv * wv);
                        }
                    }
                    if (ih0ok && iw1ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih0) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH0) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const float wv = wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride);
                            v.SetValue(co, v.GetValue(co) + xv * wv);
                        }
                    }
                    if (ih1ok && iw0ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw0));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW0);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const float wv = wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride);
                            v.SetValue(co, v.GetValue(co) + xv * wv);
                        }
                    }
                    if (ih1ok && iw1ok) {
                        const float xv = xGm.GetValue(xBaseN + static_cast<uint64_t>(ih1) * win + static_cast<uint64_t>(iw1));
                        const uint64_t wBase = wBaseCi + static_cast<uint64_t>(kH1) * kw + static_cast<uint64_t>(kW1);
                        for (uint32_t co = 0; co < cout; ++co) {
                            const float wv = wGm.GetValue(wBase + static_cast<uint64_t>(co) * wCoStride);
                            v.SetValue(co, v.GetValue(co) + xv * wv);
                        }
                    }
                }

                // Mish: x * tanh(softplus(x)), softplus(x)=log(1+exp(x))
                // t1 = exp(x)
                AscendC::Exp(t1, v, static_cast<int32_t>(cout));
                // t1 = 1 + exp(x)
                AscendC::Adds(t1, t1, 1.0f, static_cast<int32_t>(cout));
                // t1 = log(1+exp(x)) = softplus
                AscendC::Log(t1, t1, static_cast<int32_t>(cout));

                // tanh(z) = (exp(2z)-1)/(exp(2z)+1)
                // t2 = exp(2*softplus)
                AscendC::Muls(t2, t1, 2.0f, static_cast<int32_t>(cout));
                AscendC::Exp(t2, t2, static_cast<int32_t>(cout));

                // t1 = exp(2z) - 1
                AscendC::DataCopy(t1, t2, static_cast<int32_t>(cout));
                AscendC::Adds(t1, t1, -1.0f, static_cast<int32_t>(cout));
                // t2 = exp(2z) + 1
                AscendC::Adds(t2, t2, 1.0f, static_cast<int32_t>(cout));
                // t1 = tanh(z)
                AscendC::Div(t1, t1, t2, static_cast<int32_t>(cout));

                // v = x * tanh(softplus(x))
                AscendC::Mul(v, v, t1, static_cast<int32_t>(cout));

                // + add_value
                AscendC::Adds(v, v, add_value, static_cast<int32_t>(cout));
                // hardtanh clamp
                AscendC::Maxs(v, v, ht_min, static_cast<int32_t>(cout));
                AscendC::Mins(v, v, ht_max, static_cast<int32_t>(cout));
                // * scale
                AscendC::Muls(v, v, scale, static_cast<int32_t>(cout));

                // Store y[n, :, oh, ow]
                const uint64_t base =
                    static_cast<uint64_t>(ni) * outNStride +
                    static_cast<uint64_t>(oh) * outHStride +
                    static_cast<uint64_t>(ow);
                for (uint32_t co = 0; co < cout; ++co) {
                    yGm.SetValue(base + static_cast<uint64_t>(co) * outCStride, v.GetValue(co));
                }
            }
        }

        qCB.FreeTensor(cb);
        qT2.FreeTensor(t2);
        qT1.FreeTensor(t1);
        qV.FreeTensor(v);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT1;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT2;
    AscendC::TQue<AscendC::TPosition::VECIN,  1> qCB;

    AscendC::GlobalTensor<float> xGm, wGm, convBGm, yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t stride, pad, out_pad, dilation;
    uint32_t hout, wout;
    float add_value, ht_min, ht_max, scale;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv_transpose2d_mish_add_hardtanh_scaling_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dMishAddHardtanhScalingCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.stride, t.pad, t.out_pad, t.dilation,
            t.hout, t.wout,
            t.add_value, t.ht_min, t.ht_max, t.scale,
            t.blocks);
    op.Process();
}
