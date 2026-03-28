
#include "kernel_operator.h"

// Fused operator (float32 only), specialized:
// x:      [N=128,Cin=16,H=W=256]
// weight: [Cout=64,Cin=16,Kh=3,Kw=3], stride=1,pad=0,dil=1 -> conv: [128,64,254,254]
// reduce: min over channel dim (Cout) -> y_pre: [128,1,254,254]
// post: y = tanh(tanh(y_pre))
//
// tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))

class KernelConv2dMinTanhTanhCustom {
public:
    __aicore__ inline KernelConv2dMinTanhTanhCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->totalY = total_y;
        this->elemsPerBlock = elems_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * 1ull * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB scratch for tanh via vector ops on 4 lanes (scalar vehicle)
        pipe.InitBuffer(qV,   1, 4u * sizeof(float));
        pipe.InitBuffer(qTmp, 1, 4u * sizeof(float));
        pipe.InitBuffer(qEx,  1, 4u * sizeof(float));
        pipe.InitBuffer(qOne, 1, 4u * sizeof(float));
        pipe.InitBuffer(qNum, 1, 4u * sizeof(float));
        pipe.InitBuffer(qDen, 1, 4u * sizeof(float));
        pipe.InitBuffer(qT,   1, 4u * sizeof(float));
    }

    __aicore__ inline float TanhScalar(float x)
    {
        AscendC::LocalTensor<float> v   = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex  = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> t   = qT.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, 4);
        v.SetValue(0, x); v.SetValue(1, x); v.SetValue(2, x); v.SetValue(3, x);

        AscendC::Muls(tmp, v, -2.0f, 4);
        AscendC::Exp(ex, tmp, 4);
        AscendC::Sub(num, one, ex, 4);
        AscendC::Add(den, one, ex, 4);
        AscendC::Div(t, num, den, 4);
        float out = t.GetValue(0);

        qT.FreeTensor(t);
        qDen.FreeTensor(den);
        qNum.FreeTensor(num);
        qOne.FreeTensor(one);
        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qV.FreeTensor(v);
        return out;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (end > totalY) end = totalY;
        if (start >= end) return;

        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;
        constexpr uint32_t CIN = 16;
        constexpr uint32_t COUT = 64;

        const uint32_t hwOut = hout * wout; // 254*254

        for (uint32_t linear = start; linear < end; ++linear) {
            // y: [N,1,Hout,Wout] flattened over N*Hout*Wout
            uint32_t tt = linear;
            const uint32_t ni = tt / hwOut;
            tt -= ni * hwOut;
            const uint32_t ho = tt / wout;
            const uint32_t wo = tt - ho * wout;

            // valid conv => input top-left at (hi=ho, wi=wo)
            const uint32_t hi0 = ho;
            const uint32_t wi0 = wo;

            float minv = 3.402823466e+38f; // +FLT_MAX

            // reduce-min over Cout of conv outputs at (ni,co,ho,wo)
            #pragma unroll
            for (uint32_t co = 0; co < COUT; ++co) {
                float acc = bGm.GetValue(static_cast<uint64_t>(co));
                const uint64_t wBaseCo = static_cast<uint64_t>(co) * (static_cast<uint64_t>(CIN) * KH * KW);

                #pragma unroll
                for (uint32_t ci = 0; ci < CIN; ++ci) {
                    const uint64_t xPlaneBase =
                        (static_cast<uint64_t>(ni) * static_cast<uint64_t>(CIN) + static_cast<uint64_t>(ci)) *
                        static_cast<uint64_t>(hin) * static_cast<uint64_t>(win);

                    const uint64_t xBase =
                        xPlaneBase +
                        static_cast<uint64_t>(hi0) * static_cast<uint64_t>(win) +
                        static_cast<uint64_t>(wi0);

                    const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                    #pragma unroll
                    for (uint32_t kH = 0; kH < KH; ++kH) {
                        const uint64_t xRow = xBase + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                        const uint64_t wRow = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);
                        #pragma unroll
                        for (uint32_t kW = 0; kW < KW; ++kW) {
                            const float xv = xGm.GetValue(xRow + static_cast<uint64_t>(kW));
                            const float wv = wGm.GetValue(wRow + static_cast<uint64_t>(kW));
                            acc += xv * wv;
                        }
                    }
                }

                if (acc < minv) minv = acc;
            }

            float t1 = TanhScalar(minv);
            float t2 = TanhScalar(t1);

            yGm.SetValue(static_cast<uint64_t>(linear), t2);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qNum;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qDen;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_min_tanh_tanh_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dMinTanhTanhCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.total_y, t.elems_per_block);
    op.Process();
}
