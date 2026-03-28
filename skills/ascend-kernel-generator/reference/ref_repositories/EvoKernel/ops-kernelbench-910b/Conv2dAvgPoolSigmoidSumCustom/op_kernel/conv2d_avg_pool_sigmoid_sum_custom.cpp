
#include "kernel_operator.h"

// Fused operator (float32 only), specialized:
// x:      [N=128,Cin=8,H=W=384]
// weight: [Cout=64,Cin=8,Kh=3,Kw=3], stride=1,pad=0,dil=1 -> conv: [128,64,382,382]
// avgpool2d(k=4,s=4,p=0) -> [128,64,95,95]
// sigmoid, then sum over [C,H,W] -> y: [128]
//
// sigmoid(z) = 1 / (1 + exp(-z))

class KernelConv2dAvgPoolSigmoidSumCustom {
public:
    __aicore__ inline KernelConv2dAvgPoolSigmoidSumCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hconv, uint32_t wconv,
                               uint32_t hout, uint32_t wout,
                               uint32_t total_y, uint32_t n_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hconv = hconv; this->wconv = wconv;
        this->hout = hout; this->wout = wout;
        this->totalY = total_y;
        this->nPerBlock = n_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n);

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB scratch for sigmoid via vector ops on 1 element (replicated to 4)
        pipe.InitBuffer(qV,   1, 4u * sizeof(float));
        pipe.InitBuffer(qTmp, 1, 4u * sizeof(float));
        pipe.InitBuffer(qEx,  1, 4u * sizeof(float));
        pipe.InitBuffer(qOne, 1, 4u * sizeof(float));
        pipe.InitBuffer(qDen, 1, 4u * sizeof(float));
        pipe.InitBuffer(qOut, 1, 4u * sizeof(float));
    }

    __aicore__ inline float Sigmoid(float z,
                                   AscendC::LocalTensor<float>& v,
                                   AscendC::LocalTensor<float>& tmp,
                                   AscendC::LocalTensor<float>& ex,
                                   AscendC::LocalTensor<float>& one,
                                   AscendC::LocalTensor<float>& den,
                                   AscendC::LocalTensor<float>& out)
    {
        v.SetValue(0, z); v.SetValue(1, z); v.SetValue(2, z); v.SetValue(3, z);
        AscendC::Muls(tmp, v, -1.0f, 4);
        AscendC::Exp(ex, tmp, 4);
        AscendC::Add(den, one, ex, 4);
        AscendC::Div(out, one, den, 4);
        return out.GetValue(0);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t nStart = bid * nPerBlock;
        uint32_t nEnd = nStart + nPerBlock;
        if (nEnd > n) nEnd = n;
        if (nStart >= nEnd) return;

        AscendC::LocalTensor<float> v   = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex  = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> den = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> out = qOut.AllocTensor<float>();
        AscendC::Duplicate(one, 1.0f, 4);

        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;
        constexpr uint32_t PK = 4;
        constexpr uint32_t PS = 4;
        constexpr float inv16 = 1.0f / 16.0f;

        // Specialized dimensions (kept as constexpr to help compiler)
        constexpr uint32_t CIN = 8;
        constexpr uint32_t COUT = 64;
        constexpr uint32_t HOUT = 95;
        constexpr uint32_t WOUT = 95;

        for (uint32_t ni = nStart; ni < nEnd; ++ni) {
            float sumAll = 0.0f;

            for (uint32_t co = 0; co < COUT; ++co) {
                const float bias = bGm.GetValue(static_cast<uint64_t>(co));
                const uint64_t wBaseCo = static_cast<uint64_t>(co) * (static_cast<uint64_t>(CIN) * KH * KW);

                for (uint32_t ho = 0; ho < HOUT; ++ho) {
                    const uint32_t hc0 = ho * PS;
                    for (uint32_t wo = 0; wo < WOUT; ++wo) {
                        const uint32_t wc0 = wo * PS;

                        float poolSum = 0.0f;

                        #pragma unroll
                        for (uint32_t ph = 0; ph < PK; ++ph) {
                            #pragma unroll
                            for (uint32_t pw = 0; pw < PK; ++pw) {
                                const uint32_t hc = hc0 + ph;
                                const uint32_t wc = wc0 + pw;

                                float acc = bias;

                                #pragma unroll
                                for (uint32_t ci = 0; ci < CIN; ++ci) {
                                    const uint64_t xBase =
                                        ((static_cast<uint64_t>(ni) * CIN + static_cast<uint64_t>(ci)) *
                                         static_cast<uint64_t>(hin) + static_cast<uint64_t>(hc)) *
                                         static_cast<uint64_t>(win) + static_cast<uint64_t>(wc);

                                    const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                                    #pragma unroll
                                    for (uint32_t kH = 0; kH < KH; ++kH) {
                                        const uint64_t xRow =
                                            xBase + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                                        const uint64_t wRow =
                                            wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);

                                        #pragma unroll
                                        for (uint32_t kW = 0; kW < KW; ++kW) {
                                            const float xv = xGm.GetValue(xRow + static_cast<uint64_t>(kW));
                                            const float wv = wGm.GetValue(wRow + static_cast<uint64_t>(kW));
                                            acc += xv * wv;
                                        }
                                    }
                                }

                                poolSum += acc;
                            }
                        }

                        const float pooled = poolSum * inv16;
                        const float s = Sigmoid(pooled, v, tmp, ex, one, den, out);
                        sumAll += s;
                    }
                }
            }

            yGm.SetValue(static_cast<uint64_t>(ni), sumAll);
        }

        qOut.FreeTensor(out);
        qDen.FreeTensor(den);
        qOne.FreeTensor(one);
        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qV.FreeTensor(v);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qDen;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOut;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hconv, wconv;
    uint32_t hout, wout;
    uint32_t totalY;
    uint32_t nPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_avg_pool_sigmoid_sum_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dAvgPoolSigmoidSumCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hconv, t.wconv,
            t.hout, t.wout,
            t.total_y, t.n_per_block);
    op.Process();
}
