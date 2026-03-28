
#include "kernel_operator.h"

// Fused operator (float32 only), specialized:
// x:      [N=128,Cin=64,H=W=128]
// weight: [Cout=128,Cin=64,Kh=3,Kw=3], stride=1,pad=0,dil=1 -> conv: [128,128,126,126]
// post: tanh(conv - 0.5) - 0.2
// avgpool2d(k=2,s=2,p=0) -> y: [128,128,63,63]
//
// tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))

class KernelConv2dSubtractTanhSubtractAvgPoolCustom {
public:
    __aicore__ inline KernelConv2dSubtractTanhSubtractAvgPoolCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hconv, uint32_t wconv,
                               uint32_t hout, uint32_t wout,
                               float sub1, float sub2,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hconv = hconv; this->wconv = wconv;
        this->hout = hout; this->wout = wout;
        this->sub1 = sub1; this->sub2 = sub2;
        this->totalY = total_y;
        this->elemsPerBlock = elems_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB scratch: compute tanh via vector Exp/Div; store 4 pool vals
        pipe.InitBuffer(qPool, 1, 4u * sizeof(float));
        pipe.InitBuffer(qV,    1, 4u * sizeof(float));
        pipe.InitBuffer(qTmp,  1, 4u * sizeof(float));
        pipe.InitBuffer(qEx,   1, 4u * sizeof(float));
        pipe.InitBuffer(qOne,  1, 4u * sizeof(float));
        pipe.InitBuffer(qNum,  1, 4u * sizeof(float));
        pipe.InitBuffer(qDen,  1, 4u * sizeof(float));
        pipe.InitBuffer(qT,    1, 4u * sizeof(float));
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
        constexpr uint32_t POOL_K = 2;
        constexpr uint32_t POOL_S = 2;

        const uint32_t hwOut = hout * wout;      // 63*63
        const uint32_t chwOut = cout * hwOut;    // 128*hwOut

        AscendC::LocalTensor<float> pool = qPool.AllocTensor<float>();
        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex   = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one  = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num  = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den  = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> t    = qT.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, 4);

        for (uint32_t linear = start; linear < end; ++linear) {
            // y: [N,Cout,Hout,Wout]
            uint32_t tt = linear;
            const uint32_t ni = tt / chwOut;
            tt -= ni * chwOut;
            const uint32_t co = tt / hwOut;
            tt -= co * hwOut;
            const uint32_t ho = tt / wout;
            const uint32_t wo = tt - ho * wout;

            // map to conv space base for pool window
            const uint32_t hc0 = ho * POOL_S; // 0..124
            const uint32_t wc0 = wo * POOL_S; // 0..124

            const float bias = bGm.GetValue(static_cast<uint64_t>(co));
            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (64ull * KH * KW);

            // compute and pool 2x2 window of post-ops
            #pragma unroll
            for (uint32_t ph = 0; ph < POOL_K; ++ph) {
                #pragma unroll
                for (uint32_t pw = 0; pw < POOL_K; ++pw) {
                    const uint32_t hc = hc0 + ph;
                    const uint32_t wc = wc0 + pw;

                    // conv output at (ni,co,hc,wc) with valid conv => input base at (hc,wc)
                    float acc = bias;

                    #pragma unroll
                    for (uint32_t ci = 0; ci < 64; ++ci) {
                        const uint64_t xBase =
                            ((static_cast<uint64_t>(ni) * 64ull + static_cast<uint64_t>(ci)) *
                             static_cast<uint64_t>(hin) + static_cast<uint64_t>(hc)) *
                             static_cast<uint64_t>(win) + static_cast<uint64_t>(wc);

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

                    // subtract1
                    acc = acc - sub1;

                    // tanh(acc) via exp: (1 - exp(-2x))/(1 + exp(-2x))
                    v.SetValue(0, acc); v.SetValue(1, acc); v.SetValue(2, acc); v.SetValue(3, acc);
                    AscendC::Muls(tmp, v, -2.0f, 4);
                    AscendC::Exp(ex, tmp, 4);
                    AscendC::Sub(num, one, ex, 4);
                    AscendC::Add(den, one, ex, 4);
                    AscendC::Div(t, num, den, 4);
                    float tanhV = t.GetValue(0);

                    // subtract2
                    tanhV = tanhV - sub2;

                    pool.SetValue(ph * POOL_K + pw, tanhV);
                }
            }

            float sum = 0.0f;
            sum += pool.GetValue(0);
            sum += pool.GetValue(1);
            sum += pool.GetValue(2);
            sum += pool.GetValue(3);

            yGm.SetValue(static_cast<uint64_t>(linear), sum * 0.25f);
        }

        qT.FreeTensor(t);
        qDen.FreeTensor(den);
        qNum.FreeTensor(num);
        qOne.FreeTensor(one);
        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qV.FreeTensor(v);
        qPool.FreeTensor(pool);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qPool;
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
    uint32_t hconv, wconv;
    uint32_t hout, wout;
    float sub1, sub2;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_subtract_tanh_subtract_avg_pool_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dSubtractTanhSubtractAvgPoolCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hconv, t.wconv,
            t.hout, t.wout,
            t.sub1, t.sub2,
            t.total_y, t.elems_per_block);
    op.Process();
}
