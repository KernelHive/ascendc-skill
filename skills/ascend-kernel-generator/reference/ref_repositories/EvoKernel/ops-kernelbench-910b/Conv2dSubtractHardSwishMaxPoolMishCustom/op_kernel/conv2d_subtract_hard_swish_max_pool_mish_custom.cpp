
#include "kernel_operator.h"

// Specialized fused operator (float32 only):
// x: [N=128,Cin=64,H=W=128]
// weight: [Cout=128,Cin=64,Kh=3,Kw=3], stride=1,pad=0 => conv [128,128,126,126]
// subtract scalar 0.5 -> HardSwish -> MaxPool2d(k=2,s=2) => [128,128,63,63] -> Mish
//
// Mish implemented as x * tanh(softplus(x)) using vector Exp/Log/Div only:
// softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
// tanh(z) = (1 - exp(-2z)) / (1 + exp(-2z))

class KernelConv2dSubtractHardSwishMaxPoolMishCustom {
public:
    __aicore__ inline KernelConv2dSubtractHardSwishMaxPoolMishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               float sub_value,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->sub_value = sub_value;
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

        // UB scratch (tiny fixed):
        pipe.InitBuffer(qPool, 1, 4u * sizeof(float));
        pipe.InitBuffer(qV,    1, 4u * sizeof(float));
        pipe.InitBuffer(qAbs,  1, 4u * sizeof(float));
        pipe.InitBuffer(qTmp,  1, 4u * sizeof(float));
        pipe.InitBuffer(qEx,   1, 4u * sizeof(float));
        pipe.InitBuffer(qOne,  1, 4u * sizeof(float));
        pipe.InitBuffer(qNum,  1, 4u * sizeof(float));
        pipe.InitBuffer(qDen,  1, 4u * sizeof(float));
        pipe.InitBuffer(qSp,   1, 4u * sizeof(float));
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

        const uint32_t hwOut = hout * wout;      // 3969
        const uint32_t chwOut = cout * hwOut;    // 128*3969

        AscendC::LocalTensor<float> pool = qPool.AllocTensor<float>();
        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> a    = qAbs.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex   = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one  = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num  = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den  = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> sp   = qSp.AllocTensor<float>();
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

            // pool base in conv output space
            const uint32_t hc0 = ho * POOL_S; // 0..124
            const uint32_t wc0 = wo * POOL_S; // 0..124

            const float bias = bGm.GetValue(static_cast<uint64_t>(co));
            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (64ull * KH * KW);

            // Compute 4 conv points for the 2x2 pool window, fuse subtract+HardSwish
            #pragma unroll
            for (uint32_t ph = 0; ph < POOL_K; ++ph) {
                #pragma unroll
                for (uint32_t pw = 0; pw < POOL_K; ++pw) {
                    const uint32_t hc = hc0 + ph;
                    const uint32_t wc = wc0 + pw;
                    float acc = bias;

                    #pragma unroll
                    for (uint32_t ci = 0; ci < 64; ++ci) {
                        const uint64_t xBaseC =
                            ((static_cast<uint64_t>(ni) * 64ull + static_cast<uint64_t>(ci)) *
                             static_cast<uint64_t>(hin) + static_cast<uint64_t>(hc)) * static_cast<uint64_t>(win) +
                            static_cast<uint64_t>(wc);

                        const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                        #pragma unroll
                        for (uint32_t kH = 0; kH < KH; ++kH) {
                            const uint64_t xRow = xBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                            const uint64_t wRow = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);
                            #pragma unroll
                            for (uint32_t kW = 0; kW < KW; ++kW) {
                                const float xv = xGm.GetValue(xRow + static_cast<uint64_t>(kW));
                                const float wv = wGm.GetValue(wRow + static_cast<uint64_t>(kW));
                                acc += xv * wv;
                            }
                        }
                    }

                    // subtract scalar
                    acc = acc - sub_value;

                    // HardSwish: x * clamp(x+3,0,6)/6
                    float tcl = acc + 3.0f;
                    if (tcl < 0.0f) tcl = 0.0f;
                    if (tcl > 6.0f) tcl = 6.0f;
                    acc = acc * (tcl * (1.0f / 6.0f));

                    pool.SetValue(ph * POOL_K + pw, acc);
                }
            }

            // Max over 2x2
            float vmax = pool.GetValue(0);
            float v1 = pool.GetValue(1); vmax = (v1 > vmax) ? v1 : vmax;
            float v2 = pool.GetValue(2); vmax = (v2 > vmax) ? v2 : vmax;
            float v3 = pool.GetValue(3); vmax = (v3 > vmax) ? v3 : vmax;

            // Mish on scalar vmax via vector ops on 4 lanes
            v.SetValue(0, vmax); v.SetValue(1, vmax); v.SetValue(2, vmax); v.SetValue(3, vmax);

            // abs(x)
            AscendC::Abs(a, v, 4);

            // exp(-abs(x)) in ex
            AscendC::Muls(tmp, a, -1.0f, 4);
            AscendC::Exp(ex, tmp, 4);

            // log(1 + exp(-abs(x))) in tmp
            AscendC::Add(tmp, one, ex, 4);
            AscendC::Log(tmp, tmp, 4);

            // sp = max(x,0) (scalar compare per lane)
            AscendC::Duplicate(sp, 0.0f, 4);
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                float xv = v.GetValue(i);
                sp.SetValue(i, (xv > 0.0f) ? xv : 0.0f);
            }

            // softplus
            AscendC::Add(sp, sp, tmp, 4);

            // tanh(softplus): t = (1 - exp(-2sp)) / (1 + exp(-2sp))
            AscendC::Muls(tmp, sp, -2.0f, 4);
            AscendC::Exp(ex, tmp, 4);
            AscendC::Sub(num, one, ex, 4);
            AscendC::Add(den, one, ex, 4);
            AscendC::Div(t, num, den, 4);

            // out = x * tanh(softplus(x))
            AscendC::Mul(tmp, v, t, 4);
            yGm.SetValue(static_cast<uint64_t>(linear), tmp.GetValue(0));
        }

        qT.FreeTensor(t);
        qSp.FreeTensor(sp);
        qDen.FreeTensor(den);
        qNum.FreeTensor(num);
        qOne.FreeTensor(one);
        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qAbs.FreeTensor(a);
        qV.FreeTensor(v);
        qPool.FreeTensor(pool);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qPool;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qAbs;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qNum;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qDen;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qSp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    float sub_value;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_subtract_hard_swish_max_pool_mish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dSubtractHardSwishMaxPoolMishCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.sub_value,
            t.total_y, t.elems_per_block);
    op.Process();
}
