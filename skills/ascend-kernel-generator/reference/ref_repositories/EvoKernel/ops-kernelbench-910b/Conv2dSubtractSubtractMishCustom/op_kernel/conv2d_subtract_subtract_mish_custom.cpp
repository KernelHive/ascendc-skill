
#include "kernel_operator.h"

// Fused specialized operator (float32 only):
// x: [128,8,256,256], w: [64,8,3,3], b: [64] -> y: [128,64,254,254]
// y = mish(conv(x,w,b) - 0.5 - 0.2)
//
// This version computes 16 outputs (contiguous wo) per iteration and runs Mish epilogue
// as 16-lane vector ops to reduce epilogue overhead and scalar/vector switching.

class KernelConv2dSubtractSubtractMishCustom {
public:
    __aicore__ inline KernelConv2dSubtractSubtractMishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               float sub1, float sub2,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->subSum = sub1 + sub2;
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

        // 16-lane UB scratch for Mish
        pipe.InitBuffer(qV,   1, 16u * sizeof(float));
        pipe.InitBuffer(qAbs, 1, 16u * sizeof(float));
        pipe.InitBuffer(qTmp, 1, 16u * sizeof(float));
        pipe.InitBuffer(qEx,  1, 16u * sizeof(float));
        pipe.InitBuffer(qOne, 1, 16u * sizeof(float));
        pipe.InitBuffer(qNum, 1, 16u * sizeof(float));
        pipe.InitBuffer(qDen, 1, 16u * sizeof(float));
        pipe.InitBuffer(qSp,  1, 16u * sizeof(float));
        pipe.InitBuffer(qT,   1, 16u * sizeof(float));
        pipe.InitBuffer(qZero,1, 16u * sizeof(float));
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

        const uint32_t hwOut = hout * wout;      // 254*254
        const uint32_t chwOut = cout * hwOut;    // 64*hwOut

        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> a    = qAbs.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex   = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one  = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num  = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den  = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> sp   = qSp.AllocTensor<float>();
        AscendC::LocalTensor<float> t    = qT.AllocTensor<float>();
        AscendC::LocalTensor<float> zero = qZero.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, 16);
        AscendC::Duplicate(zero, 0.0f, 16);

        uint32_t linear = start;
        while (linear < end) {
            // decode linear -> (n,co,ho,wo)
            uint32_t tt = linear;
            const uint32_t ni = tt / chwOut;
            tt -= ni * chwOut;
            const uint32_t co = tt / hwOut;
            tt -= co * hwOut;
            const uint32_t ho = tt / wout;
            const uint32_t wo0 = tt - ho * wout;

            // process up to 16 contiguous wo, but do not cross row end or block end
            uint32_t span = 16;
            const uint32_t remainRow = wout - wo0;
            if (span > remainRow) span = remainRow;
            const uint32_t remainBlock = end - linear;
            if (span > remainBlock) span = remainBlock;

            // If span is small, still use vector path but pad lanes with zero and avoid stores.
            float acc0[16];
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) acc0[i] = 0.0f;

            const float bias = bGm.GetValue(static_cast<uint64_t>(co));
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                acc0[i] = bias;
            }

            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (8ull * KH * KW);

            #pragma unroll
            for (uint32_t ci = 0; ci < 8; ++ci) {
                const uint64_t xBase0 =
                    ((static_cast<uint64_t>(ni) * 8ull + static_cast<uint64_t>(ci)) *
                     static_cast<uint64_t>(hin) + static_cast<uint64_t>(ho)) *
                     static_cast<uint64_t>(win) + static_cast<uint64_t>(wo0);

                const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                #pragma unroll
                for (uint32_t kH = 0; kH < KH; ++kH) {
                    const uint64_t xRow0 = xBase0 + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                    const uint64_t wRow  = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);

                    // kW=0..2
                    const float wv0 = wGm.GetValue(wRow + 0ull);
                    const float wv1 = wGm.GetValue(wRow + 1ull);
                    const float wv2 = wGm.GetValue(wRow + 2ull);

                    // For each lane i, read x at (wo0+i + kW)
                    // Keep scalar loads but reduce address arithmetic.
                    #pragma unroll
                    for (uint32_t i = 0; i < 16; ++i) {
                        const uint64_t base = xRow0 + static_cast<uint64_t>(i);
                        const float xv0 = xGm.GetValue(base + 0ull);
                        const float xv1 = xGm.GetValue(base + 1ull);
                        const float xv2 = xGm.GetValue(base + 2ull);
                        acc0[i] += xv0 * wv0 + xv1 * wv1 + xv2 * wv2;
                    }
                }
            }

            // load to vector and subtract
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                v.SetValue(i, acc0[i] - subSum);
            }

            // Mish:
            // softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
            AscendC::Abs(a, v, 16);
            AscendC::Muls(tmp, a, -1.0f, 16);
            AscendC::Exp(ex, tmp, 16);
            AscendC::Add(tmp, one, ex, 16);
            AscendC::Log(tmp, tmp, 16);
            AscendC::Max(sp, v, zero, 16);
            AscendC::Add(sp, sp, tmp, 16);

            // tanh(sp) = (1 - exp(-2sp)) / (1 + exp(-2sp))
            AscendC::Muls(tmp, sp, -2.0f, 16);
            AscendC::Exp(ex, tmp, 16);
            AscendC::Sub(num, one, ex, 16);
            AscendC::Add(den, one, ex, 16);
            AscendC::Div(t, num, den, 16);

            AscendC::Mul(tmp, v, t, 16);

            // store only valid lanes
            const uint64_t outBase = static_cast<uint64_t>(linear);
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                if (i < span) {
                    yGm.SetValue(outBase + static_cast<uint64_t>(i), tmp.GetValue(i));
                }
            }

            linear += span;
        }

        qZero.FreeTensor(zero);
        qT.FreeTensor(t);
        qSp.FreeTensor(sp);
        qDen.FreeTensor(den);
        qNum.FreeTensor(num);
        qOne.FreeTensor(one);
        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qAbs.FreeTensor(a);
        qV.FreeTensor(v);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qAbs;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qNum;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qDen;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qSp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qZero;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    float subSum;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_subtract_subtract_mish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dSubtractSubtractMishCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.sub1, t.sub2,
            t.total_y, t.elems_per_block);
    op.Process();
}
