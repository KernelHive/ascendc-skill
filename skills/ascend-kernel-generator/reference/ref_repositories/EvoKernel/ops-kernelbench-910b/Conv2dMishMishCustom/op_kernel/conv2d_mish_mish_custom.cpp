
#include "kernel_operator.h"

// Fused operator (float32 only), specialized:
// x: [64,64,256,256]
// weight: [128,64,3,3], bias: [128]
// y: [64,128,254,254]
// y = mish(mish(conv(x,w,b)))
//
// Optimizations this round:
// - Row-preserving 16-wide processing with variable span (<=16) to avoid scalar fallback.
// - Decode (n,co,ho,wo) once, then advance along wo within the same row when possible.
// - Mish2 implemented as two consecutive in-place Mish calls on the same 16-lane UB tensors.
// - Slightly higher blockDim from tiling for better occupancy (still conservative).

class KernelConv2dMishMishCustom {
public:
    __aicore__ inline KernelConv2dMishMishCustom() {}

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
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        constexpr uint32_t L = 16;
        pipe.InitBuffer(qV,    1, L * sizeof(float));
        pipe.InitBuffer(qAbs,  1, L * sizeof(float));
        pipe.InitBuffer(qTmp,  1, L * sizeof(float));
        pipe.InitBuffer(qEx,   1, L * sizeof(float));
        pipe.InitBuffer(qOne,  1, L * sizeof(float));
        pipe.InitBuffer(qNum,  1, L * sizeof(float));
        pipe.InitBuffer(qDen,  1, L * sizeof(float));
        pipe.InitBuffer(qSp,   1, L * sizeof(float));
        pipe.InitBuffer(qT,    1, L * sizeof(float));
        pipe.InitBuffer(qZero, 1, L * sizeof(float));
    }

    __aicore__ inline void MishInplace16(AscendC::LocalTensor<float>& v,
                                        AscendC::LocalTensor<float>& a,
                                        AscendC::LocalTensor<float>& tmp,
                                        AscendC::LocalTensor<float>& ex,
                                        AscendC::LocalTensor<float>& one,
                                        AscendC::LocalTensor<float>& num,
                                        AscendC::LocalTensor<float>& den,
                                        AscendC::LocalTensor<float>& sp,
                                        AscendC::LocalTensor<float>& t,
                                        AscendC::LocalTensor<float>& z)
    {
        // softplus(x)=max(x,0)+log(1+exp(-abs(x)))
        AscendC::Abs(a, v, 16);
        AscendC::Muls(tmp, a, -1.0f, 16);
        AscendC::Exp(ex, tmp, 16);
        AscendC::Add(tmp, one, ex, 16);
        AscendC::Log(tmp, tmp, 16);
        AscendC::Max(sp, v, z, 16);
        AscendC::Add(sp, sp, tmp, 16);

        // tanh(softplus) = (1-exp(-2sp))/(1+exp(-2sp))
        AscendC::Muls(tmp, sp, -2.0f, 16);
        AscendC::Exp(ex, tmp, 16);
        AscendC::Sub(num, one, ex, 16);
        AscendC::Add(den, one, ex, 16);
        AscendC::Div(t, num, den, 16);

        AscendC::Mul(v, v, t, 16); // in-place mish
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
        constexpr uint32_t L  = 16;

        const uint32_t hwOut  = hout * wout;      // 254*254
        const uint32_t chwOut = cout * hwOut;     // 128*hwOut

        AscendC::LocalTensor<float> v    = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> a    = qAbs.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex   = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one  = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num  = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den  = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> sp   = qSp.AllocTensor<float>();
        AscendC::LocalTensor<float> t    = qT.AllocTensor<float>();
        AscendC::LocalTensor<float> z    = qZero.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, L);
        AscendC::Duplicate(z,   0.0f, L);

        uint32_t linear = start;

        while (linear < end) {
            // Decode once for current linear
            uint32_t tt0 = linear;
            const uint32_t ni = tt0 / chwOut;
            tt0 -= ni * chwOut;
            const uint32_t co = tt0 / hwOut;
            tt0 -= co * hwOut;
            const uint32_t ho = tt0 / wout;
            const uint32_t wo0 = tt0 - ho * wout;

            // Prefer vector path always: take span <= 16 without crossing row end or block end.
            uint32_t span = L;
            const uint32_t remainRow = wout - wo0;
            if (span > remainRow) span = remainRow;
            const uint32_t remainBlk = end - linear;
            if (span > remainBlk) span = remainBlk;

            float acc[L];
            const float bias = bGm.GetValue(static_cast<uint64_t>(co));
            #pragma unroll
            for (uint32_t i = 0; i < L; ++i) acc[i] = bias;

            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (static_cast<uint64_t>(cin) * KH * KW);

            for (uint32_t ci = 0; ci < cin; ++ci) {
                const uint64_t xBase0 =
                    ((static_cast<uint64_t>(ni) * static_cast<uint64_t>(cin) + static_cast<uint64_t>(ci)) *
                     static_cast<uint64_t>(hin) + static_cast<uint64_t>(ho)) *
                    static_cast<uint64_t>(win) + static_cast<uint64_t>(wo0);

                const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                #pragma unroll
                for (uint32_t kH = 0; kH < KH; ++kH) {
                    const uint64_t xRow0 = xBase0 + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                    const uint64_t wRow  = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);

                    const float w0 = wGm.GetValue(wRow + 0ull);
                    const float w1 = wGm.GetValue(wRow + 1ull);
                    const float w2 = wGm.GetValue(wRow + 2ull);

                    #pragma unroll
                    for (uint32_t i = 0; i < L; ++i) {
                        const uint64_t base = xRow0 + static_cast<uint64_t>(i);
                        const float x0 = xGm.GetValue(base + 0ull);
                        const float x1 = xGm.GetValue(base + 1ull);
                        const float x2 = xGm.GetValue(base + 2ull);
                        acc[i] += x0 * w0 + x1 * w1 + x2 * w2;
                    }
                }
            }

            // Write to UB, pad unused lanes with 0 to keep vector math deterministic
            #pragma unroll
            for (uint32_t i = 0; i < L; ++i) {
                if (i < span) v.SetValue(i, acc[i]);
                else v.SetValue(i, 0.0f);
            }

            // Two Mish applications in-place
            MishInplace16(v, a, tmp, ex, one, num, den, sp, t, z);
            MishInplace16(v, a, tmp, ex, one, num, den, sp, t, z);

            // Store only valid lanes
            const uint64_t outBase = static_cast<uint64_t>(linear);
            #pragma unroll
            for (uint32_t i = 0; i < L; ++i) {
                if (i < span) {
                    yGm.SetValue(outBase + static_cast<uint64_t>(i), v.GetValue(i));
                }
            }

            linear += span;
        }

        qZero.FreeTensor(z);
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
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_mish_mish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dMishMishCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.total_y, t.elems_per_block);
    op.Process();
}
