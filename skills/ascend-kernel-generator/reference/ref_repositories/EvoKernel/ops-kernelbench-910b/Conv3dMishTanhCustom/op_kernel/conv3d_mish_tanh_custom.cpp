
#include "kernel_operator.h"

// Fused activation for Conv3d output tensor:
// y = tanh(mish(x)) where mish(x) = x * tanh(softplus(x))
// softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
//
// float32 only; ND contiguous flattened.

class KernelConv3dMishTanhCustom {
public:
    __aicore__ inline KernelConv3dMishTanhCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t total_x,
                               uint32_t elems_per_block)
    {
        this->totalX = total_x;
        this->elemsPerBlock = elems_per_block;

        xGm.SetGlobalBuffer((__gm__ float*)x, static_cast<uint64_t>(total_x));
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(total_x));

        // UB scratch for 8-lane vector math
        pipe.InitBuffer(qV,   1, 8u * sizeof(float));
        pipe.InitBuffer(qAbs, 1, 8u * sizeof(float));
        pipe.InitBuffer(qTmp, 1, 8u * sizeof(float));
        pipe.InitBuffer(qEx,  1, 8u * sizeof(float));
        pipe.InitBuffer(qOne, 1, 8u * sizeof(float));
        pipe.InitBuffer(qNum, 1, 8u * sizeof(float));
        pipe.InitBuffer(qDen, 1, 8u * sizeof(float));
        pipe.InitBuffer(qSp,  1, 8u * sizeof(float));
        pipe.InitBuffer(qT,   1, 8u * sizeof(float));
        pipe.InitBuffer(qY,   1, 8u * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (end > totalX) end = totalX;
        if (start >= end) return;

        AscendC::LocalTensor<float> v   = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> a   = qAbs.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex  = qEx.AllocTensor<float>();
        AscendC::LocalTensor<float> one = qOne.AllocTensor<float>();
        AscendC::LocalTensor<float> num = qNum.AllocTensor<float>();
        AscendC::LocalTensor<float> den = qDen.AllocTensor<float>();
        AscendC::LocalTensor<float> sp  = qSp.AllocTensor<float>();
        AscendC::LocalTensor<float> t   = qT.AllocTensor<float>();
        AscendC::LocalTensor<float> yv  = qY.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, 8);

        uint32_t i = start;

        // Vector-8 path
        for (; (i + 8u) <= end; i += 8u) {
            // load 8 scalars
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                float xv = xGm.GetValue(static_cast<uint64_t>(i + k));
                v.SetValue(k, xv);
            }

            // ---- Mish(v) ----
            // a = abs(v)
            AscendC::Abs(a, v, 8);

            // ex = exp(-abs(v))
            AscendC::Muls(tmp, a, -1.0f, 8);
            AscendC::Exp(ex, tmp, 8);

            // tmp = log(1 + exp(-abs(v)))
            AscendC::Add(tmp, one, ex, 8);
            AscendC::Log(tmp, tmp, 8);

            // sp = max(v, 0)
            AscendC::Duplicate(sp, 0.0f, 8);
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                float xv = v.GetValue(k);
                sp.SetValue(k, (xv > 0.0f) ? xv : 0.0f);
            }

            // sp = softplus(v)
            AscendC::Add(sp, sp, tmp, 8);

            // t = tanh(softplus): (1 - exp(-2sp)) / (1 + exp(-2sp))
            AscendC::Muls(tmp, sp, -2.0f, 8);
            AscendC::Exp(ex, tmp, 8);
            AscendC::Sub(num, one, ex, 8);
            AscendC::Add(den, one, ex, 8);
            AscendC::Div(t, num, den, 8);

            // yv = mish = v * t
            AscendC::Mul(yv, v, t, 8);

            // ---- tanh(yv) ----
            // reuse tmp/ex/num/den:
            // tanh(z) = (1 - exp(-2z)) / (1 + exp(-2z))
            AscendC::Muls(tmp, yv, -2.0f, 8);
            AscendC::Exp(ex, tmp, 8);
            AscendC::Sub(num, one, ex, 8);
            AscendC::Add(den, one, ex, 8);
            AscendC::Div(yv, num, den, 8);

            // store 8
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                yGm.SetValue(static_cast<uint64_t>(i + k), yv.GetValue(k));
            }
        }

        // Scalar tail (reuse lane0 of tensors, computed with 4 ops replicated)
        for (; i < end; ++i) {
            const float x = xGm.GetValue(static_cast<uint64_t>(i));

            // pack scalar into lanes 0..3 and compute with count=4
            // (use first 4 lanes of existing 8-lane tensors)
            v.SetValue(0, x); v.SetValue(1, x); v.SetValue(2, x); v.SetValue(3, x);
            AscendC::Duplicate(one, 1.0f, 4);

            AscendC::Abs(a, v, 4);
            AscendC::Muls(tmp, a, -1.0f, 4);
            AscendC::Exp(ex, tmp, 4);
            AscendC::Add(tmp, one, ex, 4);
            AscendC::Log(tmp, tmp, 4);

            AscendC::Duplicate(sp, 0.0f, 4);
            float mx = (x > 0.0f) ? x : 0.0f;
            sp.SetValue(0, mx); sp.SetValue(1, mx); sp.SetValue(2, mx); sp.SetValue(3, mx);

            AscendC::Add(sp, sp, tmp, 4);

            AscendC::Muls(tmp, sp, -2.0f, 4);
            AscendC::Exp(ex, tmp, 4);
            AscendC::Sub(num, one, ex, 4);
            AscendC::Add(den, one, ex, 4);
            AscendC::Div(t, num, den, 4);

            AscendC::Mul(yv, v, t, 4);

            AscendC::Muls(tmp, yv, -2.0f, 4);
            AscendC::Exp(ex, tmp, 4);
            AscendC::Sub(num, one, ex, 4);
            AscendC::Add(den, one, ex, 4);
            AscendC::Div(yv, num, den, 4);

            yGm.SetValue(static_cast<uint64_t>(i), yv.GetValue(0));
        }

        qY.FreeTensor(yv);
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
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qY;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalX;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv3d_mish_tanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv3dMishTanhCustom op;
    op.Init(x, y, t.total_x, t.elems_per_block);
    op.Process();
}
