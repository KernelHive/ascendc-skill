
#include "kernel_operator.h"

// Specialized fused operator (float32 only):
// x:      [M=1024, K=8192]
// weight: [N=8192, K=8192]  (PyTorch Linear weight [out,in])
// bias:   [N=8192]
// scaling/clamp_min/clamp_max: [1]
// y:      [M,1]
//
// Reference:
//   t = x @ weight^T + bias                 # [M,N]
//   t = t * scaling
//   t = t + t                               # == t * 2
//   t = clamp(t, clamp_min, clamp_max)
//   u = logsumexp(t, dim=1, keepdim=True)   # [M,1]
//   y = u * mish(u)                         # [M,1]
//
// Note: overall scale = 2*scaling.

class KernelMatmulScaleResidualAddClampLogSumExpMishCustom {
public:
    __aicore__ inline KernelMatmulScaleResidualAddClampLogSumExpMishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                               GM_ADDR scaling, GM_ADDR clampMin, GM_ADDR clampMax,
                               GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t total_y, uint32_t rows_per_block)
    {
        M_ = M; K_ = K; N_ = N;
        totalY_ = total_y;
        rowsPerBlock_ = rows_per_block;

        const uint64_t xSize = static_cast<uint64_t>(M_) * static_cast<uint64_t>(K_);
        const uint64_t wSize = static_cast<uint64_t>(N_) * static_cast<uint64_t>(K_);
        const uint64_t bSize = static_cast<uint64_t>(N_);
        const uint64_t ySize = static_cast<uint64_t>(M_); // [M]

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm_.SetGlobalBuffer((__gm__ float*)b, bSize);
        sGm_.SetGlobalBuffer((__gm__ float*)scaling, 1);
        cminGm_.SetGlobalBuffer((__gm__ float*)clampMin, 1);
        cmaxGm_.SetGlobalBuffer((__gm__ float*)clampMax, 1);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB scratch (4-lane) for Exp/Log and Mish
        pipe_.InitBuffer(qV_,   1, 4u * sizeof(float));
        pipe_.InitBuffer(qAbs_, 1, 4u * sizeof(float));
        pipe_.InitBuffer(qTmp_, 1, 4u * sizeof(float));
        pipe_.InitBuffer(qEx_,  1, 4u * sizeof(float));
        pipe_.InitBuffer(qOne_, 1, 4u * sizeof(float));
        pipe_.InitBuffer(qNum_, 1, 4u * sizeof(float));
        pipe_.InitBuffer(qDen_, 1, 4u * sizeof(float));
        pipe_.InitBuffer(qSp_,  1, 4u * sizeof(float));
        pipe_.InitBuffer(qT_,   1, 4u * sizeof(float));

        const float scale = sGm_.GetValue(0);
        fusedMul_ = 2.0f * scale;
        clampMin_ = cminGm_.GetValue(0);
        clampMax_ = cmaxGm_.GetValue(0);
    }

    __aicore__ inline float clampf(float v) const
    {
        if (v < clampMin_) return clampMin_;
        if (v > clampMax_) return clampMax_;
        return v;
    }

    __aicore__ inline void ExpScalarTo4(float x, const AscendC::LocalTensor<float>& tmp, const AscendC::LocalTensor<float>& ex)
    {
        tmp.SetValue(0, x); tmp.SetValue(1, x); tmp.SetValue(2, x); tmp.SetValue(3, x);
        AscendC::Exp(ex, tmp, 4);
    }

    __aicore__ inline float LogScalar(float x,
                                     const AscendC::LocalTensor<float>& tmp,
                                     const AscendC::LocalTensor<float>& ex)
    {
        ex.SetValue(0, x); ex.SetValue(1, x); ex.SetValue(2, x); ex.SetValue(3, x);
        AscendC::Log(tmp, ex, 4);
        return tmp.GetValue(0);
    }

    __aicore__ inline void MishInplace4(const AscendC::LocalTensor<float>& v,
                                        const AscendC::LocalTensor<float>& a,
                                        const AscendC::LocalTensor<float>& tmp,
                                        const AscendC::LocalTensor<float>& ex,
                                        const AscendC::LocalTensor<float>& one,
                                        const AscendC::LocalTensor<float>& num,
                                        const AscendC::LocalTensor<float>& den,
                                        const AscendC::LocalTensor<float>& sp,
                                        const AscendC::LocalTensor<float>& t)
    {
        // a = abs(v)
        AscendC::Abs(a, v, 4);

        // ex = exp(-abs(v))
        AscendC::Muls(tmp, a, -1.0f, 4);
        AscendC::Exp(ex, tmp, 4);

        // tmp = log(1 + ex)
        AscendC::Add(tmp, one, ex, 4);
        AscendC::Log(tmp, tmp, 4);

        // sp = max(v, 0) (scalar per lane)
        AscendC::Duplicate(sp, 0.0f, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            float xv = v.GetValue(i);
            sp.SetValue(i, (xv > 0.0f) ? xv : 0.0f);
        }

        // sp = softplus(v)
        AscendC::Add(sp, sp, tmp, 4);

        // t = tanh(sp) = (1 - exp(-2sp)) / (1 + exp(-2sp))
        AscendC::Muls(tmp, sp, -2.0f, 4);
        AscendC::Exp(ex, tmp, 4);
        AscendC::Sub(num, one, ex, 4);
        AscendC::Add(den, one, ex, 4);
        AscendC::Div(t, num, den, 4);

        // v = v * t
        AscendC::Mul(tmp, v, t, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            v.SetValue(i, tmp.GetValue(i));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t mStart = bid * rowsPerBlock_;
        uint32_t mEnd = mStart + rowsPerBlock_;
        if (mEnd > M_) mEnd = M_;
        if (mStart >= mEnd) return;

        AscendC::LocalTensor<float> v   = qV_.AllocTensor<float>();
        AscendC::LocalTensor<float> a   = qAbs_.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
        AscendC::LocalTensor<float> ex  = qEx_.AllocTensor<float>();
        AscendC::LocalTensor<float> one = qOne_.AllocTensor<float>();
        AscendC::LocalTensor<float> num = qNum_.AllocTensor<float>();
        AscendC::LocalTensor<float> den = qDen_.AllocTensor<float>();
        AscendC::LocalTensor<float> sp  = qSp_.AllocTensor<float>();
        AscendC::LocalTensor<float> t   = qT_.AllocTensor<float>();

        AscendC::Duplicate(one, 1.0f, 4);

        for (uint32_t m = mStart; m < mEnd; ++m) {
            const uint64_t xBase = static_cast<uint64_t>(m) * static_cast<uint64_t>(K_);

            // Pass 1: max over n
            float maxv = -3.402823466e+38f;
            for (uint32_t n = 0; n < N_; ++n) {
                float acc = bGm_.GetValue(static_cast<uint64_t>(n));
                const uint64_t wBase = static_cast<uint64_t>(n) * static_cast<uint64_t>(K_);
                for (uint32_t k = 0; k < K_; ++k) {
                    const float xv = xGm_.GetValue(xBase + static_cast<uint64_t>(k));
                    const float wv = wGm_.GetValue(wBase + static_cast<uint64_t>(k));
                    acc += xv * wv;
                }
                acc *= fusedMul_;
                acc = clampf(acc);
                if (acc > maxv) maxv = acc;
            }

            // Pass 2: sum exp(val - maxv) using tensor Exp on a replicated scalar
            float sumExp = 0.0f;
            for (uint32_t n = 0; n < N_; ++n) {
                float acc = bGm_.GetValue(static_cast<uint64_t>(n));
                const uint64_t wBase = static_cast<uint64_t>(n) * static_cast<uint64_t>(K_);
                for (uint32_t k = 0; k < K_; ++k) {
                    const float xv = xGm_.GetValue(xBase + static_cast<uint64_t>(k));
                    const float wv = wGm_.GetValue(wBase + static_cast<uint64_t>(k));
                    acc += xv * wv;
                }
                acc *= fusedMul_;
                acc = clampf(acc);

                ExpScalarTo4(acc - maxv, tmp, ex);
                sumExp += ex.GetValue(0);
            }

            // lse = maxv + log(sumExp)
            float lse = maxv + LogScalar(sumExp, tmp, ex);

            // out = lse * mish(lse)
            v.SetValue(0, lse); v.SetValue(1, lse); v.SetValue(2, lse); v.SetValue(3, lse);
            MishInplace4(v, a, tmp, ex, one, num, den, sp, t);
            float out = lse * v.GetValue(0);

            yGm_.SetValue(static_cast<uint64_t>(m), out);
        }

        qT_.FreeTensor(t);
        qSp_.FreeTensor(sp);
        qDen_.FreeTensor(den);
        qNum_.FreeTensor(num);
        qOne_.FreeTensor(one);
        qEx_.FreeTensor(ex);
        qTmp_.FreeTensor(tmp);
        qAbs_.FreeTensor(a);
        qV_.FreeTensor(v);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qAbs_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qNum_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qDen_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qSp_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qT_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> cminGm_;
    AscendC::GlobalTensor<float> cmaxGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalY_{0};
    uint32_t rowsPerBlock_{1};

    float fusedMul_{0.0f};
    float clampMin_{0.0f};
    float clampMax_{0.0f};
};

extern "C" __global__ __aicore__ void matmul_scale_residual_add_clamp_log_sum_exp_mish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR scaling, GM_ADDR clamp_min, GM_ADDR clamp_max,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelMatmulScaleResidualAddClampLogSumExpMishCustom op;
    op.Init(x, weight, bias, scaling, clamp_min, clamp_max, y,
            t.M, t.K, t.N, t.total_y, t.rows_per_block);
    op.Process();
}
