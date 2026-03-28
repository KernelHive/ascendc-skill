
#include "kernel_operator.h"

// Fused operator (float32 only), specialized:
// x:      [M=1024, K=8192]
// weight: [N=8192, K=8192]  (PyTorch Linear weight [out,in])
// bias:   [N=8192]
// y:      [M=1024, N=8192]
//
// y = mish(mish(x @ weight^T + bias))
//
// Optimization in this round:
// - Keep scalar matmul (avoid strided gather tiling failure).
// - Process 4 independent outputs at once and run mish+mish as pure vector ops on 4 lanes.
// - No scalar lane GetValue/SetValue mixing on VECCALC tensors except initial pack/unpack.
// - Branch-free max(x,0) = 0.5*(x + abs(x)) to cut scalar/control pressure.

class KernelMatmulMishMishCustom {
public:
    __aicore__ inline KernelMatmulMishMishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t total_elems, uint32_t elems_per_block)
    {
        M_ = M; K_ = K; N_ = N;
        totalElems_ = total_elems;
        elemsPerBlock_ = elems_per_block;

        const uint64_t xSize = static_cast<uint64_t>(M_) * static_cast<uint64_t>(K_);
        const uint64_t wSize = static_cast<uint64_t>(N_) * static_cast<uint64_t>(K_);
        const uint64_t bSize = static_cast<uint64_t>(N_);
        const uint64_t ySize = static_cast<uint64_t>(M_) * static_cast<uint64_t>(N_);

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm_.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        // 4-lane UB scratch
        constexpr uint32_t L = 4;
        pipe_.InitBuffer(qV_,   1, L * sizeof(float));
        pipe_.InitBuffer(qAbs_, 1, L * sizeof(float));
        pipe_.InitBuffer(qTmp_, 1, L * sizeof(float));
        pipe_.InitBuffer(qEx_,  1, L * sizeof(float));
        pipe_.InitBuffer(qOne_, 1, L * sizeof(float));
        pipe_.InitBuffer(qNum_, 1, L * sizeof(float));
        pipe_.InitBuffer(qDen_, 1, L * sizeof(float));
        pipe_.InitBuffer(qSp_,  1, L * sizeof(float));
        pipe_.InitBuffer(qT_,   1, L * sizeof(float));
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

        // sp = max(v,0) = 0.5*(v + abs(v))  (branch-free vector)
        AscendC::Add(sp, v, a, 4);
        AscendC::Muls(sp, sp, 0.5f, 4);

        // sp = softplus(v)
        AscendC::Add(sp, sp, tmp, 4);

        // t = tanh(sp) via exp(-2sp): (1 - e) / (1 + e), e=exp(-2sp)
        AscendC::Muls(tmp, sp, -2.0f, 4);
        AscendC::Exp(ex, tmp, 4);
        AscendC::Sub(num, one, ex, 4);
        AscendC::Add(den, one, ex, 4);
        AscendC::Div(t, num, den, 4);

        // v = v * t
        AscendC::Mul(v, v, t, 4);
    }

    __aicore__ inline float Dot(uint32_t m, uint32_t n)
    {
        float acc = bGm_.GetValue(static_cast<uint64_t>(n));
        const uint64_t xBase = static_cast<uint64_t>(m) * static_cast<uint64_t>(K_);
        const uint64_t wBase = static_cast<uint64_t>(n) * static_cast<uint64_t>(K_);
        for (uint32_t k = 0; k < K_; ++k) {
            const float xv = xGm_.GetValue(xBase + static_cast<uint64_t>(k));
            const float wv = wGm_.GetValue(wBase + static_cast<uint64_t>(k));
            acc += xv * wv;
        }
        return acc;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = bid * elemsPerBlock_;
        uint32_t end = start + elemsPerBlock_;
        if (end > totalElems_) end = totalElems_;
        if (start >= end) return;

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

        uint32_t outIdx = start;
        while (outIdx < end) {
            // Pack up to 4 outputs into lanes
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            const uint32_t idx0 = outIdx;
            const uint32_t idx1 = (outIdx + 1 < end) ? (outIdx + 1) : idx0;
            const uint32_t idx2 = (outIdx + 2 < end) ? (outIdx + 2) : idx0;
            const uint32_t idx3 = (outIdx + 3 < end) ? (outIdx + 3) : idx0;

            const uint32_t m0 = idx0 / N_; const uint32_t n0 = idx0 - m0 * N_;
            const uint32_t m1 = idx1 / N_; const uint32_t n1 = idx1 - m1 * N_;
            const uint32_t m2 = idx2 / N_; const uint32_t n2 = idx2 - m2 * N_;
            const uint32_t m3 = idx3 / N_; const uint32_t n3 = idx3 - m3 * N_;

            acc0 = Dot(m0, n0);
            acc1 = Dot(m1, n1);
            acc2 = Dot(m2, n2);
            acc3 = Dot(m3, n3);

            v.SetValue(0, acc0);
            v.SetValue(1, acc1);
            v.SetValue(2, acc2);
            v.SetValue(3, acc3);

            MishInplace4(v, a, tmp, ex, one, num, den, sp, t);
            MishInplace4(v, a, tmp, ex, one, num, den, sp, t);

            yGm_.SetValue(static_cast<uint64_t>(idx0), v.GetValue(0));
            if (outIdx + 1 < end) yGm_.SetValue(static_cast<uint64_t>(idx1), v.GetValue(1));
            if (outIdx + 2 < end) yGm_.SetValue(static_cast<uint64_t>(idx2), v.GetValue(2));
            if (outIdx + 3 < end) yGm_.SetValue(static_cast<uint64_t>(idx3), v.GetValue(3));

            outIdx += 4;
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
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    uint32_t elemsPerBlock_{1};
};

extern "C" __global__ __aicore__ void matmul_mish_mish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelMatmulMishMishCustom op;
    op.Init(x, weight, bias, y, t.M, t.K, t.N, t.total_elems, t.elems_per_block);
    op.Process();
}
