
#include "kernel_operator.h"

constexpr float NEG_INF_F = -3.402823466e+38f; // -FLT_MAX

// Mean cross entropy loss:
// loss_i = logsumexp(logits_i) - logits_i[target_i]
// Uses a one-pass, numerically stable online logsumexp to avoid reading logits twice.
class KernelCrossEntropyLoss {
public:
    __aicore__ inline KernelCrossEntropyLoss() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR target, GM_ADDR y,
                               uint32_t N_, uint32_t C_, uint32_t tileC_,
                               uint32_t tilesPerRow_, float invN_)
    {
        N = N_;
        C = C_;
        tileC = tileC_;
        tilesPerRow = tilesPerRow_;
        invN = invN_;

        predGm.SetGlobalBuffer((__gm__ float*)predict, static_cast<uint64_t>(N) * static_cast<uint64_t>(C));
        tgtGm.SetGlobalBuffer((__gm__ int32_t*)target, static_cast<uint64_t>(N));
        outGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // Persistent UB buffers reused for all tiles.
        pipe.InitBuffer(xBuf, tileC * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileC * sizeof(float));
        pipe.InitBuffer(redBuf, tileC * sizeof(float));
        pipe.InitBuffer(scalarBuf, 32U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (static_cast<uint32_t>(AscendC::GetBlockIdx()) != 0U) return;

        if (N == 0U || C == 0U) {
            outGm.SetValue(0, 0.0f);
            return;
        }

        float total = 0.0f;
        for (uint32_t row = 0; row < N; ++row) {
            total += RowLossOnePass(row);
        }
        outGm.SetValue(0, total * invN);
    }

private:
    __aicore__ inline float ExpScalar(float v)
    {
        AscendC::LocalTensor<float> s = scalarBuf.Get<float>();
        s.SetValue(0, v);
        AscendC::Exp(s, s, 1U);
        return s.GetValue(0);
    }

    __aicore__ inline float LogScalar(float v)
    {
        AscendC::LocalTensor<float> s = scalarBuf.Get<float>();
        s.SetValue(0, v);
        AscendC::Log(s, s, 1U);
        return s.GetValue(0);
    }

    // One-pass stable logsumexp:
    // maintain m = max seen so far, s = sum(exp(x - m)).
    // when m increases to m', rescale s *= exp(m - m').
    __aicore__ inline float RowLossOnePass(uint32_t row)
    {
        int32_t t = tgtGm.GetValue(static_cast<uint64_t>(row));
        bool targetValid = (t >= 0) && (static_cast<uint32_t>(t) < C);
        uint32_t tu = targetValid ? static_cast<uint32_t>(t) : 0U;

        float m = NEG_INF_F;
        float s = 0.0f;
        float targetLogit = 0.0f;

        AscendC::LocalTensor<float> x = xBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> red = redBuf.Get<float>();

        for (uint32_t it = 0; it < tilesPerRow; ++it) {
            uint32_t cStart = it * tileC;
            uint32_t curC = C - cStart;
            if (curC > tileC) curC = tileC;

            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(C) + static_cast<uint64_t>(cStart);
            AscendC::DataCopy(x, predGm[base], curC);

            if (targetValid && tu >= cStart && tu < (cStart + curC)) {
                targetLogit = x.GetValue(tu - cStart);
            }

            AscendC::ReduceMax<float>(red, x, tmp, static_cast<int32_t>(curC));
            float tileMax = red.GetValue(0);

            float oldm = m;
            if (tileMax > m) m = tileMax;

            if (m > oldm) {
                // rescale existing sum when max increases
                s *= ExpScalar(oldm - m);
            }

            AscendC::Adds(tmp, x, -m, static_cast<int32_t>(curC));
            AscendC::Exp(tmp, tmp, static_cast<int32_t>(curC));
            AscendC::ReduceSum<float>(red, tmp, x, static_cast<int32_t>(curC));
            s += red.GetValue(0);
        }

        float logsumexp = LogScalar(s) + m;
        return logsumexp - targetLogit;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> xBuf;
    AscendC::TBuf<> tmpBuf;
    AscendC::TBuf<> redBuf;
    AscendC::TBuf<> scalarBuf;

    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<int32_t> tgtGm;
    AscendC::GlobalTensor<float> outGm;

    uint32_t N {0}, C {0}, tileC {0}, tilesPerRow {0};
    float invN {0.0f};
};

extern "C" __global__ __aicore__ void cross_entropy_loss_custom(GM_ADDR predict,
                                                                GM_ADDR target,
                                                                GM_ADDR y,
                                                                GM_ADDR workspace,
                                                                GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossEntropyLoss op;
    op.Init(predict, target, y,
            tiling_data.N,
            tiling_data.C,
            tiling_data.tileC,
            tiling_data.tilesPerRow,
            tiling_data.invN);
    op.Process();
}
