
#include "kernel_operator.h"

// Benchmark specialization for provided model:
// input x: [B, D, 7, 7], axial attention along one axis => T=7,
// dim=512, heads=8 => E=64. We fuse only attention core on [BH,T,E].
// Keep modest upper bounds and enforce in binding to avoid silent no-write.
static constexpr uint32_t MAX_T = 7;
static constexpr uint32_t MAX_E = 64;

class KernelAxialAttentionCustom {
public:
    __aicore__ inline KernelAxialAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t bh, uint32_t t, uint32_t e, float scale)
    {
        this->bh = bh;
        this->t = t;
        this->e = e;
        this->scale = scale;

        const uint64_t elems = static_cast<uint64_t>(bh) * t * e;
        qGm.SetGlobalBuffer((__gm__ float*)q, elems);
        kGm.SetGlobalBuffer((__gm__ float*)k, elems);
        vGm.SetGlobalBuffer((__gm__ float*)v, elems);
        oGm.SetGlobalBuffer((__gm__ float*)out, elems);

        // UB: Q row [E], K all [T*E], V all [T*E], scores [T], probs [T], out row [E]
        pipe.InitBuffer(bufQRow,   MAX_E * sizeof(float));
        pipe.InitBuffer(bufKAll,  (MAX_T * MAX_E) * sizeof(float));
        pipe.InitBuffer(bufVAll,  (MAX_T * MAX_E) * sizeof(float));
        pipe.InitBuffer(bufScores, MAX_T * sizeof(float));
        pipe.InitBuffer(bufProbs,  MAX_T * sizeof(float));
        pipe.InitBuffer(bufORow,   MAX_E * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // last guard: binding enforces; this prevents UB overflow if misused
        if (t > MAX_T || e > MAX_E) return;

        const uint32_t row = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (row >= bh) return;

        LoadKandV(row);

        for (uint32_t qi = 0; qi < t; ++qi) {
            LoadQRow(row, qi);
            ComputeScores();
            SoftmaxStable();
            ComputeWeightedSum();
            StoreORow(row, qi);
        }
    }

private:
    __aicore__ inline uint64_t Base(uint32_t row, uint32_t ti) const
    {
        // [BH,T,E]
        return (static_cast<uint64_t>(row) * t + ti) * e;
    }

    __aicore__ inline void LoadKandV(uint32_t row)
    {
        AscendC::LocalTensor<float> kAll = bufKAll.Get<float>();
        AscendC::LocalTensor<float> vAll = bufVAll.Get<float>();

        const uint64_t base = Base(row, 0);
        for (uint32_t tj = 0; tj < t; ++tj) {
            const uint64_t off = base + static_cast<uint64_t>(tj) * e;
            AscendC::DataCopy(kAll[tj * e], kGm[off], e);
            AscendC::DataCopy(vAll[tj * e], vGm[off], e);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadQRow(uint32_t row, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t off = Base(row, qi);
        AscendC::DataCopy(qRow, qGm[off], e);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScores()
    {
        AscendC::LocalTensor<float> qRow   = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kAll   = bufKAll.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        for (uint32_t j = 0; j < t; ++j) {
            float acc = 0.0f;
            const uint32_t base = j * e;
            for (uint32_t ei = 0; ei < e; ++ei) {
                acc += qRow(ei) * kAll(base + ei);
            }
            scores(j) = acc * scale;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftmaxStable()
    {
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        AscendC::LocalTensor<float> probs  = bufProbs.Get<float>();

        float m = scores(0);
        for (uint32_t j = 1; j < t; ++j) {
            const float v = scores(j);
            if (v > m) m = v;
        }

        for (uint32_t j = 0; j < t; ++j) probs(j) = scores(j) - m;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(probs, probs, static_cast<int32_t>(t));
        AscendC::PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        for (uint32_t j = 0; j < t; ++j) sum += probs(j);
        if (sum == 0.0f) sum = 1.0f;
        const float inv = 1.0f / sum;

        for (uint32_t j = 0; j < t; ++j) probs(j) = probs(j) * inv;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeWeightedSum()
    {
        AscendC::LocalTensor<float> probs = bufProbs.Get<float>();
        AscendC::LocalTensor<float> vAll  = bufVAll.Get<float>();
        AscendC::LocalTensor<float> oRow  = bufORow.Get<float>();

        for (uint32_t ei = 0; ei < e; ++ei) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < t; ++j) {
                acc += probs(j) * vAll(j * e + ei);
            }
            oRow(ei) = acc;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreORow(uint32_t row, uint32_t qi)
    {
        AscendC::LocalTensor<float> oRow = bufORow.Get<float>();
        const uint64_t off = Base(row, qi);
        AscendC::DataCopy(oGm[off], oRow, e);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufProbs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufORow;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t bh, t, e;
    float scale;
};

extern "C" __global__ __aicore__ void axial_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelAxialAttentionCustom op;
    op.Init(q, k, v, out, tiling_data.bh, tiling_data.t, tiling_data.e, tiling_data.scale);
    op.Process();
}
