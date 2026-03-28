
#include "kernel_operator.h"

// Specialization aligned with provided model config and common benchmarks.
static constexpr uint32_t MAX_SQ = 512;
static constexpr uint32_t MAX_SK = 256;
static constexpr uint32_t MAX_D  = 64;

class KernelCrossAttentionCustom {
public:
    __aicore__ inline KernelCrossAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t sq, uint32_t sk, uint32_t d,
                               float scale)
    {
        this->b = b;
        this->h = h;
        this->sq = sq;
        this->sk = sk;
        this->d = d;
        this->scale = scale;

        const uint64_t qElems = static_cast<uint64_t>(b) * h * sq * d;
        const uint64_t kElems = static_cast<uint64_t>(b) * h * sk * d;
        const uint64_t vElems = static_cast<uint64_t>(b) * h * sk * d;
        const uint64_t oElems = static_cast<uint64_t>(b) * h * sq * d;

        qGm.SetGlobalBuffer((__gm__ float*)q, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, kElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, vElems);
        oGm.SetGlobalBuffer((__gm__ float*)out, oElems);

        // UB buffers: K/V cached for one (b,h), plus per-q-row Q, scores/probs, and out row
        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufKAll,  (MAX_SK * MAX_D) * sizeof(float));
        pipe.InitBuffer(bufVAll,  (MAX_SK * MAX_D) * sizeof(float));
        pipe.InitBuffer(bufScores, MAX_SK * sizeof(float));
        pipe.InitBuffer(bufProbs,  MAX_SK * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Python binding enforces bounds; keep as a last guard to avoid UB overflow.
        if (sq > MAX_SQ || sk > MAX_SK || d > MAX_D) return;

        const uint32_t bh = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t head = (h == 0) ? 0 : (bh % h);
        const uint32_t batch = (h == 0) ? 0 : (bh / h);
        if (batch >= b) return;

        LoadKandV(batch, head);

        for (uint32_t qi = 0; qi < sq; ++qi) {
            LoadQRow(batch, head, qi);
            ComputeScores();
            SoftmaxStable();
            ComputeWeightedSum();
            StoreOutRow(batch, head, qi);
        }
    }

private:
    __aicore__ inline uint64_t BaseQO(uint32_t batch, uint32_t head, uint32_t si) const
    {
        // [B,H,Sq,D]
        return ((static_cast<uint64_t>(batch) * h + head) * sq + si) * d;
    }

    __aicore__ inline uint64_t BaseKV(uint32_t batch, uint32_t head, uint32_t si) const
    {
        // [B,H,Sk,D]
        return ((static_cast<uint64_t>(batch) * h + head) * sk + si) * d;
    }

    __aicore__ inline void LoadKandV(uint32_t batch, uint32_t head)
    {
        AscendC::LocalTensor<float> kAll = bufKAll.Get<float>();
        AscendC::LocalTensor<float> vAll = bufVAll.Get<float>();

        const uint64_t base = BaseKV(batch, head, 0);
        for (uint32_t ki = 0; ki < sk; ++ki) {
            const uint64_t off = base + static_cast<uint64_t>(ki) * d;
            AscendC::DataCopy(kAll[ki * d], kGm[off], d);
            AscendC::DataCopy(vAll[ki * d], vGm[off], d);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadQRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t off = BaseQO(batch, head, qi);
        AscendC::DataCopy(qRow, qGm[off], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScores()
    {
        AscendC::LocalTensor<float> qRow  = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kAll  = bufKAll.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        for (uint32_t j = 0; j < sk; ++j) {
            float acc = 0.0f;
            const uint32_t base = j * d;
            for (uint32_t di = 0; di < d; ++di) {
                acc += qRow(di) * kAll(base + di);
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
        for (uint32_t j = 1; j < sk; ++j) {
            const float v = scores(j);
            if (v > m) m = v;
        }

        for (uint32_t j = 0; j < sk; ++j) {
            probs(j) = scores(j) - m;
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Use AscendC Exp to avoid expf link issues.
        AscendC::Exp(probs, probs, sk);
        AscendC::PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        for (uint32_t j = 0; j < sk; ++j) sum += probs(j);
        if (sum == 0.0f) sum = 1.0f;
        const float inv = 1.0f / sum;

        for (uint32_t j = 0; j < sk; ++j) probs(j) = probs(j) * inv;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeWeightedSum()
    {
        AscendC::LocalTensor<float> probs  = bufProbs.Get<float>();
        AscendC::LocalTensor<float> vAll   = bufVAll.Get<float>();
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

        for (uint32_t di = 0; di < d; ++di) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < sk; ++j) {
                acc += probs(j) * vAll(j * d + di);
            }
            outRow(di) = acc;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        const uint64_t off = BaseQO(batch, head, qi);
        AscendC::DataCopy(oGm[off], outRow, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufProbs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b, h, sq, sk, d;
    float scale;
};

extern "C" __global__ __aicore__ void cross_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossAttentionCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.sq, tiling_data.sk, tiling_data.d,
            tiling_data.scale);
    op.Process();
}
