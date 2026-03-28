
#include "kernel_operator.h"

// Fused halo attention core:
// q:[B,I,D] (already scaled), k/v:[B,J,D], mask:[B,1,J] bool (True => masked)
// out:[B,I,D]
class KernelHaloAttentionCustom {
public:
    __aicore__ inline KernelHaloAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR out,
                               uint32_t B, uint32_t I, uint32_t J, uint32_t D)
    {
        this->B = B; this->I = I; this->J = J; this->D = D;
        this->totalRows = B * I;

        const uint64_t qElems  = static_cast<uint64_t>(B) * I * D;
        const uint64_t kvElems = static_cast<uint64_t>(B) * J * D;
        const uint64_t mElems  = static_cast<uint64_t>(B) * 1ULL * J;
        const uint64_t oElems  = static_cast<uint64_t>(B) * I * D;

        qGm.SetGlobalBuffer((__gm__ float*)q, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, kvElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, kvElems);
        // bool storage is 1 byte on device; treat as uint8
        mGm.SetGlobalBuffer((__gm__ uint8_t*)mask, mElems);
        oGm.SetGlobalBuffer((__gm__ float*)out, oElems);

        // UB: qRow[D], kRow[D], vRow[D], scores[J], probs[J], outRow[D]
        pipe.InitBuffer(bufQRow, D * sizeof(float));
        pipe.InitBuffer(bufKRow, D * sizeof(float));
        pipe.InitBuffer(bufVRow, D * sizeof(float));
        pipe.InitBuffer(bufScores, J * sizeof(float));
        pipe.InitBuffer(bufProbs, J * sizeof(float));
        pipe.InitBuffer(bufOutRow, D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t bnum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (bnum == 0) bnum = 1;

        // grid-stride over rows (row = b*I + i)
        for (uint32_t row = bid; row < totalRows; row += bnum) {
            const uint32_t b = row / I;
            const uint32_t i = row - b * I;
            if (b >= B) continue;

            LoadQRow(b, i);
            ComputeScores(b);     // scores[j] = dot(q[i], k[j])
            ApplyMask(b);         // set masked logits to large negative
            SoftmaxStable();      // probs = softmax(scores)
            ComputeOut(b);        // outRow = sum_j probs[j] * v[j]
            StoreOutRow(b, i);
        }
    }

private:
    __aicore__ inline uint64_t QOffset(uint32_t b, uint32_t i) const {
        return (static_cast<uint64_t>(b) * I + i) * D;
    }
    __aicore__ inline uint64_t KOffset(uint32_t b, uint32_t j) const {
        return (static_cast<uint64_t>(b) * J + j) * D;
    }
    __aicore__ inline uint64_t VOffset(uint32_t b, uint32_t j) const {
        return (static_cast<uint64_t>(b) * J + j) * D;
    }
    // mask is [B,1,J] contiguous => linear index ((b*1 + 0)*J + j) = b*J + j
    __aicore__ inline uint64_t MOffset(uint32_t b, uint32_t j) const {
        return (static_cast<uint64_t>(b) * 1ULL + 0ULL) * J + j;
    }
    __aicore__ inline uint64_t OOffset(uint32_t b, uint32_t i) const {
        return (static_cast<uint64_t>(b) * I + i) * D;
    }

    __aicore__ inline void LoadQRow(uint32_t b, uint32_t i)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        AscendC::DataCopy(qRow, qGm[QOffset(b, i)], D);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadKRow(uint32_t b, uint32_t j)
    {
        AscendC::LocalTensor<float> kRow = bufKRow.Get<float>();
        AscendC::DataCopy(kRow, kGm[KOffset(b, j)], D);
    }

    __aicore__ inline void LoadVRow(uint32_t b, uint32_t j)
    {
        AscendC::LocalTensor<float> vRow = bufVRow.Get<float>();
        AscendC::DataCopy(vRow, vGm[VOffset(b, j)], D);
    }

    __aicore__ inline void ComputeScores(uint32_t b)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kRow = bufKRow.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        for (uint32_t j = 0; j < J; ++j) {
            LoadKRow(b, j);
            AscendC::PipeBarrier<PIPE_MTE2>();

            float acc = 0.0f;
            for (uint32_t d = 0; d < D; ++d) {
                acc += qRow(d) * kRow(d);
            }
            scores(j) = acc;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyMask(uint32_t b)
    {
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        // safe large negative; softmax uses subtract-max
        const float neg = -1.0e20f;

        for (uint32_t j = 0; j < J; ++j) {
            const uint8_t mv = mGm.GetValue(static_cast<uint32_t>(MOffset(b, j)));
            if (mv != 0) { // True => masked
                scores(j) = neg;
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftmaxStable()
    {
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        AscendC::LocalTensor<float> probs = bufProbs.Get<float>();

        float m = scores(0);
        for (uint32_t j = 1; j < J; ++j) {
            const float v = scores(j);
            if (v > m) m = v;
        }

        for (uint32_t j = 0; j < J; ++j) {
            probs(j) = scores(j) - m;
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(probs, probs, static_cast<int32_t>(J));
        AscendC::PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        for (uint32_t j = 0; j < J; ++j) sum += probs(j);
        if (sum == 0.0f) sum = 1.0f;
        const float inv = 1.0f / sum;

        for (uint32_t j = 0; j < J; ++j) probs(j) = probs(j) * inv;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeOut(uint32_t b)
    {
        AscendC::LocalTensor<float> probs = bufProbs.Get<float>();
        AscendC::LocalTensor<float> vRow = bufVRow.Get<float>();
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

        for (uint32_t d = 0; d < D; ++d) outRow(d) = 0.0f;

        for (uint32_t j = 0; j < J; ++j) {
            LoadVRow(b, j);
            AscendC::PipeBarrier<PIPE_MTE2>();

            const float p = probs(j);
            for (uint32_t d = 0; d < D; ++d) {
                outRow(d) += p * vRow(d);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutRow(uint32_t b, uint32_t i)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        AscendC::DataCopy(oGm[OOffset(b, i)], outRow, D);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufProbs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, oGm;
    AscendC::GlobalTensor<uint8_t> mGm;

    uint32_t B = 0, I = 0, J = 0, D = 0;
    uint32_t totalRows = 0;
};

extern "C" __global__ __aicore__ void halo_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);
    KernelHaloAttentionCustom op;
    op.Init(q, k, v, mask, out, t.B, t.I, t.J, t.D);
    op.Process();
}
