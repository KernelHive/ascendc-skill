
#include "kernel_operator.h"

// Strided attention for Q,K,V: [B,H,S,D] float32, contiguous.
// Select keys j where j % stride == i % stride, softmax over selected keys.
// Optimization focus:
//  - Row-parallel mapping over (B*H*S) to reduce single-core serialization/pipeline gaps.
//  - Vector Exp over the full selected score vector (length cnt) to reduce scalar dispatch.
//  - UB allocations are runtime-sized (d, nsel) to avoid UB over-allocation failures.
// Constraints (host-enforced): S<=512, D<=64.

static constexpr uint32_t MAX_S = 512;
static constexpr uint32_t MAX_D = 64;

class KernelStridedAttention {
public:
    __aicore__ inline KernelStridedAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               uint32_t stride, uint32_t nsel, float scale,
                               uint32_t totalRows, uint32_t rowsPerCore)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->d = d;
        this->stride = stride;
        this->nsel = nsel;
        this->scale = scale;
        this->totalRows = totalRows;
        this->rowsPerCore = rowsPerCore;

        const uint64_t total = static_cast<uint64_t>(b) * h * s * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, total);
        kGm.SetGlobalBuffer((__gm__ float*)k, total);
        vGm.SetGlobalBuffer((__gm__ float*)v, total);
        oGm.SetGlobalBuffer((__gm__ float*)out, total);

        // UB buffers (runtime-sized, bounded by MAX_*)
        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufKRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));

        // scores/exp only need MAX_S floats each (cnt <= nsel <= MAX_S)
        pipe.InitBuffer(bufScores, MAX_S * sizeof(float));
        pipe.InitBuffer(bufExp,    MAX_S * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (s == 0 || d == 0) return;
        if (s > MAX_S || d > MAX_D) return;
        if (stride == 0 || stride > s) return;
        if (nsel == 0 || nsel > MAX_S) return;
        if (totalRows == 0 || rowsPerCore == 0) return;

        const uint32_t coreId = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t rowStart = coreId * rowsPerCore;
        const uint32_t rowEnd = (rowStart + rowsPerCore <= totalRows) ? (rowStart + rowsPerCore) : totalRows;
        if (rowStart >= rowEnd) return;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            uint32_t batch, head, qi;
            DecodeRow(row, batch, head, qi);

            const uint64_t base = BaseBH(batch, head);
            LoadQRow(base, qi);

            uint32_t cnt = FillSelectedScores(base, qi); // writes scores[0:cnt)
            if (cnt == 0) {
                ZeroOutRow();
                StoreOutRow(base, qi);
                continue;
            }

            // Softmax using vector Exp for the whole vector
            SoftmaxToExp(cnt);     // bufExp[0:cnt) = exp(scores - max), returns max in scalar path
            const float invSum = InvSumFromExp(cnt); // scalar reduction
            ComputeOutFromExp(base, qi, cnt, invSum);
            StoreOutRow(base, qi);
        }
    }

private:
    __aicore__ inline void DecodeRow(uint32_t row, uint32_t &batch, uint32_t &head, uint32_t &qi) const
    {
        // row in [0, b*h*s)
        const uint32_t hs = h * s;
        batch = row / hs;
        const uint32_t rem = row - batch * hs;
        head = rem / s;
        qi = rem - head * s;
    }

    __aicore__ inline uint64_t BaseBH(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(s) * d);
    }

    __aicore__ inline void LoadQRow(uint64_t base, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t off = base + static_cast<uint64_t>(qi) * d;
        AscendC::DataCopy(qRow, qGm[off], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline uint32_t FillSelectedScores(uint64_t base, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kRow = bufKRow.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        const uint32_t r = qi % stride;
        uint32_t idx = 0;

        for (uint32_t j = r; j < s; j += stride) {
            const uint64_t kOff = base + static_cast<uint64_t>(j) * d;
            AscendC::DataCopy(kRow, kGm[kOff], d);
            AscendC::PipeBarrier<PIPE_MTE2>();

            float acc = 0.0f;
            for (uint32_t di = 0; di < d; ++di) {
                acc += qRow(di) * kRow(di);
            }
            scores(idx) = acc * scale;
            idx++;
            if (idx >= nsel) break;
        }
        AscendC::PipeBarrier<PIPE_V>();
        return idx;
    }

    __aicore__ inline void ZeroOutRow()
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        for (uint32_t di = 0; di < d; ++di) outRow(di) = 0.0f;
        AscendC::PipeBarrier<PIPE_V>();
    }

    // bufExp[0:cnt) = exp(bufScores[0:cnt) - max(scores))
    __aicore__ inline void SoftmaxToExp(uint32_t cnt)
    {
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        AscendC::LocalTensor<float> exps   = bufExp.Get<float>();

        float m = scores(0);
        for (uint32_t i = 1; i < cnt; ++i) {
            float v = scores(i);
            if (v > m) m = v;
        }

        // exps = scores - m
        for (uint32_t i = 0; i < cnt; ++i) exps(i) = scores(i) - m;
        AscendC::PipeBarrier<PIPE_V>();

        // vector exp
        AscendC::Exp(exps, exps, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float InvSumFromExp(uint32_t cnt)
    {
        AscendC::LocalTensor<float> exps = bufExp.Get<float>();
        float sum = 0.0f;
        for (uint32_t i = 0; i < cnt; ++i) sum += exps(i);
        if (sum <= 0.0f) return 0.0f;
        return 1.0f / sum;
    }

    // out = sum_j (exp(scores_j-max)*invSum) * V[j]
    __aicore__ inline void ComputeOutFromExp(uint64_t base, uint32_t qi, uint32_t cnt, float invSum)
    {
        AscendC::LocalTensor<float> exps   = bufExp.Get<float>();
        AscendC::LocalTensor<float> vRow   = bufVRow.Get<float>();
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

        for (uint32_t di = 0; di < d; ++di) outRow(di) = 0.0f;
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t r = qi % stride;
        uint32_t idx = 0;
        for (uint32_t j = r; j < s; j += stride) {
            const float p = exps(idx) * invSum;

            const uint64_t vOff = base + static_cast<uint64_t>(j) * d;
            AscendC::DataCopy(vRow, vGm[vOff], d);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t di = 0; di < d; ++di) {
                outRow(di) = outRow(di) + p * vRow(di);
            }

            idx++;
            if (idx >= cnt) break;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutRow(uint64_t base, uint32_t qi)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        const uint64_t off = base + static_cast<uint64_t>(qi) * d;
        AscendC::DataCopy(oGm[off], outRow, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufExp;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b = 0, h = 0, s = 0, d = 0;
    uint32_t stride = 1, nsel = 0;
    float scale = 1.0f;

    uint32_t totalRows = 0;
    uint32_t rowsPerCore = 0;
};

extern "C" __global__ __aicore__ void strided_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    KernelStridedAttention op;
    op.Init(q, k, v, out,
            td.b, td.h, td.s, td.d,
            td.stride, td.nsel, td.scale,
            td.totalRows, td.rowsPerCore);
    op.Process();
}
