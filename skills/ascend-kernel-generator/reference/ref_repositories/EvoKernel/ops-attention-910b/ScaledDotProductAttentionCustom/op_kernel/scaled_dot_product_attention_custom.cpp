
#include "kernel_operator.h"

static constexpr uint32_t MAX_S = 128;
static constexpr uint32_t MAX_D = 64;

class KernelScaledDotProductAttention {
public:
    __aicore__ inline KernelScaledDotProductAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               float scale, uint32_t tasksPerBH)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->d = d;
        this->scale = scale;
        this->tasksPerBH = tasksPerBH;

        const uint64_t total = static_cast<uint64_t>(b) * h * s * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, total);
        kGm.SetGlobalBuffer((__gm__ float*)k, total);
        vGm.SetGlobalBuffer((__gm__ float*)v, total);
        oGm.SetGlobalBuffer((__gm__ float*)out, total);

        pipe.InitBuffer(bufQRow, MAX_D * sizeof(float));
        pipe.InitBuffer(bufKAll, (MAX_S * MAX_D) * sizeof(float));
        pipe.InitBuffer(bufVAll, (MAX_S * MAX_D) * sizeof(float));
        pipe.InitBuffer(bufScores, MAX_S * sizeof(float));
        pipe.InitBuffer(bufProbs, MAX_S * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Host enforces these; keep guard.
        if (s > MAX_S || d > MAX_D || tasksPerBH == 0) return;

        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t bh = block / tasksPerBH;
        const uint32_t task = block - bh * tasksPerBH;

        const uint32_t head = bh % h;
        const uint32_t batch = bh / h;
        if (batch >= b) return;

        // Split S query rows into tasksPerBH contiguous chunks.
        const uint32_t rowsPerTask = (s + tasksPerBH - 1) / tasksPerBH;
        const uint32_t qStart = task * rowsPerTask;
        uint32_t qEnd = qStart + rowsPerTask;
        if (qStart >= s) return;
        if (qEnd > s) qEnd = s;

        LoadKandV(batch, head);

        for (uint32_t qi = qStart; qi < qEnd; ++qi) {
            LoadQRow(batch, head, qi);
            ComputeScores();
            SoftmaxStable();
            ComputeWeightedSum();
            StoreOutRow(batch, head, qi);
        }
    }

private:
    __aicore__ inline uint64_t BaseOffset(uint32_t batch, uint32_t head, uint32_t si) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(s) * d) +
               static_cast<uint64_t>(si) * d;
    }

    __aicore__ inline void LoadKandV(uint32_t batch, uint32_t head)
    {
        AscendC::LocalTensor<float> kAll = bufKAll.Get<float>();
        AscendC::LocalTensor<float> vAll = bufVAll.Get<float>();

        // K and V are contiguous in D, so per-row DataCopy is efficient for these small sizes.
        for (uint32_t ki = 0; ki < s; ++ki) {
            const uint64_t off = BaseOffset(batch, head, ki);
            AscendC::DataCopy(kAll[ki * MAX_D], kGm[off], d);
            AscendC::DataCopy(vAll[ki * MAX_D], vGm[off], d);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadQRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t off = BaseOffset(batch, head, qi);
        AscendC::DataCopy(qRow, qGm[off], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScores()
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kAll = bufKAll.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        for (uint32_t j = 0; j < s; ++j) {
            float acc = 0.0f;
            const uint32_t base = j * MAX_D;
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
        AscendC::LocalTensor<float> probs = bufProbs.Get<float>();

        float m = scores(0);
        for (uint32_t j = 1; j < s; ++j) {
            const float v = scores(j);
            if (v > m) m = v;
        }

        for (uint32_t j = 0; j < s; ++j) {
            probs(j) = scores(j) - m;
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(probs, probs, s);
        AscendC::PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        for (uint32_t j = 0; j < s; ++j) {
            sum += probs(j);
        }
        const float inv = 1.0f / sum;
        for (uint32_t j = 0; j < s; ++j) {
            probs(j) = probs(j) * inv;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeWeightedSum()
    {
        AscendC::LocalTensor<float> probs = bufProbs.Get<float>();
        AscendC::LocalTensor<float> vAll = bufVAll.Get<float>();
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

        for (uint32_t di = 0; di < d; ++di) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < s; ++j) {
                acc += probs(j) * vAll(j * MAX_D + di);
            }
            outRow(di) = acc;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        const uint64_t off = BaseOffset(batch, head, qi);
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

    uint32_t b, h, s, d;
    uint32_t tasksPerBH;
    float scale;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaledDotProductAttention op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d,
            tiling_data.scale, tiling_data.tasksPerBH);
    op.Process();
}
