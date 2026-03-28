
#include "kernel_operator.h"

// Optimized local sliding-window attention core.
// Key optimization: copy K and V window blocks [wLen, D] into UB once per query,
// then compute scores and weighted sum from UB to eliminate per-token DataCopy and barriers.

static constexpr uint32_t MAX_S      = 512;
static constexpr uint32_t MAX_D      = 128;
static constexpr uint32_t MAX_WINDOW = 128;
static constexpr uint32_t MAX_WIN    = 129;   // 2*(MAX_WINDOW/2)+1

class KernelLocalAttentionCustom {
public:
    __aicore__ inline KernelLocalAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               uint32_t window_size, uint32_t max_win,
                               uint32_t total_rows, uint32_t rows_per_core,
                               float scale)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->d = d;
        this->windowSize = window_size;
        this->maxWin = max_win;
        this->totalRows = total_rows;
        this->rowsPerCore = rows_per_core;
        this->scale = scale;

        const uint64_t total = static_cast<uint64_t>(b) * h * s * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, total);
        kGm.SetGlobalBuffer((__gm__ float*)k, total);
        vGm.SetGlobalBuffer((__gm__ float*)v, total);
        oGm.SetGlobalBuffer((__gm__ float*)out, total);

        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufKWin,   (MAX_WIN * MAX_D) * sizeof(float));
        pipe.InitBuffer(bufVWin,   (MAX_WIN * MAX_D) * sizeof(float));

        pipe.InitBuffer(bufScores,  MAX_WIN * sizeof(float));
        pipe.InitBuffer(bufTmp,     MAX_WIN * sizeof(float)); // for non-overlap exp / shifted
        pipe.InitBuffer(bufOutRow,  MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (s == 0 || d == 0) return;
        if (s > MAX_S || d > MAX_D) return;
        if (windowSize == 0 || windowSize > MAX_WINDOW) return;
        if (maxWin == 0 || maxWin > MAX_WIN) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = bid * rowsPerCore;
        const uint32_t end = (start + rowsPerCore > totalRows) ? totalRows : (start + rowsPerCore);
        if (start >= end) return;

        const int32_t halfW = static_cast<int32_t>(windowSize / 2);

        for (uint32_t row = start; row < end; ++row) {
            uint32_t tmp = row;
            const uint32_t qi = tmp % s;
            tmp /= s;
            const uint32_t head = tmp % h;
            const uint32_t batch = tmp / h;

            int32_t winStart = static_cast<int32_t>(qi) - halfW;
            int32_t winEnd = static_cast<int32_t>(qi) + halfW + 1;
            if (winStart < 0) winStart = 0;
            if (winEnd > static_cast<int32_t>(s)) winEnd = static_cast<int32_t>(s);
            const uint32_t wLen = static_cast<uint32_t>(winEnd - winStart);
            if (wLen == 0 || wLen > maxWin) continue;

            LoadQRow(batch, head, qi);
            LoadKWindow(batch, head, static_cast<uint32_t>(winStart), wLen);
            ComputeScoresFromUB(wLen);
            SoftmaxStableInPlace(wLen);
            LoadVWindow(batch, head, static_cast<uint32_t>(winStart), wLen);
            ComputeOutFromUB(wLen);
            StoreOutRow(batch, head, qi);
        }
    }

private:
    __aicore__ inline uint64_t Base(uint32_t batch, uint32_t head, uint32_t si) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(s) * d) +
               static_cast<uint64_t>(si) * d;
    }

    __aicore__ inline void LoadQRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        AscendC::DataCopy(qRow, qGm[Base(batch, head, qi)], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadKWindow(uint32_t batch, uint32_t head, uint32_t winStart, uint32_t wLen)
    {
        AscendC::LocalTensor<float> kWin = bufKWin.Get<float>();
        // GM layout is contiguous along S then D, so [winStart, wLen, D] is contiguous.
        AscendC::DataCopy(kWin, kGm[Base(batch, head, winStart)], wLen * d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadVWindow(uint32_t batch, uint32_t head, uint32_t winStart, uint32_t wLen)
    {
        AscendC::LocalTensor<float> vWin = bufVWin.Get<float>();
        AscendC::DataCopy(vWin, vGm[Base(batch, head, winStart)], wLen * d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScoresFromUB(uint32_t wLen)
    {
        AscendC::LocalTensor<float> qRow   = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kWin   = bufKWin.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();

        // For each key in window, dot(Q, K) from UB.
        for (uint32_t t = 0; t < wLen; ++t) {
            float acc = 0.0f;
            const uint32_t base = t * d;

            // Unroll common head_dim to reduce scalar overhead.
            if (d == 64) {
                #pragma unroll
                for (uint32_t di = 0; di < 64; ++di) {
                    acc += qRow(di) * kWin(base + di);
                }
            } else {
                for (uint32_t di = 0; di < d; ++di) {
                    acc += qRow(di) * kWin(base + di);
                }
            }
            scores(t) = acc * scale;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftmaxStableInPlace(uint32_t n)
    {
        // bufScores becomes probs
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        AscendC::LocalTensor<float> tmp    = bufTmp.Get<float>(); // shifted/exps

        float m = scores(0);
        for (uint32_t i = 1; i < n; ++i) {
            const float v = scores(i);
            if (v > m) m = v;
        }

        for (uint32_t i = 0; i < n; ++i) tmp(i) = scores(i) - m;
        AscendC::PipeBarrier<PIPE_V>();

        // Exp forbids overlap; dst(scores) != src(tmp)
        AscendC::Exp(scores, tmp, static_cast<int32_t>(n));
        AscendC::PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        for (uint32_t i = 0; i < n; ++i) sum += scores(i);
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

        for (uint32_t i = 0; i < n; ++i) scores(i) = scores(i) * inv;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeOutFromUB(uint32_t wLen)
    {
        AscendC::LocalTensor<float> probs  = bufScores.Get<float>();
        AscendC::LocalTensor<float> vWin   = bufVWin.Get<float>();
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

        for (uint32_t di = 0; di < d; ++di) outRow(di) = 0.0f;

        // Weighted sum over window from UB.
        for (uint32_t t = 0; t < wLen; ++t) {
            const float p = probs(t);
            const uint32_t base = t * d;

            if (d == 64) {
                #pragma unroll
                for (uint32_t di = 0; di < 64; ++di) {
                    outRow(di) += p * vWin(base + di);
                }
            } else {
                for (uint32_t di = 0; di < d; ++di) {
                    outRow(di) += p * vWin(base + di);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOutRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        AscendC::DataCopy(oGm[Base(batch, head, qi)], outRow, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKWin;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVWin;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b = 0, h = 0, s = 0, d = 0;
    uint32_t windowSize = 0, maxWin = 0;
    uint32_t totalRows = 0, rowsPerCore = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void local_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelLocalAttentionCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d,
            tiling_data.window_size, tiling_data.max_win,
            tiling_data.total_rows, tiling_data.rows_per_core,
            tiling_data.scale);
    op.Process();
}
