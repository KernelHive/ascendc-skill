
#include "kernel_operator.h"

// q: [B,H,S,D] fp32
// k/v: [B,1,S,D] fp32
// y: [B,H,S,D] fp32
//
// Parallelization: each core handles one task = (b,h, qGroup), where qGroup is a tile of qi.
// This allows reusing streamed K/V tiles across multiple queries within qTile, reducing GM reads.
//
// Softmax per qi is stable two-pass:
//  1) rowMax over S (stream K)
//  2) stream K+V: denom += exp(score-rowMax), out += exp(score-rowMax)*V, then out/=denom
//
// Compute: dot uses scalar FMA loop over D<=128 to avoid heavy vector ReduceSum/barrier churn.

class KernelMultiQueryAttentionCustom {
public:
    __aicore__ inline KernelMultiQueryAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t S, uint32_t D,
                               uint32_t sTile, uint32_t qTile, float scale)
    {
        this->B = B; this->H = H; this->S = S; this->D = D;
        this->sTile = sTile; this->qTile = qTile;
        this->scale = scale;

        const uint64_t qElems = (uint64_t)B * (uint64_t)H * (uint64_t)S * (uint64_t)D;
        const uint64_t kvElems = (uint64_t)B * (uint64_t)S * (uint64_t)D; // [B,1,S,D] treated as [B,S,D]
        qGm.SetGlobalBuffer((__gm__ float*)q, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, kvElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, kvElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, qElems);

        // UB buffers:
        // qTileBuf: [qTile, D]
        // outTileBuf: [qTile, D]
        // rowMaxBuf/rowSumBuf: [qTile]
        // K/V tiles: [sTile, D]
        pipe.InitBuffer(qTileBuf, (uint32_t)((uint64_t)qTile * (uint64_t)D * sizeof(float)));
        pipe.InitBuffer(outTileBuf, (uint32_t)((uint64_t)qTile * (uint64_t)D * sizeof(float)));
        pipe.InitBuffer(rowMaxBuf, qTile * sizeof(float));
        pipe.InitBuffer(rowSumBuf, qTile * sizeof(float));

        pipe.InitBuffer(kTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));
        pipe.InitBuffer(vTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));

        // score buffer for exp input: [qTile, sTile]
        pipe.InitBuffer(scoreBuf, (uint32_t)((uint64_t)qTile * (uint64_t)sTile * sizeof(float)));
        // exp buffer: [qTile, sTile]
        pipe.InitBuffer(expBuf, (uint32_t)((uint64_t)qTile * (uint64_t)sTile * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) bnum = 1;

        const uint32_t qGroups = (S + qTile - 1) / qTile;
        const uint32_t totalTasks = B * H * qGroups;

        for (uint32_t task = bid; task < totalTasks; task += bnum) {
            const uint32_t bh = task / qGroups;
            const uint32_t qg = task - bh * qGroups;
            const uint32_t b = bh / H;
            const uint32_t h = bh - b * H;
            const uint32_t qi0 = qg * qTile;
            const uint32_t curQ = (qi0 + qTile <= S) ? qTile : (S - qi0);
            ComputeTile(b, h, qi0, curQ);
        }
    }

private:
    __aicore__ inline uint64_t QOff(uint32_t b, uint32_t h, uint32_t s, uint32_t d) const
    {
        return (((uint64_t)b * (uint64_t)H + (uint64_t)h) * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d;
    }

    __aicore__ inline uint64_t KVOff(uint32_t b, uint32_t s, uint32_t d) const
    {
        // [B,1,S,D] flattened as [B,S,D]
        return ((uint64_t)b * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d;
    }

    __aicore__ inline float Dot(const AscendC::LocalTensor<float>& qRow,
                                const AscendC::LocalTensor<float>& kRow) const
    {
        // D <= 128. Scalar FMA loop reduces vector dispatch/barriers.
        float acc = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            acc += qRow.GetValue(d) * kRow.GetValue(d);
        }
        return acc * scale;
    }

    __aicore__ inline void ComputeTile(uint32_t b, uint32_t h, uint32_t qi0, uint32_t curQ)
    {
        AscendC::LocalTensor<float> qTileLt = qTileBuf.Get<float>();     // [qTile*D]
        AscendC::LocalTensor<float> outTileLt = outTileBuf.Get<float>(); // [qTile*D]
        AscendC::LocalTensor<float> rowMaxLt = rowMaxBuf.Get<float>();   // [qTile]
        AscendC::LocalTensor<float> rowSumLt = rowSumBuf.Get<float>();   // [qTile]
        AscendC::LocalTensor<float> kTileLt = kTileBuf.Get<float>();     // [sTile*D]
        AscendC::LocalTensor<float> vTileLt = vTileBuf.Get<float>();     // [sTile*D]
        AscendC::LocalTensor<float> scoreLt = scoreBuf.Get<float>();     // [qTile*sTile]
        AscendC::LocalTensor<float> expLt = expBuf.Get<float>();         // [qTile*sTile]

        // Load Q tile
        for (uint32_t qi = 0; qi < curQ; ++qi) {
            AscendC::DataCopy(qTileLt[qi * D], qGm[QOff(b, h, qi0 + qi, 0)], D);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();

        // Init rowMax to -inf and out to 0, rowSum to 0
        for (uint32_t qi = 0; qi < curQ; ++qi) {
            rowMaxLt.SetValue(qi, -3.402823466e+38f);
            rowSumLt.SetValue(qi, 0.0f);
            // out row = 0
            AscendC::Duplicate(outTileLt[qi * D], 0.0f, (int32_t)D);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Pass 1: compute rowMax for each qi in tile, streaming K
        for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
            const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

            // Safe per-row copy; avoid unsafe burst assumptions.
            for (uint32_t j = 0; j < curS; ++j) {
                AscendC::DataCopy(kTileLt[j * D], kGm[KVOff(b, s0 + j, 0)], D);
            }
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t qi = 0; qi < curQ; ++qi) {
                AscendC::LocalTensor<float> qRow = qTileLt[qi * D];
                float rmax = rowMaxLt.GetValue(qi);
                for (uint32_t j = 0; j < curS; ++j) {
                    AscendC::LocalTensor<float> kRow = kTileLt[j * D];
                    const float sc = Dot(qRow, kRow);
                    if (sc > rmax) rmax = sc;
                }
                rowMaxLt.SetValue(qi, rmax);
            }
        }

        // Pass 2: stream K+V, compute exp(score-rowMax), accumulate denom and out
        for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
            const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

            for (uint32_t j = 0; j < curS; ++j) {
                AscendC::DataCopy(kTileLt[j * D], kGm[KVOff(b, s0 + j, 0)], D);
                AscendC::DataCopy(vTileLt[j * D], vGm[KVOff(b, s0 + j, 0)], D);
            }
            AscendC::PipeBarrier<PIPE_MTE2>();

            // Fill scoreLt[qi, j]
            for (uint32_t qi = 0; qi < curQ; ++qi) {
                AscendC::LocalTensor<float> qRow = qTileLt[qi * D];
                const float rmax = rowMaxLt.GetValue(qi);
                const uint32_t base = qi * sTile;
                for (uint32_t j = 0; j < curS; ++j) {
                    const float sc = Dot(qRow, kTileLt[j * D]) - rmax;
                    scoreLt.SetValue(base + j, sc);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();

            // expLt = exp(scoreLt) for each qi row (not in-place)
            for (uint32_t qi = 0; qi < curQ; ++qi) {
                const uint32_t base = qi * sTile;
                AscendC::Exp(expLt[base], scoreLt[base], (int32_t)curS);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // Accumulate denom and out
            for (uint32_t qi = 0; qi < curQ; ++qi) {
                const uint32_t base = qi * sTile;
                float denom = rowSumLt.GetValue(qi);

                // denom scalar accumulation (curS <= 192)
                for (uint32_t j = 0; j < curS; ++j) {
                    denom += expLt.GetValue(base + j);
                }
                rowSumLt.SetValue(qi, denom);

                // out += sum_j exp * vRow via vector Axpy per j (no extra prob scaling buffer)
                AscendC::LocalTensor<float> outRow = outTileLt[qi * D];
                for (uint32_t j = 0; j < curS; ++j) {
                    const float w = expLt.GetValue(base + j);
                    AscendC::Axpy(outRow, vTileLt[j * D], w, (int32_t)D);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Normalize and store
        for (uint32_t qi = 0; qi < curQ; ++qi) {
            const float inv = 1.0f / rowSumLt.GetValue(qi);
            AscendC::Muls(outTileLt[qi * D], outTileLt[qi * D], inv, (int32_t)D);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t qi = 0; qi < curQ; ++qi) {
            AscendC::DataCopy(yGm[QOff(b, h, qi0 + qi, 0)], outTileLt[qi * D], D);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> qTileBuf;
    AscendC::TBuf<> outTileBuf;
    AscendC::TBuf<> rowMaxBuf;
    AscendC::TBuf<> rowSumBuf;
    AscendC::TBuf<> kTileBuf;
    AscendC::TBuf<> vTileBuf;
    AscendC::TBuf<> scoreBuf;
    AscendC::TBuf<> expBuf;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, yGm;

    uint32_t B {0}, H {0}, S {0}, D {0};
    uint32_t sTile {0}, qTile {0};
    float scale {1.0f};
};

extern "C" __global__ __aicore__ void multi_query_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiQueryAttentionCustom op;
    op.Init(q, k, v, y,
            tiling_data.B, tiling_data.H, tiling_data.S, tiling_data.D,
            tiling_data.sTile, tiling_data.qTile, tiling_data.scale);
    op.Process();
}
