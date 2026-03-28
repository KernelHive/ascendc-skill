
#include "kernel_operator.h"

// Row-parallel fused attention:
// q,k,v,y: [B,H,S,D] fp32 contiguous
// Map one work-item to one row (b,h,qi). This increases parallelism (reduces pipeline gaps)
// and keeps per-core working-set small.

class KernelAdaptiveAttentionCustom {
public:
    __aicore__ inline KernelAdaptiveAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t S, uint32_t D,
                               uint32_t totalRows, uint32_t sTile, uint32_t dTile, float scale)
    {
        this->B = B; this->H = H; this->S = S; this->D = D;
        this->totalRows = totalRows;
        this->sTile = sTile; this->dTile = dTile;
        this->scale = scale;

        const uint64_t elems = (uint64_t)B * (uint64_t)H * (uint64_t)S * (uint64_t)D;
        qGm.SetGlobalBuffer((__gm__ float*)q, elems);
        kGm.SetGlobalBuffer((__gm__ float*)k, elems);
        vGm.SetGlobalBuffer((__gm__ float*)v, elems);
        yGm.SetGlobalBuffer((__gm__ float*)y, elems);

        pipe.InitBuffer(qRowBuf, D * sizeof(float));
        pipe.InitBuffer(outRowBuf, D * sizeof(float));

        pipe.InitBuffer(kTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));
        pipe.InitBuffer(vTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));

        // One buffer reused:
        //  - holds logits during max/exp
        //  - holds exp/logits then normalized probs
        pipe.InitBuffer(probBuf, sTile * sizeof(float));

        // tmpReduce[max(sTile,dTile)], redOut[8], mulTmp[dTile], prodAcc[dTile], vTmp[dTile], vAcc[dTile]
        const uint32_t redLen = (sTile > dTile) ? sTile : dTile;
        const uint32_t calcFloats = redLen + 8u + 5u * dTile + 16u;
        pipe.InitBuffer(calcBuf, calcFloats * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) bnum = 1;

        for (uint32_t row = bid; row < totalRows; row += bnum) {
            uint32_t tmp = row;
            const uint32_t qi = tmp % S;
            tmp /= S;
            const uint32_t h = tmp % H;
            const uint32_t b = tmp / H;
            ComputeRow(b, h, qi);
        }
    }

private:
    __aicore__ inline uint64_t Off(uint32_t b, uint32_t h, uint32_t s, uint32_t d) const
    {
        return (((uint64_t)b * (uint64_t)H + (uint64_t)h) * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d;
    }

    __aicore__ inline void LoadKTile(uint32_t b, uint32_t h, uint32_t s0, uint32_t curS,
                                     AscendC::LocalTensor<float>& kTile) const
    {
        AscendC::DataCopy(kTile, kGm[Off(b, h, s0, 0)], curS * D);
    }

    __aicore__ inline void LoadVTile(uint32_t b, uint32_t h, uint32_t s0, uint32_t curS,
                                     AscendC::LocalTensor<float>& vTile) const
    {
        AscendC::DataCopy(vTile, vGm[Off(b, h, s0, 0)], curS * D);
    }

    __aicore__ inline float DotQK(const AscendC::LocalTensor<float>& qRow,
                                 const AscendC::LocalTensor<float>& kRow,
                                 AscendC::LocalTensor<float>& mulTmp,
                                 AscendC::LocalTensor<float>& prodAcc,
                                 AscendC::LocalTensor<float>& tmpReduce,
                                 AscendC::LocalTensor<float>& redOut) const
    {
        // Specialize common fast path: D <= 64 (one vector mul + ReduceSum)
        if (D <= 64u) {
            AscendC::Mul(prodAcc, qRow[0], kRow[0], (int32_t)D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(redOut, prodAcc, tmpReduce, (int32_t)D);
            AscendC::PipeBarrier<PIPE_V>();
            return redOut.GetValue(0) * scale;
        }

        // General path: chunked accumulation by dTile then ReduceSum
        uint32_t d0 = 0;
        if (dTile <= D) {
            AscendC::Mul(prodAcc, qRow[0], kRow[0], (int32_t)dTile);
            d0 = dTile;
            AscendC::PipeBarrier<PIPE_V>();

            for (; d0 + dTile <= D; d0 += dTile) {
                AscendC::Mul(mulTmp, qRow[d0], kRow[d0], (int32_t)dTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(prodAcc, prodAcc, mulTmp, (int32_t)dTile);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::ReduceSum<float>(redOut, prodAcc, tmpReduce, (int32_t)dTile);
            AscendC::PipeBarrier<PIPE_V>();
            float acc = redOut.GetValue(0);

            for (; d0 < D; ++d0) {
                acc += qRow.GetValue(d0) * kRow.GetValue(d0);
            }
            return acc * scale;
        }

        float acc = 0.0f;
        for (uint32_t dd = 0; dd < D; ++dd) {
            acc += qRow.GetValue(dd) * kRow.GetValue(dd);
        }
        return acc * scale;
    }

    __aicore__ inline void ZeroOutRow(AscendC::LocalTensor<float>& outRow) const
    {
        uint32_t d0 = 0;
        for (; d0 + dTile <= D; d0 += dTile) {
            AscendC::Duplicate(outRow[d0], 0.0f, (int32_t)dTile);
        }
        for (; d0 < D; ++d0) outRow.SetValue(d0, 0.0f);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRow(uint32_t b, uint32_t h, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow  = qRowBuf.Get<float>();
        AscendC::LocalTensor<float> outRow = outRowBuf.Get<float>();
        AscendC::LocalTensor<float> kTile = kTileBuf.Get<float>();
        AscendC::LocalTensor<float> vTile = vTileBuf.Get<float>();
        AscendC::LocalTensor<float> prob  = probBuf.Get<float>();

        AscendC::LocalTensor<float> tmpAll = calcBuf.Get<float>();
        const uint32_t redLen = (sTile > dTile) ? sTile : dTile;

        AscendC::LocalTensor<float> tmpReduce = tmpAll;                       // [redLen]
        AscendC::LocalTensor<float> redOut    = tmpAll[redLen];               // [8]
        AscendC::LocalTensor<float> mulTmp    = tmpAll[redLen + 8];           // [dTile]
        AscendC::LocalTensor<float> prodAcc   = tmpAll[redLen + 8 + dTile];   // [dTile]
        AscendC::LocalTensor<float> vTmp      = tmpAll[redLen + 8 + 2u*dTile];// [dTile]
        AscendC::LocalTensor<float> vAccTmp   = tmpAll[redLen + 8 + 3u*dTile];// [dTile]

        AscendC::DataCopy(qRow, qGm[Off(b, h, qi, 0)], D);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // Pass 1: rowMax
        float rowMax = -3.402823466e+38f;
        for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
            const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);
            LoadKTile(b, h, s0, curS, kTile);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t j = 0; j < curS; ++j) {
                AscendC::LocalTensor<float> kRow = kTile[(uint32_t)((uint64_t)j * (uint64_t)D)];
                const float sc = DotQK(qRow, kRow, mulTmp, prodAcc, tmpReduce, redOut);
                if (sc > rowMax) rowMax = sc;
            }
        }

        // Pass 2: rowSum; reuse prob as logits then exp
        float rowSum = 0.0f;
        for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
            const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);
            LoadKTile(b, h, s0, curS, kTile);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t j = 0; j < curS; ++j) {
                AscendC::LocalTensor<float> kRow = kTile[(uint32_t)((uint64_t)j * (uint64_t)D)];
                prob.SetValue(j, DotQK(qRow, kRow, mulTmp, prodAcc, tmpReduce, redOut) - rowMax);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(prob, prob, (int32_t)curS);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceSum<float>(redOut, prob, tmpReduce, (int32_t)curS);
            AscendC::PipeBarrier<PIPE_V>();
            rowSum += redOut.GetValue(0);
        }
        const float invSum = 1.0f / rowSum;

        ZeroOutRow(outRow);

        // Pass 3: recompute probs and accumulate out
        for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
            const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

            LoadKTile(b, h, s0, curS, kTile);
            LoadVTile(b, h, s0, curS, vTile);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t j = 0; j < curS; ++j) {
                AscendC::LocalTensor<float> kRow = kTile[(uint32_t)((uint64_t)j * (uint64_t)D)];
                prob.SetValue(j, DotQK(qRow, kRow, mulTmp, prodAcc, tmpReduce, redOut) - rowMax);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(prob, prob, (int32_t)curS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(prob, prob, invSum, (int32_t)curS);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t d0 = 0; d0 < D; d0 += dTile) {
                const uint32_t curD = (d0 + dTile <= D) ? dTile : (D - d0);
                if (curD == dTile) {
                    AscendC::Duplicate(vAccTmp, 0.0f, (int32_t)dTile);
                    AscendC::PipeBarrier<PIPE_V>();

                    for (uint32_t j = 0; j < curS; ++j) {
                        const float w = prob.GetValue(j);
                        const uint64_t vBase = (uint64_t)j * (uint64_t)D + d0;
                        AscendC::Muls(vTmp, vTile[(uint32_t)vBase], w, (int32_t)dTile);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(vAccTmp, vAccTmp, vTmp, (int32_t)dTile);
                        AscendC::PipeBarrier<PIPE_V>();
                    }

                    AscendC::Add(outRow[d0], outRow[d0], vAccTmp, (int32_t)dTile);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for (uint32_t dd = 0; dd < curD; ++dd) {
                        float acc = outRow.GetValue(d0 + dd);
                        for (uint32_t j = 0; j < curS; ++j) {
                            acc += prob.GetValue(j) *
                                   vTile.GetValue((uint32_t)((uint64_t)j * (uint64_t)D + (d0 + dd)));
                        }
                        outRow.SetValue(d0 + dd, acc);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }

        AscendC::DataCopy(yGm[Off(b, h, qi, 0)], outRow, D);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> qRowBuf;
    AscendC::TBuf<> outRowBuf;
    AscendC::TBuf<> kTileBuf;
    AscendC::TBuf<> vTileBuf;
    AscendC::TBuf<> probBuf;
    AscendC::TBuf<> calcBuf;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, yGm;

    uint32_t B {0}, H {0}, S {0}, D {0};
    uint32_t totalRows {0};
    uint32_t sTile {0}, dTile {0};
    float scale {1.0f};
};

extern "C" __global__ __aicore__ void adaptive_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdaptiveAttentionCustom op;
    op.Init(q, k, v, y,
            tiling_data.B, tiling_data.H, tiling_data.S, tiling_data.D,
            tiling_data.totalRows,
            tiling_data.sTile, tiling_data.dTile, tiling_data.scale);
    op.Process();
}
