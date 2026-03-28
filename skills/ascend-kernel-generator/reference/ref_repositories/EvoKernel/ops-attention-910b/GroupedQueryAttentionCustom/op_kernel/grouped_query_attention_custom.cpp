
#include "kernel_operator.h"

// Fused GQA attention core:
// q: [B,H,S,D] float32
// k/v: [B,Hkv,S,D] float32 (shared across groups, H = Hkv*G)
// y: [B,H,S,D] float32
//
// Each query head h maps to kv head hk = h / G.

constexpr int32_t BUFFER_NUM = 1;

class KernelGroupedQueryAttentionCustom {
public:
    __aicore__ inline KernelGroupedQueryAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t Hkv, uint32_t G,
                               uint32_t S, uint32_t D,
                               uint32_t sTile, uint32_t dTile, float scale)
    {
        this->B = B; this->H = H; this->Hkv = Hkv; this->G = G;
        this->S = S; this->D = D;
        this->sTile = sTile; this->dTile = dTile;
        this->scale = scale;

        totalBH = B * H;

        const uint64_t qElems  = (uint64_t)B * (uint64_t)H * (uint64_t)S * (uint64_t)D;
        const uint64_t kvElems = (uint64_t)B * (uint64_t)Hkv * (uint64_t)S * (uint64_t)D;

        qGm.SetGlobalBuffer((__gm__ float*)q, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, kvElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, kvElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, qElems);

        // UB buffers (avoid misaligned GetWithOffset; allocate dedicated buffers per tensor)
        pipe.InitBuffer(qRowBuf, D * sizeof(float));
        pipe.InitBuffer(outRowBuf, D * sizeof(float));

        pipe.InitBuffer(kTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));
        pipe.InitBuffer(vTileBuf, (uint32_t)((uint64_t)sTile * (uint64_t)D * sizeof(float)));

        pipe.InitBuffer(scoreTileBuf, sTile * sizeof(float));
        pipe.InitBuffer(probTileBuf,  sTile * sizeof(float));

        const uint32_t redLen = (sTile > dTile) ? sTile : dTile;
        // tmpReduce[redLen] + redOut[8] + dotAcc[dTile] + tmpMul[dTile] + vAccTmp[dTile] + padding
        const uint32_t calcFloats = redLen + 8u + 3u * dTile + 32u;
        pipe.InitBuffer(calcBuf, calcFloats * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) bnum = 1;

        for (uint32_t bh = bid; bh < totalBH; bh += bnum) {
            const uint32_t b = bh / H;
            const uint32_t h = bh - b * H;
            const uint32_t hk = h / G; // kv head
            ComputeBHHk(b, h, hk);
        }
    }

private:
    __aicore__ inline uint64_t QOff(uint32_t b, uint32_t h, uint32_t s, uint32_t d) const
    {
        return (((uint64_t)b * (uint64_t)H + (uint64_t)h) * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d;
    }

    __aicore__ inline uint64_t KVOff(uint32_t b, uint32_t hk, uint32_t s, uint32_t d) const
    {
        return (((uint64_t)b * (uint64_t)Hkv + (uint64_t)hk) * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d;
    }

    __aicore__ inline float DotQK_FMA(const AscendC::LocalTensor<float>& qRow,
                                     const AscendC::LocalTensor<float>& kTile,
                                     AscendC::LocalTensor<float>& dotAcc,
                                     AscendC::LocalTensor<float>& tmpMul,
                                     AscendC::LocalTensor<float>& tmpReduce,
                                     AscendC::LocalTensor<float>& redOut,
                                     uint32_t j) const
    {
        const uint32_t base = j * D;

        AscendC::Duplicate(dotAcc, 0.0f, (int32_t)dTile);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t d0 = 0;
        for (; d0 + dTile <= D; d0 += dTile) {
            AscendC::Mul(tmpMul, qRow[d0], kTile[base + d0], (int32_t)dTile);
            AscendC::Add(dotAcc, dotAcc, tmpMul, (int32_t)dTile);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::ReduceSum<float>(redOut, dotAcc, tmpReduce, (int32_t)dTile);
        AscendC::PipeBarrier<PIPE_V>();
        float acc = redOut.GetValue(0);

        for (; d0 < D; ++d0) {
            acc += qRow.GetValue(d0) * kTile.GetValue(base + d0);
        }
        return acc * scale;
    }

    __aicore__ inline void ComputeBHHk(uint32_t b, uint32_t h, uint32_t hk)
    {
        AscendC::LocalTensor<float> qRow   = qRowBuf.Get<float>();
        AscendC::LocalTensor<float> outRow = outRowBuf.Get<float>();

        AscendC::LocalTensor<float> kTile = kTileBuf.Get<float>();
        AscendC::LocalTensor<float> vTile = vTileBuf.Get<float>();

        AscendC::LocalTensor<float> scoreTile = scoreTileBuf.Get<float>();
        AscendC::LocalTensor<float> probTile  = probTileBuf.Get<float>();

        AscendC::LocalTensor<float> tmpAll = calcBuf.Get<float>();
        const uint32_t redLen = (sTile > dTile) ? sTile : dTile;
        AscendC::LocalTensor<float> tmpReduce = tmpAll;                     // [redLen]
        AscendC::LocalTensor<float> redOut    = tmpAll[redLen];             // [>=1]
        AscendC::LocalTensor<float> dotAcc    = tmpAll[redLen + 8];         // [dTile]
        AscendC::LocalTensor<float> tmpMul    = tmpAll[redLen + 8 + dTile]; // [dTile]
        AscendC::LocalTensor<float> vAccTmp   = tmpAll[redLen + 8 + 2u*dTile]; // [dTile]

        for (uint32_t qi = 0; qi < S; ++qi) {
            AscendC::DataCopy(qRow, qGm[QOff(b, h, qi, 0)], D);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // Pass 1: row max
            float rowMax = -3.402823466e+38f; // -FLT_MAX
            for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
                const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

                for (uint32_t j = 0; j < curS; ++j) {
                    AscendC::DataCopy(kTile[j * D], kGm[KVOff(b, hk, s0 + j, 0)], D);
                }
                AscendC::PipeBarrier<PIPE_MTE2>();

                for (uint32_t j = 0; j < curS; ++j) {
                    float sc = DotQK_FMA(qRow, kTile, dotAcc, tmpMul, tmpReduce, redOut, j);
                    if (sc > rowMax) rowMax = sc;
                }
            }

            // Pass 2: sum exp(scores - rowMax)
            float rowSum = 0.0f;
            for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
                const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

                for (uint32_t j = 0; j < curS; ++j) {
                    AscendC::DataCopy(kTile[j * D], kGm[KVOff(b, hk, s0 + j, 0)], D);
                }
                AscendC::PipeBarrier<PIPE_MTE2>();

                for (uint32_t j = 0; j < curS; ++j) {
                    float sc = DotQK_FMA(qRow, kTile, dotAcc, tmpMul, tmpReduce, redOut, j) - rowMax;
                    scoreTile.SetValue(j, sc);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Exp(probTile, scoreTile, (int32_t)curS);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::ReduceSum<float>(redOut, probTile, tmpReduce, (int32_t)curS);
                AscendC::PipeBarrier<PIPE_V>();
                rowSum += redOut.GetValue(0);
            }
            const float invSum = 1.0f / rowSum;

            // outRow = 0
            uint32_t dInit = 0;
            for (; dInit + dTile <= D; dInit += dTile) {
                AscendC::Duplicate(outRow[dInit], 0.0f, (int32_t)dTile);
            }
            for (; dInit < D; ++dInit) outRow.SetValue(dInit, 0.0f);
            AscendC::PipeBarrier<PIPE_V>();

            // Pass 3: outRow = sum softmax * V
            for (uint32_t s0 = 0; s0 < S; s0 += sTile) {
                const uint32_t curS = (s0 + sTile <= S) ? sTile : (S - s0);

                for (uint32_t j = 0; j < curS; ++j) {
                    AscendC::DataCopy(kTile[j * D], kGm[KVOff(b, hk, s0 + j, 0)], D);
                    AscendC::DataCopy(vTile[j * D], vGm[KVOff(b, hk, s0 + j, 0)], D);
                }
                AscendC::PipeBarrier<PIPE_MTE2>();

                for (uint32_t j = 0; j < curS; ++j) {
                    float sc = DotQK_FMA(qRow, kTile, dotAcc, tmpMul, tmpReduce, redOut, j) - rowMax;
                    scoreTile.SetValue(j, sc);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Exp(probTile, scoreTile, (int32_t)curS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(probTile, probTile, invSum, (int32_t)curS);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t d0 = 0; d0 < D; d0 += dTile) {
                    const uint32_t curD = (d0 + dTile <= D) ? dTile : (D - d0);

                    if (curD == dTile) {
                        AscendC::Duplicate(vAccTmp, 0.0f, (int32_t)dTile);
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t j = 0; j < curS; ++j) {
                            const float w = probTile.GetValue(j);
                            const uint32_t vBase = j * D + d0;
                            AscendC::Axpy(vAccTmp, vTile[vBase], w, (int32_t)dTile);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outRow[d0], outRow[d0], vAccTmp, (int32_t)dTile);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        for (uint32_t dd = 0; dd < curD; ++dd) {
                            float acc = outRow.GetValue(d0 + dd);
                            for (uint32_t j = 0; j < curS; ++j) {
                                acc += probTile.GetValue(j) * vTile.GetValue(j * D + (d0 + dd));
                            }
                            outRow.SetValue(d0 + dd, acc);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }

            AscendC::DataCopy(yGm[QOff(b, h, qi, 0)], outRow, D);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> qRowBuf;
    AscendC::TBuf<> outRowBuf;
    AscendC::TBuf<> kTileBuf;
    AscendC::TBuf<> vTileBuf;
    AscendC::TBuf<> scoreTileBuf;
    AscendC::TBuf<> probTileBuf;
    AscendC::TBuf<> calcBuf;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, yGm;

    uint32_t B {0}, H {0}, Hkv {0}, G {0}, S {0}, D {0};
    uint32_t totalBH {0};
    uint32_t sTile {0}, dTile {0};
    float scale {1.0f};
};

extern "C" __global__ __aicore__ void grouped_query_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupedQueryAttentionCustom op;
    op.Init(q, k, v, y,
            tiling_data.B, tiling_data.H, tiling_data.Hkv, tiling_data.G,
            tiling_data.S, tiling_data.D,
            tiling_data.sTile, tiling_data.dTile, tiling_data.scale);
    op.Process();
}
