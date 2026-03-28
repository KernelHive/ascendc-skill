
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// q,k,v: [B,H,NB,BS,DK]
// y:     [B,H,NB,BS,DK]
class KernelBlockSparseAttention {
public:
    __aicore__ inline KernelBlockSparseAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR scale, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t NB, uint32_t BS, uint32_t DK,
                               uint32_t dkTile, uint32_t jTile, uint32_t useFastPath32)
    {
        this->B = B; this->H = H; this->NB = NB; this->BS = BS; this->DK = DK;
        this->dkTile = dkTile; this->jTile = jTile;
        this->useFastPath32 = useFastPath32;
        this->totalBlocks = B * H * NB;

        const uint64_t blockElem = (uint64_t)BS * (uint64_t)DK;
        const uint64_t totalElems = (uint64_t)this->totalBlocks * blockElem;

        qGm.SetGlobalBuffer((__gm__ float*)q, totalElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, totalElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, totalElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);

        pipe.InitBuffer(qQue, BUFFER_NUM, blockElem * sizeof(float));
        pipe.InitBuffer(kQue, BUFFER_NUM, blockElem * sizeof(float));
        pipe.InitBuffer(vQue, BUFFER_NUM, blockElem * sizeof(float));
        pipe.InitBuffer(outBuf, blockElem * sizeof(float));

        // UB layout (avoid SetValue/GetValue on scoresAcc; keep dotBuf separately):
        // scoresRow[BS] | expRow[BS] | tmpReduce[max(BS,dkTile)] | qTile[dkTile] | mulTmp[dkTile] | vTmp[dkTile] |
        // dotBuf[BS] | reduceOut[16]
        const uint32_t tmpReduceLen = (BS > dkTile) ? BS : dkTile;
        const uint32_t calcFloats = (2u * BS) + tmpReduceLen + (3u * dkTile) + BS + 16u;
        pipe.InitBuffer(calcBuf, calcFloats * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) bnum = 1;

        scaleVal = scaleGm.GetValue(0);

        bool hasCur = false;
        uint32_t curBlk = 0;
        uint32_t nextBlk = bid;

        if (nextBlk < totalBlocks) {
            CopyInBlock(nextBlk);
            hasCur = true;
            curBlk = nextBlk;
            nextBlk += bnum;
        }

        while (hasCur) {
            if (nextBlk < totalBlocks) {
                CopyInBlock(nextBlk);
            }

            ComputeBlock();
            CopyOutBlock(curBlk);

            if (nextBlk < totalBlocks) {
                curBlk = nextBlk;
                nextBlk += bnum;
            } else {
                hasCur = false;
            }
        }
    }

private:
    __aicore__ inline void CopyInBlock(uint32_t blkLinear)
    {
        const uint64_t blockElem = (uint64_t)BS * (uint64_t)DK;
        const uint64_t base = (uint64_t)blkLinear * blockElem;

        AscendC::LocalTensor<float> qLocal = qQue.AllocTensor<float>();
        AscendC::LocalTensor<float> kLocal = kQue.AllocTensor<float>();
        AscendC::LocalTensor<float> vLocal = vQue.AllocTensor<float>();

        AscendC::DataCopy(qLocal, qGm[base], (uint32_t)blockElem);
        AscendC::DataCopy(kLocal, kGm[base], (uint32_t)blockElem);
        AscendC::DataCopy(vLocal, vGm[base], (uint32_t)blockElem);

        qQue.EnQue(qLocal);
        kQue.EnQue(kLocal);
        vQue.EnQue(vLocal);
    }

    __aicore__ inline void ComputeScoresBS32(uint32_t i,
                                            const AscendC::LocalTensor<float>& qLocal,
                                            const AscendC::LocalTensor<float>& kLocal,
                                            AscendC::LocalTensor<float>& scoresRow,
                                            AscendC::LocalTensor<float>& qTile,
                                            AscendC::LocalTensor<float>& mulTmp,
                                            AscendC::LocalTensor<float>& tmpReduce,
                                            AscendC::LocalTensor<float>& dotBuf,
                                            AscendC::LocalTensor<float>& reduceOutDot)
    {
        // dotBuf[j] accumulates unscaled dot products for j in [0,32)
        AscendC::Duplicate(dotBuf, 0.0f, 32);
        // no barrier needed here; next ops overwrite/accumulate deterministically

        const uint64_t qBase = (uint64_t)i * (uint64_t)DK;
        const uint32_t dkT = dkTile;

        // DK is multiple of 32 for this path.
        for (uint32_t d = 0; d < DK; d += dkT) {
            const uint32_t curD = (d + dkT <= DK) ? dkT : (DK - d);
            AscendC::DataCopy(qTile, qLocal[(uint32_t)(qBase + d)], curD);

            // For each j, accumulate ReduceSum of elementwise product for this tile into dotBuf[j].
            // Store partial sums into UB (dotBuf) to avoid scalar read-modify-write on scoresAcc tensors.
            for (uint32_t j = 0; j < 32; ++j) {
                const uint64_t kBase = (uint64_t)j * (uint64_t)DK + d;
                AscendC::Mul(mulTmp, qTile, kLocal[(uint32_t)kBase], (int32_t)curD);
                AscendC::ReduceSum<float>(reduceOutDot, mulTmp, tmpReduce, (int32_t)curD);
                float acc = dotBuf.GetValue(j) + reduceOutDot.GetValue(0);
                dotBuf.SetValue(j, acc);
            }
        }

        AscendC::Muls(scoresRow, dotBuf, scaleVal, 32);
    }

    __aicore__ inline float DotQKSlow(uint32_t qi, uint32_t kj,
                                      const AscendC::LocalTensor<float>& qLocal,
                                      const AscendC::LocalTensor<float>& kLocal,
                                      AscendC::LocalTensor<float>& mulTmp,
                                      AscendC::LocalTensor<float>& tmpReduce,
                                      AscendC::LocalTensor<float>& reduceOut)
    {
        const uint64_t qBase = (uint64_t)qi * (uint64_t)DK;
        const uint64_t kBase = (uint64_t)kj * (uint64_t)DK;
        float acc = 0.0f;

        const uint32_t dkT = dkTile;
        uint32_t d = 0;
        for (; d + dkT <= DK; d += dkT) {
            AscendC::Mul(mulTmp, qLocal[(uint32_t)(qBase + d)], kLocal[(uint32_t)(kBase + d)], (int32_t)dkT);
            AscendC::ReduceSum<float>(reduceOut, mulTmp, tmpReduce, (int32_t)dkT);
            acc += reduceOut.GetValue(0);
        }
        for (; d < DK; ++d) {
            acc += qLocal.GetValue((uint32_t)(qBase + d)) * kLocal.GetValue((uint32_t)(kBase + d));
        }
        return acc;
    }

    __aicore__ inline void ComputeBlock()
    {
        AscendC::LocalTensor<float> qLocal = qQue.DeQue<float>();
        AscendC::LocalTensor<float> kLocal = kQue.DeQue<float>();
        AscendC::LocalTensor<float> vLocal = vQue.DeQue<float>();

        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();
        AscendC::LocalTensor<float> tmpAll = calcBuf.Get<float>();

        AscendC::LocalTensor<float> scoresRow = tmpAll;               // [BS]
        AscendC::LocalTensor<float> expRow    = tmpAll[BS];           // [BS]

        const uint32_t tmpReduceLen = (BS > dkTile) ? BS : dkTile;
        AscendC::LocalTensor<float> tmpReduce = tmpAll[2u * BS];      // [tmpReduceLen]

        AscendC::LocalTensor<float> qTile  = tmpAll[2u * BS + tmpReduceLen];                 // [dkTile]
        AscendC::LocalTensor<float> mulTmp = tmpAll[2u * BS + tmpReduceLen + dkTile];        // [dkTile]
        AscendC::LocalTensor<float> vTmp   = tmpAll[2u * BS + tmpReduceLen + 2u * dkTile];   // [dkTile]

        AscendC::LocalTensor<float> dotBuf = tmpAll[2u * BS + tmpReduceLen + 3u * dkTile];   // [BS]

        AscendC::LocalTensor<float> reduceOutMax = tmpAll[2u * BS + tmpReduceLen + 3u * dkTile + BS];      // scalar
        AscendC::LocalTensor<float> reduceOutSum = tmpAll[2u * BS + tmpReduceLen + 3u * dkTile + BS + 4];  // scalar
        AscendC::LocalTensor<float> reduceOutDot = tmpAll[2u * BS + tmpReduceLen + 3u * dkTile + BS + 8];  // scalar

        const uint32_t jT = jTile;
        const uint32_t dkT = dkTile;

        for (uint32_t i = 0; i < BS; ++i) {
            if (useFastPath32 != 0) {
                // BS is 32 for this path.
                ComputeScoresBS32(i, qLocal, kLocal, scoresRow, qTile, mulTmp, tmpReduce, dotBuf, reduceOutDot);
            } else {
                for (uint32_t j = 0; j < BS; ++j) {
                    float dot = DotQKSlow(i, j, qLocal, kLocal, mulTmp, tmpReduce, reduceOutDot);
                    scoresRow.SetValue(j, dot * scaleVal);
                }
            }

            // softmax(scoresRow) -> expRow
            AscendC::ReduceMax<float>(reduceOutMax, scoresRow, tmpReduce, (int32_t)BS);
            float rowMax = reduceOutMax.GetValue(0);

            AscendC::Adds(scoresRow, scoresRow, -rowMax, (int32_t)BS);
            AscendC::Exp(expRow, scoresRow, (int32_t)BS);

            AscendC::ReduceSum<float>(reduceOutSum, expRow, tmpReduce, (int32_t)BS);
            float invSum = 1.0f / reduceOutSum.GetValue(0);
            AscendC::Muls(expRow, expRow, invSum, (int32_t)BS);

            // out[i,:] = expRow @ V
            const uint64_t outBase = (uint64_t)i * (uint64_t)DK;

            // zero init out row
            uint32_t d0 = 0;
            for (; d0 + dkT <= DK; d0 += dkT) {
                AscendC::Duplicate(outLocal[(uint32_t)(outBase + d0)], 0.0f, (int32_t)dkT);
            }
            for (; d0 < DK; ++d0) outLocal.SetValue((uint32_t)(outBase + d0), 0.0f);

            if (useFastPath32 != 0 && BS == 32) {
                // Unrolled j-loop in tiles for BS=32 to reduce loop control overhead.
                for (uint32_t d = 0; d < DK; d += dkT) {
                    const uint32_t curD = (d + dkT <= DK) ? dkT : (DK - d);
                    auto outTile = outLocal[(uint32_t)(outBase + d)];
                    // j in [0,32)
                    for (uint32_t j = 0; j < 32; ++j) {
                        const float w = expRow.GetValue(j);
                        const uint64_t vBase = (uint64_t)j * (uint64_t)DK + d;
                        AscendC::Axpy(outTile, vLocal[(uint32_t)vBase], w, (int32_t)curD);
                    }
                }
            } else {
                for (uint32_t j0 = 0; j0 < BS; j0 += jT) {
                    const uint32_t curJT = (j0 + jT <= BS) ? jT : (BS - j0);
                    for (uint32_t d = 0; d < DK; d += dkT) {
                        const uint32_t curD = (d + dkT <= DK) ? dkT : (DK - d);
                        auto outTile = outLocal[(uint32_t)(outBase + d)];
                        for (uint32_t jj = 0; jj < curJT; ++jj) {
                            const uint32_t j = j0 + jj;
                            const float w = expRow.GetValue(j);
                            const uint64_t vBase = (uint64_t)j * (uint64_t)DK + d;
                            AscendC::Axpy(outTile, vLocal[(uint32_t)vBase], w, (int32_t)curD);
                        }
                    }
                }
            }
        }

        qQue.FreeTensor(qLocal);
        kQue.FreeTensor(kLocal);
        vQue.FreeTensor(vLocal);
    }

    __aicore__ inline void CopyOutBlock(uint32_t blkLinear)
    {
        const uint64_t blockElem = (uint64_t)BS * (uint64_t)DK;
        const uint64_t base = (uint64_t)blkLinear * blockElem;
        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();
        AscendC::DataCopy(yGm[base], outLocal, (uint32_t)blockElem);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> qQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> kQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> vQue;

    AscendC::TBuf<> outBuf;
    AscendC::TBuf<> calcBuf;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, yGm, scaleGm;

    uint32_t B, H, NB, BS, DK, totalBlocks;
    uint32_t dkTile, jTile, useFastPath32;
    float scaleVal;
};

extern "C" __global__ __aicore__ void block_sparse_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                                    GM_ADDR scale, GM_ADDR y,
                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelBlockSparseAttention op;
    op.Init(q, k, v, scale, y,
            tiling_data.B, tiling_data.H, tiling_data.NB, tiling_data.BS, tiling_data.DK,
            tiling_data.dkTile, tiling_data.jTile, tiling_data.useFastPath32);
    op.Process();
}
