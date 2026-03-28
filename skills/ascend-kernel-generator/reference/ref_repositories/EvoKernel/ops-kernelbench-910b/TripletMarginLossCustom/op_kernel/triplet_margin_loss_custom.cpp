
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTripletMarginLossCustom {
public:
    __aicore__ inline KernelTripletMarginLossCustom() {}

    __aicore__ inline void Init(GM_ADDR anchor, GM_ADDR positive, GM_ADDR negative, GM_ADDR margin,
                               GM_ADDR y,
                               uint32_t batchSize, uint32_t featSize,
                               uint32_t featTile, uint32_t featTileNum,
                               uint32_t featLast, float invBatch)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        // host sets blockDim = 1

        this->batchSize = batchSize;
        this->featSize = featSize;
        this->featTile = featTile;
        this->featTileNum = featTileNum;
        this->featLast = featLast;
        this->invBatch = invBatch;

        uint32_t totalLen = batchSize * featSize;
        aGm.SetGlobalBuffer((__gm__ float*)anchor, totalLen);
        pGm.SetGlobalBuffer((__gm__ float*)positive, totalLen);
        nGm.SetGlobalBuffer((__gm__ float*)negative, totalLen);
        mGm.SetGlobalBuffer((__gm__ float*)margin, 1);
        outGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // UB:
        // inQ: 2 buffers, each has 3*featTile floats (a/p/n)
        // workBuf: 3*featTile floats => v0, v1, reduceTmp
        pipe.InitBuffer(inQ, BUFFER_NUM, featTile * 3 * sizeof(float));
        pipe.InitBuffer(workBuf, featTile * 3 * sizeof(float));
    }

    __aicore__ inline void PrefetchTile(uint32_t gmOff, uint32_t cur, bool isPartial)
    {
        AscendC::LocalTensor<float> inLocal = inQ.AllocTensor<float>();
        // Safe tail handling: only clear when partial, and always from aligned base.
        if (isPartial) {
            AscendC::Duplicate(inLocal, 0.0f, featTile * 3);
        }
        if (cur > 0) {
            AscendC::DataCopy(inLocal[0],                  aGm[gmOff], cur);
            AscendC::DataCopy(inLocal[this->featTile],     pGm[gmOff], cur);
            AscendC::DataCopy(inLocal[2 * this->featTile], nGm[gmOff], cur);
        }
        inQ.EnQue(inLocal);
    }

    __aicore__ inline void Process()
    {
        if (batchSize == 0 || featSize == 0) {
            outGm.SetValue(0, 0.0f);
            return;
        }

        const float margin = mGm.GetValue(0);
        constexpr float eps = 1.0e-6f;

        AscendC::LocalTensor<float> work = workBuf.Get<float>();
        AscendC::LocalTensor<float> v0   = work;                     // featTile
        AscendC::LocalTensor<float> v1   = work[this->featTile];     // featTile
        AscendC::LocalTensor<float> red  = work[this->featTile * 2]; // featTile temp

        float sumLoss = 0.0f;

        for (uint32_t b = 0; b < batchSize; ++b) {
            float sumSqAP = 0.0f;
            float sumSqAN = 0.0f;

            uint32_t base = b * featSize;

            // Prefetch tile 0
            {
                uint32_t cur0 = (featTileNum == 1) ? featLast : featTile;
                bool partial0 = (featTileNum == 1) && (featLast != featTile);
                PrefetchTile(base, cur0, partial0);
            }

            for (uint32_t t = 0; t < featTileNum; ++t) {
                // Prefetch next tile
                uint32_t tNext = t + 1;
                if (tNext < featTileNum) {
                    uint32_t curN = (tNext == featTileNum - 1) ? featLast : featTile;
                    bool partialN = (tNext == featTileNum - 1) && (featLast != featTile);
                    uint32_t offN = base + tNext * featTile;
                    PrefetchTile(offN, curN, partialN);
                }

                uint32_t cur = (t == featTileNum - 1) ? featLast : featTile;

                AscendC::LocalTensor<float> tile = inQ.DeQue<float>();
                AscendC::LocalTensor<float> aL = tile;
                AscendC::LocalTensor<float> pL = tile[this->featTile];
                AscendC::LocalTensor<float> nL = tile[2 * this->featTile];

                // AP: v0 = (a - p)^2
                AscendC::Sub(v0, aL, pL, cur);
                AscendC::Mul(v0, v0, v0, cur);
                AscendC::ReduceSum<float>(v0, v0, red, cur);
                sumSqAP += v0.GetValue(0);

                // AN: v1 = (a - n)^2
                AscendC::Sub(v1, aL, nL, cur);
                AscendC::Mul(v1, v1, v1, cur);
                AscendC::ReduceSum<float>(v1, v1, red, cur);
                sumSqAN += v1.GetValue(0);

                inQ.FreeTensor(tile);
            }

            // dist = sqrt(sum + eps)
            v0.SetValue(0, sumSqAP + eps);
            v1.SetValue(0, sumSqAN + eps);
            AscendC::Sqrt(v0, v0, 1);
            AscendC::Sqrt(v1, v1, 1);

            float loss = (v0.GetValue(0) - v1.GetValue(0)) + margin;
            if (loss < 0.0f) loss = 0.0f;
            sumLoss += loss;
        }

        sumLoss *= invBatch;
        outGm.SetValue(0, sumLoss);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQ;
    AscendC::TBuf<> workBuf;

    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> pGm;
    AscendC::GlobalTensor<float> nGm;
    AscendC::GlobalTensor<float> mGm;
    AscendC::GlobalTensor<float> outGm;

    uint32_t batchSize;
    uint32_t featSize;
    uint32_t featTile;
    uint32_t featTileNum;
    uint32_t featLast;
    float invBatch;
};

extern "C" __global__ __aicore__ void triplet_margin_loss_custom(
    GM_ADDR anchor, GM_ADDR positive, GM_ADDR negative, GM_ADDR margin,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelTripletMarginLossCustom op;
    op.Init(anchor, positive, negative, margin, y,
            tiling_data.batchSize, tiling_data.featSize,
            tiling_data.featTile, tiling_data.featTileNum,
            tiling_data.featLast, tiling_data.invBatch);
    op.Process();
}
