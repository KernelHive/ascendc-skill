
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelParallelPolarizedSelfAttentionCustom {
public:
    __aicore__ inline KernelParallelPolarizedSelfAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR channel_weight, GM_ADDR spatial_weight, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W, uint32_t HW,
                               uint32_t totalPos, uint32_t blockDim, uint32_t posPerCore, uint32_t cTile)
    {
        (void)H; (void)W;
        this->B = B;
        this->C = C;
        this->HW = HW;
        this->totalPos = totalPos;
        this->blockDim = blockDim;
        this->posPerCore = posPerCore;
        this->cTile = cTile;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        this->posStart = coreIdx * posPerCore;
        const uint32_t posEnd = (this->posStart + posPerCore < totalPos) ? (this->posStart + posPerCore) : totalPos;
        this->posCount = (posEnd > this->posStart) ? (posEnd - this->posStart) : 0;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B * (uint64_t)C * (uint64_t)HW);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)B * (uint64_t)C * (uint64_t)HW);
        cwGm.SetGlobalBuffer((__gm__ float*)channel_weight, (uint64_t)B * (uint64_t)C);
        swGm.SetGlobalBuffer((__gm__ float*)spatial_weight, (uint64_t)B * (uint64_t)HW);

        // Double-buffer x, cw, y tiles
        pipe.InitBuffer(qx,  BUFFER_NUM, this->cTile * sizeof(float));
        pipe.InitBuffer(qcw, BUFFER_NUM, this->cTile * sizeof(float));
        pipe.InitBuffer(qy,  BUFFER_NUM, this->cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (posCount == 0) return;

        const uint32_t tilesPerPos = (C + cTile - 1) / cTile;

        // Pipeline over (pos, tile)
        const uint32_t totalSteps = posCount * tilesPerPos;
        for (uint32_t step = 0; step < totalSteps; ++step) {
            CopyIn(step, tilesPerPos);
            Compute(step, tilesPerPos);
            CopyOut(step, tilesPerPos);
        }
    }

private:
    __aicore__ inline void DecodeStep(uint32_t step, uint32_t tilesPerPos,
                                     uint32_t &posLocal, uint32_t &tileId) const
    {
        posLocal = step / tilesPerPos;
        tileId   = step - posLocal * tilesPerPos;
    }

    __aicore__ inline uint32_t TileLen(uint32_t tileId) const
    {
        const uint32_t base = tileId * cTile;
        const uint32_t remain = (base < C) ? (C - base) : 0;
        return (remain < cTile) ? remain : cTile;
    }

    __aicore__ inline void CopyIn(uint32_t step, uint32_t tilesPerPos)
    {
        uint32_t posLocal, tileId;
        DecodeStep(step, tilesPerPos, posLocal, tileId);

        const uint32_t tileLen = TileLen(tileId);
        if (tileLen == 0) return;

        const uint32_t pos = posStart + posLocal;
        const uint32_t b = pos / HW;
        const uint32_t hw = pos - b * HW;

        const uint64_t xBase = ((uint64_t)b * (uint64_t)C + (uint64_t)tileId * (uint64_t)cTile) * (uint64_t)HW + (uint64_t)hw;
        const uint64_t cwBase = (uint64_t)b * (uint64_t)C + (uint64_t)tileId * (uint64_t)cTile;

        // x is strided by HW (NCHW). We must gather x[b, c0:c0+tileLen, hw] with stride HW.
        // To avoid per-element GM GetValue, use DataCopy with stride via for-loop into UB (still scalar index),
        // but keep loop only over cTile (<=~512/6 aligned) and no div/mod in inner loop.
        // Additionally, cw tile is contiguous -> single DataCopy.

        AscendC::LocalTensor<float> xLocal = qx.AllocTensor<float>();
        AscendC::LocalTensor<float> cwLocal = qcw.AllocTensor<float>();

        // cw contiguous
        AscendC::DataCopy(cwLocal, cwGm[cwBase], tileLen);

        // x gather by HW stride
        for (uint32_t i = 0; i < tileLen; ++i) {
            const uint64_t off = xBase + (uint64_t)i * (uint64_t)HW;
            xLocal.SetValue(i, xGm.GetValue(off));
        }

        qx.EnQue(xLocal);
        qcw.EnQue(cwLocal);
    }

    __aicore__ inline void Compute(uint32_t step, uint32_t tilesPerPos)
    {
        uint32_t posLocal, tileId;
        DecodeStep(step, tilesPerPos, posLocal, tileId);

        const uint32_t tileLen = TileLen(tileId);
        if (tileLen == 0) return;

        const uint32_t pos = posStart + posLocal;
        const uint32_t b = pos / HW;
        const uint32_t hw = pos - b * HW;

        const float sw = swGm.GetValue((uint64_t)b * (uint64_t)HW + (uint64_t)hw);

        AscendC::LocalTensor<float> xLocal  = qx.DeQue<float>();
        AscendC::LocalTensor<float> cwLocal = qcw.DeQue<float>();
        AscendC::LocalTensor<float> yLocal  = qy.AllocTensor<float>();

        // y = x * (cw + sw)
        AscendC::Adds(cwLocal, cwLocal, sw, tileLen);
        AscendC::Mul(yLocal, xLocal, cwLocal, tileLen);

        qx.FreeTensor(xLocal);
        qcw.FreeTensor(cwLocal);
        qy.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t step, uint32_t tilesPerPos)
    {
        uint32_t posLocal, tileId;
        DecodeStep(step, tilesPerPos, posLocal, tileId);

        const uint32_t tileLen = TileLen(tileId);
        if (tileLen == 0) return;

        const uint32_t pos = posStart + posLocal;
        const uint32_t b = pos / HW;
        const uint32_t hw = pos - b * HW;

        const uint64_t yBase = ((uint64_t)b * (uint64_t)C + (uint64_t)tileId * (uint64_t)cTile) * (uint64_t)HW + (uint64_t)hw;

        AscendC::LocalTensor<float> yLocal = qy.DeQue<float>();

        // scatter by HW stride
        for (uint32_t i = 0; i < tileLen; ++i) {
            const uint64_t off = yBase + (uint64_t)i * (uint64_t)HW;
            yGm.SetValue(off, yLocal.GetValue(i));
        }

        qy.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qx;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qcw;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qy;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> cwGm;
    AscendC::GlobalTensor<float> swGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B = 0, C = 0, HW = 0;
    uint32_t totalPos = 0;
    uint32_t blockDim = 0;
    uint32_t posPerCore = 0;
    uint32_t cTile = 0;

    uint32_t posStart = 0;
    uint32_t posCount = 0;
};

extern "C" __global__ __aicore__ void parallel_polarized_self_attention_custom(
    GM_ADDR x, GM_ADDR channel_weight, GM_ADDR spatial_weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelParallelPolarizedSelfAttentionCustom op;
    op.Init(x, channel_weight, spatial_weight, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W, tiling_data.HW,
            tiling_data.totalPos, tiling_data.blockDim, tiling_data.posPerCore, tiling_data.cTile);
    op.Process();
}
