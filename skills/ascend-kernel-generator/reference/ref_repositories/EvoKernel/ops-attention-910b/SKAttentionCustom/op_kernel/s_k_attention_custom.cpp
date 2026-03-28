
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSKAttention {
public:
    __aicore__ inline KernelSKAttention() {}

    __aicore__ inline void Init(GM_ADDR attn, GM_ADDR feats, GM_ADDR y,
                               uint32_t K, uint32_t bs, uint32_t C,
                               uint32_t H, uint32_t W, uint32_t hw,
                               uint32_t totalOut,
                               uint32_t coreNum, uint32_t coreStart, uint32_t coreCount,
                               uint32_t tileNum, uint32_t tileElems, uint32_t lastTileElems)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->K = K;
        this->bs = bs;
        this->C = C;
        this->H = H;
        this->W = W;
        this->hw = hw;
        this->totalOut = totalOut;

        // With current tiling we set coreNum=1, but keep generic guards.
        uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        (void)coreNum;

        // Compute per-core segment using uniform split (robust even if host fixed coreStart/coreCount).
        uint32_t per = (totalOut + bnum - 1) / bnum;
        this->coreStart = bid * per;
        uint32_t end = this->coreStart + per;
        if (end > totalOut) end = totalOut;
        this->coreCount = (end > this->coreStart) ? (end - this->coreStart) : 0;

        // Prefer host-computed tileNum/tileElems but clamp to coreCount.
        this->tileElems = tileElems / BUFFER_NUM;
        if (this->tileElems == 0) this->tileElems = 8;
        this->tileNum = (this->coreCount + this->tileElems - 1) / this->tileElems;
        if (this->tileNum == 0) this->tileNum = 1;
        this->lastTileElems = (this->coreCount > 0) ? (this->coreCount - (this->tileNum - 1) * this->tileElems) : 0;
        if (this->lastTileElems == 0) this->lastTileElems = (this->coreCount == 0 ? 0 : this->tileElems);
        (void)lastTileElems;
        (void)coreStart;
        (void)coreCount;

        // GM sizes (elements)
        const uint64_t attnSize  = (uint64_t)K * (uint64_t)bs * (uint64_t)C;       // [K,bs,C,1,1] flattened
        const uint64_t featsSize = (uint64_t)K * (uint64_t)totalOut;               // [K,bs,C,H,W] flattened
        const uint64_t ySize     = (uint64_t)totalOut;

        attnGm.SetGlobalBuffer((__gm__ float*)attn, attnSize);
        featsGm.SetGlobalBuffer((__gm__ float*)feats, featsSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers: feats tile, acc tile, out tile
        pipe.InitBuffer(qFeats, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(accBuf, this->tileElems * sizeof(float));
        pipe.InitBuffer(qOut, BUFFER_NUM, this->tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = 0; t < this->tileNum; ++t) {
            ComputeTile(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline uint32_t GetTileLen(uint32_t t) const
    {
        if (this->coreCount == 0) return 0;
        if (t + 1 == this->tileNum) return this->lastTileElems;
        return this->tileElems;
    }

    __aicore__ inline uint32_t TileBase(uint32_t t) const
    {
        return t * this->tileElems;
    }

    __aicore__ inline void ComputeTile(uint32_t t)
    {
        uint32_t tileLen = GetTileLen(t);
        if (tileLen == 0) return;

        uint32_t baseInCore = TileBase(t);
        uint32_t start = this->coreStart + baseInCore; // global output offset
        if (start >= this->totalOut) return;

        // Clamp tileLen so we never read/write past totalOut
        if (start + tileLen > this->totalOut) {
            tileLen = this->totalOut - start;
        }
        if (tileLen == 0) return;

        AscendC::LocalTensor<float> featsLocal = qFeats.AllocTensor<float>();
        AscendC::LocalTensor<float> outLocal   = qOut.AllocTensor<float>();
        AscendC::LocalTensor<float> accLocal   = accBuf.Get<float>();

        AscendC::Duplicate(accLocal, 0.0f, tileLen);

        // For each branch k: acc += feats[k, start:start+tileLen] * attn[k, b, c]
        // Mapping global linear index i in [0,totalOut):
        // CHW = C*hw
        // b = i / CHW
        // rem = i - b*CHW
        // c = rem / hw
        const uint32_t CHW = this->C * this->hw;

        for (uint32_t k = 0; k < this->K; ++k) {
            uint64_t featsOff = (uint64_t)k * (uint64_t)this->totalOut + (uint64_t)start;
            AscendC::DataCopy(featsLocal, featsGm[featsOff], tileLen);

            // Multiply by scalar attn for each element (b,c changes with i)
            for (uint32_t j = 0; j < tileLen; ++j) {
                uint32_t i = start + j;
                uint32_t b = i / CHW;
                uint32_t rem = i - b * CHW;
                uint32_t c = rem / this->hw;

                uint64_t attnOff = (uint64_t)k * (uint64_t)this->bs * (uint64_t)this->C +
                                   (uint64_t)b * (uint64_t)this->C + (uint64_t)c;
                float a = attnGm.GetValue(attnOff);

                float fv = featsLocal.GetValue(j);
                accLocal.SetValue(j, accLocal.GetValue(j) + fv * a);
            }
        }

        AscendC::DataCopy(outLocal, accLocal, tileLen);

        qFeats.FreeTensor(featsLocal);
        qOut.EnQue(outLocal);

        // record tileLen for CopyOut via member (single-buffer pipeline)
        this->curTileLen = tileLen;
        this->curStart = start;
    }

    __aicore__ inline void CopyOut(uint32_t t)
    {
        (void)t;
        if (this->curTileLen == 0) return;
        AscendC::LocalTensor<float> outLocal = qOut.DeQue<float>();
        AscendC::DataCopy(yGm[this->curStart], outLocal, this->curTileLen);
        qOut.FreeTensor(outLocal);
        this->curTileLen = 0;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> qFeats;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qOut;
    AscendC::TBuf<> accBuf;

    AscendC::GlobalTensor<float> attnGm;
    AscendC::GlobalTensor<float> featsGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t K, bs, C, H, W, hw;
    uint32_t totalOut;

    uint32_t coreStart;
    uint32_t coreCount;

    uint32_t tileNum;
    uint32_t tileElems;
    uint32_t lastTileElems;

    // for CopyOut
    uint32_t curTileLen {0};
    uint32_t curStart {0};
};

extern "C" __global__ __aicore__ void sk_attention_custom(GM_ADDR attn, GM_ADDR feats, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSKAttention op;
    op.Init(attn, feats, y,
            tiling_data.K, tiling_data.bs, tiling_data.C,
            tiling_data.H, tiling_data.W, tiling_data.hw,
            tiling_data.totalOut,
            tiling_data.coreNum, tiling_data.coreStart, tiling_data.coreCount,
            tiling_data.tileNum, tiling_data.tileElems, tiling_data.lastTileElems);
    op.Process();
}
