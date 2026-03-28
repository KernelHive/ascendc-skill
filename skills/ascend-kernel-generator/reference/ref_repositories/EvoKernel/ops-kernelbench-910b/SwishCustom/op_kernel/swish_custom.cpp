
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes) {
    return (bytes + 31u) & ~31u;
}

class KernelSwishCustom {
public:
    __aicore__ inline KernelSwishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileSize)
    {
        totalLength_ = totalLength;
        tileSize_ = (tileSize == 0) ? 1u : tileSize;

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t usedBlocks = (blockNum == 0) ? 1u : blockNum;

        const uint32_t coreChunk = CeilDivU32(totalLength_, usedBlocks);
        const uint32_t coreStart = blockIdx * coreChunk;

        if (coreStart >= totalLength_) {
            coreLen_ = 0;
            xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, 0);
            yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y, 0);
            return;
        }

        coreLen_ = totalLength_ - coreStart;
        if (coreLen_ > coreChunk) coreLen_ = coreChunk;

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x + coreStart, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y + coreStart, coreLen_);

        if (tileSize_ > coreLen_) tileSize_ = coreLen_;
        if (tileSize_ == 0) tileSize_ = 1;

        const uint32_t xBytes = AlignUp32(tileSize_ * sizeof(DTYPE_X));
        const uint32_t yBytes = AlignUp32(tileSize_ * sizeof(DTYPE_Y));
        const uint32_t tmpBytes = AlignUp32(tileSize_ * sizeof(uint8_t));

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, xBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, yBytes);
        pipe_.InitBuffer(tmpCalcBuf_, tmpBytes);
    }

    __aicore__ inline void Process()
    {
        if (coreLen_ == 0) return;
        const uint32_t tiles = CeilDivU32(coreLen_, tileSize_);
        if (tiles == 0) return;

        // Fill pipeline: prefetch up to 2 tiles.
        CopyIn(0);
        if (tiles > 1) {
            CopyIn(1);
        }

        // Steady-state pipeline:
        // iter i: Compute(i) while next CopyIn(i+2) can overlap; CopyOut(i-1) overlaps.
        for (uint32_t i = 0; i < tiles + 1; ++i) {
            if (i < tiles) {
                Compute(i);
            }
            if (i + 2 < tiles) {
                CopyIn(i + 2);
            }
            if (i > 0) {
                CopyOut(i - 1);
            }
        }
    }

private:
    __aicore__ inline uint32_t CurLen(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * tileSize_;
        const uint32_t remain = coreLen_ - offset;
        return (remain >= tileSize_) ? tileSize_ : remain;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[tileIdx * tileSize_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        (void)tileIdx;
        const uint32_t len = CurLen(tileIdx);

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<uint8_t> tmp = tmpCalcBuf_.Get<uint8_t>();

        // Keep baseline math (no extra full-tile intermediates).
        AscendC::Sigmoid(yLocal, xLocal, tmp, len);
        AscendC::Mul(yLocal, yLocal, xLocal, len);

        outQueueY_.EnQue<DTYPE_Y>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm_[tileIdx * tileSize_], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalcBuf_;

    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;

    uint32_t totalLength_{0};
    uint32_t coreLen_{0};
    uint32_t tileSize_{0};
};

extern "C" __global__ __aicore__ void swish_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSwishCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileSize);
    op.Process();
}
