
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}
__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// softsign(x) = x / (1 + abs(x))
class KernelSoftsign {
public:
    __aicore__ inline KernelSoftsign() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileSize)
    {
        totalLength_ = totalLength;
        tileSize_ = (tileSize == 0) ? 1u : tileSize;

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength_);

        blockNum_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        blockIdx_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (blockNum_ == 0) blockNum_ = 1;

        totalTiles_ = CeilDivU32(totalLength_, tileSize_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileSize_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileSize_ * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0 || totalTiles_ == 0) return;

        uint32_t cur = blockIdx_;
        if (cur >= totalTiles_) return;

        // Prime
        CopyIn(cur);

        // Steady grid-stride double buffer: prefetch next then compute+store current.
        for (;;) {
            const uint32_t next = cur + blockNum_;
            if (next < totalTiles_) {
                CopyIn(next);
            }

            Compute(cur);
            CopyOut(cur);

            if (next >= totalTiles_) break;
            cur = next;
        }
    }

private:
    __aicore__ inline uint32_t TileOffset(uint32_t tileIdx) const { return tileIdx * tileSize_; }

    __aicore__ inline uint32_t TileLen(uint32_t tileIdx) const
    {
        const uint32_t off = TileOffset(tileIdx);
        if (off >= totalLength_) return 0;
        return MinU32(tileSize_, totalLength_ - off);
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        const uint32_t len = TileLen(tileIdx);
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[TileOffset(tileIdx)], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        const uint32_t len = TileLen(tileIdx);

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

        // Compute denom directly into yLocal to avoid extra tile-sized scratch:
        // y = reciprocal(1 + abs(x)) * x
        AscendC::Abs(yLocal, xLocal, len);                    // y = abs(x)
        AscendC::Adds(yLocal, yLocal, (DTYPE_Y)1.0f, len);    // y = 1 + abs(x)
        AscendC::Reciprocal(yLocal, yLocal, len);             // y = 1 / (1 + abs(x))
        AscendC::Mul(yLocal, yLocal, xLocal, len);            // y *= x

        outQueueY_.EnQue<DTYPE_Y>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        const uint32_t len = TileLen(tileIdx);
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm_[TileOffset(tileIdx)], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;

    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;

    uint32_t totalLength_{0};
    uint32_t tileSize_{1};
    uint32_t totalTiles_{0};
    uint32_t blockNum_{1};
    uint32_t blockIdx_{0};
};

extern "C" __global__ __aicore__ void softsign_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftsign op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileSize);
    op.Process();
}
