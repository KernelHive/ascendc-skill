
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelReluCustom {
public:
    __aicore__ inline KernelReluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength,
                                uint32_t fullTiles, uint32_t hasTail)
    {
        totalLen_  = totalLength;
        tileLen_   = tileLength;
        fullTiles_ = fullTiles;
        hasTail_   = hasTail;

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLen_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLen_);

        // Double-buffer UB: 2*X + 2*Y
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLen_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLen_ * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        if (totalLen_ == 0 || tileLen_ == 0) return;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();

        // 1) Steady-state loop over full tiles only: fixed len = tileLen_.
        // Use grid-stride over tile index to avoid per-iteration next-tile planning.
        for (uint32_t t = blockIdx; t < fullTiles_; t += blockNum) {
            const uint32_t base = t * tileLen_;
            // Full tile: len always tileLen_ (no tail checks, improves scalar/pipeline behavior).
            CopyIn(base, tileLen_);
            Compute(tileLen_);
            CopyOut(base, tileLen_);
        }

        // 2) Single tail tile (if any): only the block that owns its tile index processes it.
        if (hasTail_ == 0) return;
        const uint32_t tailTileIdx = fullTiles_;
        if (tailTileIdx % blockNum != blockIdx) return;

        const uint32_t base = tailTileIdx * tileLen_;
        if (base >= totalLen_) return;
        uint32_t len = totalLen_ - base;  // < tileLen_
        if (len == 0) return;

        CopyIn(base, len);
        Compute(len);
        CopyOut(base, len);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[gmOffset], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();
        AscendC::Relu(yLocal, xLocal, len);
        outQueueY_.EnQue<DTYPE_Y>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm_[gmOffset], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;

    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;

    uint32_t totalLen_ {0};
    uint32_t tileLen_ {0};
    uint32_t fullTiles_ {0};
    uint32_t hasTail_ {0};
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelReluCustom op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.tileLength,
            tiling_data.fullTilesPerBlock,
            tiling_data.hasTail);
    op.Process();
}
