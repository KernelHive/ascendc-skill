
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes) {
    return (bytes + 31u) & ~31u;
}

// Align element count down so (elems * sizeof(T)) is 32B aligned.
template <typename T>
__aicore__ inline uint32_t AlignDownElems32B(uint32_t elems) {
    const uint32_t bytes = elems * static_cast<uint32_t>(sizeof(T));
    const uint32_t alignedBytes = bytes & ~31u;
    return alignedBytes / static_cast<uint32_t>(sizeof(T));
}

class KernelTanhCustom {
public:
    __aicore__ inline KernelTanhCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t tileSize, uint32_t tmpSizeBytes)
    {
        totalLength_ = totalLength;
        tileSize_ = (tileSize == 0) ? 1u : tileSize;
        tmpSizeBytes_ = AlignUp32((tmpSizeBytes == 0) ? 1024u : tmpSizeBytes);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t usedBlocks = (blockNum == 0) ? 1u : blockNum;

        // Contiguous per-core chunk (better MTE burst regularity vs global grid-stride).
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

        alignedTileSize_ = AlignDownElems32B<DTYPE_X>(tileSize_);
        if (alignedTileSize_ == 0) alignedTileSize_ = tileSize_;

        // 32B padded UB allocations to reduce bank-group conflicts.
        const uint32_t xBytes = AlignUp32(tileSize_ * sizeof(DTYPE_X));
        const uint32_t yBytes = AlignUp32(tileSize_ * sizeof(DTYPE_Y));

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, xBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, yBytes);

        // temp once per core, sized by tiling, aligned.
        pipe_.InitBuffer(tmpQueue_, 1, tmpSizeBytes_);
        tmpLocal_ = tmpQueue_.AllocTensor<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (coreLen_ == 0) return;

        const uint32_t tiles = CeilDivU32(coreLen_, tileSize_);
        for (uint32_t t = 0; t < tiles; ++t) {
            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }

        tmpQueue_.FreeTensor(tmpLocal_);
    }

private:
    __aicore__ inline uint32_t CurLen(uint32_t tileIdx) const
    {
        const uint32_t off = tileIdx * tileSize_;
        const uint32_t rem = coreLen_ - off;
        return (rem >= tileSize_) ? tileSize_ : rem;
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
        const uint32_t len = CurLen(tileIdx);

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

        uint32_t computeLen = len;
        if (len == tileSize_) computeLen = alignedTileSize_;
        if (computeLen == 0) computeLen = len;

        AscendC::Tanh(yLocal, xLocal, tmpLocal_, computeLen);

        if (computeLen < len) {
            AscendC::Tanh(yLocal[computeLen], xLocal[computeLen], tmpLocal_, len - computeLen);
        }

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
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue_;

    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;
    AscendC::LocalTensor<uint8_t> tmpLocal_;

    uint32_t totalLength_{0};
    uint32_t coreLen_{0};
    uint32_t tileSize_{0};
    uint32_t alignedTileSize_{0};
    uint32_t tmpSizeBytes_{0};
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelTanhCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileSize, tiling_data.tmpSizeBytes);
    op.Process();
}
