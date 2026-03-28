
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// SELU constants (PyTorch defaults)
constexpr float SELU_SCALE_F = 1.0507009873554805f;
constexpr float SELU_SCALE_ALPHA_F = 1.7580993408473769f; // scale * alpha

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes) {
    return (bytes + 31u) & ~31u;
}

class KernelSeluCustom {
public:
    __aicore__ inline KernelSeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t blockDim, uint32_t tileLength)
    {
        totalLength_ = totalLength;
        tileLength_ = (tileLength == 0) ? 1u : tileLength;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t usedBlocks = (blockDim == 0) ? ((blockNum == 0) ? 1u : blockNum) : blockDim;

        const uint32_t coreChunk = CeilDivU32(totalLength_, usedBlocks);
        const uint32_t coreStart = blockIdx * coreChunk;

        if (coreStart >= totalLength_) {
            coreLen_ = 0;
            xGm_.SetGlobalBuffer((__gm__ float*)x, 0);
            yGm_.SetGlobalBuffer((__gm__ float*)y, 0);
            return;
        }

        coreLen_ = totalLength_ - coreStart;
        if (coreLen_ > coreChunk) coreLen_ = coreChunk;

        xGm_.SetGlobalBuffer((__gm__ float*)x + coreStart, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ float*)y + coreStart, coreLen_);

        if (tileLength_ > coreLen_) tileLength_ = coreLen_;
        if (tileLength_ == 0) tileLength_ = 1;

        const uint32_t xBytes = AlignUp32(tileLength_ * sizeof(float));
        const uint32_t yBytes = AlignUp32(tileLength_ * sizeof(float));
        const uint32_t tmpBytes = AlignUp32(tileLength_ * sizeof(float));

        pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, xBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, yBytes);
        pipe_.InitBuffer(tmpCalcBuf_, tmpBytes);
    }

    __aicore__ inline void Process()
    {
        if (coreLen_ == 0) return;

        const uint32_t tiles = CeilDivU32(coreLen_, tileLength_);
        for (uint32_t t = 0; t < tiles; ++t) {
            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline uint32_t CurLen(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * tileLength_;
        const uint32_t remain = coreLen_ - offset;
        return (remain >= tileLength_) ? tileLength_ : remain;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[tileIdx * tileLength_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        (void)tileIdx;
        const uint32_t len = CurLen(tileIdx);

        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<float> t = tmpCalcBuf_.Get<float>();

        // y = scale*x + scale*alpha*(exp(min(x,0)) - 1)
        AscendC::Muls(yLocal, xLocal, SELU_SCALE_F, len);

        AscendC::Mins(t, xLocal, 0.0f, len);
        AscendC::Exp(t, t, len);
        AscendC::Adds(t, t, -1.0f, len);
        AscendC::Muls(t, t, SELU_SCALE_ALPHA_F, len);

        AscendC::Add(yLocal, yLocal, t, len);

        outQueueY_.EnQue<float>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[tileIdx * tileLength_], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalcBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t totalLength_{0};
    uint32_t coreLen_{0};
    uint32_t tileLength_{0};
};

extern "C" __global__ __aicore__ void selu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSeluCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.blockDim, tiling_data.tileLength);
    op.Process();
}
