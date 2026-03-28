
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes) {
    return (bytes + 31u) & ~31u;
}

class KernelEluCustom {
public:
    __aicore__ inline KernelEluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t blockDim,
                               uint32_t tileElems, float alpha)
    {
        totalLength_ = totalLength;
        alpha_ = alpha;
        tileElems_ = (tileElems == 0) ? 1u : tileElems;

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

        if (tileElems_ > coreLen_) tileElems_ = coreLen_;
        if (tileElems_ == 0) tileElems_ = 1;

        const uint32_t xBytes   = AlignUp32(tileElems_ * sizeof(float));
        const uint32_t yBytes   = AlignUp32(tileElems_ * sizeof(float));
        const uint32_t tmpBytes = AlignUp32(tileElems_ * sizeof(float));

        pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, xBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, yBytes);
        pipe_.InitBuffer(tmpCalcBuf_, tmpBytes);
    }

    __aicore__ inline void Process()
    {
        if (coreLen_ == 0) return;

        const uint32_t tiles = CeilDivU32(coreLen_, tileElems_);

        // Warm-up: prefetch tile 0
        CopyIn(0);

        // Steady-state: while prefetching next, compute current; while next iteration stores previous.
        for (uint32_t t = 0; t < tiles; ++t) {
            if (t + 1 < tiles) {
                CopyIn(t + 1);
            }
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline uint32_t CurLen(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * tileElems_;
        const uint32_t remain = coreLen_ - offset;
        return (remain >= tileElems_) ? tileElems_ : remain;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[tileIdx * tileElems_], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        (void)tileIdx;
        const uint32_t len = CurLen(tileIdx);

        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<float> negLocal = tmpCalcBuf_.Get<float>();

        // Branchless ELU:
        // y = relu(x) + alpha * (exp(min(x,0)) - 1)
        AscendC::Maxs(yLocal, xLocal, 0.0f, len);
        AscendC::Mins(negLocal, xLocal, 0.0f, len);
        AscendC::Exp(negLocal, negLocal, len);
        AscendC::Adds(negLocal, negLocal, -1.0f, len);
        AscendC::Muls(negLocal, negLocal, alpha_, len);
        AscendC::Add(yLocal, yLocal, negLocal, len);

        outQueueY_.EnQue<float>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        const uint32_t len = CurLen(tileIdx);
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[tileIdx * tileElems_], yLocal, len);
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
    uint32_t tileElems_{0};
    float alpha_{1.0f};
};

extern "C" __global__ __aicore__ void elu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelEluCustom op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.blockDim,
            tiling_data.tileElems,
            tiling_data.alpha);
    op.Process();
}
