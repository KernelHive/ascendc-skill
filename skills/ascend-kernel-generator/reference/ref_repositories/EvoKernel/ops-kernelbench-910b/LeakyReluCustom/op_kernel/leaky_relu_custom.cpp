
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLeakyReluCustom {
public:
    __aicore__ inline KernelLeakyReluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength,
                                float negativeSlope)
    {
        totalLen_ = totalLength;
        tileLen_ = tileLength;
        negativeSlope_ = negativeSlope;
        negCorr_ = negativeSlope_ - 1.0f;

        xGm_.SetGlobalBuffer((__gm__ float*)x, totalLen_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalLen_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLen_ * sizeof(float));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLen_ * sizeof(float));
        pipe_.InitBuffer(tmpQueue_, 1, tileLen_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (totalLen_ == 0 || tileLen_ == 0) return;

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        // Tile-granularity grid-stride to improve load balance and occupancy.
        const uint32_t strideElems = blockNum * tileLen_;
        uint32_t base = blockIdx * tileLen_;
        if (base >= totalLen_) return;

        // Prefetch first tile
        uint32_t curLen = totalLen_ - base;
        if (curLen > tileLen_) curLen = tileLen_;
        CopyIn(base, curLen);

        while (true) {
            // Prefetch next tile early for overlap
            const uint32_t nextBase = base + strideElems;
            uint32_t nextLen = 0;
            if (nextBase < totalLen_) {
                nextLen = totalLen_ - nextBase;
                if (nextLen > tileLen_) nextLen = tileLen_;
                CopyIn(nextBase, nextLen);
            }

            Compute(curLen);
            CopyOut(base, curLen);

            if (nextLen == 0) break;
            base = nextBase;
            curLen = nextLen;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[gmOffset], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<float> tLocal = tmpQueue_.AllocTensor<float>();

        // y = x + (negativeSlope - 1) * min(x, 0)
        AscendC::Mins(tLocal, xLocal, 0.0f, len);
        AscendC::Muls(tLocal, tLocal, negCorr_, len);
        AscendC::Add(yLocal, xLocal, tLocal, len);

        tmpQueue_.FreeTensor(tLocal);
        outQueueY_.EnQue<float>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t len)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[gmOffset], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t totalLen_ {0};
    uint32_t tileLen_ {0};
    float negativeSlope_ {0.01f};
    float negCorr_ {-0.99f};
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyReluCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength, tiling_data.negativeSlope);
    op.Process();
}
