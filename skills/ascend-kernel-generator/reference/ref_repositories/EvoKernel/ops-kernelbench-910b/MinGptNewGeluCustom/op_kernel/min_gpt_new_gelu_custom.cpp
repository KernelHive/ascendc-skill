
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMinGptNewGeluCustom {
public:
    __aicore__ inline KernelMinGptNewGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t tileLength,
                               uint32_t tiles, uint32_t tanhTmpBytes)
    {
        totalLength_ = totalLength;
        tileLength_ = tileLength;
        tiles_ = tiles;
        tanhTmpBytes_ = tanhTmpBytes;

        xGm_.SetGlobalBuffer((__gm__ float*)x, totalLength_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalLength_);

        pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(float));

        // scratch for tanh intrinsic
        pipe_.InitBuffer(tanhTmpBuf_, tanhTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0) return;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        // Grid-stride over tiles for balance and minimal scalar overhead.
        for (uint32_t t = coreIdx; t < tiles_; t += coreNum) {
            const uint32_t offset = t * tileLength_;
            uint32_t count = totalLength_ - offset;
            if (count > tileLength_) count = tileLength_;
            if (count == 0) continue;

            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[offset], count);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        // minGPT/GPT2 GELU (tanh approximation):
        // y = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        constexpr float kBeta        = 0.044715f;
        constexpr float kHalf        = 0.5f;

        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tanhTmp = tanhTmpBuf_.Get<uint8_t>();

        // Reuse yLocal as scratch to reduce UB footprint:
        // yLocal = x^2
        AscendC::Mul(yLocal, xLocal, xLocal, count);
        // yLocal = beta*x^2
        AscendC::Muls(yLocal, yLocal, kBeta, count);
        // yLocal = beta*x^3
        AscendC::Mul(yLocal, yLocal, xLocal, count);
        // yLocal = x + beta*x^3
        AscendC::Add(yLocal, xLocal, yLocal, count);
        // yLocal = sqrt(2/pi) * (...)
        AscendC::Muls(yLocal, yLocal, kSqrt2OverPi, count);

        // yLocal = tanh(yLocal)
        AscendC::Tanh(yLocal, yLocal, tanhTmp, count);

        // yLocal = 0.5*(1 + tanh)
        AscendC::Adds(yLocal, yLocal, 1.0f, count);
        AscendC::Muls(yLocal, yLocal, kHalf, count);

        // y = x * yLocal
        AscendC::Mul(yLocal, xLocal, yLocal, count);

        outQueueY_.EnQue<float>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[offset], yLocal, count);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<> tanhTmpBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t totalLength_ {0};
    uint32_t tileLength_ {0};
    uint32_t tiles_ {0};
    uint32_t tanhTmpBytes_ {0};
};

extern "C" __global__ __aicore__ void min_gpt_new_gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinGptNewGeluCustom op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.tileLength,
            tiling_data.tiles,
            tiling_data.tanhTmpBytes);
    op.Process();
}
