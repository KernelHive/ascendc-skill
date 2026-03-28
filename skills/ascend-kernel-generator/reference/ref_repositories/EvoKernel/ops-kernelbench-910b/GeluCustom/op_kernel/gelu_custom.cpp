
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGeluCustom {
public:
    __aicore__ inline KernelGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t tileLength, uint32_t tanhTmpBytes)
    {
        totalLength_ = totalLength;
        tileLength_ = tileLength;
        tanhTmpBytes_ = tanhTmpBytes;

        xGm_.SetGlobalBuffer((__gm__ float*)x, totalLength_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalLength_);

        pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(float));

        pipe_.InitBuffer(tanhTmpBuf_, tanhTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0) return;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tiles = (totalLength_ + tileLength_ - 1u) / tileLength_;

        // Grid-stride over tiles: uniform iterations, reduced scalar/tail overhead.
        for (uint32_t t = coreIdx; t < tiles; t += coreNum) {
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
        // GELU(tanh approximation):
        // y = 0.5*x*(1 + tanh(alpha*(x + beta*x^3)))
        // Implement with minimal extra UB: reuse yLocal as scratch (x2 then polynomial).
        constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
        constexpr float kBeta  = 0.044715f;
        constexpr float kHalf  = 0.5f;

        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tanhTmp = tanhTmpBuf_.Get<uint8_t>();

        // yLocal = x*x  (x2)
        AscendC::Mul(yLocal, xLocal, xLocal, count);

        // yLocal = beta*x2
        AscendC::Muls(yLocal, yLocal, kBeta, count);

        // yLocal = (beta*x2)*x
        AscendC::Mul(yLocal, yLocal, xLocal, count);

        // yLocal = x + beta*x^3
        AscendC::Add(yLocal, xLocal, yLocal, count);

        // yLocal = alpha*(...)
        AscendC::Muls(yLocal, yLocal, kAlpha, count);

        // yLocal = tanh(yLocal)
        AscendC::Tanh(yLocal, yLocal, tanhTmp, count);

        // yLocal = 0.5*(1 + tanh)
        AscendC::Adds(yLocal, yLocal, 1.0f, count);
        AscendC::Muls(yLocal, yLocal, kHalf, count);

        // yLocal = x * yLocal
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
    uint32_t tanhTmpBytes_ {0};
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGeluCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength, tiling_data.tanhTmpBytes);
    op.Process();
}
