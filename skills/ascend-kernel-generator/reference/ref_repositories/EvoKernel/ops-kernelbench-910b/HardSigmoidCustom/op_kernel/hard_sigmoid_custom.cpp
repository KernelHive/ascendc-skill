
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelHardSigmoidCustom {
public:
    __aicore__ inline KernelHardSigmoidCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength)
    {
        this->totalLength = totalLength;
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        if (this->totalLength == 0) return;

        // Grid-stride over tiles for balanced work distribution and safe tail handling.
        for (uint32_t tileIdx = blockIdx; ; tileIdx += blockNum) {
            const uint32_t offset = tileIdx * this->tileLength;
            if (offset >= this->totalLength) break;

            uint32_t curLen = this->totalLength - offset;
            if (curLen > this->tileLength) curLen = this->tileLength;

            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLen)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[offset], curLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // HardSigmoid(x) = clamp(x + 3, 0, 6) / 6
        const DTYPE_Y c3   = (DTYPE_Y)(3.0f);
        const DTYPE_Y c0   = (DTYPE_Y)(0.0f);
        const DTYPE_Y c6   = (DTYPE_Y)(6.0f);
        const DTYPE_Y inv6 = (DTYPE_Y)(0.1666666716f);

        AscendC::Adds(yLocal, xLocal, c3, curLen);
        AscendC::Maxs(yLocal, yLocal, c0, curLen);
        AscendC::Mins(yLocal, yLocal, c6, curLen);
        AscendC::Muls(yLocal, yLocal, inv6, curLen);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLen)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[offset], yLocal, curLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t totalLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void hard_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelHardSigmoidCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
