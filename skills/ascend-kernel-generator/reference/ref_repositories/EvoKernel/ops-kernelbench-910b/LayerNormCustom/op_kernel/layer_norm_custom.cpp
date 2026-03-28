
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNormCustom {
public:
    __aicore__ inline KernelLayerNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols,
                               uint32_t tileLength, float eps, float invCols)
    {
        this->rows = rows;
        this->cols = cols;
        this->tileLength = tileLength;
        this->eps = eps;
        this->invCols = invCols;

        const uint64_t total64 = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
        const uint32_t total = (total64 > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : static_cast<uint32_t>(total64);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), total);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), total);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t perCore = (this->rows + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * perCore;

        uint32_t count = 0;
        if (start < this->rows) {
            const uint32_t remain = this->rows - start;
            count = (remain < perCore) ? remain : perCore;
        }
        this->rowStart = start;
        this->rowCount = count;

        pipe.InitBuffer(qX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(qY, BUFFER_NUM, this->tileLength * sizeof(float));

        // Reduction helpers
        pipe.InitBuffer(bufTmp,  this->tileLength * sizeof(float));
        pipe.InitBuffer(bufWork, this->tileLength * sizeof(float));
        pipe.InitBuffer(bufRed,  8 * sizeof(float));
        pipe.InitBuffer(bufOne,  8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->rows == 0u || this->cols == 0u || this->rowCount == 0u || this->invCols == 0.0f) {
            return;
        }

        tmp  = bufTmp.Get<float>();
        work = bufWork.Get<float>();
        red  = bufRed.Get<float>();
        one  = bufOne.Get<float>();

        for (uint32_t r = 0; r < this->rowCount; ++r) {
            const uint32_t row = this->rowStart + r;
            float mean = 0.0f;
            float invStd = 0.0f;
            ComputeMeanInvStd_TwoBuffer(row, mean, invStd);
            NormalizeRow_TwoBuffer(row, mean, invStd);
        }
    }

private:
    __aicore__ inline void PadTailInPlace(AscendC::LocalTensor<float>& tile, uint32_t validLen)
    {
        if (validLen < this->tileLength) {
            AscendC::Duplicate<float>(tile[(int32_t)validLen], 0.0f,
                                     (int32_t)this->tileLength - (int32_t)validLen);
        }
    }

    __aicore__ inline float ReduceSumFull(const AscendC::LocalTensor<float>& inFull)
    {
        AscendC::ReduceSum<float>(red, inFull, work, (int32_t)this->tileLength);
        return red.GetValue(0);
    }

    // Stats pass: compute sum and sumsq in one read pass, with 2-buffer prefetch via queues.
    __aicore__ inline void ComputeMeanInvStd_TwoBuffer(uint32_t row, float& mean, float& invStd)
    {
        const uint32_t iters = (this->cols + this->tileLength - 1u) / this->tileLength;
        const uint64_t rowBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(this->cols);

        float sum = 0.0f;
        float sumsq = 0.0f;

        if (iters == 0u) {
            mean = 0.0f;
            invStd = 1.0f;
            return;
        }

        // Prime: enqueue first tile
        {
            uint32_t len0 = this->cols;
            if (len0 > this->tileLength) len0 = this->tileLength;
            AscendC::LocalTensor<float> t0 = qX.AllocTensor<float>();
            AscendC::DataCopy<float>(t0, xGm[rowBase], len0);
            if (iters == 1u) PadTailInPlace(t0, len0);
            qX.EnQue(t0);
        }

        for (uint32_t i = 0; i < iters; ++i) {
            // Prefetch next tile into the other queue slot
            if (i + 1u < iters) {
                const uint32_t nxtOff = (i + 1u) * this->tileLength;
                uint32_t nxtLen = this->cols - nxtOff;
                if (nxtLen > this->tileLength) nxtLen = this->tileLength;
                AscendC::LocalTensor<float> tn = qX.AllocTensor<float>();
                const uint64_t gmOffNxt = rowBase + static_cast<uint64_t>(nxtOff);
                AscendC::DataCopy<float>(tn, xGm[gmOffNxt], nxtLen);
                if (i + 1u == iters - 1u) PadTailInPlace(tn, nxtLen);
                qX.EnQue(tn);
            }

            AscendC::LocalTensor<float> xTile = qX.DeQue<float>();

            sum += ReduceSumFull(xTile);
            AscendC::Mul<float>(tmp, xTile, xTile, (int32_t)this->tileLength);
            sumsq += ReduceSumFull(tmp);

            qX.FreeTensor(xTile);
        }

        mean = sum * this->invCols;
        float var = sumsq * this->invCols - mean * mean;
        if (var < 0.0f) var = 0.0f;

        one.SetValue(0, var + this->eps);
        AscendC::Sqrt<float>(one, one, 1);
        red.SetValue(0, 1.0f);
        AscendC::Div<float>(red, red, one, 1);
        invStd = red.GetValue(0);
    }

    __aicore__ inline void NormalizeRow_TwoBuffer(uint32_t row, float mean, float invStd)
    {
        const uint32_t iters = (this->cols + this->tileLength - 1u) / this->tileLength;
        const uint64_t rowBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(this->cols);
        if (iters == 0u) return;

        // Prime: enqueue first tile
        {
            uint32_t len0 = this->cols;
            if (len0 > this->tileLength) len0 = this->tileLength;
            AscendC::LocalTensor<float> t0 = qX.AllocTensor<float>();
            AscendC::DataCopy<float>(t0, xGm[rowBase], len0);
            qX.EnQue(t0);
        }

        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t colOff = i * this->tileLength;
            uint32_t curLen = this->cols - colOff;
            if (curLen > this->tileLength) curLen = this->tileLength;

            // Prefetch next tile early
            if (i + 1u < iters) {
                const uint32_t nxtOff = (i + 1u) * this->tileLength;
                uint32_t nxtLen = this->cols - nxtOff;
                if (nxtLen > this->tileLength) nxtLen = this->tileLength;
                AscendC::LocalTensor<float> tn = qX.AllocTensor<float>();
                const uint64_t gmOffNxt = rowBase + static_cast<uint64_t>(nxtOff);
                AscendC::DataCopy<float>(tn, xGm[gmOffNxt], nxtLen);
                qX.EnQue(tn);
            }

            AscendC::LocalTensor<float> xTile = qX.DeQue<float>();
            AscendC::LocalTensor<float> yTile = qY.AllocTensor<float>();

            AscendC::Adds<float>(yTile, xTile, -mean, (int32_t)curLen);
            AscendC::Muls<float>(yTile, yTile, invStd, (int32_t)curLen);

            qY.EnQue<float>(yTile);
            qX.FreeTensor(xTile);

            AscendC::LocalTensor<float> yOut = qY.DeQue<float>();
            const uint64_t gmOff = rowBase + static_cast<uint64_t>(colOff);
            AscendC::DataCopy<float>(yGm[gmOff], yOut, curLen);
            qY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> qX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufWork;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRed;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOne;

    AscendC::LocalTensor<float> tmp;
    AscendC::LocalTensor<float> work;
    AscendC::LocalTensor<float> red;
    AscendC::LocalTensor<float> one;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t tileLength = 0;
    uint32_t rowStart = 0;
    uint32_t rowCount = 0;
    float eps = 1.0e-5f;
    float invCols = 0.0f;
};

extern "C" __global__ __aicore__ void layer_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelLayerNormCustom op;
    op.Init(x, y,
            tiling_data.rows, tiling_data.cols,
            tiling_data.tileLength, tiling_data.eps, tiling_data.invCols);
    op.Process();
}
