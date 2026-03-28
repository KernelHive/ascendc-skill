
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelLogSoftmax {
public:
    __aicore__ inline KernelLogSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols, uint32_t rowsPerCore, uint32_t tileCols)
    {
        rows_ = rows;
        cols_ = cols;
        rowsPerCore_ = rowsPerCore;

        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startRow_ = blockIdx * rowsPerCore_;
        uint32_t endRow = startRow_ + rowsPerCore_;
        if (endRow > rows_) endRow = rows_;
        rowCount_ = (startRow_ >= rows_) ? 0u : (endRow - startRow_);

        const uint64_t total64 = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(cols_);
        const uint32_t totalCap = (total64 > 0x7FFFFFFFULL) ? 0x7FFFFFFF : static_cast<uint32_t>(total64);

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalCap);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalCap);

        if (cols_ == 0 || rows_ == 0 || rowCount_ == 0) {
            tileCols_ = 1;
            return;
        }

        tileCols_ = tileCols;
        if (tileCols_ == 0) tileCols_ = 1;
        if (tileCols_ > cols_) tileCols_ = cols_;

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileCols_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileCols_ * sizeof(DTYPE_Y));

        // Fixed scratch for reductions + scalar ops.
        constexpr uint32_t kReduceWorkFloats = 4096;
        constexpr uint32_t kScalarFloats = 256;
        pipe_.InitBuffer(tmpCalcBuf_, (kReduceWorkFloats + kScalarFloats) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rowCount_ == 0 || cols_ == 0) return;

        for (uint32_t r = 0; r < rowCount_; ++r) {
            const uint32_t rowIdx = startRow_ + r;
            const uint64_t base = static_cast<uint64_t>(rowIdx) * static_cast<uint64_t>(cols_);

            const float rowMax = RowReduceMaxPipelined(base);
            const float rowSum = RowReduceExpSumPipelined(base, rowMax);

            float logSum = 0.0f;
            if (rowSum > 0.0f) logSum = LogScalar(rowSum);

            RowWriteLogSoftmaxPipelined(base, rowMax, logSum);
        }
    }

private:
    __aicore__ inline uint32_t CurCols(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * tileCols_;
        const uint32_t remain = cols_ - offset;
        return remain >= tileCols_ ? tileCols_ : remain;
    }

    __aicore__ inline AscendC::LocalTensor<float> ReduceWork()
    {
        return tmpCalcBuf_.Get<float>();
    }

    __aicore__ inline AscendC::LocalTensor<float> ScalarBuf()
    {
        AscendC::LocalTensor<float> tmp = tmpCalcBuf_.Get<float>();
        constexpr uint32_t kReduceWorkFloats = 4096;
        return tmp[kReduceWorkFloats];
    }

    __aicore__ inline float LogScalar(float v)
    {
        AscendC::LocalTensor<float> s = ScalarBuf();
        s(0) = v;
        AscendC::Log(s, s, 1);
        return s(0);
    }

    __aicore__ inline float RowReduceMaxPipelined(uint64_t base)
    {
        const uint32_t tiles = CeilDivU32(cols_, tileCols_);
        if (tiles == 0) return -3.402823466e+38f;

        AscendC::LocalTensor<float> work = ReduceWork();
        AscendC::LocalTensor<float> s = ScalarBuf();

        float curMax = -3.402823466e+38f;

        // Prefetch tile 0
        {
            const uint32_t len0 = CurCols(0);
            AscendC::LocalTensor<DTYPE_X> x0 = inQueueX_.AllocTensor<DTYPE_X>();
            AscendC::DataCopy(x0, xGm_[base], len0);
            inQueueX_.EnQue(x0);
        }

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t len = CurCols(t);
            const uint64_t off = base + static_cast<uint64_t>(t) * static_cast<uint64_t>(tileCols_);

            // Prefetch next tile early
            if (t + 1 < tiles) {
                const uint32_t lenN = CurCols(t + 1);
                const uint64_t offN = base + static_cast<uint64_t>(t + 1) * static_cast<uint64_t>(tileCols_);
                AscendC::LocalTensor<DTYPE_X> xN = inQueueX_.AllocTensor<DTYPE_X>();
                AscendC::DataCopy(xN, xGm_[offN], lenN);
                inQueueX_.EnQue(xN);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();

            AscendC::ReduceMax<float>(s, xV, work, static_cast<int32_t>(len), false);
            const float tileMax = s(0);
            if (tileMax > curMax) curMax = tileMax;

            inQueueX_.FreeTensor(xV);
        }
        return curMax;
    }

    __aicore__ inline float RowReduceExpSumPipelined(uint64_t base, float rowMax)
    {
        const uint32_t tiles = CeilDivU32(cols_, tileCols_);
        if (tiles == 0) return 0.0f;

        AscendC::LocalTensor<float> work = ReduceWork();
        AscendC::LocalTensor<float> s = ScalarBuf();

        float acc = 0.0f;

        // Prefetch tile 0
        {
            const uint32_t len0 = CurCols(0);
            AscendC::LocalTensor<DTYPE_X> x0 = inQueueX_.AllocTensor<DTYPE_X>();
            AscendC::DataCopy(x0, xGm_[base], len0);
            inQueueX_.EnQue(x0);
        }

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t len = CurCols(t);
            const uint64_t off = base + static_cast<uint64_t>(t) * static_cast<uint64_t>(tileCols_);

            if (t + 1 < tiles) {
                const uint32_t lenN = CurCols(t + 1);
                const uint64_t offN = base + static_cast<uint64_t>(t + 1) * static_cast<uint64_t>(tileCols_);
                AscendC::LocalTensor<DTYPE_X> xN = inQueueX_.AllocTensor<DTYPE_X>();
                AscendC::DataCopy(xN, xGm_[offN], lenN);
                inQueueX_.EnQue(xN);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();

            AscendC::Adds(xV, xV, (float)(-rowMax), len);
            AscendC::Exp(xV, xV, len);
            AscendC::ReduceSum<float>(s, xV, work, static_cast<int32_t>(len));
            acc += s(0);

            inQueueX_.FreeTensor(xV);
        }
        return acc;
    }

    __aicore__ inline void RowWriteLogSoftmaxPipelined(uint64_t base, float rowMax, float logSum)
    {
        const uint32_t tiles = CeilDivU32(cols_, tileCols_);
        if (tiles == 0) return;

        // Prefetch tile 0
        {
            const uint32_t len0 = CurCols(0);
            AscendC::LocalTensor<DTYPE_X> x0 = inQueueX_.AllocTensor<DTYPE_X>();
            AscendC::DataCopy(x0, xGm_[base], len0);
            inQueueX_.EnQue(x0);
        }

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t len = CurCols(t);
            const uint64_t off = base + static_cast<uint64_t>(t) * static_cast<uint64_t>(tileCols_);

            // Prefetch next tile early
            if (t + 1 < tiles) {
                const uint32_t lenN = CurCols(t + 1);
                const uint64_t offN = base + static_cast<uint64_t>(t + 1) * static_cast<uint64_t>(tileCols_);
                AscendC::LocalTensor<DTYPE_X> xN = inQueueX_.AllocTensor<DTYPE_X>();
                AscendC::DataCopy(xN, xGm_[offN], lenN);
                inQueueX_.EnQue(xN);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

            AscendC::Adds(xV, xV, (float)(-rowMax), len);
            AscendC::Adds(xV, xV, (float)(-logSum), len);

            AscendC::DataCopy(yLocal, xV, len);
            outQueueY_.EnQue<DTYPE_Y>(yLocal);
            inQueueX_.FreeTensor(xV);

            AscendC::LocalTensor<DTYPE_Y> yV = outQueueY_.DeQue<DTYPE_Y>();
            AscendC::DataCopy(yGm_[off], yV, len);
            outQueueY_.FreeTensor(yV);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalcBuf_;

    AscendC::GlobalTensor<DTYPE_X> xGm_;
    AscendC::GlobalTensor<DTYPE_Y> yGm_;

    uint32_t rows_{0};
    uint32_t cols_{0};
    uint32_t rowsPerCore_{0};
    uint32_t tileCols_{1};

    uint32_t startRow_{0};
    uint32_t rowCount_{0};
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSoftmax op;
    op.Init(x, y,
            tiling_data.rows,
            tiling_data.cols,
            tiling_data.rowsPerCore,
            tiling_data.tileCols);
    op.Process();
}
