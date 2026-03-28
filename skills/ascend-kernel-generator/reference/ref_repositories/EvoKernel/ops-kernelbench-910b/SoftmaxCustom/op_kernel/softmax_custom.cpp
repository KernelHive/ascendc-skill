
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

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

        uint64_t total64 = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(cols_);
        uint32_t totalCap = (total64 > 0x7FFFFFFFULL) ? 0x7FFFFFFF : static_cast<uint32_t>(total64);

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalCap);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalCap);

        if (cols_ == 0 || rows_ == 0 || rowCount_ == 0) {
            tileCols_ = 1;
            totalTiles_ = 0;
            fullTiles_ = 0;
            lastLen_ = 0;
            hasTail_ = false;
            return;
        }

        tileCols_ = tileCols;
        if (tileCols_ == 0) tileCols_ = 1;
        if (tileCols_ > cols_) tileCols_ = cols_;

        totalTiles_ = CeilDivU32(cols_, tileCols_);
        const uint32_t rem = cols_ - (totalTiles_ - 1u) * tileCols_;
        lastLen_ = (totalTiles_ == 0) ? 0 : rem;
        hasTail_ = (totalTiles_ > 0 && lastLen_ != tileCols_);
        fullTiles_ = hasTail_ ? (totalTiles_ - 1u) : totalTiles_;

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileCols_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileCols_ * sizeof(DTYPE_Y));

        // VECCALC only used for reductions + scalar (no tile vector anymore).
        constexpr uint32_t kReduceWorkFloats = 4096;
        constexpr uint32_t kScalarFloats = 256;
        pipe_.InitBuffer(tmpCalcBuf_, (kReduceWorkFloats + kScalarFloats) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rowCount_ == 0 || cols_ == 0 || totalTiles_ == 0) return;

        for (uint32_t r = 0; r < rowCount_; ++r) {
            const uint32_t rowIdx = startRow_ + r;
            const uint64_t base = static_cast<uint64_t>(rowIdx) * static_cast<uint64_t>(cols_);

            float rowMax = RowReduceMax(base);
            float rowSum = RowReduceExpSumInplace(base, rowMax);

            if (rowSum <= 0.0f) {
                RowWriteZeros(base);
            } else {
                float invSum = ReciprocalScalar(rowSum);
                RowWriteSoftmaxInplace(base, rowMax, invSum);
            }
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<float> ReduceWork()
    {
        return tmpCalcBuf_.Get<float>(); // [0:4096)
    }

    __aicore__ inline AscendC::LocalTensor<float> ScalarBuf()
    {
        constexpr uint32_t kReduceWorkFloats = 4096;
        return tmpCalcBuf_.Get<float>()[kReduceWorkFloats];
    }

    __aicore__ inline float ReciprocalScalar(float v)
    {
        AscendC::LocalTensor<float> s = ScalarBuf();
        s(0) = v;
        AscendC::Reciprocal(s, s, 1);
        return s(0);
    }

    __aicore__ inline void PrimeLoadX(uint64_t gmOff, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm_[gmOff], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline float ReduceMaxTile(AscendC::LocalTensor<DTYPE_X> xV, uint32_t len)
    {
        AscendC::LocalTensor<float> work = ReduceWork();
        AscendC::LocalTensor<float> s = ScalarBuf();
        AscendC::ReduceMax<float>(s, xV, work, static_cast<int32_t>(len), false);
        return s(0);
    }

    __aicore__ inline float ReduceSumTile(AscendC::LocalTensor<DTYPE_X> xV, uint32_t len)
    {
        AscendC::LocalTensor<float> work = ReduceWork();
        AscendC::LocalTensor<float> s = ScalarBuf();
        // ReduceSum<float> supports non-float input and accumulates in float.
        AscendC::ReduceSum<float>(s, xV, work, static_cast<int32_t>(len));
        return s(0);
    }

    __aicore__ inline float RowReduceMax(uint64_t base)
    {
        float curMax = -3.402823466e+38f;
        if (totalTiles_ == 0) return curMax;

        PrimeLoadX(base, (fullTiles_ > 0) ? tileCols_ : lastLen_);

        for (uint32_t t = 0; t < fullTiles_; ++t) {
            if (t + 1u < totalTiles_) {
                const uint32_t nextLen = (t + 1u < fullTiles_) ? tileCols_ : lastLen_;
                const uint64_t offn = base + static_cast<uint64_t>(t + 1u) * static_cast<uint64_t>(tileCols_);
                PrimeLoadX(offn, nextLen);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();
            float tileMax = ReduceMaxTile(xV, tileCols_);
            inQueueX_.FreeTensor(xV);
            if (tileMax > curMax) curMax = tileMax;
        }

        if (hasTail_) {
            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();
            float tileMax = ReduceMaxTile(xV, lastLen_);
            inQueueX_.FreeTensor(xV);
            if (tileMax > curMax) curMax = tileMax;
        }

        return curMax;
    }

    // Sum pass: load tile, do (x - max) and exp in-place in the VECIN tensor, reduce sum.
    __aicore__ inline float RowReduceExpSumInplace(uint64_t base, float rowMax)
    {
        float acc = 0.0f;
        if (totalTiles_ == 0) return acc;

        PrimeLoadX(base, (fullTiles_ > 0) ? tileCols_ : lastLen_);

        for (uint32_t t = 0; t < fullTiles_; ++t) {
            if (t + 1u < totalTiles_) {
                const uint32_t nextLen = (t + 1u < fullTiles_) ? tileCols_ : lastLen_;
                const uint64_t offn = base + static_cast<uint64_t>(t + 1u) * static_cast<uint64_t>(tileCols_);
                PrimeLoadX(offn, nextLen);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();

            AscendC::Adds(xV, xV, (float)(-rowMax), tileCols_);
            AscendC::Exp(xV, xV, tileCols_);
            acc += ReduceSumTile(xV, tileCols_);

            inQueueX_.FreeTensor(xV);
        }

        if (hasTail_) {
            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();

            AscendC::Adds(xV, xV, (float)(-rowMax), lastLen_);
            AscendC::Exp(xV, xV, lastLen_);
            acc += ReduceSumTile(xV, lastLen_);

            inQueueX_.FreeTensor(xV);
        }

        return acc;
    }

    __aicore__ inline void RowWriteZeros(uint64_t base)
    {
        if (totalTiles_ == 0) return;

        for (uint32_t t = 0; t < fullTiles_; ++t) {
            const uint64_t off = base + static_cast<uint64_t>(t) * static_cast<uint64_t>(tileCols_);
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate(yLocal, (DTYPE_Y)0.0f, tileCols_);
            outQueueY_.EnQue<DTYPE_Y>(yLocal);
            AscendC::LocalTensor<DTYPE_Y> yV = outQueueY_.DeQue<DTYPE_Y>();
            AscendC::DataCopy(yGm_[off], yV, tileCols_);
            outQueueY_.FreeTensor(yV);
        }

        if (hasTail_) {
            const uint64_t off = base + static_cast<uint64_t>(fullTiles_) * static_cast<uint64_t>(tileCols_);
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate(yLocal, (DTYPE_Y)0.0f, lastLen_);
            outQueueY_.EnQue<DTYPE_Y>(yLocal);
            AscendC::LocalTensor<DTYPE_Y> yV = outQueueY_.DeQue<DTYPE_Y>();
            AscendC::DataCopy(yGm_[off], yV, lastLen_);
            outQueueY_.FreeTensor(yV);
        }
    }

    // Write pass: load tile, do exp(x-max) and scale in-place in VECIN tensor, then copy to VECOUT and store.
    __aicore__ inline void RowWriteSoftmaxInplace(uint64_t base, float rowMax, float invSum)
    {
        if (totalTiles_ == 0) return;

        PrimeLoadX(base, (fullTiles_ > 0) ? tileCols_ : lastLen_);

        for (uint32_t t = 0; t < fullTiles_; ++t) {
            const uint64_t off = base + static_cast<uint64_t>(t) * static_cast<uint64_t>(tileCols_);

            if (t + 1u < totalTiles_) {
                const uint32_t nextLen = (t + 1u < fullTiles_) ? tileCols_ : lastLen_;
                const uint64_t offn = base + static_cast<uint64_t>(t + 1u) * static_cast<uint64_t>(tileCols_);
                PrimeLoadX(offn, nextLen);
            }

            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

            AscendC::Adds(xV, xV, (float)(-rowMax), tileCols_);
            AscendC::Exp(xV, xV, tileCols_);
            AscendC::Muls(xV, xV, invSum, tileCols_);

            AscendC::DataCopy(yLocal, xV, tileCols_);
            inQueueX_.FreeTensor(xV);

            outQueueY_.EnQue<DTYPE_Y>(yLocal);
            AscendC::LocalTensor<DTYPE_Y> yV = outQueueY_.DeQue<DTYPE_Y>();
            AscendC::DataCopy(yGm_[off], yV, tileCols_);
            outQueueY_.FreeTensor(yV);
        }

        if (hasTail_) {
            const uint64_t off = base + static_cast<uint64_t>(fullTiles_) * static_cast<uint64_t>(tileCols_);
            AscendC::LocalTensor<DTYPE_X> xV = inQueueX_.DeQue<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY_.AllocTensor<DTYPE_Y>();

            AscendC::Adds(xV, xV, (float)(-rowMax), lastLen_);
            AscendC::Exp(xV, xV, lastLen_);
            AscendC::Muls(xV, xV, invSum, lastLen_);

            AscendC::DataCopy(yLocal, xV, lastLen_);
            inQueueX_.FreeTensor(xV);

            outQueueY_.EnQue<DTYPE_Y>(yLocal);
            AscendC::LocalTensor<DTYPE_Y> yV = outQueueY_.DeQue<DTYPE_Y>();
            AscendC::DataCopy(yGm_[off], yV, lastLen_);
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

    uint32_t totalTiles_{0};
    uint32_t fullTiles_{0};
    uint32_t lastLen_{0};
    bool hasTail_{false};
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y,
            tiling_data.rows,
            tiling_data.cols,
            tiling_data.rowsPerCore,
            tiling_data.tileCols);
    op.Process();
}
