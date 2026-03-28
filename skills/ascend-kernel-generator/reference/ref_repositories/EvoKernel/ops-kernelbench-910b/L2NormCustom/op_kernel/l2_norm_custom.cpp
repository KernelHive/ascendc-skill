
#include "kernel_operator.h"

class KernelL2NormCustom {
public:
    __aicore__ inline KernelL2NormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols,
                               uint32_t tileLength, float eps)
    {
        rows_ = rows;
        cols_ = cols;
        tileLength_ = tileLength;
        eps_ = eps;

        const uint64_t total64 = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(cols_);
        const uint32_t total = (total64 > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : static_cast<uint32_t>(total64);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), total);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), total);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t perCore = (rows_ + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * perCore;

        uint32_t count = 0;
        if (start < rows_) {
            const uint32_t remain = rows_ - start;
            count = (remain < perCore) ? remain : perCore;
        }
        rowStart_ = start;
        rowCount_ = count;

        // Persistent UB buffers (no queues, no per-tile alloc/free).
        pipe_.InitBuffer(xBuf_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, tileLength_ * sizeof(float)); // reuse for sq and y
        pipe_.InitBuffer(redBuf_, tileLength_ * sizeof(float)); // reduce scratch (size >= curLen)
        pipe_.InitBuffer(scBuf_, 64); // scalar staging
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0u || cols_ == 0u || rowCount_ == 0u) return;

        for (uint32_t r = 0; r < rowCount_; ++r) {
            const uint32_t row = rowStart_ + r;
            const float inv = ComputeInvNorm_(row);
            ScaleRow_(row, inv);
        }
    }

private:
    __aicore__ inline float ComputeInvNorm_(uint32_t row)
    {
        float acc = 0.0f;

        AscendC::LocalTensor<float> xUb  = xBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp  = tmpBuf_.Get<float>(); // squares
        AscendC::LocalTensor<float> work = redBuf_.Get<float>();
        AscendC::LocalTensor<float> sc   = scBuf_.Get<float>();

        const uint32_t iters = (cols_ + tileLength_ - 1u) / tileLength_;
        const uint64_t rowBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(cols_);

        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t colOff = i * tileLength_;
            uint32_t curLen = cols_ - colOff;
            if (curLen > tileLength_) curLen = tileLength_;

            const uint64_t gmOff = rowBase + static_cast<uint64_t>(colOff);

            // Copy only valid region; do vector ops only over curLen (no tile padding/zeroing).
            AscendC::DataCopy<float>(xUb, xGm_[gmOff], curLen);
            AscendC::Mul<float>(tmp, xUb, xUb, (int32_t)curLen);

            // ReduceSum writes result to tmp[0]; uses work as scratch.
            AscendC::ReduceSum<float>(tmp, tmp, work, (int32_t)curLen);
            acc += tmp.GetValue(0);
        }

        if (acc < eps_) acc = eps_;

        // inv = 1 / sqrt(acc) using vector primitives on 1 element.
        AscendC::Duplicate<float>(sc, 0.0f, 1);
        sc.SetValue(0, acc);
        AscendC::Sqrt<float>(sc, sc, 1);

        AscendC::Duplicate<float>(tmp, 1.0f, 1);
        AscendC::Div<float>(tmp, tmp, sc, 1);
        return tmp.GetValue(0);
    }

    __aicore__ inline void ScaleRow_(uint32_t row, float inv)
    {
        AscendC::LocalTensor<float> xUb = xBuf_.Get<float>();
        AscendC::LocalTensor<float> yUb = tmpBuf_.Get<float>();

        const uint32_t iters = (cols_ + tileLength_ - 1u) / tileLength_;
        const uint64_t rowBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(cols_);

        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t colOff = i * tileLength_;
            uint32_t curLen = cols_ - colOff;
            if (curLen > tileLength_) curLen = tileLength_;

            const uint64_t gmOff = rowBase + static_cast<uint64_t>(colOff);

            AscendC::DataCopy<float>(xUb, xGm_[gmOff], curLen);
            AscendC::Muls<float>(yUb, xUb, inv, (int32_t)curLen);
            AscendC::DataCopy<float>(yGm_[gmOff], yUb, curLen);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowCount_ = 0;
    float eps_ = 1.0e-12f;
};

extern "C" __global__ __aicore__ void l2_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelL2NormCustom op;
    op.Init(x, y,
            tiling_data.rows, tiling_data.cols,
            tiling_data.tileLength, tiling_data.eps);
    op.Process();
}
