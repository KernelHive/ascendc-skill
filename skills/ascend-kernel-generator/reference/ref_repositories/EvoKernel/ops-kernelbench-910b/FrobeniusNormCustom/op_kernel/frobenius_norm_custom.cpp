
#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

constexpr int32_t BUFFER_NUM = 1;

class KernelFrobeniusNormCustom {
public:
    __aicore__ inline KernelFrobeniusNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalLength, uint32_t tileLength)
    {
        totalLength_ = totalLength;
        tileLength_ = tileLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), totalLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), totalLength_);

        pipe_.InitBuffer(qX_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qY_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qSq_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qWork_, BUFFER_NUM, tileLength_ * sizeof(float));
        // Small scratch for scalar ops (at least 16 floats for vector safety).
        pipe_.InitBuffer(qTmp_, BUFFER_NUM, 16 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0u) return;

        float inv = ComputeInvNorm();
        ScaleAll(inv);
    }

private:
    __aicore__ inline float ComputeInvNorm()
    {
        float acc = 0.0f;
        const uint32_t iters = (totalLength_ + tileLength_ - 1u) / tileLength_;

        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t base = i * tileLength_;
            uint32_t curLen = totalLength_ - base;
            if (curLen > tileLength_) curLen = tileLength_;

            LocalTensor<float> xUb = qX_.AllocTensor<float>();
            // Vectorized GM->UB copy for valid length only.
            AscendC::DataCopy(xUb, xGm_[base], curLen);
            // Zero-pad tail for safe full-tile vector ops.
            if (curLen < tileLength_) {
                AscendC::Duplicate<float>(xUb[curLen], 0.0f, (int32_t)(tileLength_ - curLen));
            }
            qX_.EnQue(xUb);

            LocalTensor<float> xIn = qX_.DeQue<float>();

            LocalTensor<float> sq = qSq_.AllocTensor<float>();
            AscendC::Mul<float>(sq, xIn, xIn, (int32_t)tileLength_);

            LocalTensor<float> work = qWork_.AllocTensor<float>();
            AscendC::ReduceSum<float>(sq, sq, work, (int32_t)tileLength_);
            acc += sq.GetValue(0);

            qWork_.FreeTensor(work);
            qSq_.FreeTensor(sq);
            qX_.FreeTensor(xIn);
        }

        // Match PyTorch semantics: no eps clamp.
        // inv = 1 / sqrt(acc). If acc==0, this becomes inf and x*inf yields NaNs for zero elements.
        LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
        AscendC::Duplicate<float>(tmp, 0.0f, 16);
        tmp.SetValue(0, acc);
        AscendC::Sqrt<float>(tmp, tmp, 1);

        LocalTensor<float> one = qSq_.AllocTensor<float>();
        AscendC::Duplicate<float>(one, 1.0f, 1);
        AscendC::Div<float>(one, one, tmp, 1);

        float inv = one.GetValue(0);
        qSq_.FreeTensor(one);
        qTmp_.FreeTensor(tmp);
        return inv;
    }

    __aicore__ inline void ScaleAll(float invNorm)
    {
        const uint32_t iters = (totalLength_ + tileLength_ - 1u) / tileLength_;

        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t base = i * tileLength_;
            uint32_t curLen = totalLength_ - base;
            if (curLen > tileLength_) curLen = tileLength_;

            LocalTensor<float> xUb = qX_.AllocTensor<float>();
            AscendC::DataCopy(xUb, xGm_[base], curLen);
            if (curLen < tileLength_) {
                AscendC::Duplicate<float>(xUb[curLen], 0.0f, (int32_t)(tileLength_ - curLen));
            }
            qX_.EnQue(xUb);

            LocalTensor<float> xIn = qX_.DeQue<float>();
            LocalTensor<float> yUb = qY_.AllocTensor<float>();

            AscendC::Muls<float>(yUb, xIn, invNorm, (int32_t)tileLength_);

            qY_.EnQue(yUb);
            qX_.FreeTensor(xIn);

            LocalTensor<float> yOut = qY_.DeQue<float>();
            // Vectorized UB->GM for valid length only (no OOB on last tile).
            AscendC::DataCopy(yGm_[base], yOut, curLen);
            qY_.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qY_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSq_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qWork_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qTmp_;

    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    uint32_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
};

extern "C" __global__ __aicore__ void frobenius_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelFrobeniusNormCustom op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
