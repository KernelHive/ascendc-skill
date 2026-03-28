
#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

constexpr int32_t BUFFER_NUM = 1;

class KernelInstanceNormCustom {
public:
    __aicore__ inline KernelInstanceNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t n, uint32_t c, uint32_t hw, uint32_t planes,
                               uint32_t tileLength, uint32_t reduceTmpLen,
                               float invHw, float eps)
    {
        n_ = n;
        c_ = c;
        hw_ = hw;
        planes_ = planes;
        tileLength_ = tileLength;
        reduceTmpLen_ = reduceTmpLen;
        invHw_ = invHw;
        eps_ = eps;

        const uint64_t total = static_cast<uint64_t>(planes_) * static_cast<uint64_t>(hw_);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), total);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), total);

        pipe_.InitBuffer(qX_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qY_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(qTmp_, BUFFER_NUM, tileLength_ * sizeof(float));   // for x^2
        pipe_.InitBuffer(qRedDst_, BUFFER_NUM, 32U * sizeof(float));
        pipe_.InitBuffer(qRedTmp_, BUFFER_NUM, reduceTmpLen_ * sizeof(float));
        pipe_.InitBuffer(qScalar_, BUFFER_NUM, 32U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (planes_ == 0u || hw_ == 0u) return;

        const uint32_t block = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        if (blockNum == 0u) return;

        // round-robin over planes for load balance, no overlap across blocks
        for (uint32_t p = block; p < planes_; p += blockNum) {
            NormalizePlane(p);
        }
    }

private:
    __aicore__ inline void NormalizePlane(uint32_t p)
    {
        const uint32_t base = p * hw_;  // plane stride is exactly hw (never tile-aligned)

        LocalTensor<float> redDst = qRedDst_.AllocTensor<float>();
        LocalTensor<float> redTmp = qRedTmp_.AllocTensor<float>();
        LocalTensor<float> scalar = qScalar_.AllocTensor<float>();

        float sum = 0.0f;
        float sumsq = 0.0f;

        // pass1: mean/var stats
        uint32_t offset = 0;
        while (offset < hw_) {
            uint32_t cur = hw_ - offset;
            if (cur > tileLength_) cur = tileLength_;

            LocalTensor<float> xUb = qX_.AllocTensor<float>();
            AscendC::DataCopy(xUb, xGm_[base + offset], cur);
            // no padding needed: only compute on cur
            qX_.EnQue(xUb);

            LocalTensor<float> xIn = qX_.DeQue<float>();

            AscendC::ReduceSum<float>(redDst, xIn, redTmp, (int32_t)cur);
            sum += redDst.GetValue(0);

            LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
            AscendC::Mul<float>(tmp, xIn, xIn, (int32_t)cur);
            AscendC::ReduceSum<float>(redDst, tmp, redTmp, (int32_t)cur);
            sumsq += redDst.GetValue(0);

            qTmp_.FreeTensor(tmp);
            qX_.FreeTensor(xIn);

            offset += cur;
        }

        // mean
        scalar.SetValue(0, sum);
        AscendC::Muls<float>(scalar, scalar, invHw_, 1);
        const float mean = scalar.GetValue(0);

        // E[x^2]
        scalar.SetValue(0, sumsq);
        AscendC::Muls<float>(scalar, scalar, invHw_, 1);
        const float ex2 = scalar.GetValue(0);

        float var = ex2 - mean * mean;
        if (var < 0.0f) var = 0.0f;

        // rstd = 1 / sqrt(var + eps) (use Div+Sqrt for accuracy)
        scalar.SetValue(0, var + eps_);
        AscendC::Sqrt<float>(scalar, scalar, 1);
        redDst.SetValue(0, 1.0f);
        AscendC::Div<float>(redDst, redDst, scalar, 1);
        const float rstd = redDst.GetValue(0);

        // pass2: normalize
        offset = 0;
        while (offset < hw_) {
            uint32_t cur = hw_ - offset;
            if (cur > tileLength_) cur = tileLength_;

            LocalTensor<float> xUb = qX_.AllocTensor<float>();
            AscendC::DataCopy(xUb, xGm_[base + offset], cur);
            qX_.EnQue(xUb);

            LocalTensor<float> xIn = qX_.DeQue<float>();
            LocalTensor<float> yUb = qY_.AllocTensor<float>();

            // y = (x - mean) * rstd
            AscendC::Duplicate<float>(yUb, mean, (int32_t)cur);
            AscendC::Sub<float>(yUb, xIn, yUb, (int32_t)cur);
            AscendC::Muls<float>(yUb, yUb, rstd, (int32_t)cur);

            qY_.EnQue(yUb);
            qX_.FreeTensor(xIn);

            LocalTensor<float> yOut = qY_.DeQue<float>();
            AscendC::DataCopy(yGm_[base + offset], yOut, cur);
            qY_.FreeTensor(yOut);

            offset += cur;
        }

        qScalar_.FreeTensor(scalar);
        qRedTmp_.FreeTensor(redTmp);
        qRedDst_.FreeTensor(redDst);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qY_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qTmp_;

    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qRedDst_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qRedTmp_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qScalar_;

    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;

    uint32_t n_ = 0, c_ = 0, hw_ = 0, planes_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t reduceTmpLen_ = 0;
    float invHw_ = 0.0f;
    float eps_ = 1e-5f;
};

extern "C" __global__ __aicore__ void instance_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    KernelInstanceNormCustom op;
    op.Init(x, y,
            td.n, td.c, td.hw, td.planes,
            td.tileLength, td.reduceTmpLen,
            td.invHw, td.eps);
    op.Process();
}
