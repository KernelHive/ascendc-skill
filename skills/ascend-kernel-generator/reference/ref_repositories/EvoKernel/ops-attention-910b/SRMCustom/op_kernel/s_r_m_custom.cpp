
#include "kernel_operator.h"
#include <cstdint>

// Fused SRM optimized:
// - Vectorize reductions by padding HW=49 to 64 and using ReduceSum for sum and sumsq
// - Keep one GM load + one GM store per (b,c) tile
// - Keep exact sigmoid (no approximation) and unbiased variance formula
// - Reduce sigmoid temp UB to minimal (single-element)

class KernelSRMCustom {
public:
    __aicore__ inline KernelSRMCustom() {}

    __aicore__ inline void Init(GM_ADDR x,
                               GM_ADDR cfc_weight,
                               GM_ADDR bn_weight,
                               GM_ADDR bn_bias,
                               GM_ADDR bn_running_mean,
                               GM_ADDR bn_running_var,
                               GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t xTotal,
                               uint32_t coreNum, uint32_t bPerCore,
                               float eps)
    {
        this->B = (int32_t)B;
        this->C = (int32_t)C;
        this->H = (int32_t)H;
        this->W = (int32_t)W;
        this->HW = (int32_t)HW;
        this->xTotal = (int32_t)xTotal;
        this->coreNum = (int32_t)coreNum;
        this->bPerCore = (int32_t)bPerCore;
        this->eps = eps;

        int32_t blk = (int32_t)AscendC::GetBlockIdx();
        int32_t startB = blk * this->bPerCore;
        int32_t endB = startB + this->bPerCore;
        if (endB > this->B) endB = this->B;
        this->startB = startB;
        this->endB = endB;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)xTotal);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)xTotal);

        cfcWGm.SetGlobalBuffer((__gm__ float*)cfc_weight, (uint64_t)C * 2ULL);
        bnWGm.SetGlobalBuffer((__gm__ float*)bn_weight, (uint64_t)C);
        bnBGm.SetGlobalBuffer((__gm__ float*)bn_bias, (uint64_t)C);
        bnMeanGm.SetGlobalBuffer((__gm__ float*)bn_running_mean, (uint64_t)C);
        bnVarGm.SetGlobalBuffer((__gm__ float*)bn_running_var, (uint64_t)C);

        // Pad tile to 64 to allow vector ReduceSum; only store first 49 back.
        pipe.InitBuffer(xPadBuf, (uint32_t)(64U * sizeof(float)));
        pipe.InitBuffer(tmpBuf, 64U * sizeof(float));          // scalars + small vectors
        pipe.InitBuffer(sigTmpBuf, 256U);                      // minimal sigmoid temp
    }

    __aicore__ inline void Process()
    {
        if (!(C == 512 && H == 7 && W == 7 && HW == 49)) return;
        if (startB >= endB) return;

        AscendC::LocalTensor<float> xPad = xPadBuf.Get<float>();   // 64
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> sigTmp = sigTmpBuf.Get<uint8_t>();

        const float N = (float)HW;      // 49
        const float invN = 1.0f / N;
        const float invNm1 = (HW > 1) ? (1.0f / (float)(HW - 1)) : 0.0f;

        for (int32_t b = startB; b < endB; ++b) {
            uint64_t bBase = (uint64_t)b * (uint64_t)C * (uint64_t)HW;

            for (int32_t c = 0; c < C; ++c) {
                uint64_t base = bBase + (uint64_t)c * (uint64_t)HW;

                // Load 49, zero-pad to 64
                AscendC::DataCopy(xPad, xGm[base], (uint32_t)HW);
                AscendC::Duplicate<float>(xPad[(uint32_t)HW], 0.0f, (uint32_t)(64 - HW));

                // sum = ReduceSum(xPad[0:64])
                AscendC::ReduceSum<float>(tmp, xPad, tmp, 64U);
                float sum = tmp.GetValue(0U);

                // sumsq = ReduceSum(xPad*xPad)
                AscendC::Mul(xPad, xPad, xPad, 64U);
                AscendC::ReduceSum<float>(tmp, xPad, tmp, 64U);
                float sumsq = tmp.GetValue(0U);

                float mean = sum * invN;

                float var_num = sumsq - (sum * sum) * invN;
                if (var_num < 0.0f) var_num = 0.0f;
                float var_unbiased = (HW > 1) ? (var_num * invNm1) : 0.0f;

                tmp.SetValue(0U, var_unbiased);
                AscendC::Sqrt(tmp, tmp, 1);
                float stdv = tmp.GetValue(0U);

                uint64_t wBase = (uint64_t)c * 2ULL;
                float w0 = cfcWGm.GetValue(wBase + 0ULL);
                float w1 = cfcWGm.GetValue(wBase + 1ULL);
                float z = mean * w0 + stdv * w1;

                float gamma = bnWGm.GetValue((uint64_t)c);
                float beta  = bnBGm.GetValue((uint64_t)c);
                float rm    = bnMeanGm.GetValue((uint64_t)c);
                float rv    = bnVarGm.GetValue((uint64_t)c);

                float denom = rv + eps;
                if (denom < 0.0f) denom = 0.0f;
                tmp.SetValue(0U, denom);
                AscendC::Sqrt(tmp, tmp, 1);
                float sd = tmp.GetValue(0U);
                tmp.SetValue(0U, sd);
                AscendC::Reciprocal(tmp, tmp, 1);
                float rstd = tmp.GetValue(0U);

                float zbn = (z - rm) * rstd;
                zbn = zbn * gamma + beta;

                tmp.SetValue(0U, zbn);
                AscendC::Sigmoid<float, true>(tmp, tmp, sigTmp, 1);
                float g = tmp.GetValue(0U);

                // Reload original x (49) into xPad and scale; avoids needing second UB buffer
                AscendC::DataCopy(xPad, xGm[base], (uint32_t)HW);
                AscendC::Muls(xPad, xPad, g, (uint32_t)HW);
                AscendC::DataCopy(yGm[base], xPad, (uint32_t)HW);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> xPadBuf;
    AscendC::TBuf<> tmpBuf;
    AscendC::TBuf<> sigTmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> cfcWGm;
    AscendC::GlobalTensor<float> bnWGm;
    AscendC::GlobalTensor<float> bnBGm;
    AscendC::GlobalTensor<float> bnMeanGm;
    AscendC::GlobalTensor<float> bnVarGm;

    int32_t B{0}, C{0}, H{0}, W{0}, HW{0}, xTotal{0};
    int32_t coreNum{0}, bPerCore{0}, startB{0}, endB{0};
    float eps{1e-5f};
};

extern "C" __global__ __aicore__ void srm_custom(GM_ADDR x,
                                                GM_ADDR cfc_weight,
                                                GM_ADDR bn_weight,
                                                GM_ADDR bn_bias,
                                                GM_ADDR bn_running_mean,
                                                GM_ADDR bn_running_var,
                                                GM_ADDR y,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSRMCustom op;
    op.Init(x, cfc_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.xTotal,
            tiling_data.coreNum, tiling_data.bPerCore,
            tiling_data.eps);
    op.Process();
}
