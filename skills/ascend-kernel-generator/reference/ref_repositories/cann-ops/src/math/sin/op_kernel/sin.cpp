/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file sin.cpp
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <class T, class ComputeStrategy>
class KernelSin
{
public:
    __aicore__ inline KernelSin() {}

    template <bool IsExistBigCore>
    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR y,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t bigCoreLoopNum,
                                uint32_t smallCoreLoopNum,
                                uint32_t ubPartDataNum,
                                uint32_t smallCoreTailDataNum,
                                uint32_t bigCoreTailDataNum,
                                uint32_t tailBlockNum,
                                AscendC::TPipe* pipe)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreNum - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * coreNum;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(xBuf, this->ubPartDataNum * sizeof(float));
            pipe->InitBuffer(yBuf, this->ubPartDataNum * sizeof(float));
        }
        strategy.InitBufImpl(pipe, this->ubPartDataNum);
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = PreDeQueCastX();
        AscendC::LocalTensor<float> yLocal = PreAllocateY();

        strategy.ComputeImpl(xLocal, yLocal, this->processDataNum);

        PostReleaseCastEnQue(xLocal, yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline AscendC::LocalTensor<float> PreDeQueCastX()
    {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            return xLocal;
        } else {
            AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
            AscendC::LocalTensor<T> xOrigin = inQueueX.DeQue<T>();
            AscendC::Cast(xLocal, xOrigin, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            inQueueX.FreeTensor(xOrigin);
            return xLocal;
        }
    }

    __aicore__ inline AscendC::LocalTensor<float> PreAllocateY()
    {
        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            return yLocal;
        } else {
            AscendC::LocalTensor<float> yLocal = yBuf.Get<float>();
            return yLocal;
        }
    }

    __aicore__ inline void PostReleaseCastEnQue(AscendC::LocalTensor<float>& xLocal,
                                                AscendC::LocalTensor<float>& yLocal)
    {
        if constexpr (std::is_same_v<T, float>) {
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
        } else {
            AscendC::LocalTensor<T> yTarget = outQueueY.AllocTensor<T>();
            AscendC::Cast(yTarget, yLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            outQueueY.EnQue(yTarget);
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf, yBuf;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    ComputeStrategy strategy;
};

class HighPerfStrategy
{
public:
    __aicore__ inline HighPerfStrategy() {}

    __aicore__ inline void InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum)
    {
        pipe->InitBuffer(tmpBuf1, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf2, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf3, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf4, tileDataNum * sizeof(float));
    }

    __aicore__ inline void ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       uint32_t processDataNum);

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
};

constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375f;

constexpr float SIN_RES_MULIT_SCA = 2.604926501e-6f;
constexpr float SIN_RES_ADDICT_UP = -0.0001980894471f;
constexpr float SIN_2ADDS = 0.008333049340f;
constexpr float SIN_3ADDS = -0.1666665792f;

constexpr float pi_0 = 3.14160156f;
constexpr float pi_1 = -8.9071691e-06f;
constexpr float pi_2 = -1.74122761e-09f;
constexpr float pi_3 = 1.24467439e-13f;

__aicore__ inline void HighPerfStrategy::ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                                     AscendC::LocalTensor<float>& yLocal,
                                                     uint32_t processDataNum)
{
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::LocalTensor<float> tmpTensor3 = tmpBuf3.Get<float>();
    AscendC::LocalTensor<float> tmpTensor4 = tmpBuf4.Get<float>();

    const AscendC::LocalTensor<float>& input_x = xLocal;
    const AscendC::LocalTensor<float>& x_vmul = tmpTensor2;
    const AscendC::LocalTensor<float>& x_vmul0 = yLocal;
    const AscendC::LocalTensor<float>& round_pi_div = tmpTensor1;
    const AscendC::LocalTensor<float>& round_pi_div0 = tmpTensor3;
    const AscendC::LocalTensor<float>& round_pi_div0_1 = tmpTensor2;
    const AscendC::LocalTensor<float>& round_pi_div1 = yLocal;
    const AscendC::LocalTensor<float>& fix = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_1 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_1 = xLocal;
    const AscendC::LocalTensor<float>& fix_2 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_3 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_3 = xLocal;
    const AscendC::LocalTensor<float>& fix_4 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_5 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_5 = xLocal;
    const AscendC::LocalTensor<float>& fix_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& x_fixed_6 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_7 = xLocal;
    const AscendC::LocalTensor<float>& x_fixed_7 = yLocal;
    const AscendC::LocalTensor<float>& x_pow = tmpTensor2;
    const AscendC::LocalTensor<float>& kover2 = xLocal;
    const AscendC::LocalTensor<float>& kover2floor = tmpTensor3;
    const AscendC::LocalTensor<float>& kover2floorm4 = xLocal;
    const AscendC::LocalTensor<float>& k2 = tmpTensor3;
    const AscendC::LocalTensor<float>& sign = tmpTensor4;
    const AscendC::LocalTensor<float>& sign_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& res_up = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_1 = xLocal;
    const AscendC::LocalTensor<float>& res_up_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_3 = xLocal;
    const AscendC::LocalTensor<float>& res_up_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_5 = xLocal;
    const AscendC::LocalTensor<float>& res_up_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_7 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_up_8 = xLocal;
    const AscendC::LocalTensor<float>& res_sign = yLocal;
    const AscendC::LocalTensor<float>& res_mins = tmpTensor1;
    const AscendC::LocalTensor<float>& res_maxs = yLocal;

    AscendC::Muls(x_vmul, input_x, PI_FOR_X_TODIV, processDataNum);
    AscendC::Muls(x_vmul0, x_vmul, 1.0f / 2048.0f, processDataNum);
    AscendC::Cast(round_pi_div, x_vmul, AscendC::RoundMode::CAST_ROUND, processDataNum);
    AscendC::Cast(round_pi_div0, x_vmul0, AscendC::RoundMode::CAST_ROUND, processDataNum);
    AscendC::Muls(round_pi_div0_1, round_pi_div0, 2048.0f, processDataNum);
    AscendC::Sub(round_pi_div1, round_pi_div, round_pi_div0_1, processDataNum);
    AscendC::Muls(fix, round_pi_div0_1, pi_0, processDataNum);
    AscendC::Sub(x_fixed, input_x, fix, processDataNum);
    AscendC::Muls(fix_1, round_pi_div1, pi_0, processDataNum);
    AscendC::Sub(x_fixed_1, x_fixed, fix_1, processDataNum);
    AscendC::Muls(fix_2, round_pi_div0_1, pi_1, processDataNum);
    AscendC::Sub(x_fixed_2, x_fixed_1, fix_2, processDataNum);
    AscendC::Muls(fix_3, round_pi_div1, pi_1, processDataNum);
    AscendC::Sub(x_fixed_3, x_fixed_2, fix_3, processDataNum);
    AscendC::Muls(fix_4, round_pi_div0_1, pi_2, processDataNum);
    AscendC::Sub(x_fixed_4, x_fixed_3, fix_4, processDataNum);
    AscendC::Muls(fix_5, round_pi_div1, pi_2, processDataNum);
    AscendC::Sub(x_fixed_5, x_fixed_4, fix_5, processDataNum);
    AscendC::Muls(fix_6, round_pi_div0_1, pi_3, processDataNum);
    AscendC::Sub(x_fixed_6, x_fixed_5, fix_6, processDataNum);
    AscendC::Muls(fix_7, round_pi_div1, pi_3, processDataNum);
    AscendC::Sub(x_fixed_7, x_fixed_6, fix_7, processDataNum);
    AscendC::Mul(x_pow, x_fixed_7, x_fixed_7, processDataNum);
    AscendC::Muls(kover2, round_pi_div, 0.5f, processDataNum);
    AscendC::Cast(kover2floor, kover2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    AscendC::Muls(kover2floorm4, kover2floor, 4.0f, processDataNum);
    AscendC::Muls(k2, round_pi_div, -2.0f, processDataNum);
    AscendC::Add(sign, kover2floorm4, k2, processDataNum);
    AscendC::Adds(sign_1, sign, 1.0f, processDataNum);
    AscendC::Muls(res_up, x_pow, SIN_RES_MULIT_SCA, processDataNum);
    AscendC::Adds(res_up_1, res_up, SIN_RES_ADDICT_UP, processDataNum);
    AscendC::Mul(res_up_2, res_up_1, x_pow, processDataNum);
    AscendC::Adds(res_up_3, res_up_2, SIN_2ADDS, processDataNum);
    AscendC::Mul(res_up_4, res_up_3, x_pow, processDataNum);
    AscendC::Adds(res_up_5, res_up_4, SIN_3ADDS, processDataNum);
    AscendC::Mul(res_up_6, res_up_5, x_pow, processDataNum);
    AscendC::Adds(res_up_7, res_up_6, 1.0f, processDataNum);
    AscendC::Mul(res_up_8, res_up_7, x_fixed_7, processDataNum);
    AscendC::Mul(res_sign, res_up_8, sign_1, processDataNum);
    AscendC::Mins(res_mins, res_sign, 1.0f, processDataNum);
    AscendC::Maxs(res_maxs, res_mins, -1.0f, processDataNum);
}

class HighPrecStrategy
{
public:
    __aicore__ inline HighPrecStrategy() {}

    __aicore__ inline void InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum)
    {
        pipe->InitBuffer(tmpBuf1, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf2, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf3, tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf4, tileDataNum * sizeof(float));
    }

    __aicore__ inline void ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       uint32_t processDataNum);

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
};

constexpr float PI_V4_0 = 1.5708008f;
constexpr float PI_V4_1 = -0.0000044535846f;
constexpr float PI_V4_2 = -8.706138e-10f;
constexpr float PI_V4_3 = 1.5703125f;
constexpr float PI_12 = 0.0004837513f;
constexpr float PI_22 = 0.000000075495336f;
constexpr float PI_32 = 2.5579538e-12f;
constexpr float PI_42 = 5.389786e-15f;
constexpr float PI_52 = 5.166901e-19f;
constexpr float PI_62 = 3.281839e-22f;

constexpr float INV_HALF_PI = 0.63661975f;

constexpr float SCOEF_4 = 0.0000027183114939898219064f;
constexpr float SCOEF_3 = -0.000198393348360966317347f;
constexpr float SCOEF_2 = 0.0083333293858894631756f;
constexpr float SCOEF_1 = -0.166666666416265235595f;

constexpr float CCOEF_4 = 0.0000243904487962774090654f;
constexpr float CCOEF_3 = -0.00138867637746099294692f;
constexpr float CCOEF_2 = 0.0416666233237390631894f;
constexpr float CCOEF_1 = -0.499999997251031003120f;

__aicore__ inline void HighPrecStrategy::ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                                     AscendC::LocalTensor<float>& yLocal,
                                                     uint32_t processDataNum)
{
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::LocalTensor<float> tmpTensor3 = tmpBuf3.Get<float>();
    AscendC::LocalTensor<float> tmpTensor4 = tmpBuf4.Get<float>();

    const AscendC::LocalTensor<float>& input_x = xLocal;
    const AscendC::LocalTensor<float>& x_scaled = yLocal;
    const AscendC::LocalTensor<float>& x_overpi = tmpTensor3;
    const AscendC::LocalTensor<float>& n = tmpTensor2;
    const AscendC::LocalTensor<float>& n0 = tmpTensor1;
    const AscendC::LocalTensor<float>& n0_1 = tmpTensor3;
    const AscendC::LocalTensor<float>& n0_2 = tmpTensor1;
    const AscendC::LocalTensor<float>& n1 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_1 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_1 = yLocal;
    const AscendC::LocalTensor<float>& fix_2 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_2 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_3 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_3 = yLocal;
    const AscendC::LocalTensor<float>& fix_4 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_4 = tmpTensor2;
    const AscendC::LocalTensor<float>& remain_x = yLocal;
    const AscendC::LocalTensor<float>& temp = tmpTensor2;
    const AscendC::LocalTensor<float>& n2 = yLocal;
    const AscendC::LocalTensor<float>& n0_3 = tmpTensor2;
    const AscendC::LocalTensor<float>& n1_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& fix_5 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_5 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_6 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_6 = xLocal;
    const AscendC::LocalTensor<float>& fix_7 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_7 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_8 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_8 = xLocal;
    const AscendC::LocalTensor<float>& fix_9 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_9 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_10 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_10 = xLocal;
    const AscendC::LocalTensor<float>& fix_11 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_11 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_12 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_12 = xLocal;
    const AscendC::LocalTensor<float>& fix_13 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_13 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_14 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_14 = xLocal;
    const AscendC::LocalTensor<float>& fix_15 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_15 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_16 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_16 = xLocal;
    const AscendC::LocalTensor<float>& fix_17 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_17 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_18 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_18 = xLocal;
    const AscendC::LocalTensor<float>& fix_19 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_19 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_20 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_20 = xLocal;
    const AscendC::LocalTensor<float>& fix_21 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_21 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_22 = xLocal;
    const AscendC::LocalTensor<float>& x_fix_22 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_23 = tmpTensor3;
    const AscendC::LocalTensor<float>& x_fix_23 = xLocal;
    const AscendC::LocalTensor<float>& fix_24 = tmpTensor2;
    const AscendC::LocalTensor<float>& x_fix_24 = tmpTensor1;
    const AscendC::LocalTensor<float>& fix_25 = tmpTensor2;
    const AscendC::LocalTensor<float>& x_fix_25 = xLocal;
    const AscendC::LocalTensor<float>& x_pow = tmpTensor2;
    const AscendC::LocalTensor<float>& sin_poly = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& sin_poly_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_3 = tmpTensor1;
    const AscendC::LocalTensor<float>& sin_poly_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_5 = tmpTensor1;
    const AscendC::LocalTensor<float>& sin_poly_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_7 = tmpTensor4;
    const AscendC::LocalTensor<float>& sin_poly_8 = tmpTensor1;
    const AscendC::LocalTensor<float>& cos_poly = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_1 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_3 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_5 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_7 = tmpTensor2;
    const AscendC::LocalTensor<float>& half_n2 = tmpTensor4;
    const AscendC::LocalTensor<float>& half4_n2 = tmpTensor3;
    const AscendC::LocalTensor<float>& n_half2 = xLocal;
    const AscendC::LocalTensor<float>& n_half4 = tmpTensor4;
    const AscendC::LocalTensor<float>& k1 = tmpTensor3;
    const AscendC::LocalTensor<float>& k2 = xLocal;
    const AscendC::LocalTensor<float>& sign = tmpTensor4;
    const AscendC::LocalTensor<float>& sign_1 = xLocal;
    const AscendC::LocalTensor<float>& ifcos = tmpTensor4;
    const AscendC::LocalTensor<float>& ifsin = yLocal;
    const AscendC::LocalTensor<float>& ifsin_1 = tmpTensor3;
    const AscendC::LocalTensor<float>& temp1 = yLocal;
    const AscendC::LocalTensor<float>& cos_poly_8 = tmpTensor1;
    const AscendC::LocalTensor<float>& res = tmpTensor2;
    const AscendC::LocalTensor<float>& res_1 = yLocal;

    AscendC::Muls(x_scaled, input_x, 1.0f / 2048.0f, processDataNum);
    AscendC::Muls(x_overpi, x_scaled, INV_HALF_PI, processDataNum);
    AscendC::Cast(n, x_overpi, AscendC::RoundMode::CAST_RINT, processDataNum);
    AscendC::Muls(n0, x_overpi, 1.0f / 2048.0f, processDataNum);
    AscendC::Cast(n0_1, n0, AscendC::RoundMode::CAST_RINT, processDataNum);
    AscendC::Muls(n0_2, n0_1, 2048.0f, processDataNum);
    AscendC::Sub(n1, n, n0_2, processDataNum);
    AscendC::Muls(fix, n0_2, PI_V4_0, processDataNum);
    AscendC::Sub(x_fix, x_scaled, fix, processDataNum);
    AscendC::Muls(fix_1, n1, PI_V4_0, processDataNum);
    AscendC::Sub(x_fix_1, x_fix, fix_1, processDataNum);
    AscendC::Muls(fix_2, n0_2, PI_V4_1, processDataNum);
    AscendC::Sub(x_fix_2, x_fix_1, fix_2, processDataNum);
    AscendC::Muls(fix_3, n1, PI_V4_1, processDataNum);
    AscendC::Sub(x_fix_3, x_fix_2, fix_3, processDataNum);
    AscendC::Muls(fix_4, n0_2, PI_V4_2, processDataNum);
    AscendC::Sub(x_fix_4, x_fix_3, fix_4, processDataNum);
    AscendC::Muls(remain_x, x_fix_4, 2048.0f, processDataNum);
    AscendC::Muls(temp, remain_x, INV_HALF_PI, processDataNum);
    AscendC::Cast(n2, temp, AscendC::RoundMode::CAST_RINT, processDataNum);
    AscendC::Muls(n0_3, n0_2, 2048.0f, processDataNum);
    AscendC::Muls(n1_1, n1, 2048.0f, processDataNum);
    AscendC::Muls(fix_5, n0_3, PI_V4_3, processDataNum);
    AscendC::Sub(x_fix_5, input_x, fix_5, processDataNum);
    AscendC::Muls(fix_6, n1_1, PI_V4_3, processDataNum);
    AscendC::Sub(x_fix_6, x_fix_5, fix_6, processDataNum);
    AscendC::Muls(fix_7, n0_3, PI_12, processDataNum);
    AscendC::Sub(x_fix_7, x_fix_6, fix_7, processDataNum);
    AscendC::Muls(fix_8, n2, PI_V4_3, processDataNum);
    AscendC::Sub(x_fix_8, x_fix_7, fix_8, processDataNum);
    AscendC::Muls(fix_9, n1_1, PI_12, processDataNum);
    AscendC::Sub(x_fix_9, x_fix_8, fix_9, processDataNum);
    AscendC::Muls(fix_10, n0_3, PI_22, processDataNum);
    AscendC::Sub(x_fix_10, x_fix_9, fix_10, processDataNum);
    AscendC::Muls(fix_11, n2, PI_12, processDataNum);
    AscendC::Sub(x_fix_11, x_fix_10, fix_11, processDataNum);
    AscendC::Muls(fix_12, n1_1, PI_22, processDataNum);
    AscendC::Sub(x_fix_12, x_fix_11, fix_12, processDataNum);
    AscendC::Muls(fix_13, n0_3, PI_32, processDataNum);
    AscendC::Sub(x_fix_13, x_fix_12, fix_13, processDataNum);
    AscendC::Muls(fix_14, n2, PI_22, processDataNum);
    AscendC::Sub(x_fix_14, x_fix_13, fix_14, processDataNum);
    AscendC::Muls(fix_15, n1_1, PI_32, processDataNum);
    AscendC::Sub(x_fix_15, x_fix_14, fix_15, processDataNum);
    AscendC::Muls(fix_16, n0_3, PI_42, processDataNum);
    AscendC::Sub(x_fix_16, x_fix_15, fix_16, processDataNum);
    AscendC::Muls(fix_17, n2, PI_32, processDataNum);
    AscendC::Sub(x_fix_17, x_fix_16, fix_17, processDataNum);
    AscendC::Muls(fix_18, n1_1, PI_42, processDataNum);
    AscendC::Sub(x_fix_18, x_fix_17, fix_18, processDataNum);
    AscendC::Muls(fix_19, n0_3, PI_52, processDataNum);
    AscendC::Sub(x_fix_19, x_fix_18, fix_19, processDataNum);
    AscendC::Muls(fix_20, n2, PI_42, processDataNum);
    AscendC::Sub(x_fix_20, x_fix_19, fix_20, processDataNum);
    AscendC::Muls(fix_21, n1_1, PI_52, processDataNum);
    AscendC::Sub(x_fix_21, x_fix_20, fix_21, processDataNum);
    AscendC::Muls(fix_22, n0_3, PI_62, processDataNum);
    AscendC::Sub(x_fix_22, x_fix_21, fix_22, processDataNum);
    AscendC::Muls(fix_23, n2, PI_52, processDataNum);
    AscendC::Sub(x_fix_23, x_fix_22, fix_23, processDataNum);
    AscendC::Muls(fix_24, n1_1, PI_62, processDataNum);
    AscendC::Sub(x_fix_24, x_fix_23, fix_24, processDataNum);
    AscendC::Muls(fix_25, n2, PI_62, processDataNum);
    AscendC::Sub(x_fix_25, x_fix_24, fix_25, processDataNum);
    AscendC::Mul(x_pow, x_fix_25, x_fix_25, processDataNum);
    AscendC::Muls(sin_poly, x_pow, SCOEF_4, processDataNum);
    AscendC::Adds(sin_poly_1, sin_poly, SCOEF_3, processDataNum);
    AscendC::Mul(sin_poly_2, x_pow, sin_poly_1, processDataNum);
    AscendC::Adds(sin_poly_3, sin_poly_2, SCOEF_2, processDataNum);
    AscendC::Mul(sin_poly_4, x_pow, sin_poly_3, processDataNum);
    AscendC::Adds(sin_poly_5, sin_poly_4, SCOEF_1, processDataNum);
    AscendC::Mul(sin_poly_6, x_pow, sin_poly_5, processDataNum);
    AscendC::Adds(sin_poly_7, sin_poly_6, 1.0f, processDataNum);
    AscendC::Mul(sin_poly_8, x_fix_25, sin_poly_7, processDataNum);
    AscendC::Muls(cos_poly, x_pow, CCOEF_4, processDataNum);
    AscendC::Adds(cos_poly_1, cos_poly, CCOEF_3, processDataNum);
    AscendC::Mul(cos_poly_2, x_pow, cos_poly_1, processDataNum);
    AscendC::Adds(cos_poly_3, cos_poly_2, CCOEF_2, processDataNum);
    AscendC::Mul(cos_poly_4, x_pow, cos_poly_3, processDataNum);
    AscendC::Adds(cos_poly_5, cos_poly_4, CCOEF_1, processDataNum);
    AscendC::Mul(cos_poly_6, x_pow, cos_poly_5, processDataNum);
    AscendC::Adds(cos_poly_7, cos_poly_6, 1.0f, processDataNum);
    AscendC::Muls(half_n2, n2, 0.5f, processDataNum);
    AscendC::Muls(half4_n2, n2, 0.25f, processDataNum);
    AscendC::Cast(n_half2, half_n2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    AscendC::Cast(n_half4, half4_n2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    AscendC::Muls(k1, n_half2, -2.0f, processDataNum);
    AscendC::Muls(k2, n_half4, 4.0f, processDataNum);
    AscendC::Add(sign, k1, k2, processDataNum);
    AscendC::Adds(sign_1, sign, 1.0f, processDataNum);
    AscendC::Add(ifcos, n2, k1, processDataNum);
    AscendC::Muls(ifsin, ifcos, -1.0f, processDataNum);
    AscendC::Adds(ifsin_1, ifsin, 1.0f, processDataNum);
    AscendC::Mul(temp1, sin_poly_8, ifsin_1, processDataNum);
    AscendC::Mul(cos_poly_8, cos_poly_7, ifcos, processDataNum);
    AscendC::Add(res, temp1, cos_poly_8, processDataNum);
    AscendC::Mul(res_1, res, sign_1, processDataNum);
}

extern "C" __global__ __aicore__ void sin(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    using ComputeStrategy = HighPerfStrategy;
#else
    using ComputeStrategy = HighPrecStrategy;
#endif

    KernelSin<DTYPE_X, ComputeStrategy> op;
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(1)) {
        op.Init<true>(x,
                      y,
                      tiling_data.smallCoreDataNum,
                      tiling_data.bigCoreDataNum,
                      tiling_data.bigCoreLoopNum,
                      tiling_data.smallCoreLoopNum,
                      tiling_data.ubPartDataNum,
                      tiling_data.smallCoreTailDataNum,
                      tiling_data.bigCoreTailDataNum,
                      tiling_data.tailBlockNum,
                      &pipe);
    } else if (TILING_KEY_IS(0)) {
        op.Init<false>(x,
                       y,
                       tiling_data.smallCoreDataNum,
                       tiling_data.bigCoreDataNum,
                       tiling_data.bigCoreLoopNum,
                       tiling_data.smallCoreLoopNum,
                       tiling_data.ubPartDataNum,
                       tiling_data.smallCoreTailDataNum,
                       tiling_data.bigCoreTailDataNum,
                       tiling_data.tailBlockNum,
                       &pipe);
    }
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void sin_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling)
{
    sin<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif