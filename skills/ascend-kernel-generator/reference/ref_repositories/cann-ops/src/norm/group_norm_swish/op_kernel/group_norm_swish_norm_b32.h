/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_swish_norm_b32.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_NORM_B32_H
#define GROUP_NORM_SWISH_NORM_B32_H

#include "group_norm_swish_base.h"

namespace GroupNormSwish {
using namespace AscendC;

template <typename T1, typename T2> class GroupNormSwishNormB32 : public GroupNormSwishBase<T1, T2> {
public:
    __aicore__ inline GroupNormSwishNormB32(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                const GroupNormSwishTilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore(const int64_t groupNum);
    __aicore__ inline void Compute(const int64_t groupNum);
    __aicore__ inline void ComputeSmallHW(const int64_t groupNum);
    __aicore__ inline void ComputeNormHW(const int64_t groupNum);
    __aicore__ inline void ComputeMeanAndRstd(const int64_t groupId);
    __aicore__ inline void ComputeSumTwoPass(const int64_t num, const int64_t index);
    __aicore__ inline void ComputeSmallHWInner(const LocalTensor<float> &gammaLocal,
                                               const LocalTensor<float> &betaLocal, const float mean, const float rstd,
                                               const int64_t dBegin, const int64_t dNum);
    __aicore__ inline void ComputeNormHWInner(const float scale, const float bias, const int64_t calcNum);

private:
    int32_t loopDNum = 0;
    int32_t loopDTimes = 0;
    int32_t loopDTail = 0;
};


template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                           GM_ADDR mean, GM_ADDR rstd,
                                                           const GroupNormSwishTilingData *tilingData, TPipe *pipeIn)
{
    GroupNormSwishBase<T1, T2>::InitGlobal(x, gamma, beta, y, mean, rstd, tilingData, pipeIn);
    GroupNormSwishBase<T1, T2>::InitLocalB32(this->tiling->shapeCAlign, this->tiling->groupPerCoreAlign,
                                             this->tiling->loopTimesAlign * 2);
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishNormB32<T1, T2>::Process()
{
    if (this->blockIdx >= this->usedBlock) {
        return;
    } else if (this->blockIdx == this->usedBlock - 1) {
        ProcessPerCore(this->tiling->groupLastCore);
    } else {
        ProcessPerCore(this->tiling->groupPerCore);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ProcessPerCore(const int64_t groupNum)
{
    this->CopyInGammaAndBeta(0, this->tiling->shapeCAlign, this->tiling->shapeC);
    Compute(groupNum);
    this->CastMeanAndRstd(groupNum);
    this->CopyOutMeanAndRstd(groupNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::Compute(const int64_t groupNum)
{
    if (this->tiling->hwNum <= this->tiling->numPerLoop && this->tiling->hwNum % 8 == 0) {
        loopDNum = this->tiling->numPerLoop / this->tiling->hwNum;
        loopDTimes = this->CeilDiv(this->tiling->shapeD, loopDNum);
        loopDTail = this->CeilRem(this->tiling->shapeD, loopDNum);
        ComputeSmallHW(groupNum);
    } else {
        loopDTimes = this->CeilDiv(this->tiling->hwNum, this->tiling->numPerLoop);
        loopDTail = this->CeilRem(this->tiling->hwNum, this->tiling->numPerLoop);
        ComputeNormHW(groupNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeSmallHW(const int64_t groupNum)
{
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    for (int64_t groupId = 0; groupId < groupNum; groupId++) {
        int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
        int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
        ComputeMeanAndRstd(groupId);
        float mean = this->sum[0];
        float rstd = this->sum[1];
        mean = mean / this->tiling->numPerGroup;
        rstd = float(1.0) / sqrt(rstd + this->tiling->epsilon);
        this->meanOut.SetValue(groupId, mean);
        this->rstdOut.SetValue(groupId, rstd);
        int64_t xOffset = groupId * this->tiling->numPerGroup;
        for (int64_t i = 0; i < loopDTimes - 1; i++) {
            this->CopyInX(xOffset, loopDNum * this->tiling->hwNum);
            ComputeSmallHWInner(gammaLocal, betaLocal, mean, rstd, channelIdGlobal + i * loopDNum, loopDNum);
            this->CopyOutY(xOffset, loopDNum * this->tiling->hwNum);
            xOffset += loopDNum * this->tiling->hwNum;
        }
        this->CopyInX(xOffset, loopDTail * this->tiling->hwNum);
        ComputeSmallHWInner(gammaLocal, betaLocal, mean, rstd, channelIdGlobal + this->tiling->shapeD - loopDTail,
                            loopDTail);
        this->CopyOutY(xOffset, loopDTail * this->tiling->hwNum);
    }
    this->inQueueGamma.template FreeTensor(gammaLocal);
    this->inQueueBeta.template FreeTensor(betaLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeNormHW(const int64_t groupNum)
{
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    for (int64_t groupId = 0; groupId < groupNum; groupId++) {
        ComputeMeanAndRstd(groupId);
        float mean = this->sum[0];
        float rstd = this->sum[1];
        mean = mean / this->tiling->numPerGroup;
        rstd = float(1.0) / sqrt(rstd + this->tiling->epsilon);
        this->meanOut.SetValue(groupId, mean);
        this->rstdOut.SetValue(groupId, rstd);
        int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
        int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
        int64_t xOffset = groupId * this->tiling->numPerGroup;
        for (int64_t i = 0; i < this->tiling->shapeD; i++) {
            float gamma = gammaLocal.GetValue(channelIdGlobal + i);
            float beta = betaLocal.GetValue(channelIdGlobal + i);
            float scale = rstd * gamma;
            float bias = -scale * mean + beta;
            for (int64_t j = 0; j < loopDTimes - 1; j++) {
                this->CopyInX(xOffset, this->tiling->numPerLoop);
                ComputeNormHWInner(scale, bias, this->tiling->numPerLoop);
                this->CopyOutY(xOffset, this->tiling->numPerLoop);
                xOffset += this->tiling->numPerLoop;
            }
            this->CopyInX(xOffset, loopDTail);
            ComputeNormHWInner(scale, bias, loopDTail);
            this->CopyOutY(xOffset, loopDTail);
            xOffset += loopDTail;
        }
    }
    this->inQueueGamma.template FreeTensor(gammaLocal);
    this->inQueueBeta.template FreeTensor(betaLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeMeanAndRstd(const int64_t groupId)
{
    int64_t offset = groupId * this->tiling->numPerGroup;
    for (int64_t i = 0; i < this->tiling->loopTimes - 1; i++) {
        this->CopyInX(offset, this->tiling->numPerLoop);
        offset += this->tiling->numPerLoop;
        ComputeSumTwoPass(this->tiling->numPerLoop, i);
    }
    this->CopyInX(offset, this->tiling->numTailLoop);
    ComputeSumTwoPass(this->tiling->numTailLoop, this->tiling->loopTimes - 1);

    this->c[0] = 0;
    this->c[1] = 0;
    this->sum[0] = 0;
    this->sum[1] = 0;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    this->TwoPassSumOneLoop(this->tiling->loopTimes);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeSumTwoPass(const int64_t num, const int64_t index)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
    this->ReduceSumCustom(this->meanUb[index * 2], xUb, this->x2Ub32, num);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    float mean = this->meanUb(index * 2) + this->meanUb(index * 2 + 1);
    mean = mean / num;
    Adds(this->x2Ub32, xUb, -mean, num);
    pipe_barrier(PIPE_V);
    Mul(this->x2Ub32, this->x2Ub32, this->x2Ub32, num);
    pipe_barrier(PIPE_V);
    this->ReduceSumCustom(this->rstdUb[index * 2], this->x2Ub32, this->x2Ub32, num);
    this->inQueueX.template FreeTensor(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeSmallHWInner(const LocalTensor<float> &gammaLocal,
                                                                          const LocalTensor<float> &betaLocal,
                                                                          const float mean, const float rstd,
                                                                          const int64_t dBegin, const int64_t dNum)
{
    int64_t calcNum = dNum * this->tiling->hwNum;
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
    auto dstUb = yUb;
    if (this->tiling->activateSwish) {
        dstUb = xUb;
    }
    for (int64_t i = 0; i < dNum; i++) {
        float gamma = gammaLocal.GetValue(dBegin + i);
        float beta = betaLocal.GetValue(dBegin + i);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        Muls(xUb[i * this->tiling->hwNum], xUb[i * this->tiling->hwNum], scale, this->tiling->hwNum);
        pipe_barrier(PIPE_V);
        Adds(dstUb[i * this->tiling->hwNum], xUb[i * this->tiling->hwNum], bias, this->tiling->hwNum);
    }
    pipe_barrier(PIPE_V);
    if (this->tiling->activateSwish) {
        Muls(yUb, xUb, -this->tiling->swishScale, calcNum);
        pipe_barrier(PIPE_V);
        Exp(yUb, yUb, calcNum);
        pipe_barrier(PIPE_V);
        Adds(yUb, yUb, this->scalarOne, calcNum);
        pipe_barrier(PIPE_V);
        Div(yUb, xUb, yUb, calcNum);
        pipe_barrier(PIPE_V);
    }
    this->outQueueY.template EnQue(yUb);
    this->inQueueX.template FreeTensor(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishNormB32<T1, T2>::ComputeNormHWInner(const float scale, const float bias,
                                                                         const int64_t calcNum)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
    Muls(xUb, xUb, scale, calcNum);
    pipe_barrier(PIPE_V);
    if (this->tiling->activateSwish) {
        Adds(xUb, xUb, bias, calcNum);
        pipe_barrier(PIPE_V);
        Muls(yUb, xUb, -this->tiling->swishScale, calcNum);
        pipe_barrier(PIPE_V);
        Exp(yUb, yUb, calcNum);
        pipe_barrier(PIPE_V);
        Adds(yUb, yUb, this->scalarOne, calcNum);
        pipe_barrier(PIPE_V);
        Div(yUb, xUb, yUb, calcNum);
        pipe_barrier(PIPE_V);
    } else {
        Adds(yUb, xUb, bias, calcNum);
        pipe_barrier(PIPE_V);
    }
    this->outQueueY.template EnQue(yUb);
    this->inQueueX.template FreeTensor(xUb);
}
} // namespace GroupNormSwish
#endif // GROUP_NORM_SWISH_NORM_B32_H