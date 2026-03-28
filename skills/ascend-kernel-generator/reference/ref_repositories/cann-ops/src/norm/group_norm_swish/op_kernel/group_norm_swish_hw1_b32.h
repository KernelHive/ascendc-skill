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
 * \file group_norm_swish_hw1_b32.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_HW1_B32_H
#define GROUP_NORM_SWISH_HW1_B32_H

#include "group_norm_swish_base.h"

namespace GroupNormSwish {
using namespace AscendC;

template <typename T1, typename T2> class GroupNormSwishHW1B32 : public GroupNormSwishBase<T1, T2> {
public:
    __aicore__ inline GroupNormSwishHW1B32(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                const GroupNormSwishTilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore(const int64_t groupNum);
    __aicore__ inline void ComputeOneLoop(const int64_t groupNum);
    __aicore__ inline void AccumulateXandX2OneLoop(const int64_t groupId, const LocalTensor<float> &xUb);
    __aicore__ inline void ComputeOneLoopInner(const int64_t groupId, const LocalTensor<float> &gammaLocal,
                                               const LocalTensor<float> &betaLocal, const LocalTensor<float> &xUb);
    __aicore__ inline void ComputeMultipleLoop(const int64_t groupNum);
    __aicore__ inline void ComputeMultipleLoopInner(const int64_t groupBegin, const int64_t groupEnd);
    __aicore__ inline void ComputeMeanAndRstd(const int64_t groupId);
    __aicore__ inline void ComputeMeanAndRstdInner(const int64_t groupId, const int64_t loopTimeBegin,
                                                   const int64_t loopTimeEnd);
    __aicore__ inline void ComputeSumTwoPass(const int64_t num, const int64_t index);
    __aicore__ inline void ComputeGroupNormSwish(const int64_t gmmmaOffset, const float mean, const float rstd,
                                                 const int64_t calcNum);
    __aicore__ inline void ComputeEqual(const int64_t groupNum);
    __aicore__ inline void ComputeEqualMeanSameType();
    __aicore__ inline void ComputeEqualRstdSameType();
    __aicore__ inline void ComputeEqualYSameType(const int64_t groupNum);

private:
    int32_t loopDTimes = 0;
    int32_t loopDTail = 0;
    int32_t loopGTimes = 0;
    int32_t loopGTail = 0;
    int32_t loopETimes = 0;
    int32_t loopETail = 0;
    int32_t loopENum = 8192;
    int32_t outQueSize = 0;
    int32_t meanBufSize = 512;
};


template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                          GM_ADDR mean, GM_ADDR rstd,
                                                          const GroupNormSwishTilingData *tilingData, TPipe *pipeIn)
{
    if (tilingData->shapeC == tilingData->numGroups) {
        this->tiling = tilingData;
        this->blockIdx = GetBlockIdx();
        this->usedBlock = GetBlockNum();
        this->numRec = float(1.0) / float(this->tiling->numPerGroup);
        this->pipe = pipeIn;

        int64_t xGMOffset = this->blockIdx * this->tiling->groupPerCore;
        this->xGm.SetGlobalBuffer((__gm__ T1 *)x + xGMOffset);
        this->gammaGm.SetGlobalBuffer((__gm__ T2 *)gamma);
        this->betaGm.SetGlobalBuffer((__gm__ T2 *)beta);
        this->yGm.SetGlobalBuffer((__gm__ T1 *)y + xGMOffset);
        this->meanGm.SetGlobalBuffer((__gm__ T2 *)mean + xGMOffset);
        this->rstdGm.SetGlobalBuffer((__gm__ T2 *)rstd + xGMOffset);

        this->pipe->InitBuffer(this->inQueueX, 2, loopENum * sizeof(float));
        this->pipe->InitBuffer(this->x2Buf32, loopENum * sizeof(float));
        this->x2Ub32 = this->x2Buf32.template Get<float>();
    } else {
        GroupNormSwishBase<T1, T2>::InitGlobal(x, gamma, beta, y, mean, rstd, tilingData, pipeIn);
        outQueSize = 2048;
        // Init TQue
        this->pipe->InitBuffer(this->inQueueGamma, 1, this->tiling->numPerLoop * sizeof(float));
        this->pipe->InitBuffer(this->inQueueBeta, 1, this->tiling->numPerLoop * sizeof(float));
        this->pipe->InitBuffer(this->inQueueX, 1, this->tiling->numPerLoop * sizeof(T1));
        this->pipe->InitBuffer(this->outQueueY, 1, this->tiling->numPerLoop * sizeof(T1));
        this->pipe->InitBuffer(this->outQueueMean, 1, outQueSize * sizeof(float));
        this->pipe->InitBuffer(this->outQueueRstd, 1, outQueSize * sizeof(float));
        // Init TBuf
        this->pipe->InitBuffer(this->x2Buf32, this->tiling->numPerLoop * sizeof(float));
        this->pipe->InitBuffer(this->meanBuf32, meanBufSize * 2 * sizeof(float));
        this->pipe->InitBuffer(this->rstdBuf32, meanBufSize * 2 * sizeof(float));
        // Init LocalTensor
        this->x2Ub32 = this->x2Buf32.template Get<float>();
        this->meanUb = this->meanBuf32.template Get<float>();
        this->rstdUb = this->rstdBuf32.template Get<float>();
        this->meanOut = this->outQueueMean.template AllocTensor<float>();
        this->rstdOut = this->outQueueRstd.template AllocTensor<float>();
    }
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishHW1B32<T1, T2>::Process()
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
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ProcessPerCore(const int64_t groupNum)
{
    if (this->tiling->shapeC == this->tiling->numGroups) {
        ComputeEqual(groupNum);
    } else if (this->tiling->shapeD <= this->tiling->numPerLoop && groupNum <= outQueSize) {
        ComputeOneLoop(groupNum);
        this->CastMeanAndRstd(groupNum);
        this->CopyOutMeanAndRstd(groupNum);
    } else {
        loopGTimes = this->CeilDiv(groupNum, outQueSize);
        loopGTail = this->CeilRem(groupNum, outQueSize);
        loopDTimes = this->CeilDiv(this->tiling->shapeD, this->tiling->numPerLoop);
        loopDTail = this->CeilRem(this->tiling->shapeD, this->tiling->numPerLoop);
        ComputeMultipleLoop(groupNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeOneLoop(const int64_t groupNum)
{
    for (int64_t groupId = 0; groupId < groupNum; groupId++) {
        int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
        this->CopyInGammaAndBeta(groupIdGlobal * this->tiling->shapeD, this->tiling->numPerLoop, this->tiling->shapeD);
        LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
        LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
        this->CopyInX(groupId * this->tiling->numPerGroup, this->tiling->numPerGroup);
        LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
        AccumulateXandX2OneLoop(groupId, xUb);
        ComputeOneLoopInner(groupId, gammaLocal, betaLocal, xUb);
        this->inQueueX.template FreeTensor(xUb);
        this->inQueueGamma.template FreeTensor(gammaLocal);
        this->inQueueBeta.template FreeTensor(betaLocal);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::AccumulateXandX2OneLoop(const int64_t groupId,
                                                                             const LocalTensor<float> &xUb)
{
    this->ReduceSumCustom(this->meanUb, xUb, this->x2Ub32, this->tiling->numPerGroup);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    float mean = this->meanUb(0) + this->meanUb(1);
    mean = mean / this->tiling->numPerGroup;
    Adds(this->x2Ub32, xUb, -mean, this->tiling->numPerGroup);
    pipe_barrier(PIPE_V);
    Mul(this->x2Ub32, this->x2Ub32, this->x2Ub32, this->tiling->numPerGroup);
    pipe_barrier(PIPE_V);
    this->ReduceSumCustom(this->rstdUb, this->x2Ub32, this->x2Ub32, this->tiling->numPerGroup);
}

template <typename T1, typename T2>
__aicore__ inline void
GroupNormSwishHW1B32<T1, T2>::ComputeOneLoopInner(const int64_t groupId, const LocalTensor<float> &gammaLocal,
                                                  const LocalTensor<float> &betaLocal, const LocalTensor<float> &xUb)
{
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float mean = this->meanUb(0) + this->meanUb(1);
    float rstd = this->rstdUb(0) + this->rstdUb(1);
    mean = mean / this->tiling->numPerGroup;
    rstd = rstd / this->tiling->numPerGroup;
    rstd = float(1.0) / sqrt(rstd + this->tiling->epsilon);
    this->meanOut.SetValue(groupId, mean);
    this->rstdOut.SetValue(groupId, rstd);

    Adds(xUb, xUb, -mean, this->tiling->shapeD);
    pipe_barrier(PIPE_V);
    Muls(xUb, xUb, rstd, this->tiling->shapeD);
    pipe_barrier(PIPE_V);
    Mul(xUb, xUb, gammaLocal, this->tiling->shapeD);
    pipe_barrier(PIPE_V);
    if (this->tiling->activateSwish) {
        Add(xUb, xUb, betaLocal, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
        Muls(yUb, xUb, -this->tiling->swishScale, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
        Exp(yUb, yUb, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
        Adds(yUb, yUb, this->scalarOne, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
        Div(yUb, xUb, yUb, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
    } else {
        Add(yUb, xUb, betaLocal, this->tiling->shapeD);
        pipe_barrier(PIPE_V);
    }
    this->outQueueY.template EnQue(yUb);
    this->CopyOutY(groupId * this->tiling->numPerGroup, this->tiling->numPerGroup);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeMultipleLoop(const int64_t groupNum)
{
    for (int64_t i = 0; i < loopGTimes - 1; i++) {
        ComputeMultipleLoopInner(i * outQueSize, (i + 1) * outQueSize);
        this->CastMeanAndRstd(outQueSize);
        this->CopyOutMeanAndRstdWithOffset(i * outQueSize, outQueSize);
    }
    ComputeMultipleLoopInner(groupNum - loopGTail, groupNum);
    this->CastMeanAndRstd(loopGTail);
    this->CopyOutMeanAndRstdWithOffset(groupNum - loopGTail, loopGTail);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeMultipleLoopInner(const int64_t groupBegin,
                                                                              const int64_t groupEnd)
{
    int64_t xOffset, gammaOffset;
    for (int64_t groupId = groupBegin; groupId < groupEnd; groupId++) {
        int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
        int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
        ComputeMeanAndRstd(groupId);
        float mean = this->sum[0];
        float rstd = this->sum[1];
        mean = mean / this->tiling->numPerGroup;
        rstd = float(1.0) / sqrt(rstd + this->tiling->epsilon);
        this->meanOut.SetValue(groupId - groupBegin, mean);
        this->rstdOut.SetValue(groupId - groupBegin, rstd);
        for (int64_t i = 0; i < loopDTimes - 1; i++) {
            xOffset = groupId * this->tiling->numPerGroup + i * this->tiling->numPerLoop;
            gammaOffset = channelIdGlobal + i * this->tiling->numPerLoop;
            this->CopyInX(xOffset, this->tiling->numPerLoop);
            ComputeGroupNormSwish(gammaOffset, mean, rstd, this->tiling->numPerLoop);
            this->CopyOutY(xOffset, this->tiling->numPerLoop);
        }
        xOffset = groupId * this->tiling->numPerGroup + (loopDTimes - 1) * this->tiling->numPerLoop;
        gammaOffset = channelIdGlobal + (loopDTimes - 1) * this->tiling->numPerLoop;
        this->CopyInX(xOffset, loopDTail);
        ComputeGroupNormSwish(gammaOffset, mean, rstd, loopDTail);
        this->CopyOutY(xOffset, loopDTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeMeanAndRstd(const int64_t groupId)
{
    this->c[0] = 0;
    this->c[1] = 0;
    this->sum[0] = 0;
    this->sum[1] = 0;
    int64_t loopTimesInterGroup = this->CeilDiv(this->tiling->loopTimes, meanBufSize);
    int64_t loopTailInterGroup = this->CeilRem(this->tiling->loopTimes, meanBufSize);
    for (int64_t i = 0; i < loopTimesInterGroup - 1; i++) {
        ComputeMeanAndRstdInner(groupId, i * meanBufSize, (i + 1) * meanBufSize);
    }
    ComputeMeanAndRstdInner(groupId, this->tiling->loopTimes - loopTailInterGroup, this->tiling->loopTimes);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeMeanAndRstdInner(const int64_t groupId,
                                                                             const int64_t loopTimeBegin,
                                                                             const int64_t loopTimeEnd)
{
    for (int64_t i = loopTimeBegin; i < loopTimeEnd - 1; i++) {
        this->CopyInX(groupId * this->tiling->numPerGroup + i * this->tiling->numPerLoop, this->tiling->numPerLoop);
        ComputeSumTwoPass(this->tiling->numPerLoop, i - loopTimeBegin);
    }
    int64_t xOffset = groupId * this->tiling->numPerGroup + (loopTimeEnd - 1) * this->tiling->numPerLoop;
    if (loopTimeEnd == this->tiling->loopTimes) {
        this->CopyInX(xOffset, this->tiling->numTailLoop);
        ComputeSumTwoPass(this->tiling->numTailLoop, loopTimeEnd - 1 - loopTimeBegin);
    } else {
        this->CopyInX(xOffset, this->tiling->numPerLoop);
        ComputeSumTwoPass(this->tiling->numPerLoop, loopTimeEnd - 1 - loopTimeBegin);
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    this->TwoPassSumMulLoop(loopTimeBegin, loopTimeEnd);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeSumTwoPass(const int64_t num, const int64_t index)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
    // compute mean
    this->ReduceSumCustom(this->meanUb[index * 2], xUb, this->x2Ub32, num);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float mean = this->meanUb(index * 2) + this->meanUb(index * 2 + 1);
    mean = mean / num;
    this->meanUb(index * 2) = mean;
    // two-pass
    Adds(this->x2Ub32, xUb, -mean, num);
    pipe_barrier(PIPE_V);
    this->inQueueX.template FreeTensor(xUb);
    Mul(this->x2Ub32, this->x2Ub32, this->x2Ub32, num);
    pipe_barrier(PIPE_V);
    this->ReduceSumCustom(this->rstdUb[index * 2], this->x2Ub32, this->x2Ub32, num);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeGroupNormSwish(const int64_t gmmmaOffset, const float mean,
                                                                           const float rstd, const int64_t calcNum)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();


    this->CopyInGammaAndBeta(gmmmaOffset, this->tiling->numPerLoop, calcNum);
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    if constexpr (!std::is_same_v<T2, float>) {
        this->CastGammaAndBeta(gammaLocal, betaLocal, this->tiling->numPerLoop);
    }
    Adds(xUb, xUb, -mean, calcNum);
    pipe_barrier(PIPE_V);
    Muls(xUb, xUb, rstd, calcNum);
    pipe_barrier(PIPE_V);
    Mul(xUb, xUb, gammaLocal, calcNum);
    pipe_barrier(PIPE_V);
    this->inQueueGamma.template FreeTensor(gammaLocal);
    if (this->tiling->activateSwish) {
        Add(xUb, xUb, betaLocal, calcNum);
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
        Add(yUb, xUb, betaLocal, calcNum);
        pipe_barrier(PIPE_V);
    }
    this->inQueueX.template FreeTensor(xUb);
    this->inQueueBeta.template FreeTensor(betaLocal);
    this->outQueueY.template EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeEqual(const int64_t groupNum)
{
    loopETimes = this->CeilDiv(groupNum, loopENum);
    loopETail = this->CeilRem(groupNum, loopENum);
    ComputeEqualMeanSameType();
    ComputeEqualRstdSameType();
    ComputeEqualYSameType(groupNum);
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeEqualMeanSameType()
{
    int64_t xOffset = 0;
    for (int64_t i = 0; i < loopETimes - 1; i++) {
        LocalTensor<T1> xUb = this->inQueueX.template AllocTensor<T1>();
        DataCopy(xUb, this->xGm[xOffset], loopENum);
        pipe_barrier(PIPE_ALL);
        this->inQueueX.template EnQue(xUb);
        LocalTensor<T2> meanUb = this->inQueueX.template DeQue<T2>();
        DataCopy(this->meanGm[xOffset], meanUb, loopENum);
        pipe_barrier(PIPE_ALL);
        this->inQueueX.template FreeTensor(meanUb);
        xOffset += loopENum;
    }
    LocalTensor<T1> xUb = this->inQueueX.template AllocTensor<T1>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
    DataCopyPad(xUb, this->xGm[xOffset], {1, static_cast<uint16_t>(loopETail * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
#endif
#else
    int64_t copyNumAlign = this->CeilDiv(loopETail, this->elementsPerBlockT1) * this->elementsPerBlockT1;
    DataCopy(xUb, this->xGm[xOffset], copyNumAlign);
#endif
    pipe_barrier(PIPE_ALL);
    this->inQueueX.template EnQue(xUb);
    LocalTensor<T2> meanUb = this->inQueueX.template DeQue<T2>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
    DataCopyPad(this->meanGm[xOffset], meanUb, {1, static_cast<uint16_t>(loopETail * sizeof(T1)), 0, 0});
#endif
#else
    this->CopyOutWithOutPadT1(this->meanGm[xOffset], meanUb, loopETail);
#endif
    this->inQueueX.template FreeTensor(meanUb);
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeEqualRstdSameType()
{
    LocalTensor<float> rstdUb = this->inQueueX.template AllocTensor<float>();
    float rstd = float(1.0) / sqrt(this->tiling->epsilon);
    Duplicate(rstdUb, rstd, loopENum);
    pipe_barrier(PIPE_ALL);
    for (int64_t i = 0; i < loopETimes - 1; i++) {
        DataCopy(this->rstdGm[i * loopENum], rstdUb, loopENum);
    }
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
    DataCopyPad(this->rstdGm[(loopETimes - 1) * loopENum], rstdUb,
                {1, static_cast<uint16_t>(loopETail * sizeof(T2)), 0, 0});
#endif
#else
    int64_t copyNumAlign = this->CeilDiv(loopETail, this->elementsPerBlockT2) * this->elementsPerBlockT2;
    DataCopy(this->rstdGm[(loopETimes - 1) * loopENum], rstdUb, copyNumAlign);
#endif
    this->inQueueX.template FreeTensor(rstdUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishHW1B32<T1, T2>::ComputeEqualYSameType(const int64_t groupNum)
{
    int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore) % this->tiling->numGroups;
    int64_t movingNum = this->GetMin(this->tiling->numGroups - groupIdGlobal, loopENum);
    int64_t movedNum = 0;
    while (movedNum < groupNum) {
        LocalTensor<T2> xUb = this->inQueueX.template AllocTensor<T2>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
        DataCopyPad(xUb, this->betaGm[groupIdGlobal], {1, static_cast<uint16_t>(movingNum * sizeof(T2)), 0, 0},
                    {false, 0, 0, 0});
#endif
#else
        int64_t copyNumAlign = this->CeilDiv(movingNum, this->elementsPerBlockT2) * this->elementsPerBlockT2;
        DataCopy(xUb, this->betaGm[groupIdGlobal], copyNumAlign);
#endif
        pipe_barrier(PIPE_ALL);
        if (this->tiling->activateSwish) {
            Muls(this->x2Ub32, xUb, -this->tiling->swishScale, movingNum);
            pipe_barrier(PIPE_V);
            Exp(this->x2Ub32, this->x2Ub32, movingNum);
            pipe_barrier(PIPE_V);
            Adds(this->x2Ub32, this->x2Ub32, this->scalarOne, movingNum);
            pipe_barrier(PIPE_V);
            Div(xUb, xUb, this->x2Ub32, movingNum);
            pipe_barrier(PIPE_ALL);
        }
        LocalTensor<T1> xUbFp16 = xUb.template ReinterpretCast<T1>();
#if __CCE_AICORE__ == 220
#ifndef __CCE_KT_TEST__
        DataCopyPad(this->yGm[movedNum], xUbFp16, {1, static_cast<uint16_t>(movingNum * sizeof(T2)), 0, 0});
#endif
#else
        this->CopyOutWithOutPadT2(this->yGm[movedNum], xUbFp16, movingNum);
#endif
        movedNum += movingNum;
        groupIdGlobal = (groupIdGlobal + movingNum) % this->tiling->numGroups;
        movingNum = this->GetMin(this->tiling->numGroups - groupIdGlobal, loopENum);
        this->inQueueX.template FreeTensor(xUb);
    }
}

} // namespace GroupNormSwish
#endif // GROUP_NORM_SWISH_HW1_B32_H