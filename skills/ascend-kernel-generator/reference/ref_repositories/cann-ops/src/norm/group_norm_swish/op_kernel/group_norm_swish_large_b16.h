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
 * \file group_norm_swish_large_b16.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_LARGE_B16_H
#define GROUP_NORM_SWISH_LARGE_B16_H

#include "group_norm_swish_base.h"

namespace GroupNormSwish {
using namespace AscendC;

template <typename T1, typename T2> class GroupNormSwishLargeB16 : public GroupNormSwishBase<T1, T2> {
public:
    __aicore__ inline GroupNormSwishLargeB16(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                const GroupNormSwishTilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore(const int64_t groupNum);
    __aicore__ inline void Compute(const int64_t groupBegin, const int64_t groupEnd);
    __aicore__ inline void CalcMeanAndRstd(const int64_t groupId);
    __aicore__ inline void CalcMeanAndRstdInner(const int64_t groupId, const int64_t loopTimesBegin,
                                                const int64_t loopTimesEnd);
    __aicore__ inline void ComputeSum(const int64_t num, const int64_t index);
    __aicore__ inline void CalcGroupNormSwish(const int64_t groupBegin, const int64_t groupId);
    __aicore__ inline void CalcGroupNormSwishInner(const float scale, const float bias, const int64_t num);

private:
    int64_t loopDTimes = 0;
    int64_t loopDTail = 0;
    int64_t loopGTimes = 0;
    int64_t loopGTail = 0;
    int32_t inQueSize = 256;                                   // inQueSize
    int32_t outQueSize = 256;                                  // outQueSize
    int32_t meanBufSize = (4096 - inQueSize - outQueSize) / 2; // bufSize
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                            GM_ADDR mean, GM_ADDR rstd,
                                                            const GroupNormSwishTilingData *tilingData, TPipe *pipeIn)
{
    GroupNormSwishBase<T1, T2>::InitGlobal(x, gamma, beta, y, mean, rstd, tilingData, pipeIn);
    GroupNormSwishBase<T1, T2>::InitLocal(inQueSize, outQueSize, meanBufSize * 2);
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishLargeB16<T1, T2>::Process()
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
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::ProcessPerCore(const int64_t groupNum)
{
    loopGTimes = this->CeilDiv(groupNum, outQueSize);
    loopGTail = this->CeilRem(groupNum, outQueSize);
    loopDTimes = this->CeilDiv(this->tiling->hwNum, this->tiling->numPerLoop);
    loopDTail = this->CeilRem(this->tiling->hwNum, this->tiling->numPerLoop);
    for (int64_t i = 0; i < loopGTimes - 1; i++) {
        Compute(i * outQueSize, (i + 1) * outQueSize);
        this->CastMeanAndRstd(outQueSize);
        this->CopyOutMeanAndRstdWithOffset(i * outQueSize, outQueSize);
    }
    Compute(groupNum - loopGTail, groupNum);
    this->CastMeanAndRstd(loopGTail);
    this->CopyOutMeanAndRstdWithOffset(groupNum - loopGTail, loopGTail);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::Compute(const int64_t groupBegin, const int64_t groupEnd)
{
    for (int64_t groupId = groupBegin; groupId < groupEnd; groupId++) {
        CalcMeanAndRstd(groupId);
        CalcGroupNormSwish(groupBegin, groupId);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::CalcMeanAndRstd(const int64_t groupId)
{
    this->c[0] = 0;
    this->c[1] = 0;
    this->sum[0] = 0;
    this->sum[1] = 0;
    int64_t loopTimesInterGroup = this->CeilDiv(this->tiling->loopTimes, meanBufSize);
    int64_t loopTailInterGroup = this->CeilRem(this->tiling->loopTimes, meanBufSize);
    for (int64_t i = 0; i < loopTimesInterGroup - 1; i++) {
        CalcMeanAndRstdInner(groupId, i * meanBufSize, (i + 1) * meanBufSize);
    }
    CalcMeanAndRstdInner(groupId, this->tiling->loopTimes - loopTailInterGroup, this->tiling->loopTimes);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::CalcMeanAndRstdInner(const int64_t groupId,
                                                                            const int64_t loopTimesBegin,
                                                                            const int64_t loopTimesEnd)
{
    for (int64_t i = loopTimesBegin; i < loopTimesEnd - 1; i++) {
        this->CopyInX(groupId * this->tiling->numPerGroup + i * this->tiling->numPerLoop, this->tiling->numPerLoop);
        ComputeSum(this->tiling->numPerLoop, i - loopTimesBegin);
    }
    if (loopTimesEnd == this->tiling->loopTimes) {
        this->CopyInX((groupId + 1) * this->tiling->numPerGroup - this->tiling->numTailLoop, this->tiling->numTailLoop);
        ComputeSum(this->tiling->numTailLoop, loopTimesEnd - 1 - loopTimesBegin);
    } else {
        this->CopyInX(groupId * this->tiling->numPerGroup + (loopTimesEnd - 1) * this->tiling->numPerLoop,
                      this->tiling->numPerLoop);
        ComputeSum(this->tiling->numPerLoop, loopTimesEnd - 1 - loopTimesBegin);
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    this->TwoPassSumMulLoop(loopTimesBegin, loopTimesEnd);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::ComputeSum(const int64_t num, const int64_t index)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
#ifndef __CCE_KT_TEST__
    Cast(this->x1Ub32, xUb, RoundMode::CAST_NONE, num);
#endif
    pipe_barrier(PIPE_V);
    this->inQueueX.template FreeTensor(xUb);
    // 第一次计算均值
    this->ReduceSumCustom(this->meanUb[index * 2], this->x1Ub32, this->x2Ub32, num);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float mean = this->meanUb(index * 2) + this->meanUb(index * 2 + 1);
    mean = mean / num;
    Adds(this->x2Ub32, this->x1Ub32, -mean, num);
    pipe_barrier(PIPE_V);
    // 对均值进行修正
    this->ReduceSumCustom(this->meanUb[index * 2], this->x2Ub32, this->x2Ub32, num);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float mean1 = this->meanUb(index * 2) + this->meanUb(index * 2 + 1);
    mean1 = mean1 / num + mean;
    this->meanUb(index * 2) = mean1;
    // two-pass计算方差
    Adds(this->x2Ub32, this->x1Ub32, -mean1, num);
    pipe_barrier(PIPE_V);
    Mul(this->x2Ub32, this->x2Ub32, this->x2Ub32, num);
    pipe_barrier(PIPE_V);
    this->ReduceSumCustom(this->rstdUb[index * 2], this->x2Ub32, this->x2Ub32, num);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::CalcGroupNormSwish(const int64_t groupBegin,
                                                                          const int64_t groupId)
{
    int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
    int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
    int64_t startChannelId = channelIdGlobal;
    int64_t xOffset = groupId * this->tiling->numPerGroup;

    float mean = this->sum[0];
    float rstd = this->sum[1];
    mean = mean / this->tiling->numPerGroup;
    rstd = float(1.0) / sqrt(rstd + this->tiling->epsilon);
    this->meanOut.SetValue(groupId - groupBegin, mean);
    this->rstdOut.SetValue(groupId - groupBegin, rstd);
    this->CopyInGammaAndBeta(channelIdGlobal, inQueSize, inQueSize);
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    if constexpr (!std::is_same_v<T2, float>) {
        this->CastGammaAndBeta(gammaLocal, betaLocal, inQueSize);
        pipe_barrier(PIPE_V);
    }
    Muls(gammaLocal, gammaLocal, rstd, inQueSize);
    pipe_barrier(PIPE_V);
    Muls(this->meanUb, gammaLocal, -mean, inQueSize);
    pipe_barrier(PIPE_V);
    Add(betaLocal, this->meanUb, betaLocal, inQueSize);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (int64_t i = 0; i < this->tiling->shapeD; i++) {
        int64_t currentId = channelIdGlobal + i;
        if (currentId >= startChannelId + inQueSize) {
            this->inQueueGamma.template FreeTensor(gammaLocal);
            this->inQueueBeta.template FreeTensor(betaLocal);
            this->CopyInGammaAndBeta(currentId, inQueSize, inQueSize);
            gammaLocal = this->inQueueGamma.template DeQue<float>();
            betaLocal = this->inQueueBeta.template DeQue<float>();
            if constexpr (!std::is_same_v<T2, float>) {
                this->CastGammaAndBeta(gammaLocal, betaLocal, inQueSize);
                pipe_barrier(PIPE_V);
            }
            startChannelId = currentId;
            Muls(gammaLocal, gammaLocal, rstd, inQueSize);
            pipe_barrier(PIPE_V);
            Muls(this->meanUb, gammaLocal, -mean, inQueSize);
            pipe_barrier(PIPE_V);
            Add(betaLocal, this->meanUb, betaLocal, inQueSize);
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
        }
        for (int64_t j = 0; j < loopDTimes - 1; j++) {
            this->CopyInX(xOffset, this->tiling->numPerLoop);
            CalcGroupNormSwishInner(gammaLocal(currentId - startChannelId), betaLocal(currentId - startChannelId),
                                    this->tiling->numPerLoop);
            this->CopyOutY(xOffset, this->tiling->numPerLoop);
            xOffset += this->tiling->numPerLoop;
        }
        this->CopyInX(xOffset, loopDTail);
        CalcGroupNormSwishInner(gammaLocal(currentId - startChannelId), betaLocal(currentId - startChannelId),
                                loopDTail);
        this->CopyOutY(xOffset, loopDTail);
        xOffset += loopDTail;
    }
    this->inQueueGamma.template FreeTensor(gammaLocal);
    this->inQueueBeta.template FreeTensor(betaLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishLargeB16<T1, T2>::CalcGroupNormSwishInner(const float scale, const float bias,
                                                                               const int64_t num)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
#ifndef __CCE_KT_TEST__
    Cast(this->x1Ub32, xUb, RoundMode::CAST_NONE, num);
#endif
    pipe_barrier(PIPE_V);
    this->inQueueX.template FreeTensor(xUb);
    Muls(this->x1Ub32, this->x1Ub32, scale, num);
    pipe_barrier(PIPE_V);
    Adds(this->x1Ub32, this->x1Ub32, bias, num);
    pipe_barrier(PIPE_V);
    this->ComputeSwishB16(num);
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
#ifndef __CCE_KT_TEST__
    Cast(yUb, this->x1Ub32, this->GetRoundMode(), num);
#endif
    pipe_barrier(PIPE_V);
    this->outQueueY.template EnQue(yUb);
}

} // namespace GroupNormSwish
#endif // GROUP_NORM_SWISH_LARGE_B16_H