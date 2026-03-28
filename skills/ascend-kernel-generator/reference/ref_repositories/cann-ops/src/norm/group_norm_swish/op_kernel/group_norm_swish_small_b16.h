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
 * \file group_norm_swish_small_b16.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_SMALL_B16_H
#define GROUP_NORM_SWISH_SMALL_B16_H

#include "group_norm_swish_base.h"

namespace GroupNormSwish {
using namespace AscendC;

template <typename T1, typename T2> class GroupNormSwishSmallB16 : public GroupNormSwishBase<T1, T2> {
public:
    __aicore__ inline GroupNormSwishSmallB16(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                const GroupNormSwishTilingData *tilingData, TPipe *pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore(const int64_t groupNum);
    __aicore__ inline void Compute(const int64_t groupNum);
    __aicore__ inline void ComputeOneLoop(const int64_t groupNum);
    __aicore__ inline void ComputeMeanAndRstd(const int64_t groupId);
    __aicore__ inline void ComputeOneLoopInner(const int64_t groupId, const LocalTensor<float> &gammaLocal,
                                               const LocalTensor<float> &betaLocal);
    __aicore__ inline void ComputeMulLoop(const int64_t groupNum);
    __aicore__ inline void ComputeMulLoopInner(const int64_t groupId, const LocalTensor<float> &gammaLocal,
                                               const LocalTensor<float> &betaLocal);
    __aicore__ inline void CalcGroupNormSwish(const float scale, const float bias);
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                            GM_ADDR mean, GM_ADDR rstd,
                                                            const GroupNormSwishTilingData *tilingData, TPipe *pipeIn)
{
    GroupNormSwishBase<T1, T2>::InitGlobal(x, gamma, beta, y, mean, rstd, tilingData, pipeIn);
    GroupNormSwishBase<T1, T2>::InitLocal(this->tiling->shapeCAlign, this->tiling->groupPerCoreAlign, oneBlockNum);
}

template <typename T1, typename T2> __aicore__ inline void GroupNormSwishSmallB16<T1, T2>::Process()
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
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ProcessPerCore(const int64_t groupNum)
{
    this->CopyInGammaAndBeta(0, this->tiling->shapeCAlign, this->tiling->shapeC);
    Compute(groupNum);
    this->CastMeanAndRstd(groupNum);
    this->CopyOutMeanAndRstd(groupNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::Compute(const int64_t groupNum)
{
    if (this->tiling->hwNum % 8 == 0) {
        ComputeOneLoop(groupNum);
    } else {
        ComputeMulLoop(groupNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ComputeOneLoop(const int64_t groupNum)
{
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    if constexpr (!std::is_same_v<T2, float>) {
        this->CastGammaAndBeta(gammaLocal, betaLocal, this->tiling->shapeCAlign);
    }
    for (int64_t groupId = 0; groupId < groupNum; groupId++) {
        this->CopyInX(groupId * this->tiling->numPerGroup, this->tiling->numPerGroup);
        ComputeMeanAndRstd(groupId);
        ComputeOneLoopInner(groupId, gammaLocal, betaLocal);
        this->CopyOutY(groupId * this->tiling->numPerGroup, this->tiling->numPerGroup);
    }
    this->inQueueGamma.template FreeTensor(gammaLocal);
    this->inQueueBeta.template FreeTensor(betaLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ComputeMeanAndRstd(const int64_t groupId)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
#ifndef __CCE_KT_TEST__
    Cast(this->x1Ub32, xUb, RoundMode::CAST_NONE, this->tiling->numPerGroup);
#endif
    pipe_barrier(PIPE_V);
    this->inQueueX.template FreeTensor(xUb);

    this->ReduceSumCustom(this->meanUb, this->x1Ub32, this->x2Ub32, this->tiling->numPerGroup);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    float mean = this->meanUb(0) + this->meanUb(1);
    mean = mean / this->tiling->numPerGroup;
    Adds(this->x2Ub32, this->x1Ub32, -mean, this->tiling->numPerGroup);
    pipe_barrier(PIPE_V);
    Mul(this->x2Ub32, this->x2Ub32, this->x2Ub32, this->tiling->numPerGroup);
    pipe_barrier(PIPE_V);
    this->ReduceSumCustom(this->rstdUb, this->x2Ub32, this->x2Ub32, this->tiling->numPerGroup);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ComputeOneLoopInner(const int64_t groupId,
                                                                           const LocalTensor<float> &gammaLocal,
                                                                           const LocalTensor<float> &betaLocal)
{
    int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
    int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
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
    for (int64_t i = 0; i < this->tiling->shapeD; i++) {
        int64_t xOffset = i * this->tiling->hwNum;
        float gamma = gammaLocal.GetValue(channelIdGlobal + i);
        float beta = betaLocal.GetValue(channelIdGlobal + i);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        Muls(this->x1Ub32[xOffset], this->x1Ub32[xOffset], scale, this->tiling->hwNum);
        pipe_barrier(PIPE_V);
        Adds(this->x1Ub32[xOffset], this->x1Ub32[xOffset], bias, this->tiling->hwNum);
    }
    pipe_barrier(PIPE_V);
    this->ComputeSwishB16(this->tiling->numPerGroup);
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
#ifndef __CCE_KT_TEST__
    Cast(yUb, this->x1Ub32, this->GetRoundMode(), this->tiling->numPerGroup);
#endif
    pipe_barrier(PIPE_V);
    this->outQueueY.template EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ComputeMulLoop(const int64_t groupNum)
{
    LocalTensor<float> gammaLocal = this->inQueueGamma.template DeQue<float>();
    LocalTensor<float> betaLocal = this->inQueueBeta.template DeQue<float>();
    if constexpr (!std::is_same_v<T2, float>) {
        this->CastGammaAndBeta(gammaLocal, betaLocal, this->tiling->shapeCAlign);
    }
    for (int64_t groupId = 0; groupId < groupNum; groupId++) {
        this->CopyInX(groupId * this->tiling->numPerGroup, this->tiling->numPerGroup);
        ComputeMeanAndRstd(groupId);
        ComputeMulLoopInner(groupId, gammaLocal, betaLocal);
    }
    this->inQueueGamma.template FreeTensor(gammaLocal);
    this->inQueueBeta.template FreeTensor(betaLocal);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::ComputeMulLoopInner(const int64_t groupId,
                                                                           const LocalTensor<float> &gammaLocal,
                                                                           const LocalTensor<float> &betaLocal)
{
    int64_t groupIdGlobal = (this->blockIdx * this->tiling->groupPerCore + groupId) % this->tiling->numGroups;
    int64_t channelIdGlobal = groupIdGlobal * this->tiling->shapeD;
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
    for (int64_t i = 0; i < this->tiling->shapeD; i++) {
        int64_t xOffset = groupId * this->tiling->numPerGroup + i * this->tiling->hwNum;
        this->CopyInX(xOffset, this->tiling->hwNum);
        float gamma = gammaLocal.GetValue(channelIdGlobal + i);
        float beta = betaLocal.GetValue(channelIdGlobal + i);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        CalcGroupNormSwish(scale, bias);
        this->CopyOutY(xOffset, this->tiling->hwNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSwishSmallB16<T1, T2>::CalcGroupNormSwish(const float scale, const float bias)
{
    LocalTensor<T1> xUb = this->inQueueX.template DeQue<T1>();
#ifndef __CCE_KT_TEST__
    Cast(this->x1Ub32, xUb, RoundMode::CAST_NONE, this->tiling->hwNum);
#endif
    pipe_barrier(PIPE_V);
    this->inQueueX.template FreeTensor(xUb);
    Muls(this->x1Ub32, this->x1Ub32, scale, this->tiling->hwNum);
    pipe_barrier(PIPE_V);
    Adds(this->x1Ub32, this->x1Ub32, bias, this->tiling->hwNum);
    pipe_barrier(PIPE_V);
    this->ComputeSwishB16(this->tiling->hwNum);
    LocalTensor<T1> yUb = this->outQueueY.template AllocTensor<T1>();
#ifndef __CCE_KT_TEST__
    Cast(yUb, this->x1Ub32, this->GetRoundMode(), this->tiling->hwNum);
#endif
    pipe_barrier(PIPE_V);
    this->outQueueY.template EnQue(yUb);
}
} // namespace GroupNormSwish
#endif // GROUP_NORM_SWISH_SMALL_B16_H