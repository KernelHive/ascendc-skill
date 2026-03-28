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
 * \file cross_entropy_loss_fp32.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_FP32_H
#define CROSS_ENTROPY_LOSS_FP32_H

#include "cross_entropy_loss_base.h"
#include "cross_entropy_loss.h"

using namespace AscendC;
namespace CrossEntropyLossCustom {

template <>
__aicore__ inline void CrossEntropyLoss<float>::GetLogProbOut(const LocalTensor<float>& inputBuf, uint64_t& offset, const float& logBatchSum, 
                                                                 float& batchLogProb, const uint64_t& target, float& smoothingLoss, bool isIgnore)
{
    AscendC::LocalTensor<float> workLocal = this->calcBuf.template GetWithOffset<float>(NUM_1024, this->workLocalOffset);
    uint64_t weightOffset = 0;
    for (size_t i = 0; i < this->ubLoopNum; ++i)
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_MTE2));
        set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        this->CopyIn(inputBuf, offset, uint32_t(this->inputUbSize));
        AscendC::Adds(inputBuf, inputBuf, -logBatchSum, this->inputUbSize);
        if (target >= offset && target < offset + this->inputUbSize) {
            batchLogProb = inputBuf(target - offset);
        }
        this->CopyOut(inputBuf, this->probOutBuf, offset, this->inputUbSize);
        if (this->isSmoothing && !isIgnore) {
            GetSmoothingLoss(inputBuf, workLocal, smoothingLoss, this->inputUbSize, weightOffset);
            weightOffset += this->inputUbSize;
        }
        offset += this->inputUbSize;
    }
    if (this->ubTailNum != 0) {
        event_t eventMTE3MTE2 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_MTE2));
        set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        this->CopyIn(inputBuf, offset, uint32_t(this->ubTailNum));
        AscendC::Adds(inputBuf, inputBuf, -logBatchSum, this->ubTailNum);
        if (target >= offset) {
            batchLogProb = inputBuf(target - offset);
        }
        this->CopyOut(inputBuf, this->probOutBuf, offset, this->ubTailNum);
        if (this->isSmoothing && !isIgnore) {
            GetSmoothingLoss(inputBuf, workLocal, smoothingLoss, this->ubTailNum, weightOffset);
            weightOffset += this->inputUbSize;
        }
        offset += this->ubTailNum;
    }
}

template <>
__aicore__ inline void CrossEntropyLoss<float>::GetNoneLoss()
{
    uint32_t len = this->batchNum;
    if (this->isSmoothing) {
        float smoothingScale = this->labelSmoothing / this->targetNum;
        AscendC::Muls(this->smoothingLossLocal, this->smoothingLossLocal, smoothingScale, len);
        AscendC::Muls(this->lnLocal, this->lnLocal, float(this->labelSmoothing - 1), len);
        AscendC::Add(this->lnLocal, this->lnLocal, this->smoothingLossLocal, len);
    } else {
        AscendC::Muls(this->lnLocal, this->lnLocal, float(-1.0), len);
    }
    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    if (len % BLOCK_32 == 0) {
        AscendC::DataCopy(this->lossGm, this->lnLocal, len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPad(this->lossGm, this->lnLocal, copyParams);
    }
}

template <>
__aicore__ inline void CrossEntropyLoss<float>::CalcMeanLoss()
{
    AscendC::LocalTensor<float> syncLocal = this->calcBuf.template Get<float>(NUM_192);
    AscendC::DataCopy(syncLocal, this->workspaceGm, NUM_192);
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    AscendC::WholeReduceSum(syncLocal, syncLocal, this->usedCoreNum, 1, 1, 1, 8);
    AscendC::WholeReduceSum(syncLocal[NUM_64], syncLocal[NUM_64], this->usedCoreNum, 1, 1, 1, 8);
    float loss = 0.0;
    float weightSum = syncLocal(NUM_64);
    float lnSum = syncLocal(0);
    if (this->isSmoothing) {
        float smoothingScale = this->labelSmoothing / this->targetNum;
        AscendC::WholeReduceSum(syncLocal[NUM_128], syncLocal[NUM_128], this->usedCoreNum, 1, 1, 1, 8);
        loss = ((this->labelSmoothing - 1) * lnSum + syncLocal(NUM_128) * smoothingScale) / weightSum;
    } else {
        loss = -lnSum / weightSum;
    }
    this->lossGm.SetValue(0, loss);
}

template <>
__aicore__ inline void CrossEntropyLoss<float>::CalcSumLoss()
{  
    AscendC::LocalTensor<float> syncLocal = this->calcBuf.template Get<float>(NUM_192);
    AscendC::DataCopy(syncLocal, this->workspaceGm, NUM_192);
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    AscendC::WholeReduceSum(syncLocal, syncLocal, this->usedCoreNum, 1, 1, 1, 8);
    float lnSum = syncLocal(0);
    float loss = 0.0;
    if (this->isSmoothing) {
        float smoothingScale = this->labelSmoothing / this->targetNum;
        AscendC::WholeReduceSum(syncLocal[NUM_128], syncLocal[NUM_128], this->usedCoreNum, 1, 1, 1, 8);
        loss = (this->labelSmoothing - 1) * lnSum + syncLocal(NUM_128) * smoothingScale;
    } else {
        loss = -lnSum;
    }
    this->lossGm.SetValue(0, loss);
}

template <>
__aicore__ inline void CrossEntropyLoss<float>::ProcessFp32()
{
    uint64_t offset = 0;
    uint64_t batchTarget = 0;
    uint64_t batchTargetOffset = 0;
    float batchWeight = 0.0;
    float rowMax = 0.0;
    float batchLogProb = 0.0;
    float logBatchSum = 0.0;
    float batchSum = 0.0;
    float smoothingLoss = 0.0;
    bool isIgnore = false;
    this->inputLocal = this->inQueue.template AllocTensor<float>();
    for (size_t batchIdx = 0; batchIdx < this->batchNum; ++batchIdx) {
        this->reduceRes.SetValue(0, MIN_FLT);
        GetRowMax(this->inputLocal, offset);
        rowMax = this->reduceRes(0);

        batchTarget = this->targetGm.GetValue(this->startBatchIndex + batchIdx);
        bool isIgnore = batchTarget == this->ignoreIndex;
        if (this->defaultWeight == NUM_1) {
            batchWeight = 1.0;
        } else {
            batchWeight = this->weightGm.GetValue(batchTarget);
        }
        this->weightLocal.SetValue(batchIdx, batchWeight);
        batchSum = 0;
        GetExpSum(this->inputLocal, offset, batchSum, rowMax);
        this->reduceCalc.SetValue(0, batchSum);
        AscendC::Log(this->reduceRes, this->reduceCalc, 1);
        logBatchSum = this->reduceRes(0) + rowMax;
        batchTargetOffset = batchTarget + batchIdx * this->targetNum;
        smoothingLoss = 0.0;
        GetLogProbOut(this->inputLocal, offset, logBatchSum, batchLogProb, batchTargetOffset, smoothingLoss, isIgnore);
        this->smoothingLossLocal.SetValue(batchIdx, -smoothingLoss);
        if (isIgnore) {
            this->lnLocal.SetValue(batchIdx, float(0.0));
            this->weightLocal.SetValue(batchIdx, float(0.0));
            continue;
        }
        this->lnLocal.SetValue(batchIdx, batchLogProb);
    }
    AscendC::Mul(this->lnLocal, this->lnLocal, this->weightLocal, this->batchNum);
    if (this->reduction == REDUCTION_MEAN) {
        GetMeanLoss(this->inputLocal);
    } else if (this->reduction == REDUCTION_SUM) {
        GetSumLoss();
    } else if (this->reduction == REDUCTION_NONE) {
        GetNoneLoss();
    }
    this->inQueue.FreeTensor(this->inputLocal);
}

} // namespace CrossEntropyLossCustom
#endif  // CROSS_ENTROPY_LOSS_FP32_H