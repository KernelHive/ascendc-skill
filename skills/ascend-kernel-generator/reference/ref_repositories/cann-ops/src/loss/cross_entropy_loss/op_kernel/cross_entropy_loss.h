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
 * \file cross_entropy_loss.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "cross_entropy_loss_base.h"

using namespace AscendC;
namespace CrossEntropyLossCustom {

template <typename OriT>
class CrossEntropyLoss : public CrossEntropyLossBase<OriT> {
public:
    __aicore__ inline CrossEntropyLoss(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, 
                                GM_ADDR zloss, GM_ADDR lseForZloss, GM_ADDR workspace, const CrossEntropyLossTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessFp32();
protected:
    __aicore__ inline void GetRowMax(const LocalTensor<float>& inputBuf, uint64_t offset);
    __aicore__ inline void GetExpSum(const LocalTensor<float>& inputBuf, uint64_t offset, float& batchSum, const float& rowMax);
    __aicore__ inline void GetLogProbOut(const LocalTensor<float>& inputBuf, uint64_t& offset, const float& logBatchSum, 
                                         float& batchLogProb, const uint64_t& target, float& smoothingLoss, bool isIgnore);
    __aicore__ inline void GetSmoothingLoss(const LocalTensor<float>& inputBuf, const LocalTensor<float>& workLocal, float& smoothingLoss, const uint64_t len, const uint64_t offset);
    __aicore__ inline void GetSmoothingLossSum();
    __aicore__ inline void GetLnWeightSum(const LocalTensor<float>& inputBuf);
    __aicore__ inline void CalcMeanLoss();
    __aicore__ inline void CalcSumLoss();
    __aicore__ inline void GetNoneLoss();
    __aicore__ inline void GetSumLoss();
    __aicore__ inline void GetMeanLoss(const LocalTensor<float>& inputBuf);
};

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::Init(GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, 
                                                    GM_ADDR zloss, GM_ADDR lseForZloss, GM_ADDR workspace, const CrossEntropyLossTilingData& tilingData)
{
    CrossEntropyLossBase<OriT>::InitTiling(tilingData);
    CrossEntropyLossBase<OriT>::InitGlobalTensor(input, target, weight, loss, logProb, workspace);
    CrossEntropyLossBase<OriT>::InitUB();
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::Process()
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
    this->inputLocal = this->inQueue.template AllocTensor<OriT>();
    bool isIgnore = false;
    for (size_t batchIdx = 0; batchIdx < this->batchNum; ++batchIdx) {
        this->reduceRes.SetValue(0, MIN_FLT);
        GetRowMax(this->castTmpBuf, offset);
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
        GetExpSum(this->castTmpBuf, offset, batchSum, rowMax);
        this->reduceCalc.SetValue(0, batchSum);
        AscendC::Log(this->reduceRes, this->reduceCalc, 1);
        logBatchSum = this->reduceRes(0) + rowMax;
        batchTargetOffset = batchTarget + batchIdx * this->targetNum;
        smoothingLoss = 0.0;
        GetLogProbOut(this->castTmpBuf, offset, logBatchSum, batchLogProb, batchTargetOffset, smoothingLoss, isIgnore);
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
        GetMeanLoss(this->castTmpBuf);
    } else if (this->reduction == REDUCTION_SUM) {
        GetSumLoss();
    } else if (this->reduction == REDUCTION_NONE) {
        GetNoneLoss();
    }
    this->inQueue.FreeTensor(this->inputLocal);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetMeanLoss(const LocalTensor<float>& inputBuf)
{    
    GetLnWeightSum(inputBuf);
    SyncAll<true>();
    if (this->coreIndex == 0) {
        CalcMeanLoss();
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetSumLoss()
{
    uint32_t len = this->batchNum;
    AscendC::LocalTensor<float> workLocal = this->calcBuf.template GetWithOffset<float>(this->lnTmpBufSize, this->workLocalOffset);
    this->GetReduceSum(this->lnLocal, workLocal, this->reduceRes, len);
    this->reduceRes(0) = (len > NUM_4096 && len < FP32_128_REPEAT) ? 
                          this->reduceRes(0) + this->reduceRes(1) : this->reduceRes(0);
    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    AscendC::DataCopyExtParams copyParams{1, 4, 0, 0, 0};
    AscendC::DataCopyPad(this->workspaceGm[this->coreIndex], this->reduceRes, copyParams);
    if (this->isSmoothing) {
        GetSmoothingLossSum();
    }
    SyncAll<true>();

    if (this->coreIndex == 0) {
        CalcSumLoss();
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetNoneLoss()
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
    AscendC::Cast(this->probOutBuf, this->lnLocal, AscendC::RoundMode::CAST_RINT, len);
    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    if (len % BLOCK_32 == 0) {
        AscendC::DataCopy(this->lossGm, this->probOutBuf, len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPad(this->lossGm, this->probOutBuf, copyParams);
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetRowMax(const LocalTensor<float>& inputBuf, uint64_t offset)
{
    AscendC::LocalTensor<float> workLocal = this->calcBuf.template GetWithOffset<float>(NUM_1024, this->workLocalOffset);
    for (size_t i = 0; i < this->ubLoopNum; ++i) {
        this->CopyIn(inputBuf, offset, uint32_t(this->inputUbSize));
        offset += this->inputUbSize;
        uint32_t castOffset = 0;
        for (size_t j = 0; j < this->vecLoopNum; ++j) {
            this->GetReduceMax(inputBuf[castOffset], workLocal, this->reduceRes, this->reduceCalc, FP32_128_REPEAT);
            castOffset += FP32_128_REPEAT;
        }
        if (this->vecTailNum != 0) {
            this->GetReduceMax(inputBuf[castOffset], workLocal, this->reduceRes, this->reduceCalc, this->vecTailNum);
        }
    }
    if (this->ubTailNum != 0) {
        this->CopyIn(inputBuf, offset, uint32_t(this->ubTailNum));
        offset += this->ubTailNum;
        uint32_t tailOffset = 0;
        for (size_t i = 0; i < this->tailVecLoopNum; ++i) {
            this->GetReduceMax(inputBuf[tailOffset], workLocal, this->reduceRes, this->reduceCalc, FP32_128_REPEAT);
            tailOffset += FP32_128_REPEAT;
        }
        if (this->tailVecTailNum != 0) {
            this->GetReduceMax(inputBuf[tailOffset], workLocal, this->reduceRes, this->reduceCalc, this->tailVecTailNum);
        }
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetExpSum(const LocalTensor<float>& inputBuf, uint64_t offset, float& batchSum, const float& rowMax)
{
    AscendC::LocalTensor<float> workLocal = this->calcBuf.template GetWithOffset<float>(NUM_1024, this->workLocalOffset);
    for (size_t i=0; i<this->ubLoopNum; ++i) {
        this->CopyIn(inputBuf, offset, uint32_t(this->inputUbSize));
        offset += this->inputUbSize;
        AscendC::Adds(inputBuf, inputBuf, -rowMax, this->inputUbSize);
        pipe_barrier(PIPE_V);
        AscendC::Exp(inputBuf, inputBuf, this->inputUbSize);
        pipe_barrier(PIPE_V);
        uint32_t ubOffset = 0;
        for (size_t j=0; j<this->vecLoopNum; ++j) {
            this->GetReduceSum(inputBuf[ubOffset], workLocal, this->reduceCalc, FP32_128_REPEAT);
            event_t eventVS = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            batchSum += this->reduceCalc(0);
            ubOffset += FP32_128_REPEAT;
        }
        if (this->vecTailNum != 0) {
            this->GetReduceSum(inputBuf[ubOffset], workLocal, this->reduceCalc, this->vecTailNum);
            event_t eventVS = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            batchSum += this->vecTailNum > NUM_4096 ? this->reduceCalc(0) + this->reduceCalc(1) : this->reduceCalc(0);
        }
    }
    if (this->ubTailNum != 0) {
        this->CopyIn(inputBuf, offset, uint32_t(this->ubTailNum));
        offset += this->ubTailNum;
        AscendC::Adds(inputBuf, inputBuf, -rowMax, this->ubTailNum);
        pipe_barrier(PIPE_V);
        AscendC::Exp(inputBuf, inputBuf, this->ubTailNum);
        pipe_barrier(PIPE_V);
        uint32_t tailOffset = 0;
        for (size_t j=0; j<this->tailVecLoopNum; ++j) {
            this->GetReduceSum(inputBuf[tailOffset], workLocal, this->reduceCalc, FP32_128_REPEAT);
            event_t eventVS = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            batchSum += this->reduceCalc(0);
            tailOffset += FP32_128_REPEAT;
        }
        if (this->tailVecTailNum != 0) {
            this->GetReduceSum(inputBuf[tailOffset], workLocal, this->reduceCalc, this->tailVecTailNum);
            event_t eventVS = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            batchSum += this->tailVecTailNum > NUM_4096 ? this->reduceCalc(0) + this->reduceCalc(1) : this->reduceCalc(0);
        }
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetLogProbOut(const LocalTensor<float>& inputBuf, uint64_t& offset, const float& logBatchSum, 
                                                                 float& batchLogProb, const uint64_t& target, float& smoothingLoss, bool isIgnore)
{
    AscendC::LocalTensor<float> workLocal = this->calcBuf.template GetWithOffset<float>(NUM_1024, this->workLocalOffset);
    uint64_t weightOffset = 0;
    for (size_t i = 0; i < this->ubLoopNum; ++i)
    {
        this->CopyIn(inputBuf, offset, uint32_t(this->inputUbSize));
        AscendC::Adds(inputBuf, inputBuf, -logBatchSum, this->inputUbSize);
        pipe_barrier(PIPE_V);
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

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetSmoothingLoss(const LocalTensor<float>& inputBuf, const LocalTensor<float>& workLocal, float& smoothingLoss, const uint64_t len, const uint64_t weightOffset)
{
    if (this->defaultWeight == 0) {
        this->CopyWeightIn(weightOffset, uint32_t(len));
        AscendC::Mul(inputBuf, inputBuf, this->weight4SmoothingBuf, len);
    }
    uint32_t loopNum = len / FP32_128_REPEAT;
    uint32_t tailNum = len % FP32_128_REPEAT;
    uint64_t smoothingOffset = 0;
    for (size_t i = 0; i < loopNum; ++i) {
        this->GetReduceSum(inputBuf[smoothingOffset], workLocal, this->reduceCalc, FP32_128_REPEAT);
        smoothingLoss += this->reduceCalc(0);
        smoothingOffset += FP32_128_REPEAT;
    }
    if (tailNum != 0) {
        this->GetReduceSum(inputBuf[smoothingOffset], workLocal, this->reduceCalc, tailNum);
        smoothingLoss += tailNum > NUM_4096 ? this->reduceCalc(0) + this->reduceCalc(1) : this->reduceCalc(0);
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetLnWeightSum(const LocalTensor<float>& inputBuf)
{
    AscendC::DataCopyExtParams copyParams{1,4,0,0,0};
    AscendC::LocalTensor<float> weightCalcLocal = this->calcBuf.template GetWithOffset<float>(NUM_1024, this->workLocalOffset);
    this->GetReduceSum(this->lnLocal, inputBuf, this->reduceRes, this->batchNum);
    this->reduceRes(0) = (this->batchNum > NUM_4096 && this->batchNum < FP32_128_REPEAT) ? 
                          this->reduceRes(0) + this->reduceRes(1) : this->reduceRes(0);

    this->GetReduceSum(this->weightLocal, weightCalcLocal, this->reduceCalc, this->batchNum);
    this->reduceCalc(0) = (this->batchNum > NUM_4096 && this->batchNum < FP32_128_REPEAT) ?
                           this->reduceCalc(0) + this->reduceCalc(1) : this->reduceCalc(0);

    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    AscendC::DataCopyPad(this->workspaceGm[this->coreIndex], this->reduceRes, copyParams);
    AscendC::DataCopyPad(this->workspaceGm[NUM_64 + this->coreIndex], this->reduceCalc, copyParams);
    if (this->isSmoothing) {
        GetSmoothingLossSum();
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::GetSmoothingLossSum()
{
    AscendC::DataCopyExtParams copyParams{1,4,0,0,0};
    this->GetReduceSum(this->smoothingLossLocal, this->smoothingLossLocal, this->castTmpBuf, this->batchNum);
    this->castTmpBuf(0) = (this->batchNum > NUM_4096 && this->batchNum < FP32_128_REPEAT) ? 
                           this->castTmpBuf(0) + this->castTmpBuf(1) : this->castTmpBuf(0);

    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    AscendC::DataCopyPad(this->workspaceGm[NUM_128 + this->coreIndex], this->castTmpBuf, copyParams);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::CalcMeanLoss()
{
    AscendC::LocalTensor<float> syncLocal = this->calcBuf.template Get<float>(NUM_192);
    AscendC::LocalTensor<OriT> meanLoss = this->calcBuf.template GetWithOffset<OriT>(BLOCK_32, NUM_192 * FP32_BYTE_LEN);
    AscendC::DataCopy(syncLocal, this->workspaceGm, NUM_192);
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    AscendC::WholeReduceSum(syncLocal, syncLocal, this->usedCoreNum, 1, 1, 1, 8);
    AscendC::WholeReduceSum(syncLocal[NUM_64], syncLocal[NUM_64], this->usedCoreNum, 1, 1, 1, 8);
    float loss = 0.0;
    float weightSum = syncLocal(64);
    float lnSum = syncLocal(0);
    if (this->isSmoothing) {
        float smoothingScale = this->labelSmoothing / this->targetNum;
        AscendC::WholeReduceSum(syncLocal[NUM_128], syncLocal[NUM_128], this->usedCoreNum, 1, 1, 1, 8);
        loss = ((this->labelSmoothing - 1) * lnSum + syncLocal(128) * smoothingScale) / weightSum;
    } else {
        loss = -lnSum / weightSum;
    }
    syncLocal.SetValue(0, loss);
    AscendC::Cast(meanLoss, syncLocal, AscendC::RoundMode::CAST_RINT, 1);
    this->lossGm.SetValue(0, meanLoss(0));
}

template <typename OriT>
__aicore__ inline void CrossEntropyLoss<OriT>::CalcSumLoss()
{
    AscendC::LocalTensor<float> syncLocal = this->calcBuf.template Get<float>(NUM_192);
    AscendC::LocalTensor<OriT> sumLoss = this->calcBuf.template GetWithOffset<OriT>(BLOCK_32, NUM_192 * FP32_BYTE_LEN);
    AscendC::DataCopy(syncLocal, this->workspaceGm, NUM_192);
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    AscendC::WholeReduceSum(syncLocal, syncLocal, this->usedCoreNum, 1, 1, 1, 8);
    if (this->isSmoothing) {
        float smoothingScale = this->labelSmoothing / this->targetNum;
        AscendC::WholeReduceSum(syncLocal[NUM_128], syncLocal[NUM_128], this->usedCoreNum, 1, 1, 1, 8);
        float loss = (this->labelSmoothing - 1) * syncLocal(0) + syncLocal(NUM_128) * smoothingScale;
        syncLocal.SetValue(0, loss);
    } else {
        AscendC::Muls(syncLocal, syncLocal, float(-1.0), 1);
    }
    AscendC::Cast(sumLoss, syncLocal, AscendC::RoundMode::CAST_RINT, 1);
    this->lossGm.SetValue(0, sumLoss(0));
}
} // namespace CrossEntropyLossCustom
#endif // CROSS_ENTROPY_LOSS_H