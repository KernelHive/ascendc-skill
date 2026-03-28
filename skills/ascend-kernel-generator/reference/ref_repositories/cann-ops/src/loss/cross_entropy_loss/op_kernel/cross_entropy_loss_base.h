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
 * \file cross_entropy_loss_base.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_BASE_H
#define CROSS_ENTROPY_LOSS_BASE_H

#include "kernel_operator.h"

using namespace AscendC;
namespace CrossEntropyLossCustom {

constexpr uint32_t DOUBEL_BUFFER = 1;
constexpr uint32_t BLOCK_32 = 32;
constexpr uint32_t FP32_ONE_REPEAT = 64;
constexpr uint32_t FP32_128_REPEAT = 8192;
constexpr uint32_t FP32_PER_BLOCK = 8;
constexpr uint32_t FP32_BYTE_LEN = 4;
constexpr uint32_t HALF_ONE_REPEAT = 2;
constexpr float MIN_FLT =  -3.4028235e+38;

constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_192 = 192;
constexpr uint32_t NUM_1024 = 1024;
constexpr uint32_t NUM_4096 = 4096;
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_MEAN = 1;
constexpr uint32_t REDUCTION_SUM = 2;

template <typename OriT>
class CrossEntropyLossBase {
public:
    __aicore__ inline CrossEntropyLossBase() {};

protected:
    __aicore__ inline void InitTiling(const CrossEntropyLossTilingData& tilingData);
    __aicore__ inline void InitGlobalTensor(GM_ADDR input, GM_ADDR target, GM_ADDR weight, 
                                            GM_ADDR loss, GM_ADDR logProb, GM_ADDR workspace);
    __aicore__ inline void InitUB();
    __aicore__ inline void CopyIn(const LocalTensor<float>& castLocal, const uint64_t offset, const uint32_t len);
    __aicore__ inline void CopyOut(const LocalTensor<float>& srcLocal, const LocalTensor<OriT>& dstLocal, 
                                   const uint64_t offset, const uint32_t len);
    __aicore__ inline void CopyWeightIn(const uint64_t offset, const uint32_t len);
    __aicore__ inline void GetReduceSum(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                        const LocalTensor<float>& dstLocal, const uint32_t len);
    __aicore__ inline void ReduceSumSmall(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                          const LocalTensor<float>& dstLocal, const uint32_t len);
    __aicore__ inline void GetReduceMax(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                        const LocalTensor<float>& resTmpBuf, const LocalTensor<float>& dstLocal, const uint32_t len);
    __aicore__ inline void ReduceMaxSmall(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                          const LocalTensor<float>& dstLocal, const uint32_t len);
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    GlobalTensor<OriT> inputGm;
    GlobalTensor<int64_t> targetGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<OriT> logProbGm;
    GlobalTensor<OriT> lossGm;
    GlobalTensor<float> workspaceGm;

    AscendC::LocalTensor<OriT> inputLocal;
    AscendC::LocalTensor<float> castTmpBuf;
    AscendC::LocalTensor<OriT> probOutBuf;
    AscendC::LocalTensor<float> weight4SmoothingBuf;
    AscendC::LocalTensor<float> reduceCalc;
    AscendC::LocalTensor<float> reduceRes;
    AscendC::LocalTensor<float> lnLocal;
    AscendC::LocalTensor<float> weightLocal;
    AscendC::LocalTensor<float> smoothingLossLocal;

    // tiling data
    uint64_t targetNum;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t frontBatchNum;
    uint64_t tailBatchNum;
    uint64_t inputUbSize;
    uint64_t castTmpBufByte;
    uint64_t lnTmpBufSize;
    uint64_t weightTmpBufSize;
    uint64_t weight4SmoothingBufSize;
    uint64_t totalTmpBufByte;
    uint64_t ubLoopNum;
    uint64_t ubTailNum;
    uint64_t vecLoopNum;
    uint64_t vecTailNum;
    uint64_t tailVecLoopNum;
    uint64_t tailVecTailNum;
    uint64_t reduction;
    int64_t ignoreIndex;
    float labelSmoothing;
    uint32_t defaultWeight;

    // global params
    uint32_t inputTypeSize;
    uint32_t coreIndex;
    uint32_t batchNum;
    uint64_t workLocalOffset;
    uint64_t startBatchIndex;
    uint32_t usedCoreNum;
    bool isSmoothing;
};

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::InitTiling(const CrossEntropyLossTilingData& tilingData) 
{
    this->targetNum = tilingData.targetNum;
    this->frontCoreNum = tilingData.frontCoreNum;
    this->frontBatchNum = tilingData.frontBatchNum;
    this->tailCoreNum = tilingData.tailCoreNum;
    this->tailBatchNum = tilingData.tailBatchNum;
    this->inputUbSize = tilingData.inputUbSize;
    this->castTmpBufByte = tilingData.castTmpBufByte;
    this->lnTmpBufSize = tilingData.lnTmpBufSize;
    this->weightTmpBufSize = tilingData.weightTmpBufSize;
    this->weight4SmoothingBufSize = tilingData.weight4SmoothingBufSize;
    this->totalTmpBufByte = tilingData.totalTmpBufByte;
    this->ubLoopNum = tilingData.ubLoopNum;
    this->ubTailNum = tilingData.ubTailNum;
    this->vecLoopNum = tilingData.vecLoopNum;
    this->vecTailNum = tilingData.vecTailNum;
    this->tailVecLoopNum = tilingData.tailVecLoopNum;
    this->tailVecTailNum = tilingData.tailVecTailNum;
    this->reduction = tilingData.reduction;
    this->ignoreIndex = tilingData.ignoreIndex;
    this->labelSmoothing = tilingData.labelSmoothing;
    this->defaultWeight = tilingData.defaultWeight;
    this->usedCoreNum = this->frontCoreNum + this->tailCoreNum;
    this->isSmoothing = this->labelSmoothing > 0 ? true : false;
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::InitGlobalTensor(GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss,
                            GM_ADDR logProb, GM_ADDR workspace)
{
    this->coreIndex = GetBlockIdx();
    uint64_t coreAddrOffset;
    uint64_t totalLenPerCore;
    if (this->tailCoreNum == 0 || this->coreIndex < this->frontCoreNum) {
        this->batchNum = this->frontBatchNum;
        totalLenPerCore = this->frontBatchNum * this->targetNum;
        coreAddrOffset = this->coreIndex * this->batchNum *this->targetNum;
        this->startBatchIndex = this->coreIndex * this->frontBatchNum;
    } else {
        this->batchNum = this->tailBatchNum;
        totalLenPerCore = this->tailBatchNum * this->targetNum;
        coreAddrOffset = this->frontCoreNum * this->frontBatchNum * this->targetNum + (this->coreIndex - this->frontCoreNum) * this->batchNum * this->targetNum;
        this->startBatchIndex = this->frontCoreNum * this->frontBatchNum + (this->coreIndex - this->frontCoreNum) * this->tailBatchNum;
    }
    inputGm.SetGlobalBuffer((__gm__ OriT *)(input) + coreAddrOffset, totalLenPerCore);
    targetGm.SetGlobalBuffer((__gm__ int64_t *)(target));
    weightGm.SetGlobalBuffer((__gm__ float *)(weight));
    lossGm.SetGlobalBuffer((__gm__ OriT *)(loss) + this->startBatchIndex, this->batchNum);
    logProbGm.SetGlobalBuffer((__gm__ OriT *)(logProb) + coreAddrOffset, totalLenPerCore);
    workspaceGm.SetGlobalBuffer((__gm__ float *)(workspace));
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::InitUB()
{
    this->inputTypeSize = sizeof(OriT);
    this->pipe.InitBuffer(this->inQueue, DOUBEL_BUFFER, this->inputUbSize * this->inputTypeSize);

    this->pipe.InitBuffer(this->calcBuf, this->totalTmpBufByte);
    uint32_t castTmpBufSize = this->castTmpBufByte / FP32_BYTE_LEN;
    this->workLocalOffset = this->castTmpBufByte;
    this->castTmpBuf = this->calcBuf.template Get<float>(castTmpBufSize);
    this->probOutBuf = this->calcBuf.template GetWithOffset<OriT>(castTmpBufSize, this->workLocalOffset);
    this->workLocalOffset += castTmpBufSize * this->inputTypeSize;
    this->weight4SmoothingBuf = this->calcBuf.template GetWithOffset<float>(weight4SmoothingBufSize, this->workLocalOffset);
    this->workLocalOffset += weight4SmoothingBufSize * FP32_BYTE_LEN;
    this->reduceCalc = this->calcBuf.template GetWithOffset<float>(FP32_PER_BLOCK, this->workLocalOffset);
    this->workLocalOffset += BLOCK_32;
    this->reduceRes = this->calcBuf.template GetWithOffset<float>(FP32_PER_BLOCK, this->workLocalOffset);
    this->workLocalOffset += BLOCK_32;
    this->lnLocal = this->calcBuf.template GetWithOffset<float>(this->lnTmpBufSize, this->workLocalOffset);
    this->workLocalOffset += this->lnTmpBufSize * FP32_BYTE_LEN;
    this->weightLocal = this->calcBuf.template GetWithOffset<float>(this->weightTmpBufSize, this->workLocalOffset);
    this->workLocalOffset += this->weightTmpBufSize * FP32_BYTE_LEN;
    this->smoothingLossLocal = this->calcBuf.template GetWithOffset<float>(this->weightTmpBufSize, this->workLocalOffset);
    this->workLocalOffset += this->weightTmpBufSize * FP32_BYTE_LEN;
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::CopyIn(const LocalTensor<float>& castLocal, const uint64_t offset, const uint32_t len)
{
    if (len == FP32_128_REPEAT || len % BLOCK_32 == 0) {
        DataCopy(this->inputLocal, inputGm[offset], len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPadExtParams<OriT> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(this->inputLocal, inputGm[offset], copyParams, padParams);
    }
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);

    AscendC::Cast(castLocal, this->inputLocal, AscendC::RoundMode::CAST_NONE, len);
    event_t eventVMTE2 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    pipe_barrier(PIPE_V);
}

template <>
__aicore__ inline void CrossEntropyLossBase<float>::CopyIn(const LocalTensor<float>& castLocal, const uint64_t offset, const uint32_t len)
{
    event_t eventVMTE2 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    if (len == FP32_128_REPEAT || len % BLOCK_32 == 0) {
        DataCopy(this->inputLocal, inputGm[offset], len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(this->inputLocal, inputGm[offset], copyParams, padParams);
    }
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::CopyWeightIn(const uint64_t offset, const uint32_t len)
{
    event_t eventVMTE2 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
    if (len == FP32_128_REPEAT || len % BLOCK_32 == 0) {
        DataCopy(this->weight4SmoothingBuf, weightGm[offset], len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * FP32_BYTE_LEN, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(this->weight4SmoothingBuf, weightGm[offset], copyParams, padParams);
    }
    event_t eventMTE2V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
    wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::CopyOut(const LocalTensor<float>& srcLocal, const LocalTensor<OriT>& dstLocal, 
                                                           const uint64_t offset, const uint32_t len)
{
    event_t eventMTE3V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
    wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_RINT, len);
    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    if (len == FP32_128_REPEAT || len % BLOCK_32 == 0) {
        AscendC::DataCopy(this->logProbGm[offset], dstLocal, len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPad(this->logProbGm[offset], dstLocal, copyParams);
    }
}

template <>
__aicore__ inline void CrossEntropyLossBase<float>::CopyOut(const LocalTensor<float>& srcLocal, const LocalTensor<float>& dstLocal, 
                                                           const uint64_t offset, const uint32_t len)
{
    event_t eventVMTE3 = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
    if (len == FP32_128_REPEAT || len % BLOCK_32 == 0) {
        AscendC::DataCopy(this->logProbGm[offset], srcLocal, len);
    } else {
        AscendC::DataCopyExtParams copyParams{1, len * this->inputTypeSize, 0, 0, 0};
        AscendC::DataCopyPad(this->logProbGm[offset], srcLocal, copyParams);
    }
    event_t eventMTE3V = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
    wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::GetReduceSum(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                                                const LocalTensor<float>& dstLocal, const uint32_t len)
{
    if (len == FP32_128_REPEAT) {
        AscendC::BlockReduceSum(workLocal, srcLocal, 128, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::BlockReduceSum(workLocal, workLocal, 16, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::BlockReduceSum(workLocal, workLocal, 2, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::WholeReduceSum(dstLocal, workLocal, 16, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
    } else if (len <= NUM_4096) {
        ReduceSumSmall(srcLocal, workLocal, dstLocal, len);
    } else {
        AscendC::BlockReduceSum(workLocal, srcLocal, 64, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        AscendC::BlockReduceSum(workLocal, workLocal, 8, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        AscendC::WholeReduceSum(dstLocal, workLocal, 64, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
        ReduceSumSmall(srcLocal[NUM_4096], workLocal, dstLocal[1], len - NUM_4096);
    }
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::ReduceSumSmall(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                                                const LocalTensor<float>& dstLocal, const uint32_t len)
{
    uint32_t repeat = len / 64;
    uint32_t tailNum = len % 64;
    if (repeat > 0) {
        AscendC::WholeReduceSum(workLocal, srcLocal, 64, repeat, 1, 1, 8);
        pipe_barrier(PIPE_V);
    }
    if (tailNum != 0) {
        AscendC::WholeReduceSum(workLocal[repeat], srcLocal[len - tailNum], tailNum, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
        repeat += 1;
    }
    AscendC::WholeReduceSum(dstLocal, workLocal, repeat, 1, 1, 1, 8);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::GetReduceMax(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                                                const LocalTensor<float>& dstLocal, const LocalTensor<float>& resTmpBuf, const uint32_t len)
{
    if (len == FP32_128_REPEAT) {
        AscendC::BlockReduceMax(workLocal, srcLocal, 128, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::BlockReduceMax(workLocal, workLocal, 16, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::BlockReduceMax(workLocal, workLocal, 2, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::WholeReduceMax(resTmpBuf, workLocal, 16, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        pipe_barrier(PIPE_V);
    } else if (len <= NUM_4096) {
        ReduceMaxSmall(srcLocal, workLocal, resTmpBuf, len);
        pipe_barrier(PIPE_V);
    } else {
        AscendC::BlockReduceMax(workLocal, srcLocal, 64, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);

        AscendC::BlockReduceMax(workLocal, workLocal, 8, 64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        
        AscendC::WholeReduceMax(resTmpBuf, workLocal, 64, 1, 1, 1, 8);
        pipe_barrier(PIPE_V);
        
        AscendC::Max(dstLocal, dstLocal, resTmpBuf, 1, 1, {1,1,1,8,8,8});
        pipe_barrier(PIPE_V);
        
        ReduceMaxSmall(srcLocal[NUM_4096], workLocal, resTmpBuf, len - NUM_4096);
        pipe_barrier(PIPE_V);
    }
    AscendC::Max(dstLocal, dstLocal, resTmpBuf, 1, 1, {1,1,1,8,8,8});
    pipe_barrier(PIPE_V);
}

template <typename OriT>
__aicore__ inline void CrossEntropyLossBase<OriT>::ReduceMaxSmall(const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
                                                                const LocalTensor<float>& dstLocal, const uint32_t len)
{
    uint32_t repeat = len / 64;
    uint32_t tailNum = len % 64;
    if (repeat > 0) {
        AscendC::WholeReduceMax(workLocal, srcLocal, 64, repeat, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        pipe_barrier(PIPE_V);
    }
    if (tailNum != 0) {
        AscendC::WholeReduceMax(workLocal[repeat], srcLocal[len - tailNum], tailNum, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        pipe_barrier(PIPE_V);
        repeat += 1;
    }
    AscendC::WholeReduceMax(dstLocal, workLocal, repeat, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
}  
} // namespace CrossEntropyLossCustom

#endif  // CROSS_ENTROPY_LOSS_BASE_H