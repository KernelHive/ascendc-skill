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
 * @file snrm2_aiv.h
 */
#ifndef SNRM2_AIV_H
#define SNRM2_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace Snrm2 {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class Snrm2AIV {
public:
    __aicore__ inline Snrm2AIV(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SetZeroBuf();
    __aicore__ inline void ClearGM(GlobalTensor<T> dstGM);
    __aicore__ inline void SingleIteration(uint32_t currOffset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t currOffset, uint32_t dataCount);
    __aicore__ inline void CopyIn(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount);
    __aicore__ inline void ComputeSquareAndReduceSum(uint32_t dataCount);
    __aicore__ inline void ComputeReduceSumAndSqrt();
    __aicore__ inline void CopyOut(GlobalTensor<T> dstGM);
    __aicore__ inline void PostProcess();

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    GlobalTensor<T> workGM;

    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> zeroBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> outQueue;

    uint32_t n = 0;  // total elements num(float32)
    uint32_t useCoreNum = 0;
    uint32_t calNum = 0;
    uint32_t startOffset = 0;
    uint32_t maxDataCount = 0;
    uint32_t totalVecCoreNum = 40;
    int32_t vecIdx;
    int32_t blockNum;

    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void Snrm2AIV<T>::Init(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    inGM.SetGlobalBuffer((__gm__ T *)x, this->n);
    outGM.SetGlobalBuffer((__gm__ T *)result, 1);

    workGM.SetGlobalBuffer((__gm__ T *)(workSpace), this->blockNum);

    this->maxDataCount = 90 * 1024 / BYTENUM_PER_FLOAT32;  // 90kb / 4b

    // Compute the min space size for ReduceSum
    int firstMaxRepeat = maxDataCount / elementsPerRepeat;  // 376
    int iter1OutputCount = firstMaxRepeat;
    int iter1AlignEnd = (iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    int finalWorkLocalNeedSize = iter1AlignEnd;

    pipe.InitBuffer(workBuf, finalWorkLocalNeedSize * sizeof(T));  // 1504bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, elementsPerBlock * sizeof(T));
    pipe.InitBuffer(zeroBuf, UB_BYTENUM_PER_BLOCK);  // 32bytes

    // Set zero for ubuf
    SetZeroBuf();
    // Clear workspace
    ClearGM(workGM[this->vecIdx]);

    return;
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::SetZeroBuf()
{
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    Duplicate<T>(zeroLocal, 0, this->elementsPerBlock);

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::ClearGM(GlobalTensor<T> dstGM)
{
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
    DataCopyPad(dstGM, zeroLocal, copyParams);

    AscendC::PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    this->n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));
    this->useCoreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + sizeof(uint32_t)));
    this->startOffset =
        (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 2 * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
    this->calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 2 * sizeof(uint32_t) +
                                         this->totalVecCoreNum * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::SingleIteration(uint32_t currOffset, uint32_t dataCount)
{
    CopyIn(inGM, currOffset, dataCount);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    ComputeSquareAndReduceSum(dataCount);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    CopyOut(workGM[this->vecIdx]);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::SingleIterationAligned(uint32_t currOffset, uint32_t dataCount)
{
    uint32_t dataCountAligned = (dataCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;

    CopyInPad(inGM, currOffset, dataCount);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    ComputeSquareAndReduceSum(dataCountAligned);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    CopyOut(workGM[this->vecIdx]);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::PostProcess()
{
    CopyInPad(workGM, 0, this->blockNum);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    ComputeReduceSumAndSqrt();
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    CopyOut(outGM);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::Process()
{
    if (this->calNum <= 0) {
        return;
    }

    uint32_t repeatTimes = this->calNum / this->maxDataCount;
    uint32_t remainNum = this->calNum % this->maxDataCount;

    uint32_t currOffset = this->startOffset;

    SetAtomicAdd<T>();
    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, this->maxDataCount);
        currOffset += this->maxDataCount;
    }

    if (remainNum > 0) {
        SingleIterationAligned(currOffset, remainNum);
    }
    SetAtomicNone();
    SyncAll();

    if (this->vecIdx == 0) {
        PostProcess();
    }

    return;
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::CopyIn(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, srcGM[offset], dataCount);
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::CopyInPad(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;

    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, srcGM[offset], copyParams, padParams);

    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::ComputeSquareAndReduceSum(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Mul(inLocal, inLocal, inLocal, dataCount);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSum(outLocal, inLocal, workLocal, dataCount);

    outQueue.EnQue<T>(outLocal);
    inQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::ComputeReduceSumAndSqrt()
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    uint32_t dupNum = ((this->blockNum - 1) / this->elementsPerBlock + 1) * this->elementsPerBlock;

    ReduceSum(outLocal, inLocal, workLocal, dupNum);
    AscendC::PipeBarrier<PIPE_V>();
    Sqrt(outLocal, outLocal, this->elementsPerBlock);

    outQueue.EnQue<T>(outLocal);
    inQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void Snrm2AIV<T>::CopyOut(GlobalTensor<T> dstGM)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
    DataCopyPad(dstGM, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}
}

#endif  // SNRM2_AIV_H