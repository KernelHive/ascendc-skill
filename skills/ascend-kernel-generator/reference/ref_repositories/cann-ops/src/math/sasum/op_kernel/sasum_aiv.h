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
 * @file sasum_aiv.h
 */
#ifndef SASUM_AIV_H
#define SASUM_AIV_H

#include <type_traits>
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

namespace Sasum {

template <typename T>
class SasumAIV {
public:
    __aicore__ inline SasumAIV(){};
    __aicore__ inline void Init(GM_ADDR inGM, GM_ADDR outGM, uint32_t n, uint32_t offset, uint32_t calNum);
    __aicore__ inline void Process();
    __aicore__ inline void SingleIteration(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut();

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TBuf<TPosition::VECCALC> workBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t computeNum;
    uint32_t startOffset;
    uint32_t maxDataCount;
};

template <typename T>
__aicore__ inline void SasumAIV<T>::Init(GM_ADDR inDevice, GM_ADDR outDevice, uint32_t n, uint32_t offset,
                                         uint32_t calNum)
{
    inGM.SetGlobalBuffer((__gm__ T *)inDevice, n);
    outGM.SetGlobalBuffer((__gm__ T *)outDevice, 1);
    computeNum = calNum;
    startOffset = offset;

    maxDataCount = 80 * 1024 / 4;  // 80kb
    // compute the minimum workspace for ReduceSum
    int typeSize = 4;
    int elementsPerBlock = 32 / typeSize;
    int elementsPerRepeat = 256 / typeSize;
    int firstMaxRepeat = maxDataCount / 64;  // 376

    int iter1OutputCount = firstMaxRepeat;
    int iter1AlignEnd = (iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    int finalWorkLocalNeedSize = iter1AlignEnd;

    uint32_t byteLen = finalWorkLocalNeedSize * sizeof(T);  // 1504
    pipe.InitBuffer(workBuf, byteLen + 32);

    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, 8 * sizeof(T));

    LocalTensor<T> workLocal = workBuf.Get<T>(8);
    Duplicate<float>(workLocal, 0.0, 8);
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, workLocal, copyParams);

    SyncAll();

    return;
}

template <typename T>
__aicore__ inline void SasumAIV<T>::Process()
{
    SetAtomicAdd<T>();

    uint32_t repeatTimes = computeNum / maxDataCount;
    uint32_t remainNum = computeNum % maxDataCount;
    uint32_t maxCopyPadNum = (UINT16_MAX + 1) / sizeof(T);

    uint32_t currOffset = startOffset;
    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, maxDataCount);
        currOffset += maxDataCount;
    }
    if (remainNum > 0) {
        if (remainNum >= maxCopyPadNum) {
            SingleIteration(currOffset, maxCopyPadNum);
            currOffset += maxCopyPadNum;
            remainNum -= maxCopyPadNum;
        }

        if (remainNum > 0) {
            SingleIterationAligned(currOffset, remainNum);
        }
    }
    SetAtomicNone();
    return;
}

template <typename T>
__aicore__ inline void SasumAIV<T>::SingleIteration(uint32_t offset, uint32_t dataCount)
{
    CopyIn(offset, dataCount);
    Compute(dataCount);
    CopyOut();
}

template <typename T>
__aicore__ inline void SasumAIV<T>::SingleIterationAligned(uint32_t offset, uint32_t dataCount)
{
    uint32_t dataCountAligned = (dataCount + 7) / 8 * 8;
    CopyInPad(offset, dataCount);
    Compute(dataCountAligned);
    CopyOut();
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGM[offset], dataCount);
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyInPad(uint32_t offset, uint32_t dataCount)
{
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};  // need to modify
    uint8_t paddingNum = 8 - dataCount % 8;
    DataCopyPadParams padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGM[offset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::Compute(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Abs(inLocal, inLocal, dataCount);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSum(outLocal, inLocal, workLocal, dataCount);

    outQueue.EnQue<T>(outLocal);
    inQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyOut()
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}
}
#endif  // SASUM_AIV_H