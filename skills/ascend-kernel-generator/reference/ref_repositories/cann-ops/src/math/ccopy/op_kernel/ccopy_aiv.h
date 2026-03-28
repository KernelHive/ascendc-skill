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
 * @file ccopy_aiv.h
 */
#ifndef CCOPY_AIV_H
#define CCOPY_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class CcopyAIV {
public:
    __aicore__ inline CcopyAIV(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SingleIteration(uint32_t currOffset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t currOffset, uint32_t dataCount);

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;

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
__aicore__ inline void CcopyAIV<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    inGM.SetGlobalBuffer((__gm__ T *)x, this->n);
    outGM.SetGlobalBuffer((__gm__ T *)y, this->n);

    this->maxDataCount = 90 * 1024 / BYTENUM_PER_FLOAT32;  // 90kb / 4b

    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));

    return;
}

template <typename T>
__aicore__ inline void CcopyAIV<T>::ParseTilingData(GM_ADDR tilingGm)
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
__aicore__ inline void CcopyAIV<T>::SingleIteration(uint32_t currOffset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGM[currOffset], dataCount);
    inQueue.EnQue<T>(inLocal);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    LocalTensor<T> outLocal = inQueue.DeQue<T>();
    DataCopy(outGM[currOffset], outLocal, dataCount);
    inQueue.FreeTensor(outLocal);
    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void CcopyAIV<T>::SingleIterationAligned(uint32_t currOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGM[currOffset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);

    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

    LocalTensor<T> outLocal = inQueue.DeQue<T>();
    DataCopyPad(outGM[currOffset], outLocal, copyParams);
    inQueue.FreeTensor(outLocal);
    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void CcopyAIV<T>::Process()
{
    if (this->calNum <= 0) {
        return;
    }

    uint32_t repeatTimes = this->calNum / this->maxDataCount;
    uint32_t remainNum = this->calNum % this->maxDataCount;

    uint32_t currOffset = this->startOffset;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, this->maxDataCount);
        currOffset += this->maxDataCount;
    }

    if (remainNum > 0) {
        SingleIterationAligned(currOffset, remainNum);
    }
    return;
}

#endif  // CCOPY_AIV_H