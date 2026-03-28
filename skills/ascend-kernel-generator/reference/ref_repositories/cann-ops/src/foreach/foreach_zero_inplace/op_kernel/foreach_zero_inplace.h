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
 * \file foreach_zero_inplace.h
 * \brief
 */
 
#ifndef FOREACH_ZERO_INPLACE_N_D_H
#define FOREACH_ZERO_INPLACE_N_D_H

#include "kernel_operator.h"

namespace ForeachZeroInplace {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class ForeachZeroInplaceND {
public:
    __aicore__ inline ForeachZeroInplaceND(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;

    GlobalTensor<T> inTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;

    int64_t blockIdx = 0;

    uint32_t maxDataCount = {0};
    // tiling params
    uint64_t inputsTensorUbSize = 0;

    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};
};

template <typename T>
__aicore__ inline void ForeachZeroInplaceND<T>::Init(GM_ADDR x, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    inTensorsPtr = x;

    ParseTilingData(tilingData);
    pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);

    maxDataCount = inputsTensorUbSize / sizeof(T);
}

template <typename T>
__aicore__ inline void ForeachZeroInplaceND<T>::Process() {
    for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == tensorStart) {
            cursorStart = tensorStartOffset;
        }
        if (i == tensorEnd) {
            cursorEnd = tensorEndOffset;
        }

        dataCount = cursorEnd - cursorStart + 1;

        inTensorsGM.SetGlobalBuffer(GetTensorAddr(i, inTensorsPtr) + cursorStart);

        SingleTensorProcess(dataCount);
    }
}

template <typename T>
__aicore__ inline void ForeachZeroInplaceND<T>::SingleTensorProcess(int64_t dataCount) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;
    
    for (uint32_t i = 0; i < copyTimes; i++) {
        ComputeAndCopyOut(i, maxDataCount, false);
    }

    if (copyTimesRemainder > 0){
        ComputeAndCopyOut(copyTimes, copyTimesRemainder, true);
    }
}

template <typename T>
__aicore__ inline void ForeachZeroInplaceND<T>::ParseTilingData(
    const ForeachCommonTilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachZeroInplaceND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();
    
    Duplicate(dataLocal, (T)0, dataCount);

    // Transport can be performed only after the Muls is complete.
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
        DataCopyPad(inTensorsGM[index * maxDataCount], dataLocal, copyParams);
    } else {
        DataCopy(inTensorsGM[index * maxDataCount], dataLocal, dataCount);
    }
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    dataQueue.FreeTensor(dataLocal);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachZeroInplaceND<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}
}  // namespace ForeachZeroInplace

#endif  // FOREACH_ZERO_INPLACE_N_D_H