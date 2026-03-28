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
 * \file foreach_pow_scalar_and_tensor.h
 * \brief
 */

#ifndef FOREACH_POW_SCALAR_AND_TENSOR_H
#define FOREACH_POW_SCALAR_AND_TENSOR_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/math/kernel_operator_power_intf.h"

namespace ForeachPowScalarAndTensor {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;

template<typename T>
class InnerComputer
{
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal,
        LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
        T scalarVal,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        Power<T, false>(outLocal, scalarVal, dataLocal, dataCount);
    }
};

template<>
class InnerComputer<bfloat16_t>
{
private:
    __aicore__ inline void ComputerPerCast(
        LocalTensor<bfloat16_t> &dataLocal,
        LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        float scalarVal,
        uint32_t maxCastDataCount, uint16_t index, int64_t dataCount) {        
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Power<float, false>(float32Tensor[maxCastDataCount], scalarVal, float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * maxCastDataCount], float32Tensor[maxCastDataCount], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t> &dataLocal,
        LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        float scalarVal,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        uint32_t castTimes;
        uint32_t castTimesRemainder;

        if (maxCastDataCount != 0) {
            castTimes = dataCount / maxCastDataCount;
            castTimesRemainder = dataCount % maxCastDataCount;
        }

        for (uint32_t i = 0; i < castTimes; i++) {           
            ComputerPerCast(
              dataLocal, outLocal, float32Tensor,
              scalarVal, maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputerPerCast(
              dataLocal, outLocal, float32Tensor, 
              scalarVal, maxCastDataCount, castTimes, castTimesRemainder);
        }
    }
};

template <typename T>
class ForeachPowScalarAndTensorND {
public:
    __aicore__ inline ForeachPowScalarAndTensorND(){};
    __aicore__ inline void Init(GM_ADDR scalar, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder);
    __aicore__ inline __gm__ T* GetInputTensorAddr(uint16_t index);
    __aicore__ inline __gm__ T* GetOutputTensorAddr(uint16_t index);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;
    int64_t blockIdx = 0;
    using TT = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    TT scalarVal = 0;

    uint32_t maxDataCount = {0};
    // tiling params
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;
    uint32_t maxCastDataCount = {0};
};

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::Init(
    GM_ADDR scalar, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    inTensorPtr = inputs;
    outTensorPtr = outputs;
    ParseTilingData(tilingData);
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    if (std::is_same_v<T, bfloat16_t>) {
        scalarVal = inScalarGM.GetValue(0);
    } else {
        scalarVal = T(inScalarGM.GetValue(0));
    }
    // Init for bfloat16
    if (std::is_same<T, bfloat16_t>::value) {
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        maxCastDataCount = inputsTensorUbSize / sizeof(float);        
    } else {
        pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::Process() {
    LocalTensor<float> float32Tensor; 
    if (std::is_same<T, bfloat16_t>::value) {
        float32Tensor = float32Queue.DeQue<float>(); 
    }
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
        inTensorGM.SetGlobalBuffer(GetInputTensorAddr(i) + cursorStart);
        outTensorGM.SetGlobalBuffer(GetOutputTensorAddr(i) + cursorStart);
        SingleTensorProcess(dataCount, float32Tensor);
    }
    if (std::is_same<T, bfloat16_t>::value) {
        float32Queue.FreeTensor(float32Tensor);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;

    for (uint32_t i = 0; i < copyTimes; i++) {
        CopyIn(i, maxDataCount, false);
        ComputeAndCopyOut(i, maxDataCount, float32Tensor, false);
    }

    if (copyTimesRemainder > 0) {
        CopyIn(copyTimes, copyTimesRemainder, true);
        ComputeAndCopyOut(copyTimes, copyTimesRemainder, float32Tensor, true);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::ParseTilingData(
    const ForeachCommonTilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorGM[index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorGM[index * maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void ForeachPowScalarAndTensorND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount, 
    LocalTensor<float> &float32Tensor, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    InnerComputer<T> computer;
    computer.Compute(
      dataLocal, outLocal, float32Tensor,
      scalarVal, maxCastDataCount, dataCount);
    
    dataQueue.FreeTensor(dataLocal);
    outQueue.EnQue<T>(outLocal);

    LocalTensor<T> retLocal = outQueue.DeQue<T>();

    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
        DataCopyPad(outTensorGM[index * maxDataCount], retLocal, copyParams);
    } else {
        DataCopy(outTensorGM[index * maxDataCount], retLocal, dataCount);
    }
    outQueue.FreeTensor(retLocal);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachPowScalarAndTensorND<T>::GetInputTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

template <typename T>
__aicore__ inline __gm__ T* ForeachPowScalarAndTensorND<T>::GetOutputTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(outTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

}  // namespace ForeachPowScalarAndTensor

#endif  // FOREACH_POW_SCALAR_AND_TENSOR_H