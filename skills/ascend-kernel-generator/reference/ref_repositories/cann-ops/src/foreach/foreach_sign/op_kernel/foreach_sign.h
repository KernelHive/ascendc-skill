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
 * \file foreach_sign.h
 * \brief
 */
 
#ifndef FOREACH_SIGN_H
#define FOREACH_SIGN_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachSign {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr uint8_t COPY_SPACE_MULTIPLE = 4;
constexpr uint8_t INPUT_PARAMETER_COUNT = 4;

struct TensorInfo {
    LocalTensor<float> float32Tensor;
    LocalTensor<half> float16Tensor;
    uint32_t maxCastDataCount;

    __aicore__ inline TensorInfo() {}
};

template<typename T>
class InnerComputer
{
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal, LocalTensor<T> &outLocal, struct TensorInfo &tensorInfo, int64_t dataCount) {
        Sign<T, false>(outLocal, dataLocal, dataCount);
    }
};

template<>
class InnerComputer<bfloat16_t>
{
private:
    __aicore__ inline void ComputerPerCast(
        LocalTensor<bfloat16_t> &dataLocal, LocalTensor<bfloat16_t> &outLocal, struct TensorInfo &tensorInfo,
        uint16_t index, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Cast(tensorInfo.float32Tensor, dataLocal[index * tensorInfo.maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Sign<float, false>(tensorInfo.float32Tensor[tensorInfo.maxCastDataCount], tensorInfo.float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * tensorInfo.maxCastDataCount], tensorInfo.float32Tensor[tensorInfo.maxCastDataCount],
            RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t> &dataLocal, LocalTensor<bfloat16_t> &outLocal,
        struct TensorInfo &tensorInfo, int64_t dataCount) {
        uint32_t castTimes = tensorInfo.maxCastDataCount == 0 ? -1 : (dataCount / tensorInfo.maxCastDataCount);
        uint32_t castTimesRemainder = tensorInfo.maxCastDataCount == 0 ? -1 : (dataCount % tensorInfo.maxCastDataCount);

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, i, tensorInfo.maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, castTimes, castTimesRemainder);
        }
    }
};

template<>
class InnerComputer<int32_t>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<int32_t> &dataLocal, LocalTensor<int32_t> &outLocal, struct TensorInfo &tensorInfo,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Maxs(dataLocal, dataLocal, -1, dataCount);
        PipeBarrier<PIPE_V>();
        Mins(outLocal, dataLocal, 1, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template<>
class InnerComputer<int8_t>
{
private:
    __aicore__ inline void ComputerPerCast(
        LocalTensor<int8_t> &dataLocal, LocalTensor<int8_t> &outLocal, struct TensorInfo &tensorInfo,
        uint16_t index, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Cast(tensorInfo.float16Tensor, dataLocal[index * tensorInfo.maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Sign<half, false>(tensorInfo.float16Tensor[tensorInfo.maxCastDataCount], tensorInfo.float16Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * tensorInfo.maxCastDataCount], tensorInfo.float16Tensor[tensorInfo.maxCastDataCount],
            RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Compute(
        LocalTensor<int8_t> &dataLocal, LocalTensor<int8_t> &outLocal, struct TensorInfo &tensorInfo,
        int64_t dataCount) {
        uint32_t castTimes = tensorInfo.maxCastDataCount == 0 ? -1 : (dataCount / tensorInfo.maxCastDataCount);
        uint32_t castTimesRemaind = tensorInfo.maxCastDataCount == 0 ? -1 : (dataCount % tensorInfo.maxCastDataCount);
        for (uint32_t i = 0; i < castTimes; i++) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, i, tensorInfo.maxCastDataCount);
        }

        if (castTimesRemaind > 0) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, castTimes, castTimesRemaind);
        }
    }
};

template<>
class InnerComputer<int64_t>
{
private:
    __aicore__ inline void ComputerPerCast(
        LocalTensor<int64_t> &dataLocal, LocalTensor<int64_t> &outLocal, struct TensorInfo &tensorInfo,
        uint16_t index, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Cast(tensorInfo.float32Tensor, dataLocal[index * tensorInfo.maxCastDataCount], RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
        Sign<float, false>(tensorInfo.float32Tensor[tensorInfo.maxCastDataCount], tensorInfo.float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * tensorInfo.maxCastDataCount], tensorInfo.float32Tensor[tensorInfo.maxCastDataCount],
            RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Compute(
        LocalTensor<int64_t> &dataLocal, LocalTensor<int64_t> &outLocal, struct TensorInfo &tensorInfo,
        int64_t dataCounts) {
        uint32_t castTimes = tensorInfo.maxCastDataCount == 0 ? -1 : (dataCounts / tensorInfo.maxCastDataCount);
        uint32_t castTimesRemainder =
            tensorInfo.maxCastDataCount == 0 ? -1 : (dataCounts % tensorInfo.maxCastDataCount);

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, i, tensorInfo.maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputerPerCast(dataLocal, outLocal, tensorInfo, castTimes, castTimesRemainder);
        }
    }
};

template <typename T>
class ForeachSignND {
public:
    __aicore__ inline ForeachSignND(){};
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor, LocalTensor<half> &float16Tensor);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount,
                                            struct TensorInfo &tensorInfo, bool isRemainder);
    __aicore__ inline __gm__ T* GetInputTensorAddr(uint16_t index);
    __aicore__ inline __gm__ T* GetOutputTensorAddr(uint16_t index);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;
    int64_t blockIdx = 0;

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
__aicore__ inline void ForeachSignND<T>::Init(
    GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData) {
    blockIdx = GetBlockIdx();
    inTensorPtr = inputs;
    outTensorPtr = outputs;
    ParseTilingData(tilingData);
    
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int64_t>) {
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT * 2);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        maxCastDataCount = inputsTensorUbSize * INPUT_PARAMETER_COUNT / sizeof(float);
    } else if (std::is_same_v<T, int8_t>){
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT * 2);
        LocalTensor<half> float16Tensor = float32Queue.AllocTensor<half>();
        float32Queue.EnQue(float16Tensor);
        maxCastDataCount = inputsTensorUbSize * INPUT_PARAMETER_COUNT / sizeof(half);
    } else {
        pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
    }
}

template <typename T>
__aicore__ inline void ForeachSignND<T>::Process() {
    LocalTensor<float> float32Tensor;
    LocalTensor<half> float16Tensor;
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int64_t>) {
        float32Tensor = float32Queue.DeQue<float>();
    } else if (std::is_same_v<T, int8_t>) {
        float16Tensor = float32Queue.DeQue<half>();
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
        SingleTensorProcess(dataCount, float32Tensor, float16Tensor);
    }
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int64_t>) {
        float32Queue.FreeTensor(float32Tensor);
    } else if (std::is_same_v<T, int8_t>) {
        float32Queue.FreeTensor(float16Tensor);
    }
}

template <typename T>
__aicore__ inline void ForeachSignND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor, LocalTensor<half> &float16Tensor) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;

    struct TensorInfo tensorInfo;
    tensorInfo.float32Tensor = float32Tensor;
    tensorInfo.float16Tensor = float16Tensor;
    tensorInfo.maxCastDataCount = maxCastDataCount;

    for (uint32_t i = 0; i < copyTimes; i++) {
        CopyIn(i, maxDataCount, false);
        ComputeAndCopyOut(i, maxDataCount, tensorInfo, false);
    }

    if (copyTimesRemainder > 0) {
        CopyIn(copyTimes, copyTimesRemainder, true);
        ComputeAndCopyOut(copyTimes, copyTimesRemainder, tensorInfo, true);
    }
}

template <typename T>
__aicore__ inline void ForeachSignND<T>::ParseTilingData(
    const ForeachCommonTilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachSignND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder) {
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
__aicore__ inline void ForeachSignND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount,
                                                        struct TensorInfo &tensorInfo, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    InnerComputer<T> computer;
    computer.Compute(
      dataLocal, outLocal, tensorInfo,
      dataCount);
    
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
__aicore__ inline __gm__ T* ForeachSignND<T>::GetInputTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

template <typename T>
__aicore__ inline __gm__ T* ForeachSignND<T>::GetOutputTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(outTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

}  // namespace ForeachSign

#endif  // FOREACH_SIGN_H