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
 * \file foreach_copy.h
 * \brief
 */
#ifndef FOREACH_COPY_N_D_H
#define FOREACH_COPY_N_D_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachCopy {
    using namespace AscendC;

    constexpr int32_t BUFFER_NUM = 1;

    template<typename T>

    class ForeachCopyND {
    public:
        __aicore__ inline ForeachCopyND(){};
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                    const ForeachCommonTilingData* tilingData);
        __aicore__ inline void Process();

    private:
        template <typename T1, typename T2>
        __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
            if (b == 0) {
                return a;
            }
            return (a + b - 1) / b;
        };
        __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
        __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
        __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
        __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount,
            LocalTensor<float> &float32Tensor, bool isRemainder);
        __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;

        GlobalTensor<T> inTensorsGM;
        GlobalTensor<T> outTensorsGM;

        GM_ADDR inTensorsPtr = nullptr;
        GM_ADDR outTensorsPtr = nullptr;

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
    __aicore__ inline void ForeachCopyND<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
            const ForeachCommonTilingData* tilingData) {
        blockIdx = GetBlockIdx();
        inTensorsPtr = x;
        outTensorsPtr = y;
        ParseTilingData(tilingData);
        pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
    }

    template <typename T>
    __aicore__ inline void ForeachCopyND<T>::Process() {
        /*将中间量预留出来*/
        LocalTensor<float> float32Tensor;

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
            outTensorsGM.SetGlobalBuffer(GetTensorAddr(i, outTensorsPtr) + cursorStart);
            SingleTensorProcess(dataCount, float32Tensor);
        }
    }

    template <typename T>
    __aicore__ inline void ForeachCopyND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor) {
        // Batch handling and calculation.
        uint32_t copyTimes = dataCount / maxDataCount;
        uint32_t copyTimesRemainder = dataCount % maxDataCount;
        uint32_t tempDataCount = maxDataCount;

        if (copyTimesRemainder > 0) {
            copyTimes++;
        }

        for (uint32_t i = 0; i < copyTimes; i++) {
            bool isRemainder = false;
            if (i == copyTimes -1 && copyTimesRemainder > 0) {
                tempDataCount = copyTimesRemainder;
                isRemainder = true;
            }
            CopyIn(i, tempDataCount, isRemainder);
            ComputeAndCopyOut(i, tempDataCount, float32Tensor, isRemainder);
        }
    }

    template <typename T>
    __aicore__ inline void ForeachCopyND<T>::ParseTilingData(
            const ForeachCommonTilingData* tilingData) {
        inputsTensorUbSize = tilingData->inputsTensorUbSize;
        tensorDataCountList = tilingData->tensorDataCountList;
        tensorStart = tilingData->tensorStartList[blockIdx];
        tensorEnd = tilingData->tensorEndList[blockIdx];
        tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
    }

    template <typename T>
    __aicore__ inline void ForeachCopyND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(dataLocal, inTensorsGM[index * maxDataCount], copyParams, padParams);
        } else {
            DataCopy(dataLocal, inTensorsGM[index * maxDataCount], dataCount);
        }
        dataQueue.EnQue(dataLocal);
    }

    template <typename T>
    __aicore__ inline void ForeachCopyND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = dataQueue.DeQue<T>();

        event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);

        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPad(outTensorsGM[index * maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(outTensorsGM[index * maxDataCount], dataLocal, dataCount);
        }

        event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        dataQueue.FreeTensor(dataLocal);
    }

    template <typename T>
    __aicore__ inline __gm__ T* ForeachCopyND<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);  // notice: diff with abs
        return reinterpret_cast<__gm__ T*>(*(retPtr + index));
    }
}  // namespace ForeachCopy

#endif  // FOREACH_COPY_N_D_H