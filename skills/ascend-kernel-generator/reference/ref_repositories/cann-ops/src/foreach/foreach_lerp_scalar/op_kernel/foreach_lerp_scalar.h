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
 * \file foreach_lerp_scalar.h
 * \brief
 */

#ifndef FOREACH_LERP_SCALAR_H
#define FOREACH_LERP_SCALAR_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachLerpScalar {
    using namespace AscendC;

    constexpr int32_t BUFFER_NUM = 2;

    constexpr uint8_t COPY_SPACE_MULTIPLE = 9;

    constexpr uint8_t INPUT_PARAMETER_COUNT = 2;

    constexpr float FLOAT_NUM_NEG = -0.5;
    constexpr float FLOAT_NUM_POS = 0.5;
    constexpr float FLOAT_NUM_ONE = 1.0;

    template<typename T>
    class InnerComputer {
    private:
        __aicore__ inline void ComputePerCast(
            LocalTensor<T> &x1Local,
            LocalTensor<T> &x2Local,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, float localWeight,
            uint16_t index, int64_t dataCount) {
                PipeBarrier<PIPE_V>();
                Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
                PipeBarrier<PIPE_V>();
                Cast(float32Tensor[maxCastDataCount], x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
                PipeBarrier<PIPE_V>();
                Sub(float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], float32Tensor, dataCount);
                PipeBarrier<PIPE_V>();
                if (localWeight < FLOAT_NUM_POS && localWeight > FLOAT_NUM_NEG) {
                    Axpy(float32Tensor, float32Tensor[maxCastDataCount], localWeight, dataCount);
                    PipeBarrier<PIPE_V>();
                    Cast(x1Local[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
                } else {
                    localWeight = localWeight - FLOAT_NUM_ONE;
                    PipeBarrier<PIPE_V>();
                    Cast(float32Tensor, x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
                    PipeBarrier<PIPE_V>();
                    Axpy(float32Tensor, float32Tensor[maxCastDataCount], localWeight, dataCount);
                    PipeBarrier<PIPE_V>();
                    Cast(x2Local[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
                }
                
        }
    public:
        __aicore__ inline void Compute(
                LocalTensor<T> &x1Local,
                LocalTensor<T> &x2Local,
                LocalTensor<float> &float32Tensor,
                float weightVal,
                uint32_t maxCastDataCount,
                int64_t dataCount) {
            uint32_t castTimes = 0;
            uint32_t castTimesRemainder = 0;
            if (maxCastDataCount == 0) {
                castTimes = -1;
                castTimesRemainder = -1;
            } else {
                castTimes = dataCount / maxCastDataCount;
                castTimesRemainder = dataCount % maxCastDataCount;
            }

            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    x1Local, x2Local, float32Tensor,
                    maxCastDataCount, weightVal, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(
                    x1Local, x2Local, float32Tensor,
                    maxCastDataCount, weightVal, castTimes, castTimesRemainder);
            }
        }
    };
    template<>
    class InnerComputer<float> {
    public:
        __aicore__ inline void Compute(
            LocalTensor<float> &x1Local,
            LocalTensor<float> &x2Local,
            LocalTensor<float> &float32Tensor,
            float weightVal,
            uint32_t maxCastDataCount,
            int64_t dataCount) { 
            PipeBarrier<PIPE_V>();    
            
            if (weightVal < FLOAT_NUM_POS && weightVal > FLOAT_NUM_NEG) {
                Sub(x2Local, x2Local, x1Local, dataCount);
                PipeBarrier<PIPE_V>();
                Axpy(x1Local, x2Local, weightVal, dataCount);
                PipeBarrier<PIPE_V>();
            } else {
                Sub(x1Local, x2Local, x1Local, dataCount);
                PipeBarrier<PIPE_V>();
                weightVal = weightVal - FLOAT_NUM_ONE; 
                PipeBarrier<PIPE_V>();
                Axpy(x2Local, x1Local, weightVal, dataCount);
                PipeBarrier<PIPE_V>();
            }
        }
    };



    template <typename T>
    class ForeachLerpScalarND {
    public:
        __aicore__ inline ForeachLerpScalarND(){};
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
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
        __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR gmAddr);

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
        TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
        TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

        GlobalTensor<T> x1TensorGM;
        GlobalTensor<T> x2TensorGM;
        GlobalTensor<T> yTensorGM;
        GlobalTensor<DTYPE_WEIGHT> weightGM;

        GM_ADDR x1TensorPtr = nullptr;
        GM_ADDR x2TensorPtr = nullptr;
        GM_ADDR yTensorPtr = nullptr;
        int64_t blockIdx = 0;
        float weightVal = 0.0;

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
    __aicore__ inline void ForeachLerpScalarND<T>::Init(
            GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
            const ForeachCommonTilingData* tilingData) {
        blockIdx = GetBlockIdx();
        x1TensorPtr = x1;
        x2TensorPtr = x2;
        yTensorPtr = y;
        ParseTilingData(tilingData);
        weightGM.SetGlobalBuffer((__gm__ DTYPE_WEIGHT*)weight, 1);

        weightVal = float(weightGM.GetValue(0));

        // Init for bfloat16
        #if __CCE_AICORE__ == 220
        if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            pipe.InitBuffer(x1Queue, BUFFER_NUM, totalTensorUbSize);
            pipe.InitBuffer(x2Queue, BUFFER_NUM, totalTensorUbSize);
            pipe.InitBuffer(yQueue, BUFFER_NUM, totalTensorUbSize);
            maxDataCount = totalTensorUbSize / sizeof(T);
            pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            pipe.InitBuffer(x1Queue, BUFFER_NUM, inputsTensorUbSize);
            pipe.InitBuffer(x2Queue, BUFFER_NUM, inputsTensorUbSize);
            pipe.InitBuffer(yQueue, BUFFER_NUM, inputsTensorUbSize);
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
        #else
        if (std::is_same<T, half>::value) {
            uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            pipe.InitBuffer(x1Queue, BUFFER_NUM, totalTensorUbSize);
            pipe.InitBuffer(x2Queue, BUFFER_NUM, totalTensorUbSize);
            pipe.InitBuffer(yQueue, BUFFER_NUM, totalTensorUbSize);
            maxDataCount = totalTensorUbSize / sizeof(T);
            pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            pipe.InitBuffer(x1Queue, BUFFER_NUM, inputsTensorUbSize);
            pipe.InitBuffer(x2Queue, BUFFER_NUM, inputsTensorUbSize);
            pipe.InitBuffer(yQueue, BUFFER_NUM, inputsTensorUbSize);
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
        #endif
    }

    template <typename T>
    __aicore__ inline void ForeachLerpScalarND<T>::Process() {
        /*将中间量预留出来*/
        LocalTensor<float> float32Tensor;
        #if __CCE_AICORE__ == 220
        if (std::is_same<T, bfloat16_t>::value) {
            float32Tensor = float32Queue.DeQue<float>(); 
        }
        #endif
        if (std::is_same<T, half>::value) {
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
            x1TensorGM.SetGlobalBuffer(GetTensorAddr(i, x1TensorPtr) + cursorStart);
            x2TensorGM.SetGlobalBuffer(GetTensorAddr(i, x2TensorPtr) + cursorStart);
            yTensorGM.SetGlobalBuffer(GetTensorAddr(i, yTensorPtr) + cursorStart);
            SingleTensorProcess(dataCount, float32Tensor);
        }
        #if __CCE_AICORE__ == 220
        if (std::is_same<T, bfloat16_t>::value) {
            float32Queue.FreeTensor(float32Tensor);
        }
        #endif
        if (std::is_same<T, half>::value) {
            float32Queue.FreeTensor(float32Tensor);
        }
    }

    template <typename T>
    __aicore__ inline void ForeachLerpScalarND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor) {
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
    __aicore__ inline void ForeachLerpScalarND<T>::ParseTilingData(
            const ForeachCommonTilingData* tilingData) {
        inputsTensorUbSize = tilingData->inputsTensorUbSize;
        tensorDataCountList = tilingData->tensorDataCountList;
        tensorStart = tilingData->tensorStartList[blockIdx];
        tensorEnd = tilingData->tensorEndList[blockIdx];
        tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
    }

    template <typename T>
    __aicore__ inline void ForeachLerpScalarND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(x1Local, x1TensorGM[index * maxDataCount], copyParams, padParams);
            DataCopyPad(x2Local, x2TensorGM[index * maxDataCount], copyParams, padParams);
        } else {
            DataCopy(x1Local, x1TensorGM[index * maxDataCount], dataCount);
            DataCopy(x2Local, x2TensorGM[index * maxDataCount], dataCount);
        }
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    template <typename T>
    __aicore__ inline void ForeachLerpScalarND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount,
        LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> x1Local = x1Queue.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.DeQue<T>();

        InnerComputer<T> computer;
        computer.Compute(
            x1Local,
            x2Local,
            float32Tensor,
            weightVal,
            maxCastDataCount,
            dataCount);

        if (weightVal < FLOAT_NUM_POS && weightVal > FLOAT_NUM_NEG) {
            x2Queue.FreeTensor(x2Local);
            event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);
            if (isRemainder) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
                DataCopyPad(yTensorGM[index * maxDataCount], x1Local, copyParams);
            } else {
                DataCopy(yTensorGM[index * maxDataCount], x1Local, dataCount);
            }
            event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            x1Queue.FreeTensor(x1Local);
        } else {
            x1Queue.FreeTensor(x1Local);
            event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);
            if (isRemainder) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
                DataCopyPad(yTensorGM[index * maxDataCount], x2Local, copyParams);
            } else {
                DataCopy(yTensorGM[index * maxDataCount], x2Local, dataCount);
            }
            event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            x2Queue.FreeTensor(x2Local);
        }

    }

    template <typename T>
    __aicore__ inline __gm__ T* ForeachLerpScalarND<T>::GetTensorAddr(uint16_t index, GM_ADDR gmAddr) {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(gmAddr);
        uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
    }
}  // namespace ForeachLerpScalar

#endif  // FOREACH_LERP_SCALAR_H