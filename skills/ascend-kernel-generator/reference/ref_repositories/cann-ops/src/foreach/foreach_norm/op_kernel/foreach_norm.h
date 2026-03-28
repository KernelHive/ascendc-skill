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
 * \file foreach_norm.h
 * \brief
 */

 
#ifndef FOREACH_NORM_N_D_H
#define FOREACH_NORM_N_D_H

#include "kernel_operator.h"

namespace ForeachNorm {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;
constexpr uint8_t COPY_SPACE_MULTIPLE = 9;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint8_t NORM_MODEL_CODE = 2;

template <typename T1, typename T2>
__aicore__ inline T1 CeilA2B(T1 a, T2 b) {
    return (a + b - 1) / b;
};

template <typename T>
__aicore__ inline void SetValueAdapter(LocalTensor<T> & outLocal, float value, uint16_t index) {
    outLocal.SetValue(index, T(value));
};

template <>
__aicore__ inline void SetValueAdapter<bfloat16_t>(LocalTensor<bfloat16_t> & outLocal, float value, uint16_t index) {
    outLocal.SetValue(index, ToBfloat16(value));
};

// modelCode:
// 0 ord=0 Calculate the count of not-zero element in each tensor. (not used now)
// 1 ord=1 AbsAndNotNeedPower NotNeedSqrt (now is **default** as we now only consider ord=1 || ord=2)
// 2 ord=2 MulSelf(we don't need abs this time) Sqrt(not Power(self,1/ord))
// 3 ord=+inf Calculate the max Abs(element) in each tensor. (not used now)
// 4 ord=-inf Calculate the min Abs(element) in each tensor. (not used now)
// 5 ord=else... the default operator(not used now)

// this is actually ord=1
template <typename P, uint8_t modelCode>
class NormAdapter {
    public:
    __aicore__ inline void AbsAndPowerAdapt(LocalTensor<P> &dataLocal, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Abs(dataLocal, dataLocal, dataCount);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ReciprocalPowerAdapt(LocalTensor<P> &dataLocal, LocalTensor<P> &outLocal, int64_t dataCount) {
        uint64_t mask = 64;
        uint32_t repeatTimes = 4;
        CopyRepeatParams copyRepeatParams{1, 1, 8, 8};
        Copy(outLocal, dataLocal, mask, repeatTimes, copyRepeatParams);
    }
};

template <typename P>
class NormAdapter<P,NORM_MODEL_CODE> {
    public:
    __aicore__ inline void AbsAndPowerAdapt(LocalTensor<P> &dataLocal, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Mul(dataLocal, dataLocal, dataLocal, dataCount);
        PipeBarrier<PIPE_V>(); 
    }
    __aicore__ inline void ReciprocalPowerAdapt(LocalTensor<P> &dataLocal, LocalTensor<P> &outLocal, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Sqrt(outLocal, dataLocal, dataCount);
        PipeBarrier<PIPE_V>(); 
    }
};

template<typename T, typename P, uint8_t modelCode>
class InnerComputer {
private:
    __aicore__ inline void SquareAndReduceRound1ComputePerCast(NormAdapter<float, modelCode> &normAdapter,
        LocalTensor<T> &dataLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, uint16_t index, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        normAdapter.AbsAndPowerAdapt(float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(float32Tensor, float32Tensor, float32Tensor, dataCount);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<float>(float32Tensor, float32Tensor.GetValue(0), maxCastDataCount + index);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
public:
    __aicore__ inline void SquareAndReduceRound1Compute(NormAdapter<float, modelCode> &normAdapter,
        LocalTensor<T> &dataLocal, LocalTensor<P> &tempLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, int64_t dataCount, uint16_t tempIndex) {
        uint32_t castTimes = 0;
        uint32_t castDatacountRemainder = 0;
        if (maxCastDataCount == 0) {
            castTimes = -1;
            castDatacountRemainder = -1;
        } else {
            castTimes = dataCount / maxCastDataCount;
            castDatacountRemainder = dataCount % maxCastDataCount;
        }

        for (uint32_t i = 0; i < castTimes; i++) {
            SquareAndReduceRound1ComputePerCast(
            normAdapter, dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
        }
        if (castDatacountRemainder > 0) {
            SquareAndReduceRound1ComputePerCast(
            normAdapter, dataLocal, float32Tensor, maxCastDataCount, castTimes, castDatacountRemainder);
            castTimes++;
        }
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(float32Tensor, float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], castTimes);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<P>(tempLocal, float32Tensor.GetValue(0), tempIndex);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    __aicore__ inline void ReduceRound2AndSqrtCompute(NormAdapter<float, modelCode> &normAdapter,
        LocalTensor<P> &dataLocal, LocalTensor<T> &outLocal, int64_t dataCount) {
        if (dataCount > 1) {
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(dataLocal, dataLocal, dataLocal, dataCount);
        }
        PipeBarrier<PIPE_V>();
        normAdapter.ReciprocalPowerAdapt(dataLocal, dataLocal, 1);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, dataLocal, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_V>();
    }
};

template<typename P, uint8_t modelCode>
class InnerComputer<float, P, modelCode> {
public:
    __aicore__ inline void SquareAndReduceRound1Compute(NormAdapter<float, modelCode> &normAdapter,
        LocalTensor<float> &dataLocal, LocalTensor<P> &tempLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, int64_t dataCount, uint16_t tempIndex) {
        PipeBarrier<PIPE_V>();
        normAdapter.AbsAndPowerAdapt(dataLocal, dataCount);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(dataLocal, dataLocal, dataLocal, dataCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<P>(tempLocal, dataLocal.GetValue(0), tempIndex);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
    __aicore__ inline void ReduceRound2AndSqrtCompute(NormAdapter<float, modelCode> &normAdapter,
        LocalTensor<float> &dataLocal, LocalTensor<float> &outLocal, int64_t dataCount) {
        if (dataCount > 1) {
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(dataLocal, dataLocal, dataLocal, dataCount);
        }
        PipeBarrier<PIPE_V>();
        normAdapter.ReciprocalPowerAdapt(dataLocal, outLocal, 1);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T, typename P, uint8_t modelCode>
class ForeachNormND {
public:
    __aicore__ inline ForeachNormND(){};
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace,
                                const ForeachReduceTilingData* tilingData) {
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        inTensorPtr = inputs;
        outTensorPtr = output;
        workTensorPtr = workspace;
        ParseTilingData(tilingData);

        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>( & ((__gm__ P*)workTensorPtr)[MAX_CORE_CONT + MAX_TENSOR_CONT]), MAX_CORE_CONT * DEFAULT_SYNCALL_NEED_SIZE);
        constexpr int32_t EACH_CORE_HANDLE_NUM = BYTE_BLOCK / sizeof(int32_t);

        workTensorGM.SetGlobalBuffer( (__gm__ P*)workTensorPtr, MAX_CORE_CONT+MAX_TENSOR_CONT);

        if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
            pipe.InitBuffer(outQueue, BUFFER_NUM, BYTE_BLOCK);
            maxDataCount = totalTensorUbSize / sizeof(T);
            pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
            pipe.InitBuffer(outQueue, BUFFER_NUM, BYTE_BLOCK);
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
        pipe.InitBuffer(calcBuf, byteLen);
    }

    __aicore__ inline void Process() {
        // Stage1 Square and ReduceRound1
        if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            float32Tensor = float32Queue.DeQue<float>(); 
        }

        for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
            if (tensorDataCountList[i] == 0) {
                continue;
            }
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
            inTensorGM.SetGlobalBuffer(GetTensorAddr(i,inTensorPtr) + cursorStart);

            // coreMiddleOffset : describe this core's offset for middle value of tensor
            SingleTensorProcess(dataCount, coreMiddleOffset+i-tensorStart);
        }

        // Sync All Cores
        uint16_t flagId = 1;
        constexpr uint8_t mode = 0;
        CrossCoreSetFlag<mode, PIPE_MTE3>(flagId);
        CrossCoreWaitFlag(flagId);

        // Stage2 Reduce2 and sqrt
        for (uint16_t i = blockIdx; i<totalTensorCount; i+=needCoreNum) {
            outTensorGM.SetGlobalBuffer(GetTensorAddr(i,outTensorPtr));
            if (tensorDataCountList[i] == 0) {
                OutputZero();
                continue;
            }
            CopyInStage2(tensorMiddleCountList[i], tensorMiddleStartList[i]);
            ReduceRound2AndSqrt(tensorMiddleCountList[i], tensorMiddleStartList[i]);
        }
        if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
            float32Queue.FreeTensor(float32Tensor);
        }
    }

private:
    __aicore__ inline void ParseTilingData(const ForeachReduceTilingData* tilingData) {
        inputsTensorUbSize = tilingData->inputsTensorUbSize;
        needCoreNum = tilingData->needCoreNum;
        totalTensorCount = tilingData->totalTensorCount;
        tensorDataCountList = tilingData->tensorDataCountList;
        tensorStart = tilingData->tensorStartList[blockIdx];
        tensorEnd = tilingData->tensorEndList[blockIdx];
        tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
        // for reduce
        tensorMiddleStartList = tilingData->tensorMiddleStartList;
        tensorMiddleCountList = tilingData->tensorMiddleCountList;
        coreMiddleOffset = tilingData->coreMiddleOffsetList[blockIdx];
    }

    __aicore__ inline void SingleTensorProcess(int64_t dataCount, uint16_t offset) {
        // Batch handling and calculation.
        uint32_t copyTimes = dataCount / maxDataCount;
        uint32_t datacountRemainder = dataCount % maxDataCount;

        if (datacountRemainder > 0) {
            copyTimes++;
        }
        LocalTensor<P> tempLocal = calcBuf.Get<P>(CeilA2B(copyTimes, BYTE_BLOCK / sizeof(P)) * BYTE_BLOCK / sizeof(P));
        uint32_t tempDataCount=maxDataCount;
        for (uint32_t i = 0; i < copyTimes; i++) {
            if (i == copyTimes - 1 && datacountRemainder > 0) {
                tempDataCount = datacountRemainder;
            }
            CopyInStage1(i,tempDataCount);
            SquareAndReduceRound1(i, tempDataCount, tempLocal);
        }

        PipeBarrier<PIPE_V>();
        ReduceSum<P>(tempLocal, tempLocal, tempLocal, copyTimes);
        PipeBarrier<PIPE_V>();

        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(P)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位       
        DataCopyPad(workTensorGM[offset], tempLocal, copyParams);

        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    }

    // CopyIn, Compute and CopyOut
    __aicore__ inline void CopyInStage1(uint16_t index, int64_t dataCount) {
        LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorGM[index * maxDataCount], copyParams, padParams);

        dataQueue.EnQue(dataLocal);
    }

    __aicore__ inline void SquareAndReduceRound1(uint16_t index, int64_t dataCount, LocalTensor<P> & tempLocal) {
        LocalTensor<T> dataLocal = dataQueue.DeQue<T>();

        computer.SquareAndReduceRound1Compute(
            normAdapter,
            dataLocal,
            tempLocal,
            float32Tensor,
            maxCastDataCount,
            dataCount,
            index);

        dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void CopyInStage2(uint16_t dataCount, uint16_t offset) {
        LocalTensor<P> dataLocal = dataQueue.AllocTensor<P>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(P)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
        DataCopyPadExtParams<P> padParams{true, 0, 0, 0};
        DataCopyPad(dataLocal, workTensorGM[offset], copyParams, padParams);

        dataQueue.EnQue(dataLocal);
    }

    __aicore__ inline void ReduceRound2AndSqrt(uint16_t dataCount, uint16_t offset) {
        LocalTensor<P> dataLocal = dataQueue.DeQue<P>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        computer.ReduceRound2AndSqrtCompute(normAdapter, dataLocal, outLocal, dataCount);

        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);

        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
        DataCopyPad(outTensorGM, outLocal, copyParams2);

        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        dataQueue.FreeTensor(dataLocal);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void OutputZero() {
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        SetValueAdapter(outLocal, float(0.0), 0);
        
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);

        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);

        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位       
        DataCopyPad(outTensorGM, outLocal, copyParams2);

        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(retPtr + index));
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> calcBuf;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<P> workTensorGM;
    GlobalTensor<int32_t> syncGlobal;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;
    GM_ADDR workTensorPtr = nullptr;
    GM_ADDR bufferTensorPtr = nullptr;

    uint64_t blockNum = 0;
    uint64_t blockIdx = 0;
    uint32_t byteLen = 1024;

    uint32_t maxDataCount = 0;
    // tiling params
    uint64_t inputsTensorUbSize = 0;
    uint16_t needCoreNum = 0;
    uint16_t totalTensorCount = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};
    // tiling param for Reduce Op
    const uint16_t* tensorMiddleCountList = nullptr;
    const uint16_t* tensorMiddleStartList = nullptr;
    uint16_t coreMiddleOffset = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;
    LocalTensor<float> float32Tensor;
    uint32_t maxCastDataCount = {0};
    InnerComputer<T, P, modelCode> computer;
    NormAdapter<P, modelCode> normAdapter;
};

} // namespace ForeachNorm

#endif // FOREACH_NORM_N_D_H