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
 * \file foreach_reduce.h
 * \brief
 */

 
#ifndef KERNEL_FOREACH_REDUCE_N_D_H
#define KERNEL_FOREACH_REDUCE_N_D_H

#include "kernel_foreach_base_v2.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;

template <typename T>
__aicore__ inline void SetValueAdapter(LocalTensor<T> & outLocal, float value, uint16_t index) {
    outLocal.SetValue(index, T(value));
};

template <>
__aicore__ inline void SetValueAdapter<bfloat16_t>(LocalTensor<bfloat16_t> & outLocal, float value, uint16_t index) {
    outLocal.SetValue(index, ToBfloat16(value));
};

template <typename T, typename P, typename Predicate, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, typename Tiling=ForeachReduceTilingData>
class KernelForeadhReduce : public KernelForeachBaseV2<T, Tiling>{
protected:
    using Base = KernelForeachBaseV2<T, Tiling>;

    explicit __aicore__ inline KernelForeadhReduce(Predicate &p): Base(), pred(p) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const Tiling* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, uint16_t offset);
private:
    __aicore__ inline void CopyInStage1(uint32_t index, int64_t dataCount);
    __aicore__ inline void CopyInFromWorkspace(uint16_t dataCount, uint16_t offset);
    __aicore__ inline void Copy2Workspace(uint32_t index, int64_t dataCount);
    __aicore__ inline void ComputeRound1(uint16_t index, int64_t dataCount, LocalTensor<P>& tempLocal);
    __aicore__ inline void ComputeRound2(uint16_t dataCount, uint16_t offset);
    __aicore__ inline void ReduceCompute(LocalTensor<P>& dstLocal, LocalTensor<P>& srcLocal, LocalTensor<P>& workLocal, int32_t count);
    __aicore__ inline void InitCoreParams(GM_ADDR x);
    __aicore__ inline void OutputZero();

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount);
    __aicore__ inline void CopyInPlusStage1(uint32_t index, int64_t dataCount);
    __aicore__ inline void BeforeProcess();
    __aicore__ inline void AfterProcess();
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart);

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> calcBuf;
    TQue<QuePosition::VECIN, 1> float32Queue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<P> workTensorGM;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;
    GM_ADDR workTensorPtr = nullptr;

    LocalTensor<float> float32Tensor;

    uint32_t byteLen = 1024;

    uint32_t maxCastDataCount = {0};
private:
    Predicate &pred;
};

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const Tiling* tilingData) {
    Base::Init(tilingData);
    Base::ParseReduceTilingData(tilingData);

    inTensorPtr = x;
    outTensorPtr = y;
    workTensorPtr = workspace;
    workTensorGM.SetGlobalBuffer( (__gm__ P*)workTensorPtr, MAX_CORE_CONT + MAX_TENSOR_CONT);

    if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        uint64_t totalTensorUbSize = Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        Base::pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        Base::pipe.InitBuffer(outQueue, BUFFER_NUM, BYTE_BLOCK);
        Base::maxDataCount = totalTensorUbSize / sizeof(T);
        Base::pipe.InitBuffer(float32Queue, 1, Base::inputsTensorUbSize * INPUT_PARAMETER_COUNT);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        maxCastDataCount = Base::inputsTensorUbSize / sizeof(float);
    } else {
        Base::pipe.InitBuffer(dataQueue, BUFFER_NUM, Base::inputsTensorUbSize);
        Base::pipe.InitBuffer(outQueue, BUFFER_NUM, BYTE_BLOCK);
        Base::maxDataCount = Base::inputsTensorUbSize / sizeof(T);
    }
    Base::pipe.InitBuffer(calcBuf, byteLen);
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void  KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::Process() {
     /*将中间量预留出来*/
    if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        float32Tensor = float32Queue.DeQue<float>(); 
    }

    BeforeProcess();

    // stage1
    for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
        if ( Base::tensorDataCountList[i] == 0) {
            continue;
        }
        int64_t cursorStart = 0;
        int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == Base::tensorStart) {
            cursorStart = Base::tensorStartOffset;
        }
        if (i == Base::tensorEnd) {
            cursorEnd = Base::tensorEndOffset;
        }

        dataCount = cursorEnd - cursorStart + 1;

        inTensorGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorPtr) + cursorStart);
        ProcessPlusInLoop(i, cursorStart);
        // coreMiddleOffset : describe this core's offset for middle value of tensor
        SingleTensorProcess(dataCount, Base::coreMiddleOffset + i - Base::tensorStart);
    }

    AfterProcess();

    // Sync All Cores
    uint16_t flagId = 1;
    constexpr uint8_t mode = 0;
    CrossCoreSetFlag<mode, PIPE_MTE3>(flagId);
    CrossCoreWaitFlag(flagId);

    // Stage2 
    for (uint16_t i = Base::blockIdx; i < Base::totalTensorCount; i += Base::needCoreNum) {
        outTensorGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorPtr));
        if (Base::tensorDataCountList[i] == 0) {
            OutputZero();
            continue;
        }
        CopyInFromWorkspace(Base::tensorMiddleCountList[i], Base::tensorMiddleStartList[i]);
        ComputeRound2(Base::tensorMiddleCountList[i], Base::tensorMiddleStartList[i]);
    }

    if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        float32Queue.FreeTensor(float32Tensor);
    }
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::SingleTensorProcess(int64_t dataCount, uint16_t offset) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / Base::maxDataCount;
    uint32_t datacountRemainder = dataCount % Base::maxDataCount;

    if (datacountRemainder > 0) {
        copyTimes++;
    }

    int32_t tempLocalCount = Base::CeilA2B(copyTimes, BYTE_BLOCK / sizeof(P)) * BYTE_BLOCK / sizeof(P);
    LocalTensor<P> tempLocal = calcBuf.Get<P>(Base::CeilA2B(copyTimes, BYTE_BLOCK / sizeof(P)) * BYTE_BLOCK / sizeof(P));
    uint32_t tempDataCount = Base::maxDataCount;
    for (uint32_t i = 0; i < copyTimes; i++) {
        if (i == copyTimes - 1 && datacountRemainder > 0) {
            tempDataCount = datacountRemainder;
        }
        CopyInStage1(i, tempDataCount);
        CopyInPlusStage1(i, tempDataCount);
        ComputeRound1(i, tempDataCount, tempLocal);
    }

    ReduceCompute(tempLocal, tempLocal, tempLocal, copyTimes);

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(P)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位       
    DataCopyPad(workTensorGM[offset], tempLocal, copyParams);

    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::CopyInStage1(uint32_t index, int64_t dataCount) {
    LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dataLocal, inTensorGM[index * Base::maxDataCount], copyParams, padParams);

    dataQueue.EnQue(dataLocal);
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::CopyInFromWorkspace(uint16_t dataCount, uint16_t offset) {
    LocalTensor<P> dataLocal = dataQueue.AllocTensor<P>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(P)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
    DataCopyPadExtParams<P> padParams{true, 0, 0, 0};
    DataCopyPad(dataLocal, workTensorGM[offset], copyParams, padParams);

    dataQueue.EnQue(dataLocal);
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::Copy2Workspace(uint32_t index, int64_t dataCount) {
    return;
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::ComputeRound1(uint16_t index, int64_t dataCount, LocalTensor<P>& tempLocal) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::ComputeRound1)>) {
        pred.ComputeRound1(index, dataCount, tempLocal);
    }
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::ComputeRound2(uint16_t dataCount, uint16_t offset) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::ComputeRound2)>) {
        pred.ComputeRound2(dataCount, offset);
    }
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::ReduceCompute(LocalTensor<P>& dstLocal, LocalTensor<P>& srcLocal, LocalTensor<P>& workLocal, int32_t count) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::ReduceCompute)>);
    pred.ReduceCompute(dstLocal, srcLocal, workLocal, count);
}

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::OutputZero() {
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

template <typename T, typename P, typename Predicate , int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline bool KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::CopyOut(uint32_t index, int64_t dataCount) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::CopyOut)>) {
        pred.CopyOut(index, dataCount);
    }
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::CopyInPlusStage1(uint32_t index, int64_t dataCount) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::CopyInPlusStage1)>) {
        pred.CopyInPlusStage1(index, dataCount);
    }
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::BeforeProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::BeforeProcess)>) {
        pred.BeforeProcess();
    }
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::AfterProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::AfterProcess)>) {
        pred.AfterProcess();
    }
}

template <typename T, typename P, typename Predicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void KernelForeadhReduce<T, P, Predicate, bufferNum, paramsCount, Tiling>::ProcessPlusInLoop(uint32_t index, uint64_t cursorStart){
    if (std::is_member_function_pointer_v<decltype(&Predicate::ProcessPlusInLoop)>) {
        pred.ProcessPlusInLoop(index, cursorStart);
    }
}
}  // namespace OpKernel
}  // namespace Common

#endif // KERNEL_FOREACH_REDUCE_N_D_H