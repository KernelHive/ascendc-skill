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
 * \file foreach_one_scalar_binary_level_zero_api.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_BINARY_LEVEL_ZERO_API
#define FOREACH_ONE_SCALAR_BINARY_LEVEL_ZERO_API

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int16_t MAX_REPEATS = 255;
constexpr int16_t BYTES_PER_REPEAT = 256;
constexpr int8_t BYTES_PER_BLOCK = 32;

template <typename T>
using OneScalarBinaryLevelZeroApiOp = 
    void (const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, 
    uint64_t, const uint8_t, const BinaryRepeatParams&);

template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal,
	    LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
	    LocalTensor<P> &oneBlockData,
	    uint32_t maxCastDataCount,
	    int64_t dataCount,
        uint64_t elementsPerRepeat) {
        uint32_t totalRepeatCnt = 0;
        uint32_t totalRepeatCntRemainder = 0;
        if (elementsPerRepeat == 0) {
            totalRepeatCnt = 0;
            totalRepeatCntRemainder = 0;
        } else {
            totalRepeatCnt = dataCount / elementsPerRepeat;
            totalRepeatCntRemainder = dataCount % elementsPerRepeat; // should calc
        }
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS; // limit by L0 API, should calc
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS; // should calc
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            op(outLocal[offset], dataLocal[offset], oneBlockData, elementsPerRepeat, MAX_REPEATS, {1, 1, 0, 8, 8, 0});
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            op(outLocal[offset], dataLocal[offset], oneBlockData, elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 0, 8, 8, 0});
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            op(outLocal[offset], dataLocal[offset], oneBlockData, totalRepeatCntRemainder, 1, {1, 1, 0, 8, 8, 0});
        }
    }
};

template <OneScalarBinaryLevelZeroApiOp<float> *op, uint8_t paramsCount>
class InnerComputer<half, float, op, paramsCount> {
public: 
    __aicore__ inline void Compute(
        LocalTensor<half> &dataLocal,
        LocalTensor<half> &outLocal,
        LocalTensor<float> &float32Tensor,
        LocalTensor<float> oneBlockData,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        uint64_t elementsPerRepeat) {
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
                dataLocal, outLocal, float32Tensor, oneBlockData,
                maxCastDataCount, i, maxCastDataCount, elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(
                dataLocal, outLocal, float32Tensor, oneBlockData,
                maxCastDataCount, castTimes, castTimesRemainder, elementsPerRepeat);       
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<half> &dataLocal,
	    LocalTensor<half> &outLocal,
        LocalTensor<float> &float32Tensor,
        LocalTensor<float> &oneBlockData,
        uint32_t maxCastDataCount, 
        uint32_t index,
        int64_t dataCount,
        uint64_t elementsPerRepeat) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = 0;
        uint32_t totalRepeatCntRemainder = 0;

        if (elementsPerRepeat == 0) {
            totalRepeatCnt = 0;
            totalRepeatCntRemainder = 0;
        } else {
            totalRepeatCnt = dataCount / elementsPerRepeat;
            totalRepeatCntRemainder = dataCount % elementsPerRepeat;
        }

        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, elementsPerRepeat, MAX_REPEATS, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, totalRepeatCntRemainder, 1, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
template <OneScalarBinaryLevelZeroApiOp<float> *op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount> {
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t> &dataLocal,
        LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        LocalTensor<float> oneBlockData,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        uint64_t elementsPerRepeat) {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(
                dataLocal, outLocal, float32Tensor, oneBlockData,
                maxCastDataCount, i, maxCastDataCount, elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(
                dataLocal, outLocal, float32Tensor, oneBlockData,
                maxCastDataCount, castTimes, castTimesRemainder, elementsPerRepeat);       
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<bfloat16_t> &dataLocal,
	    LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        LocalTensor<float> &oneBlockData,
        uint32_t maxCastDataCount, 
        uint32_t index,
        int64_t dataCount,
        uint64_t elementsPerRepeat) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat;
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, elementsPerRepeat, MAX_REPEATS, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor[offset], oneBlockData, totalRepeatCntRemainder, 1, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};
#endif


template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachOneScalarBinaryLevelZeroApi : public KernelForeachUnary<T, ForeachOneScalarBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = OneScalarBinaryLevelZeroApiOp<P>;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

    __aicore__ inline ForeachOneScalarBinaryLevelZeroApi() : Base(*this) {};

protected:
    TBuf<QuePosition::VECCALC> scalarOneBlockBuf;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    LocalTensor<P> scalarOneBlockLM;
    uint64_t elementsPerRepeat = BYTES_PER_REPEAT / sizeof(P);
    P scalarValue = 0;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> inLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();
        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            inLocal,
            outLocal,
            float32Tensor,
            scalarOneBlockLM,
            Base::maxCastDataCount,
            dataCount,
            elementsPerRepeat);
        Base::dataQueue.FreeTensor(inLocal);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> retLocal = Base::outQueue.template DeQue<T>();

        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], retLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], retLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::outQueue.FreeTensor(retLocal);
        return true;
    }
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}
    friend Base;
};

template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    if (
	#if __CCE_AICORE__ == 220
	std::is_same_v<T, bfloat16_t> || 
	#endif
	std::is_same_v<T, half>) {
        Base::inTensorsPtr = x;
        Base::outTensorsPtr = y;
        Base::Base::blockIdx = GetBlockIdx();
        Base::Base::ParseTilingData(tilingData);
        Base::Base::totalTensorUbSize = Base::Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        Base::Base::maxDataCount = Base::Base::totalTensorUbSize / sizeof(T);        
        Base::Base::maxCastDataCount = Base::Base::inputsTensorUbSize / sizeof(float);
    
        Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, Base::Base::totalTensorUbSize);
        if (needCopyOut) {
            Base::Base::pipe.InitBuffer(Base::outQueue, bufferNum, Base::Base::totalTensorUbSize);
        }
        Base::Base::pipe.InitBuffer(Base::float32Queue, 1, Base::Base::inputsTensorUbSize * paramsCount);
        LocalTensor<float> float32Tensor = Base::float32Queue.template AllocTensor<float>();
        Base::float32Queue.EnQue(float32Tensor);
    } else {
        Base::Init(x, y, workspace, tilingData);
    }

    Base::pipe.InitBuffer(scalarOneBlockBuf, BYTES_PER_BLOCK);
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    scalarValue = P(inScalarGM.GetValue(0));
    scalarOneBlockLM = scalarOneBlockBuf.Get<P>();
    Duplicate(scalarOneBlockLM, scalarValue, BYTES_PER_BLOCK / sizeof(P));
}

template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>::Process() {
    if (
    #if __CCE_AICORE__ == 220
    std::is_same_v<T, bfloat16_t> || 
    #endif
    std::is_same_v<T, half>) {
        LocalTensor<float> float32Tensor;
        float32Tensor = Base::float32Queue.template DeQue<float>(); 

        BeforeProcess();
        for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
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
            Base::inTensorsGM.SetGlobalBuffer(Base::Base::GetTensorAddr(i, Base::inTensorsPtr) + cursorStart);
            Base::outTensorsGM.SetGlobalBuffer(Base::Base::GetTensorAddr(i, Base::outTensorsPtr) + cursorStart);
            Base::SingleTensorProcess(dataCount, float32Tensor);
        }
        AfterProcess();
        Base::float32Queue.FreeTensor(float32Tensor);
    } else {
        Base::Process();
    }
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_BINARY_LEVEL_ZERO_API