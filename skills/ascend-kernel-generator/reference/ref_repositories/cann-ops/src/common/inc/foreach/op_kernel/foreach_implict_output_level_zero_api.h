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
 * \file foreach_implict_output_level_zero_api.h
 * \brief
 */

#ifndef FOREACH_IMPLICT_OUTPUT_LEVEL_ZERO_API
#define FOREACH_IMPLICT_OUTPUT_LEVEL_ZERO_API

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int16_t MAX_REPEATS = 255;
constexpr int16_t BYTES_PER_REPEAT = 256;
constexpr int8_t BYTES_PER_BLOCK = 32;
constexpr int8_t STRIDES_PER_REPEAT = 8;

template <typename T>
using ImplictOutputLevelZeroApiOp = 
    void (const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, 
    uint64_t, const uint8_t, const BinaryRepeatParams&);

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        LocalTensor<P> &oneBlockData,
        uint64_t elementsPerRepeat) {        
        uint32_t totalRepeats = 0;
        uint32_t divisible = 0;
        if (elementsPerRepeat == 0) {
            totalRepeats = -1;
            divisible = -1;
        } else {
            totalRepeats = dataCount / elementsPerRepeat;
            divisible = dataCount % elementsPerRepeat;
        }
        uint32_t outerRepeats = totalRepeats / MAX_REPEATS;

        uint32_t offset = 0;
        for (uint32_t i = 0; i < outerRepeats; i ++) {
            op(dataLocal[offset], oneBlockData, dataLocal[offset], elementsPerRepeat, MAX_REPEATS,
                {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (dataCount - (outerRepeats * MAX_REPEATS * elementsPerRepeat) > 0) {
            uint8_t curRepeat = totalRepeats - outerRepeats * MAX_REPEATS;
            if (curRepeat > 0) {
                op(dataLocal[offset], oneBlockData, dataLocal[offset], elementsPerRepeat, curRepeat,
                    {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
                offset += curRepeat * elementsPerRepeat;
            }

            if (divisible > 0) {
                uint32_t remain = dataCount - elementsPerRepeat * totalRepeats;
                op(dataLocal[offset], oneBlockData, dataLocal[offset], remain, 1,
                    {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            }
        }
    }
};

#if __CCE_AICORE__ == 220
template <ImplictOutputLevelZeroApiOp<float> *op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount> {
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t> &dataLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        LocalTensor<float> oneBlockData,
        uint64_t elementsPerRepeat) {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(
                dataLocal, float32Tensor,
                maxCastDataCount, i, maxCastDataCount, oneBlockData, elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(
                dataLocal, float32Tensor,
                maxCastDataCount, castTimes, castTimesRemainder, oneBlockData, elementsPerRepeat);       
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<bfloat16_t> &dataLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, uint32_t index, int64_t dataCount,
        LocalTensor<float> oneBlockData,
        uint64_t elementsPerRepeat) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat; // should calc
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS; // limit by L0 API, should calc
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS; // should calc

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, MAX_REPEATS, {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();   
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], totalRepeatCntRemainder, 1, {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
        }
        
        PipeBarrier<PIPE_V>();
        Cast(dataLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};
#endif

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT>
class ForeachImplictOutputLevelZeroApi : public KernelForeachUnary<T, ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum, paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum, paramsCount, false>;
    using Operator = ImplictOutputLevelZeroApiOp<P>;

    __aicore__ inline ForeachImplictOutputLevelZeroApi() : Base(*this) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData, P duplicatedNum);
    using Base::Process;

protected:
    LocalTensor<P> scalarOneBlockUB;
    // for repeat in one block
    TQue<QuePosition::VECIN, 1> scalarOneBlockQueue;
    uint64_t elementsPerRepeat = BYTES_PER_REPEAT / sizeof(P);

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();

        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount,
            scalarOneBlockUB,
            elementsPerRepeat);

        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void BeforeProcess() {
        scalarOneBlockQueue.DeQue<T>();
    }

    __aicore__ inline void AfterProcess() {
        scalarOneBlockQueue.FreeTensor(scalarOneBlockUB);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P> *op, int32_t bufferNum, uint8_t paramsCount>
__aicore__ inline void ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData, P duplicatedNum) {
    Base::Init(x, y, workspace, tilingData);
    Base::pipe.InitBuffer(scalarOneBlockQueue, 1, BYTES_PER_BLOCK);
    scalarOneBlockUB = scalarOneBlockQueue.AllocTensor<P>();
    Duplicate(scalarOneBlockUB, duplicatedNum, BYTES_PER_BLOCK / sizeof(P));
    scalarOneBlockQueue.EnQue(scalarOneBlockUB);
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_UNARY_H