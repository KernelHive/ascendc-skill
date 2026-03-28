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
 * \file foreach_implict_output.h
 * \brief
 */

#ifndef FOREACH_IMPLICT_OUTPUT
#define FOREACH_IMPLICT_OUTPUT

#include "kernel_foreach_elewise.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;
constexpr bool NEED_TEMP_BUF = false;

template <typename T>
using ImplictOutputOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&, const LocalTensor<T>&);

template <typename T, typename P, ImplictOutputOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        const LocalTensor<P> &tempTensor) {
        op(dataLocal, dataLocal, dataCount, tempTensor);
    }
};

#if __CCE_AICORE__ == 220
    template <ImplictOutputOp<float> *op, uint8_t paramsCount>
    class InnerComputer<bfloat16_t, float, op, paramsCount> {
    public:
        __aicore__ inline void Compute(
            LocalTensor<bfloat16_t> &dataLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount,
            const LocalTensor<float> &tempTensor) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;
            
            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount, tempTensor);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder, tempTensor);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            LocalTensor<bfloat16_t> &dataLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount, const LocalTensor<float> &tempTensor) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, dataCount, tempTensor);
            PipeBarrier<PIPE_V>();
            Cast(dataLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        }
    };
#endif

template <typename T, typename P, ImplictOutputOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needTempBuf=NEED_TEMP_BUF, typename Tiling=ForeachCommonTilingData>
class ForeachImplictOutputV2 : public KernelForeachElewise<T, ForeachImplictOutputV2<T, P, op, bufferNum, paramsCount, needTempBuf, Tiling>, bufferNum, paramsCount, false, Tiling> {
public:
    using Base = KernelForeachElewise<T, ForeachImplictOutputV2<T, P, op, bufferNum, paramsCount, needTempBuf, Tiling>, bufferNum, paramsCount, false, Tiling>;
    using Operator = ImplictOutputOp<P>;

    __aicore__ inline ForeachImplictOutputV2() : Base(*this) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const Tiling* tilingData);
    using Base::Process;

protected:
    // 中间临时空间
    LocalTensor<P> tempTensor;
    TQue<QuePosition::VECIN, 1> tempQueue;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount,
            tempTensor);

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
        if (needTempBuf) {
            tempQueue.DeQue<T>();
        }
    }

    __aicore__ inline void AfterProcess() {
        if (needTempBuf) {
            tempQueue.FreeTensor(tempTensor);
        }
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, ImplictOutputOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needTempBuf, typename Tiling>
__aicore__ inline void ForeachImplictOutputV2<T, P, op, bufferNum, paramsCount, needTempBuf, Tiling>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const Tiling* tilingData) {                           
    Base::Init(x, y, workspace, tilingData);
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_UNARY_H