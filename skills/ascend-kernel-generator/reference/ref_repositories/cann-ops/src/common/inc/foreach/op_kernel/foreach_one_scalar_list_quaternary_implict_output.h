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
 * \file foreach_one_scalar_list_quaternary_implict_output.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_LIST_QUATERNARY_IMPLICT_OUTPUT_H
#define FOREACH_ONE_SCALAR_LIST_QUATERNARY_IMPLICT_OUTPUT_H

#include "foreach_one_scalar_quaternary_implict_output.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T, OneScalarQuaternaryImplictOutputOp<T> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT>
class ForeachOneScalarListQuaternaryImplictOutput : public KernelForeachUnary<T, ForeachOneScalarListQuaternaryImplictOutput<T, op, bufferNum, paramsCount>, bufferNum, paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarListQuaternaryImplictOutput<T, op, bufferNum, paramsCount>, bufferNum, paramsCount, false>;
    using Operator = OneScalarQuaternaryImplictOutputOp<T>;
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline ForeachOneScalarListQuaternaryImplictOutput() : Base(*this) {};

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_2;
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_3;
    GlobalTensor<T> inTensorsGM_2;
    GlobalTensor<T> inTensorsGM_3;
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    GM_ADDR inTensorsPtr_2 = nullptr;
    GM_ADDR inTensorsPtr_3 = nullptr;
    float scalarValue = 0.0;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> inLocal1 = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> inLocal2 = InQueue_2.DeQue<T>();
        LocalTensor<T> inLocal3 = InQueue_3.DeQue<T>();

        InnerComputer<T, op> computer;
        computer.Compute(
            inLocal1, inLocal2, inLocal3, float32Tensor,
            scalarValue, Base::maxCastDataCount, dataCount);
        
        InQueue_2.FreeTensor(inLocal2);
        InQueue_3.FreeTensor(inLocal3);

        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], inLocal1, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], inLocal1, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        Base::dataQueue.FreeTensor(inLocal1);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> inLocal2 = InQueue_2.AllocTensor<T>();
        LocalTensor<T> inLocal3 = InQueue_3.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal2, inTensorsGM_2[index * Base::maxDataCount], copyParams, padParams);
            DataCopyPad(inLocal3, inTensorsGM_3[index * Base::maxDataCount], copyParams, padParams);
        } else {
            DataCopy(inLocal2, inTensorsGM_2[index * Base::maxDataCount], dataCount);
            DataCopy(inLocal3, inTensorsGM_3[index * Base::maxDataCount], dataCount);
        } 
        InQueue_2.EnQue(inLocal2);
        InQueue_3.EnQue(inLocal3);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, OneScalarQuaternaryImplictOutputOp<T> *op, int32_t bufferNum, uint8_t paramsCount>
__aicore__ inline void ForeachOneScalarListQuaternaryImplictOutput<T, op, bufferNum, paramsCount>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    Base::Base::blockIdx = GetBlockIdx();
    Base::Base::ParseTilingData(tilingData);
    Base::inTensorsPtr = x1;
    inTensorsPtr_2 = x2;
    inTensorsPtr_3 = x3;
    Base::outTensorsPtr = y;
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);

    #if __CCE_AICORE__ == 220
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        uint64_t totalTensorUbSize = Base::Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, totalTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_2, bufferNum, totalTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_3, bufferNum, totalTensorUbSize);
        Base::Base::maxDataCount = totalTensorUbSize / sizeof(T);

        Base::Base::pipe.InitBuffer(Base::float32Queue, 1, Base::Base::inputsTensorUbSize * paramsCount);
        LocalTensor<float> float32Tensor = Base::float32Queue.template AllocTensor<float>();
        Base::float32Queue.EnQue(float32Tensor);
        Base::Base::maxCastDataCount = Base::Base::inputsTensorUbSize / sizeof(float);
    } else {
        Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_3, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::maxDataCount = Base::Base::inputsTensorUbSize / sizeof(T);
    }
    #else
    if (std::is_same_v<T, half>) {
        uint64_t totalTensorUbSize = Base::Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, totalTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_2, bufferNum, totalTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_3, bufferNum, totalTensorUbSize);
        Base::Base::maxDataCount = totalTensorUbSize / sizeof(T);

        Base::Base::pipe.InitBuffer(Base::float32Queue, 1, Base::Base::inputsTensorUbSize * paramsCount);
        LocalTensor<float> float32Tensor = Base::float32Queue.template AllocTensor<float>();
        Base::float32Queue.EnQue(float32Tensor);
        Base::Base::maxCastDataCount = Base::Base::inputsTensorUbSize / sizeof(float);
    } else {
        Base::Base::pipe.InitBuffer(Base::dataQueue, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::pipe.InitBuffer(InQueue_3, bufferNum, Base::Base::inputsTensorUbSize);
        Base::Base::maxDataCount = Base::Base::inputsTensorUbSize / sizeof(T);
    }
    #endif
}

template <typename T, OneScalarQuaternaryImplictOutputOp<T> *op, int32_t bufferNum, uint8_t paramsCount>
__aicore__ inline void ForeachOneScalarListQuaternaryImplictOutput<T, op, bufferNum, paramsCount>::Process() {
    LocalTensor<float> float32Tensor;
    #if __CCE_AICORE__ == 220
    if (std::is_same_v<T, bfloat16_t>) {
        float32Tensor = Base::float32Queue.template DeQue<float>(); 
    }
    #endif
    if (std::is_same_v<T, half>) {
        float32Tensor = Base::float32Queue.template DeQue<float>(); 
    }
    for (uint16_t i = Base::Base::tensorStart; i <= Base::Base::tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = Base::Base::tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == Base::Base::tensorStart) {
            cursorStart = Base::Base::tensorStartOffset;
        }
        if (i == Base::Base::tensorEnd) {
            cursorEnd = Base::Base::tensorEndOffset;
        }
        dataCount = cursorEnd - cursorStart + 1;
        scalarValue = float(inScalarGM.GetValue(i));
        Base::inTensorsGM.SetGlobalBuffer(Base::Base::GetTensorAddr(i, Base::inTensorsPtr) + cursorStart);
        inTensorsGM_2.SetGlobalBuffer(Base::Base::GetTensorAddr(i, inTensorsPtr_2) + cursorStart);
        inTensorsGM_3.SetGlobalBuffer(Base::Base::GetTensorAddr(i, inTensorsPtr_3) + cursorStart);
        Base::outTensorsGM.SetGlobalBuffer(Base::Base::GetTensorAddr(i, Base::outTensorsPtr) + cursorStart);
        Base::SingleTensorProcess(dataCount, float32Tensor);
    }
    #if __CCE_AICORE__ == 220
    if (std::is_same_v<T, bfloat16_t>) {
        Base::float32Queue.template FreeTensor(float32Tensor);
    }
    #endif
    if (std::is_same_v<T, half>) {
        Base::float32Queue.template FreeTensor(float32Tensor);
    }
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_LIST_QUATERNARY_IMPLICT_OUTPUT_H