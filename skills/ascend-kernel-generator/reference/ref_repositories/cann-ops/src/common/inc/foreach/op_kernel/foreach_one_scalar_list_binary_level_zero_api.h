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
 * \file foreach_one_scalar_list_binary_level_zero_api.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_LIST_BINARY_H_LEVEL_ZERO_API
#define FOREACH_ONE_SCALAR_LIST_BINARY_H_LEVEL_ZERO_API

#define DTYPE_SCALAR  DTYPE_SCALARS

#include "foreach_one_scalar_binary_level_zero_api.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachOneScalarListBinaryLevelZeroApi : public KernelForeachUnary<T, ForeachOneScalarListBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarListBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = OneScalarBinaryLevelZeroApiOp<P>;
    __aicore__ inline ForeachOneScalarListBinaryLevelZeroApi() : Base(*this) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
            const ForeachCommonTilingData* tilingData);
    using Base::Process;

protected:
    TBuf<QuePosition::VECCALC> scalarOneBlockBuf;
    GlobalTensor<DTYPE_SCALARS> inScalarGM;
    LocalTensor<P> scalarOneBlockLM;
    uint64_t elementsPerRepeat = BYTES_PER_REPEAT / sizeof(P);
    P scalarVal = 0;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            outLocal,
            float32Tensor,
            scalarOneBlockLM,
            Base::maxCastDataCount,
            dataCount,
            elementsPerRepeat);

        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {
        scalarVal = P(inScalarGM.GetValue(index));
        Duplicate(scalarOneBlockLM, scalarVal, BYTES_PER_BLOCK / sizeof(P));
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> returnLocal = Base::outQueue.template DeQue<T>();
        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], returnLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], returnLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::outQueue.FreeTensor(returnLocal);
        return true;
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    friend Base;
};

template <typename T, typename P, OneScalarBinaryLevelZeroApiOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarListBinaryLevelZeroApi<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    Base::Init(x, y, workspace, tilingData);
    Base::pipe.InitBuffer(scalarOneBlockBuf, BYTES_PER_BLOCK);
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALARS*)scalar, 1);
    scalarOneBlockLM = scalarOneBlockBuf.template Get<P>();
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_LIST_BINARY_H_LEVEL_ZERO_API