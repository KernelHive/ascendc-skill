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
 * \file foreach_one_scalar_quaternary.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_QUATERNARY_H
#define FOREACH_ONE_SCALAR_QUATERNARY_H

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr uint32_t FACTOR_FOR_CAST = 2;

template <typename T>
using OneScalarQuaternaryOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const T&, const int32_t&);

template <typename T, typename P, OneScalarQuaternaryOp<P> *op>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &inLocal_1, 
        LocalTensor<T> &inLocal_2, 
        LocalTensor<T> &inLocal_3, 
        LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
        T scalarVal,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        op(outLocal, inLocal_1, inLocal_2, inLocal_3, scalarVal, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
template <OneScalarQuaternaryOp<float> *op>
class InnerComputer<bfloat16_t, float, op> {
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t> &inLocal_1,
        LocalTensor<bfloat16_t> &inLocal_2,
        LocalTensor<bfloat16_t> &inLocal_3,
        LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        bfloat16_t scalarVal,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(
                inLocal_1, inLocal_2, inLocal_3, outLocal, float32Tensor,
                scalarVal, maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(
                inLocal_1, inLocal_2, inLocal_3, outLocal, float32Tensor,
                scalarVal, maxCastDataCount, castTimes, castTimesRemainder);
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<bfloat16_t> &inLocal_1,
        LocalTensor<bfloat16_t> &inLocal_2,
        LocalTensor<bfloat16_t> &inLocal_3,
        LocalTensor<bfloat16_t> &outLocal,
        LocalTensor<float> &float32Tensor,
        bfloat16_t scalarVal, uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor[maxCastDataCount * FACTOR_FOR_CAST], inLocal_3[index * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
        PipeBarrier<PIPE_V>();
        op(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount * FACTOR_FOR_CAST], ToFloat(scalarVal), dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, maxCastDataCount);
        PipeBarrier<PIPE_V>();
    }
};
#endif

template <typename T, typename P, OneScalarQuaternaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachOneScalarQuaternary : public KernelForeachUnary<T, ForeachOneScalarQuaternary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarQuaternary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = OneScalarQuaternaryOp<P>;
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline ForeachOneScalarQuaternary() : Base(*this) {};
    using Base::Process;

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_2;
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_3;
    GlobalTensor<T> inTensorsGM_2;
    GlobalTensor<T> inTensorsGM_3;
    GlobalTensor<T> inScalarGM;
    GM_ADDR inTensorsPtr_2 = nullptr;
    GM_ADDR inTensorsPtr_3 = nullptr;
    T scalarVal = 0;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> inLocal_1 = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> inLocal_2 = InQueue_2.DeQue<T>();
        LocalTensor<T> inLocal_3 = InQueue_3.DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op> computer;
        computer.Compute(
            inLocal_1, inLocal_2, inLocal_3, outLocal, float32Tensor,
            scalarVal, Base::maxCastDataCount, dataCount);
        
        Base::outQueue.template EnQue<T>(outLocal);
        Base::dataQueue.FreeTensor(inLocal_1);
        InQueue_2.FreeTensor(inLocal_2);
        InQueue_3.FreeTensor(inLocal_3);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> inLocal_2 = InQueue_2.AllocTensor<T>();
        LocalTensor<T> inLocal_3 = InQueue_3.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], copyParams, padParams);
            DataCopyPad(inLocal_3, inTensorsGM_3[index * Base::maxDataCount], copyParams, padParams);
        } else {
            DataCopy(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], dataCount);
            DataCopy(inLocal_3, inTensorsGM_3[index * Base::maxDataCount], dataCount);
        } 
        InQueue_2.EnQue(inLocal_2);
        InQueue_3.EnQue(inLocal_3);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {
        scalarVal = inScalarGM.GetValue(index);
        inTensorsGM_2.SetGlobalBuffer(Base::GetTensorAddr(index, inTensorsPtr_2) + cursorStart);
        inTensorsGM_3.SetGlobalBuffer(Base::GetTensorAddr(index, inTensorsPtr_3) + cursorStart);
    }

    friend Base;
};

template <typename T, typename P, OneScalarQuaternaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarQuaternary<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    Base::Init(x1, y, workspace, tilingData);
    inTensorsPtr_2 = x2;
    inTensorsPtr_3 = x3;

    inScalarGM.SetGlobalBuffer((__gm__ T*)scalar, 1);
    #if __CCE_AICORE__ == 220
    if (std::is_same_v<T, bfloat16_t>) {
        uint64_t totalTensorUbSize = Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        Base::pipe.InitBuffer(InQueue_2, bufferNum, totalTensorUbSize);
        Base::pipe.InitBuffer(InQueue_3, bufferNum, totalTensorUbSize);
    } else {
        Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
        Base::pipe.InitBuffer(InQueue_3, bufferNum, Base::inputsTensorUbSize);
    }
    #else
    Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
    Base::pipe.InitBuffer(InQueue_3, bufferNum, Base::inputsTensorUbSize);
    #endif
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_QUATERNARY_H