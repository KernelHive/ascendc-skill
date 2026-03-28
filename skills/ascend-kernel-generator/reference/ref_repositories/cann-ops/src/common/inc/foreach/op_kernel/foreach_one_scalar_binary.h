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
 * \file foreach_one_scalar_binary.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_BINARY_H
#define FOREACH_ONE_SCALAR_BINARY_H

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T>
using OneScalarBinaryOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const T&, const int32_t&);

template <typename T, typename P, OneScalarBinaryOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        const LocalTensor<T> &dataLocal,
        const LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
        T scalarVal,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        op(outLocal, dataLocal, scalarVal, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
    template <OneScalarBinaryOp<float> *op, uint8_t paramsCount>
    class InnerComputer<bfloat16_t, float, op, paramsCount> {
    public:
        __aicore__ inline void Compute(
            const LocalTensor<bfloat16_t> &dataLocal,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            float scalarVal,
            uint32_t maxCastDataCount,
            int64_t dataCount) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;
            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    dataLocal, outLocal, float32Tensor,
                    scalarVal, 
                    maxCastDataCount, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(
                    dataLocal, outLocal, float32Tensor,
                    scalarVal,
                    maxCastDataCount, castTimes, castTimesRemainder);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            const LocalTensor<bfloat16_t> &dataLocal,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            float scalarVal,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, scalarVal, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
        }
    };
#endif

template <typename T, typename P, OneScalarBinaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachOneScalarBinary : public KernelForeachUnary<T, ForeachOneScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = OneScalarBinaryOp<P>;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline ForeachOneScalarBinary() : Base(*this) {};
    using Base::Process;

protected:
    GlobalTensor<DTYPE_SCALAR> inScalarGM;
    #if __CCE_AICORE__ == 220
        using TT = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
        TT scalarVal = 0;
    #else 
        T scalarVal = 0;
    #endif

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            outLocal,
            float32Tensor,
            scalarVal,
            Base::maxCastDataCount,
            dataCount);

        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, OneScalarBinaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    Base::Init(x, y, workspace, tilingData);

    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALAR*)scalar, 1);
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            scalarVal = inScalarGM.GetValue(0);
        } else {
            scalarVal = T(inScalarGM.GetValue(0));
        }
    #else 
        scalarVal = T(inScalarGM.GetValue(0));
    #endif
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_BINARY_H