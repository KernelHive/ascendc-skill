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
 * \file foreach_triangle.h
 * \brief
 */

#ifndef FOREACH_TRIANGLE_H
#define FOREACH_TRIANGLE_H

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T>
using TriangleOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const uint32_t);

template <typename T, typename P, TriangleOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        const LocalTensor<T> &x1Local,
        const LocalTensor<T> &yLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        op(yLocal, x1Local, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
    template <TriangleOp<float> *op, uint8_t paramsCount>
    class InnerComputer<bfloat16_t, float, op, paramsCount> {
    public:
        __aicore__ inline void Compute(
            const LocalTensor<bfloat16_t> &x1Local,
            const LocalTensor<bfloat16_t> &yLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;

            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    x1Local, yLocal, float32Tensor,
                    maxCastDataCount, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(x1Local, yLocal, float32Tensor,
                    maxCastDataCount, castTimes, castTimesRemainder);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            const LocalTensor<bfloat16_t> &x1Local,
            const LocalTensor<bfloat16_t> &yLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        }
    };
#endif

template <typename T, typename P, TriangleOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachTriangle : public KernelForeachUnary<T, ForeachTriangle<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachTriangle<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = TriangleOp<P>;

    __aicore__ inline ForeachTriangle() : Base(*this) {};
    using Base::Init;
    using Base::Process;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            outLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount);

        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_UNARY_H