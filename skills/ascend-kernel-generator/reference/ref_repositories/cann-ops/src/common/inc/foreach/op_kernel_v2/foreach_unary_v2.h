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
 * \file foreach_unary.h
 * \brief
 */

#ifndef FOREACH_UNARY_V2_H
#define FOREACH_UNARY_V2_H

#include "kernel_foreach_elewise.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;
constexpr bool NEED_TEMP_BUF = false;

template <typename T>
using UnaryOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const uint32_t, const LocalTensor<T>&);

template <typename T, typename P, UnaryOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        const LocalTensor<T> &x1Local,
        const LocalTensor<T> &yLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount,
        const LocalTensor<P> &tempTensor) {
        PipeBarrier<PIPE_V>();
        op(yLocal, x1Local, dataCount, tempTensor);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
    template <UnaryOp<float> *op, uint8_t paramsCount>
    class InnerComputer<bfloat16_t, float, op, paramsCount> {
    public:
        __aicore__ inline void Compute(
            const LocalTensor<bfloat16_t> &x1Local,
            const LocalTensor<bfloat16_t> &yLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount,
            const LocalTensor<float> &tempTensor) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;

            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    x1Local, yLocal, float32Tensor,
                    maxCastDataCount, i, maxCastDataCount, tempTensor);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(x1Local, yLocal, float32Tensor,
                    maxCastDataCount, castTimes, castTimesRemainder, tempTensor);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            const LocalTensor<bfloat16_t> &x1Local,
            const LocalTensor<bfloat16_t> &yLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount, const LocalTensor<float> &tempTensor) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, dataCount, tempTensor);
            PipeBarrier<PIPE_V>();
            Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        }
    };
#endif

template <typename T, typename P, UnaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT,
          bool needCopyOut=NEED_COPY_OUT, bool needTempBuf=NEED_TEMP_BUF, typename Tiling=ForeachCommonTilingData>
class ForeachUnaryV2 : public KernelForeachElewise<T, ForeachUnaryV2<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling>, bufferNum, paramsCount, needCopyOut, Tiling> {
public:
    using Base = KernelForeachElewise<T, ForeachUnaryV2<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling>, bufferNum, paramsCount, needCopyOut, Tiling>;
    using Operator = UnaryOp<P>;

    __aicore__ inline ForeachUnaryV2() : Base(*this) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const Tiling* tilingData);
    using Base::Process;

protected:
    // 中间临时空间
    LocalTensor<P> tempTensor;
    TQue<QuePosition::VECIN, 1> tempQueue;

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
            dataCount,
            tempTensor);

        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<T>(outLocal);
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

template <typename T, typename P, UnaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut, bool needTempBuf, typename Tiling>
__aicore__ inline void ForeachUnaryV2<T, P, op, bufferNum, paramsCount, needCopyOut, needTempBuf, Tiling>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const Tiling* tilingData) {                           
    Base::Init(x, y, workspace, tilingData);
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_UNARY_V2_H