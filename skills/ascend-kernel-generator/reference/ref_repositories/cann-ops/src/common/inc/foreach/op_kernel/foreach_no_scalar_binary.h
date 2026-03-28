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
 * \file foreach_no_scalar_binary.h
 * \brief
 */

#ifndef FOREACH_NO_SCALAR_BINARY_H
#define FOREACH_NO_SCALAR_BINARY_H

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T>
using NoScalarBinaryOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&);

template <typename T, typename P, NoScalarBinaryOp<P> *op>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &inLocal_1,
        LocalTensor<T> &inLocal_2,
        const LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        op(outLocal, inLocal_1, inLocal_2, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
    template <NoScalarBinaryOp<float> *op>
    class InnerComputer<bfloat16_t, float, op> {
    public:
        __aicore__ inline void Compute(
            LocalTensor<bfloat16_t> &inLocal_1,
            LocalTensor<bfloat16_t> &inLocal_2,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;
            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    inLocal_1, inLocal_2, outLocal, float32Tensor,
                    maxCastDataCount, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(
                    inLocal_1, inLocal_2, outLocal, float32Tensor,
                    maxCastDataCount, castTimes, castTimesRemainder);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            LocalTensor<bfloat16_t> &inLocal_1,
            LocalTensor<bfloat16_t> &inLocal_2,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            op(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        }
    };
#endif

template <typename T, typename P, NoScalarBinaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachNoScalarBinary : public KernelForeachUnary<T, ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = NoScalarBinaryOp<P>;
    __aicore__ inline void Init(GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline ForeachNoScalarBinary() : Base(*this) {};
    using Base::Process;

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_2;
    GlobalTensor<T> inTensorsGM_2;
    GM_ADDR inTensorsPtr_2 = nullptr;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> inLocal_2 = InQueue_2.DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op> computer;
        computer.Compute(
            dataLocal,
            inLocal_2, 
            outLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount);

        Base::dataQueue.FreeTensor(dataLocal);
        InQueue_2.FreeTensor(inLocal_2);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> inLocal_2 = InQueue_2.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], copyParams, padParams);
        } else {
            DataCopy(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], dataCount);
        } 
        InQueue_2.EnQue(inLocal_2);
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {
        inTensorsGM_2.SetGlobalBuffer(Base::GetTensorAddr(index, inTensorsPtr_2) + cursorStart);
    }

    friend Base;
};

template <typename T, typename P, NoScalarBinaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData) {
    Base::Init(inputs_1, y, workspace, tilingData);
    inTensorsPtr_2 = inputs_2;
    #if __CCE_AICORE__ == 220
        if (std::is_same<T, bfloat16_t>::value) {
            uint64_t totalTensorUbSize = Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            Base::pipe.InitBuffer(InQueue_2, bufferNum, totalTensorUbSize);
        } else {
            Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
        }
    #else 
        Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
    #endif
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_NO_SCALAR_BINARY_H