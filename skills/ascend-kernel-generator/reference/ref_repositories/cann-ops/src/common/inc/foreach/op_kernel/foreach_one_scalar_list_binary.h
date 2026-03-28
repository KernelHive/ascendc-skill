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
 * \file foreach_one_scalar_list_binary.h
 * \brief
 */

#ifndef FOREACH_ONE_SCALAR_LIST_BINARY_H
#define FOREACH_ONE_SCALAR_LIST_BINARY_H

#define DTYPE_SCALAR  DTYPE_SCALARS

#include "kernel_foreach_unary.h"
#include "foreach_one_scalar_binary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T, typename P, OneScalarBinaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachOneScalarListBinary : public KernelForeachUnary<T, ForeachOneScalarListBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachOneScalarListBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = OneScalarBinaryOp<P>;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline ForeachOneScalarListBinary() : Base(*this) {};
    using Base::Process;

protected:
    GlobalTensor<DTYPE_SCALARS> inScalarGM;
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

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {
#if __CCE_AICORE__ == 220
            if (std::is_same_v<T, bfloat16_t>) {
                scalarVal = inScalarGM.GetValue(index);
            } else {
                scalarVal = T(inScalarGM.GetValue(index));
            }
#else 
            scalarVal = T(inScalarGM.GetValue(index));
#endif
    }

    friend Base;
};

template <typename T, typename P, OneScalarBinaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachOneScalarListBinary<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, const ForeachCommonTilingData* tilingData) {
    Base::Init(x, y, workspace, tilingData);
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALARS*)scalar, 1);
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_ONE_SCALAR_LIST_BINARY_H
