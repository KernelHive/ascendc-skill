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
 * \file foreach_addcdiv_scalar.cpp
 * \brief
 */

#include "kernel_operator.h"

 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_quaternary_implict_output.h"
 
using namespace AscendC;
using namespace Common::OpKernel;
 
template <typename T>
__aicore__ void AddcDivScalarAdapterForFloat(
    const LocalTensor<T>& tensor1Local, 
    const LocalTensor<T>& tensor2Local,
    const LocalTensor<T>& tensor3Local,
    const LocalTensor<T>& float32Tensor,
    const T scalarVal,
    const uint32_t maxCastDataCount,
    const int64_t dataCount) {
    Div(tensor2Local, tensor2Local, tensor3Local, dataCount);
    PipeBarrier<PIPE_V>();
    Axpy<T, T>(tensor1Local, tensor2Local, scalarVal, dataCount);
}
 
template <typename T>
__aicore__ void ComputerPerCastForAddcDivScalar(
        const LocalTensor<T> &tensor1Local, 
        const LocalTensor<T> &tensor2Local, 
        const LocalTensor<T> &tensor3Local, 
        const LocalTensor<float> &float32Tensor,
        const float scalarVal, const uint32_t maxCastDataCount, 
        const uint32_t index, const int64_t dataCount) {
    Cast(float32Tensor, tensor2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor[maxCastDataCount], tensor3Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor[maxCastDataCount * 2], tensor1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    // input + scalar_tensor * (tensor1 / tensor2)
    PipeBarrier<PIPE_V>();
    Div(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
    PipeBarrier<PIPE_V>();
    Axpy<float, float>(float32Tensor[maxCastDataCount * 2], float32Tensor, scalarVal, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(tensor1Local[index * maxCastDataCount], float32Tensor[maxCastDataCount * 2], RoundMode::CAST_RINT, dataCount);
}
 
template <typename T>
__aicore__ void AddcDivScalarAdapter(
    const LocalTensor<T>& tensor1Local, 
    const LocalTensor<T>& tensor2Local,
    const LocalTensor<T>& tensor3Local,
    const LocalTensor<float>& float32Tensor,
    const float scalarVal,
    const uint32_t maxCastDataCount,
    const int64_t dataCount) {
    
    uint32_t castTimes = dataCount / maxCastDataCount;
    uint32_t castTimesRemainder = dataCount % maxCastDataCount;
    for (uint32_t i = 0; i < castTimes; i++) {
        ComputerPerCastForAddcDivScalar<T>(
            tensor1Local, tensor2Local, tensor3Local,
            float32Tensor, scalarVal, maxCastDataCount, i, maxCastDataCount);
    }
    if (castTimesRemainder > 0) {
        ComputerPerCastForAddcDivScalar<T>(
            tensor1Local, tensor2Local, tensor3Local,
            float32Tensor, scalarVal, maxCastDataCount, castTimes, castTimesRemainder);
    }
}
 
extern "C" __global__ __aicore__ void foreach_addcdiv_scalar(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3, GM_ADDR scalar,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarQuaternaryImplictOutput<half, AddcDivScalarAdapter<half>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarQuaternaryImplictOutput<float, AddcDivScalarAdapterForFloat<float>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachOneScalarQuaternaryImplictOutput<bfloat16_t, AddcDivScalarAdapter<bfloat16_t>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}
 