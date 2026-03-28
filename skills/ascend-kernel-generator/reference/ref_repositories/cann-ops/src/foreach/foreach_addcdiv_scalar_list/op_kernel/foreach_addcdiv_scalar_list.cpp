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
 * \file foreach_addcdiv_scalar_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_quaternary.h"
 
using namespace AscendC;
using namespace Common::OpKernel;
 
constexpr uint8_t BYTE_PER_BLOCK = 32;
 
template <typename T>
__aicore__ void AddcDivListFloatAdapter(
    const LocalTensor<T>& dstLocal, 
    const LocalTensor<T>& tensor1Local,
    const LocalTensor<T>& tensor2Local,
    const LocalTensor<T>& tensor3Local,
    const T& scalarVal,
    const int32_t& uValue) {
    Div(tensor2Local, tensor2Local, tensor3Local, uValue);
    PipeBarrier<PIPE_V>();
    Axpy<T, T>(tensor1Local, tensor2Local, scalarVal, uValue);
    if(dstLocal.GetPhyAddr() != tensor1Local.GetPhyAddr()){
        PipeBarrier<PIPE_V>();
        if (uValue * sizeof(T) % BYTE_PER_BLOCK == 0) {
            DataCopy(dstLocal, tensor1Local, uValue);
        } else {
            int32_t dataCountInBlock = BYTE_PER_BLOCK / sizeof(T);
            DataCopy(dstLocal, tensor1Local, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
        }
    }
}
 
extern "C" __global__ __aicore__ void foreach_addcdiv_scalar_list(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3, GM_ADDR scalar,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarQuaternary<half, half, AddcDivListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarQuaternary<float, float, AddcDivListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachOneScalarQuaternary<bfloat16_t, float, AddcDivListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}
 