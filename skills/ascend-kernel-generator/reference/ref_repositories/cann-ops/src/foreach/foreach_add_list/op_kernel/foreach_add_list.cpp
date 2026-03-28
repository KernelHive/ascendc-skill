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
 * \file foreach_add_list.cpp
 * \brief
 */

 #include "kernel_operator.h"

 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
 #include "foreach_one_scalar_ternary.h"
 
 using namespace AscendC;
 using namespace Common::OpKernel;
 
 constexpr uint8_t BYTE_PER_BLOCK = 32;
 
 template <typename T>
 __aicore__ void AddListNormalAdapter(
     const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal1, const LocalTensor<T>& srcLocal2, const T& scalarVal, const int32_t& uValue) {
     Muls(srcLocal2, srcLocal2, scalarVal, uValue);
     PipeBarrier<PIPE_V>();
     Add(dstLocal, srcLocal1, srcLocal2, uValue);
 }
 
 template <typename T>
 __aicore__ void AddListFloatAdapter(
     const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal1, const LocalTensor<T>& srcLocal2, const T& scalarVal, const int32_t& uValue) {
     Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);
     if(dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()){
         PipeBarrier<PIPE_V>();
         if (uValue * sizeof(T) % BYTE_PER_BLOCK == 0) {
             DataCopy(dstLocal, srcLocal1, uValue);
         } else {
             int32_t dataCountInBlock = BYTE_PER_BLOCK / sizeof(T);
             DataCopy(dstLocal, srcLocal1, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
         }
     }
 }
 
extern "C" __global__ __aicore__ void foreach_add_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR alpha, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarTernary<half, half, AddListFloatAdapter<half>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarTernary<float, float, AddListFloatAdapter<float>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ == 220
else if (TILING_KEY_IS(3)) {
    ForeachOneScalarTernary<int, int, AddListNormalAdapter<int>> op;
    op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
    op.Process();
} 
else if (TILING_KEY_IS(4)) {
    ForeachOneScalarTernary<bfloat16_t, float, AddListFloatAdapter<float>> op;
    op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
    op.Process();
}
#endif
}
