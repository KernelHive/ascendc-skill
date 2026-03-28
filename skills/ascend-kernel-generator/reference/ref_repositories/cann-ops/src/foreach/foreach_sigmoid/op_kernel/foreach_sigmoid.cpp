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
 * \file foreach_sigmoid.cpp
 * \brief
 */

 #include "kernel_operator.h" 
 #include "lib/sigmoid/kernel_operator_sigmoid_intf.h"
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_triangle.h"
 
using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachTriangle<half, half, Sigmoid<half, false>, 2> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachTriangle<float, float, Sigmoid<float, false>, 2> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachTriangle<bfloat16_t, float, Sigmoid<float, false>, 2> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
#endif
}