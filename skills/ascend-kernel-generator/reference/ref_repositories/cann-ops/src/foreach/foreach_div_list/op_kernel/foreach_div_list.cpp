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
 * \file foreach_div_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_no_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;
 
 extern "C" __global__ __aicore__ void foreach_div_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachNoScalarBinary<half, half, Div> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNoScalarBinary<float, float, Div> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
        else if (TILING_KEY_IS(4)) {
            ForeachNoScalarBinary<bfloat16_t, float, Div> op;
            op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
            op.Process();
        }
#endif
}
