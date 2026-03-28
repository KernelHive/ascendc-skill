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
 * \file foreach_lerp_scalar.cpp
 * \brief
 */

#include "foreach_lerp_scalar.h"

using namespace ForeachLerpScalar;

extern "C" __global__ __aicore__ void foreach_lerp_scalar(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachLerpScalarND<half> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachLerpScalarND<float> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachLerpScalarND<bfloat16_t> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    }
    #endif
}
