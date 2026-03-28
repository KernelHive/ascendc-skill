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
 * \file foreach_zero_inplace.cpp
 * \brief
 */

#include "foreach_zero_inplace.h"

using namespace ForeachZeroInplace;

extern "C" __global__ __aicore__ void foreach_zero_inplace(GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachZeroInplaceND<half> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachZeroInplaceND<float> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachZeroInplaceND<int> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachZeroInplaceND<bfloat16_t> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    } 
#endif
    else if (TILING_KEY_IS(5)) {
        ForeachZeroInplaceND<int16_t> op;
        op.Init(x, userWS, &tilingData);
        op.Process();
    }
}
