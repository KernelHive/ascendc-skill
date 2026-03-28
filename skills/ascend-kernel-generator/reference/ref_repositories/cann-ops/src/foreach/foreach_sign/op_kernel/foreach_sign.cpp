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
 * \file foreach_sign.cpp
 * \brief
 */

 #include "foreach_sign.h"

 using namespace ForeachSign;
 
 extern "C" __global__ __aicore__ void foreach_sign(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, 
     GM_ADDR tiling) {
     GET_TILING_DATA(tilingData, tiling);
 
     //foreach(vector) not need workspace
     GM_ADDR userWS = nullptr;
 
     if (TILING_KEY_IS(1)) {
         ForeachSignND<half> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(2)) {
         ForeachSignND<float> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(3)) {
         ForeachSignND<int32_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(4)) {
         ForeachSignND<bfloat16_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(7)) {
         ForeachSignND<int8_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(10)) {
         ForeachSignND<int64_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     }
 }