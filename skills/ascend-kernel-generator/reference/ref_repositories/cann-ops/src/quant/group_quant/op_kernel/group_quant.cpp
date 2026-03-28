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
 * \file group_quant.cpp
 * \brief
 */

#include "group_quant_base.h"

using namespace GroupQuant;

extern "C" __global__ __aicore__ void group_quant(GM_ADDR x, GM_ADDR scale, GM_ADDR groupIndex, GM_ADDR offset,
                                                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        // The dtype of input offset must be same as scale.
        // Input offset is optional, so set offset dtype by scale dtype.
        GroupQuant::GroupQuantBase<DTYPE_X, DTYPE_SCALE, DTYPE_GROUP_INDEX, DTYPE_SCALE, DTYPE_Y> op;
        op.Init(x, scale, groupIndex, offset, y, &tilingData);
        op.Process();
    }
}