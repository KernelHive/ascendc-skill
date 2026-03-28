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
 * \file upsample_trilinear3d_backward.cpp
 * \brief
 */
#include "upsample_trilinear3d_backward.h"

using namespace UpsampleTrilinear3dBackward;

extern "C" __global__ __aicore__ void upsample_trilinear3d_backward(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const UpsampleTrilinear3dBackwardTilingData *__restrict tilingData = &tiling_data;
    const TCubeTiling *__restrict matmulTilingW = &(tilingData->matmulTilingW);
    const TCubeTiling *__restrict matmulTilingH = &(tilingData->matmulTilingH);
    const TCubeTiling *__restrict matmulTilingD = &(tilingData->matmulTilingD);
    GM_ADDR userWs = GetUserWorkspace(workspace);
#define INIT_AND_PROCESS                          \
    REGIST_MATMUL_OBJ(&op.pipe,                   \
        GetSysWorkSpacePtr(),                     \
        op.matmulW,                               \
        matmulTilingW,                            \
        op.matmulH,                               \
        matmulTilingH,                            \
        op.matmulD,                               \
        matmulTilingD);                           \
    op.Init(input, output, userWs, &tiling_data); \
    op.Process()

    if (TILING_KEY_IS(1)) {
        if (tilingData->dataType == 1) {
            UpsampleTrilinear3dBackwardND<half> op;
            INIT_AND_PROCESS;
        } else if (tilingData->dataType == 2) {
            UpsampleTrilinear3dBackwardND<float> op;
            INIT_AND_PROCESS;
        } else if (tilingData->dataType == 3) {
            UpsampleTrilinear3dBackwardND<bfloat16_t> op;
            INIT_AND_PROCESS;
        }
    }
}
