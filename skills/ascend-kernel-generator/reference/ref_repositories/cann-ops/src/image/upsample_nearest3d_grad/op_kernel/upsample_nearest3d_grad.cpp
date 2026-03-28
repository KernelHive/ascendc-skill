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
 * \file upsample_nearest3d_grad.cpp
 * \brief
 */

#include "upsample_nearest3d_grad.h"

using namespace UpsampleNearest3dGrad;

extern "C" __global__ __aicore__ void upsample_nearest3d_grad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const UpsampleNearest3dGradTilingData *__restrict tiling_data = &tilingData;
    const TCubeTiling *__restrict matmulTilingWTiling = &(tiling_data->matmulTilingW);
    const TCubeTiling *__restrict matmulTilingHTiling = &(tiling_data->matmulTilingH);
    const TCubeTiling *__restrict matmulTilingDTiling = &(tiling_data->matmulTilingD);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#define INIT_AND_PROCESS                       \
    REGIST_MATMUL_OBJ(&op.pipe,                \
        GetSysWorkSpacePtr(),                  \
        op.matmulW,                            \
        matmulTilingWTiling,                   \
        op.matmulH,                            \
        matmulTilingHTiling,                   \
        op.matmulD,                            \
        matmulTilingDTiling);                  \
    op.Init(x, y, false, userWS, &tilingData); \
    op.Process()

    if (TILING_KEY_IS(1)) {
        if (tiling_data->dataType == 1) {
            UpsampleNearest3dGradND<half> op;
            INIT_AND_PROCESS;
        } else if (tiling_data->dataType == 2) {
            UpsampleNearest3dGradND<float> op;
            INIT_AND_PROCESS;
        } else if (tiling_data->dataType == 3) {
            UpsampleNearest3dGradND<bfloat16_t> op;
            INIT_AND_PROCESS;
        }
    }
}
