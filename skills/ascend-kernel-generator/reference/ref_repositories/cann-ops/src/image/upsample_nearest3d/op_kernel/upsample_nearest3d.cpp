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
 * \file upsample_nearest3d.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200
#include "upsample_nearest3d_310p.h"
#else
#include "upsample_nearest3d.h"
#endif

using namespace UpsampleNearest3d;

extern "C" __global__ __aicore__ void upsample_nearest3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const UpsampleNearest3dTilingData *__restrict tiling_data = &tilingData;

    GM_ADDR userWS = GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                       \
    op.Init(x, y, false, userWS, &tilingData); \
    op.Process()

#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(1)) {
        if (tiling_data->dataType == 1) {
            UpsampleNearest3dND310p<half> op;
            INIT_AND_PROCESS;
        } else if (tiling_data->dataType == 2) {
            UpsampleNearest3dND310p<float> op;
            INIT_AND_PROCESS;
        }
    }
#else
    if (TILING_KEY_IS(1)) {
        if (tiling_data->dataType == 1) {
            UpsampleNearest3dND<half> op;
            INIT_AND_PROCESS;
        } else if (tiling_data->dataType == 2) {
            UpsampleNearest3dND<float> op;
            INIT_AND_PROCESS;
        } else if (tiling_data->dataType == 3) {
            UpsampleNearest3dND<bfloat16_t> op;
            INIT_AND_PROCESS;
        }
    }
#endif
}
