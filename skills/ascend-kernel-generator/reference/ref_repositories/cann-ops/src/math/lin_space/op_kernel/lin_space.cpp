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
 * \file lin_space.cpp
 * \brief
 */
#include "lin_space_half_and_float.h"
#include "lin_space_with_big_shape.h"
#include "lin_space_need_cast.h"
#include "lin_space_with_big_shape_and_cast.h"
#include "lin_space_small_shape.h"
#include "lin_space_small_shape_and_cast.h"

using namespace LinSpace;

extern "C" __global__ __aicore__ void lin_space(
    GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(101)) {
        LinSpace::LinSpaceHalfAndFloat<float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        LinSpace::LinSpaceWithBigShape<float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        LinSpace::LinSpaceSmallShape<float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(201)) {
        LinSpace::LinSpaceHalfAndFloat<half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(202)) {
        LinSpace::LinSpaceWithBigShape<half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(203)) {
        LinSpace::LinSpaceSmallShape<half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(301)) {
        LinSpace::LinSpaceNeedCast<int8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(302)) {
        LinSpace::LinSpaceWithBigShapeAndCast<int8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(303)) {
        LinSpace::LinSpaceSmallShapeAndCast<int8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(401)) {
        LinSpace::LinSpaceNeedCast<uint8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(402)) {
        LinSpace::LinSpaceWithBigShapeAndCast<uint8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(403)) {
        LinSpace::LinSpaceSmallShapeAndCast<uint8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(501)) {
        LinSpace::LinSpaceNeedCast<int16_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(502)) {
        LinSpace::LinSpaceWithBigShapeAndCast<int16_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(503)) {
        LinSpace::LinSpaceSmallShapeAndCast<int16_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(601)) {
        LinSpace::LinSpaceNeedCast<int32_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(602)) {
        LinSpace::LinSpaceWithBigShapeAndCast<int32_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(603)) {
        LinSpace::LinSpaceSmallShapeAndCast<int32_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#if ORIG_DTYPE_START == DT_BF16
    else if (TILING_KEY_IS(701)) {
        LinSpace::LinSpaceNeedCast<bfloat16_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(702)) {
        LinSpace::LinSpaceWithBigShapeAndCast<bfloat16_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(703)) {
        LinSpace::LinSpaceSmallShapeAndCast<bfloat16_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#endif
}
