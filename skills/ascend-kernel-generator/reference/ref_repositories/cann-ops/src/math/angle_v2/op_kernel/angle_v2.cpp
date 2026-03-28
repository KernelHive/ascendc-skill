/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file angle_v2.cpp
 */
#include "angle_v2_complex.h"
#include "angle_v2_u8.h"
#include "angle_v2_int.h"
#include "angle_v2.h"

using namespace AngleV2N;

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        AngleV2N::AngleV2Complex<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AngleV2N::AngleV2<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        AngleV2N::AngleV2<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        AngleV2N::AngleV2Int<int8_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        AngleV2N::AngleV2Int<int16_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        AngleV2N::AngleV2Int<int32_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        AngleV2N::AngleV2Int<int64_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
