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
 * \file dua_quantize_add_layer_norm.cpp
 * \brief
 */
#include "dua_quantize_add_layer_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void dua_quantize_add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroPoints1, GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2,
    GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1000)) {
        KernelDuaQuantizeAddLayerNormSingleRow<bfloat16_t, 1000> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        KernelDuaQuantizeAddLayerNormSingleRow<bfloat16_t, 1001> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x, workspace, &tilingData);
        op.Process();
    }
}