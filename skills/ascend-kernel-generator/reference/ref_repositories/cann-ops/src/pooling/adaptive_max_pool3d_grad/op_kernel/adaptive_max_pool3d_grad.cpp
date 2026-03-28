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
 * \file adaptive_max_pool3d_grad.cpp
 * \brief
 */

#include "adaptive_max_pool3d_grad_normal.h"
#include "adaptive_max_pool3d_grad_scatter.h"
#include "adaptive_max_pool3d_grad_scatter_overlap.h"

using namespace AdaptiveMaxPool3DGrad;
#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
} while (0)
extern "C" __global__ __aicore__ void adaptive_max_pool3d_grad(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if (g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    if (TILING_KEY_IS(0)) { // Normal New Kernel
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, false);
    }  else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, true);
    }
    if (TILING_KEY_IS(2)) { // Scatter Kernel
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradScatter, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y);
    } else if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradScatterOverlap, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y);
    }
}