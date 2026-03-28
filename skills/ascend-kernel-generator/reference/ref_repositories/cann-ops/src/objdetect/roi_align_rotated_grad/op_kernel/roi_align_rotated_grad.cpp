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
 * @file roi_align_rotated_grad.cpp
 */
#include "roi_align_rotated_grad.h"

extern "C" __global__ __aicore__ void roi_align_rotated_grad(
    GM_ADDR grad_output,
    GM_ADDR rois,
    GM_ADDR grad_input,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    const RoiAlignRotatedGradTilingData *__restrict tilingDevice = &tilingData;
    KernelRoiAlignRotatedGrad op;
    op.Init(grad_output, rois, grad_input, tilingDevice);
    op.Process();
}