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
 * \file pad_v4_grad.cpp
 * \brief
 */
#include "pad_v4_grad_h_pad.h"
#include "pad_v4_grad_h_w_bf16_pad.h"
#include "pad_v4_grad_h_w_pad.h"
#include "pad_v4_grad_large_h_small_w_bf16_pad.h"
#include "pad_v4_grad_large_h_small_w_pad.h"
#include "pad_v4_grad_mini_h_w_pad.h"
#include "pad_v4_grad_small_h_large_w_bf16_pad.h"
#include "pad_v4_grad_small_h_large_w_pad.h"
#include "pad_v4_grad_w_pad.h"

extern "C" __global__ __aicore__ void pad_v4_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(1000)) {
        PadV4GradPadMiniHW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2000)) {
        PadV4GradPadMiniHW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3000)) {
        PadV4GradPadMiniHW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1100)) {
        PadV4GradPadSamllHLargeW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2100)) {
        PadV4GradPadSamllHLargeW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3100)) {
        PadV4GradPadSamllHLargeWBf16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1010)) {
        PadV4GradLargeHSmallW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2010)) {
        PadV4GradLargeHSmallW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3010)) {
        PadV4GradLargeHSmallWBf16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1101)) {
        PadV4GradPadW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2101)) {
        PadV4GradPadW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3101)) {
        PadV4GradPadW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1110)) {
        PadV4GradPadH<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2110)) {
        PadV4GradPadH<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3110)) {
        PadV4GradPadH<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1111)) {
        PadV4GradPadHW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2111)) {
        PadV4GradPadHW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3111)) {
        PadV4GradPadHWBf16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
}