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
 * @file inplace_attn_softmax.cpp
 */
#include "kernel_operator.h"
#include "inplace_attn_softmax_bash.h"
#include "inplace_attn_softmax_bigshape.h"
#include "inplace_attn_softmax.h"
using namespace AscendC;
using namespace InplaceAttnSoftmaxOpt;

extern "C" __global__ __aicore__ void inplace_attn_softmax(GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) 
{
    GET_TILING_DATA(tiling_data, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    if (GetBlockIdx() >= tiling_data.realCoreNum) {
        return;
    }
    TPipe pipe;
    if (TILING_KEY_IS(101)) {
        InplaceAttnSoftmax<half, half, true, false> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(111)) {
        InplaceAttnSoftmaxBigShape<half, half, true, true> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(201)) {
        InplaceAttnSoftmax<bfloat16_t, bfloat16_t, true, false> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(211)) {
        InplaceAttnSoftmaxBigShape<bfloat16_t, bfloat16_t, true, true> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(301)) {
        InplaceAttnSoftmax<float, float, false, false> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    }
    else if (TILING_KEY_IS(311)) {
        InplaceAttnSoftmaxBigShape<float, float, false, true> op(&pipe);
        op.Init(x, workspace, &tiling_data);
        op.Process();
    } 
}