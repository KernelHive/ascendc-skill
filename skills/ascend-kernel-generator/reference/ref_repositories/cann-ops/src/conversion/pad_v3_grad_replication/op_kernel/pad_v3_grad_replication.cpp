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
 * @file pad_v3_grad_replication.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_v3_grad_replication.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void pad_v3_grad_replication(GM_ADDR x, GM_ADDR paddings, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        PadV3GradReplication<float, false> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    } else if (TILING_KEY_IS(2)) {
        PadV3GradReplication<half, true> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    } else if (TILING_KEY_IS(3)) {
        PadV3GradReplication<bfloat16_t, true> op(tilingData);
        op.Init(x, z, workspace);
        op.Process();
        return;
    }
}