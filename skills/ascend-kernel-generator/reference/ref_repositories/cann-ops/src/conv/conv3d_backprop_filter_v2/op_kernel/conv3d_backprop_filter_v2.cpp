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
 * \file conv3d_backprop_filter_v2.cpp
 * \brief
 */
#include "conv3d_backprop_filter_v2.h"
#include "conv3d_backprop_filter_v2_init_output.h"
#include "conv3d_dw_v2_basic_block.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void conv3d_backprop_filter_v2(GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop,
                                                                GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    SetSysWorkspace(workSpace);
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    ENABLE_DETERMINISTIC();
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
#if __CCE_AICORE__ == 220
    if constexpr (FORMAT_X == FORMAT_NCDHW) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    }
#endif

    if (TILING_KEY_IS(0)) {
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
#endif
        Conv3dDwInitOutput<DTYPE_Y> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();

        Conv3dDw<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        op.ProcessWithDeterministic();
#else
        op.Process();
#endif
    } else if (TILING_KEY_IS(1)) {
        Conv3dDwInitOutput<DTYPE_Y> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();

        // 基本块Tiling模板, M和K绑核
        Conv3dDwBasicBlockSplitMK<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        Conv3dDwInitOutput<DTYPE_Y> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();

        // 基本块Tiling模板, N和K绑核
        Conv3dDwBasicBlockSplitKN<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        Conv3dDwInitOutput<DTYPE_Y> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();

        // 基本块Tiling模板, M和N绑核
        Conv3dDwBasicBlockSplitMN<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    }
}