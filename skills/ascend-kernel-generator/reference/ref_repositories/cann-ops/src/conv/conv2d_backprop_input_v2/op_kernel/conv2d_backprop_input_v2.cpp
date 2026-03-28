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
 * \file conv2d_backprop_input_v2.cpp
 * \brief
 */

#include "conv3d_backprop_input_v2/conv3d_backprop_input_v2.h"
#include "conv3d_backprop_input_v2/conv3d_dx_v2_basic_block.h"
#include "conv3d_backprop_input_v2/conv3d_backprop_input_v2_init_output.h"

#ifndef Y_FORMAT_3D
#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_NC1HWC0
#define Y_FORMAT_3D FORMAT_NDC1HWC0
#else
#define Y_FORMAT_3D FORMAT_NCDHW
#endif
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void conv2d_backprop_input_v2(GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop,
                                                               GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }

    GM_ADDR usrWsp = GetUserWorkspace(workSpace);
    if (usrWsp == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
#if __CCE_AICORE__ == 220
    if constexpr (Y_FORMAT_3D == FORMAT_NCDHW) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    } else if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    }
#endif

    if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        // init output with L0C now
        Conv3dDxInitOutput<DTYPE_Y, Y_FORMAT_3D, InitOutputFlag::L0_INIT> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();
    }

    if (TILING_KEY_IS(0)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_LT_HKWK>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_GE_HKWK>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::HKWK_EQ_ONE>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    } if (TILING_KEY_IS(100)) {
        Conv3dDxBasicBlockSplitMN<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_LT_HKWK>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        Conv3dDxBasicBlockSplitMN<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::BASEK_GE_HKWK>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        Conv3dDxBasicBlockSplitMN<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, Y_FORMAT_3D,
                 Convolution3DBackprop::B2Condition::HKWK_EQ_ONE>
            op;
        op.Init(filter, out_backprop, y, usrWsp, &tilingData);
        op.Process();
    }
}