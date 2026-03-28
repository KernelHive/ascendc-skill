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
 * \file circular_pad.cpp
 * \brief
 */
#include "circular_pad_2d.h"
#include "circular_pad_3d.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void circular_pad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(211)) {
        CircularPad2D<int8_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(212)) {
        CircularPad2D<int8_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(221)) {
        CircularPad2D<half> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(222)) {
        CircularPad2D<half> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(231)) {
        CircularPad2D<bfloat16_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(232)) {
        CircularPad2D<bfloat16_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(241)) {
        CircularPad2D<float> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(242)) {
        CircularPad2D<float> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(251)) {
        CircularPad2D<int32_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(252)) {
        CircularPad2D<int32_t> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }

    else if (TILING_KEY_IS(311)) {
        CircularPad3D<int8_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(312)) {
        CircularPad3D<int8_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(321)) {
        CircularPad3D<half> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(322)) {
        CircularPad3D<half> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(331)) {
        CircularPad3D<bfloat16_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(332)) {
        CircularPad3D<bfloat16_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(341)) {
        CircularPad3D<float> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(342)) {
        CircularPad3D<float> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(351)) {
        CircularPad3D<int32_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(352)) {
        CircularPad3D<int32_t> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    } else {
        return;
    }
}