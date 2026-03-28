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
 * \file circular_pad_grad.cpp
 * \brief
 */
#include "circular_pad_grad_2d.h"
#include "circular_pad_grad_3d.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void circular_pad_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);

    // 2D
    if (TILING_KEY_IS(211)) {
        CircularPadGrad2D<float, float, false> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(212)) {
        CircularPadGrad2D<float, float, false> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(221)) {
        CircularPadGrad2D<half, float, true> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(222)) {
        CircularPadGrad2D<half, float, true> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(231)) {
        CircularPadGrad2D<bfloat16_t, float, true> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(232)) {
        CircularPadGrad2D<bfloat16_t, float, true> op(&pipe);
        op.Init2D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }

    // 3D
    else if (TILING_KEY_IS(311)) {
        CircularPadGrad3D<float, float, false> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(312)) {
        CircularPadGrad3D<float, float, false> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(321)) {
        CircularPadGrad3D<half, float, true> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(322)) {
        CircularPadGrad3D<half, float, true> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    }
    else if (TILING_KEY_IS(331)) {
        CircularPadGrad3D<bfloat16_t, float, true> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessSmallShape();
    }
    else if (TILING_KEY_IS(332)) {
        CircularPadGrad3D<bfloat16_t, float, true> op(&pipe);
        op.Init3D(x, paddings, y, usrWorkspace, tiling_data);
        op.ProcessBigShape();
    } else {
        return;
    }
}