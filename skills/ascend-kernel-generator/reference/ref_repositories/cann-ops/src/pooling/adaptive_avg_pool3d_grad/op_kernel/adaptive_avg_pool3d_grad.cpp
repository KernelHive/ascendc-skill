/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad.cpp
 * \brief
 */

#include "adaptive_avg_pool3d_grad_float.h"
#include "adaptive_avg_pool3d_grad_cast.h"
#include "adaptive_avg_pool3d_grad_nc_large_cast.h"
#include "adaptive_avg_pool3d_grad_nc_large_float.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void adaptive_avg_pool3d_grad(GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradFloat<float> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(10)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradCast<half> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(20)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradCast<bfloat16_t> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeFloat<float> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(11)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeCast<half> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    } else if (TILING_KEY_IS(21)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeCast<bfloat16_t> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    }
}