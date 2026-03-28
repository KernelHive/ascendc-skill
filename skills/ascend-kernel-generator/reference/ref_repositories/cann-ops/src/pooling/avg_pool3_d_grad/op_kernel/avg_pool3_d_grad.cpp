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
 * \file avg_pool3_d_grad.cpp
 * \brief
 */
#include "avg_pool3_d_grad_no_cast.h"
#include "avg_pool3_d_grad_cast.h"
#include "avg_pool3_d_grad_t.h"
#include "avg_pool3_d_grad_t_cast.h"

using namespace AvgPool3DGrad;

extern "C" __global__ __aicore__ void avg_pool3_d_grad(GM_ADDR orig_input_shape, GM_ADDR grads, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    #define INIT_AND_PROCESS                \
        op.Init(grads, output, tiling_data, userWorkspace);\
        op.Process()

    if (TILING_KEY_IS(1000)) {
        AvgPool3DGrad::KernelAvgPool3DGradCast<DTYPE_GRADS> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
        AvgPool3DGrad::KernelAvgPool3DGradNoCast<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3000)) {
        AvgPool3DGrad::KernelAvgPool3DGradT<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(4000)) {
        AvgPool3DGrad::KernelAvgPool3DGradTCast<DTYPE_GRADS> op;
        INIT_AND_PROCESS;
    }
}