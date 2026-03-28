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
 * \file instance_norm_v3.cpp
 * \brief
 */

#include "instance_norm_nchw_kernel.h"
#include "instance_norm_nchw_kernel_cut_reduce.h"

extern "C" __global__ __aicore__ void instance_norm_v3(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
    GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                   \
    op.Init(x, gamma, beta, y, mean, variance, usrWorkspace, &tilingData); \
    op.Process()

    // SingleRowDynamic
    if (TILING_KEY_IS(1)) {
        KernelInstanceNormNCHW<DTYPE_X, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelInstanceNormNCHWCutReduce<DTYPE_X, 1> op(&pipe);
        INIT_AND_PROCESS;
    }
}