/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_fused_matmul_softmax_grad_base.h"
#include "inplace_fused_matmul_softmax_grad.h"
#include "inplace_fused_matmul_softmax_grad_big_shape.h"

using namespace InplaceFusedMatmulSoftmaxGradOpt;

extern "C" __global__ __aicore__ void inplace_fused_matmul_softmax_grad(
    GM_ADDR softmaxOutput, 
    GM_ADDR gradOutput, 
    GM_ADDR values, 
    GM_ADDR workspace, 
    GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (GetBlockIdx() >= tilingData.baseTilingData.realCoreNum) {
        using mt = MMType<half>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        return;
    }
    
    if (TILING_KEY_IS(11)) {
        using mt = MMType<half>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, half, true, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        using mt = MMType<half>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, half, false, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(21)) {
        using mt = MMType<bfloat16_t>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, bfloat16_t, true, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(22)) {
        using mt = MMType<bfloat16_t>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, bfloat16_t, false, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(31)) {
        using mt = MMType<float>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, float, true, false> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(32)) {
        using mt = MMType<float>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGrad<mt, float, false, false> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111)) { 
        using mt = MMType<half>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, half, true, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(112)) {
        using mt = MMType<half>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, half, false, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(121)) {
        using mt = MMType<bfloat16_t>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, bfloat16_t, true, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(122)) {
        using mt = MMType<bfloat16_t>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, bfloat16_t, false, true> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(131)) {
        using mt = MMType<float>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, float, true, false> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(132)) {
        using mt = MMType<float>::MT;
        mt mm1;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tilingData.cubeTilingData);
        InplaceFusedMatmulSoftmaxGradBigShape<mt, float, false, false> op(mm1);
        op.Init(softmaxOutput, gradOutput, values, userWS, &tilingData, &pipe);
        op.Process();
    }
}