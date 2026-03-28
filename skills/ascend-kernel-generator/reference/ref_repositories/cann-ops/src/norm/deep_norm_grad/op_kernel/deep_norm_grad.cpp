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
 * \file deep_norm_grad.cpp
 * \brief
 */
#include "deep_norm_grad_merge_n.h"
#include "deep_norm_grad_cut_d.h"
#include "deep_norm_grad_large_n_small_d.h"

#define GENERAL_OP_IMPL(templateClass, ...)                                                       \
    do {                                                                                          \
        templateClass<__VA_ARGS__> op;                                                            \
        op.Init(dy, x, gx, rstd, mean, gamma, dx, dgx, dgamma, dbeta, tiling_data, usrWorkspace); \
        op.Process();                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void deep_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean,
    GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    //       mergeN  cutD  LargeNSmallD
    // fp32   0       1     2
    // fp16  10      11    12
    // bf16  20      21    22
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(KernelDeepNormGradMergeN, float);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(KernelDeepNormGradCutD, float);
    } else if (TILING_KEY_IS(2)) {
        GENERAL_OP_IMPL(KernelDeepNormGradLargeNSmallD, float);
    } else if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelDeepNormGradMergeN, half);
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(KernelDeepNormGradCutD, half);
    } else if (TILING_KEY_IS(12)) {
        GENERAL_OP_IMPL(KernelDeepNormGradLargeNSmallD, half);
    } else {
#if __CCE_AICORE__ == 220
        if (TILING_KEY_IS(20)) {
            GENERAL_OP_IMPL(KernelDeepNormGradMergeN, bfloat16_t);
        } else if (TILING_KEY_IS(21)) {
            GENERAL_OP_IMPL(KernelDeepNormGradCutD, bfloat16_t);
        } else if (TILING_KEY_IS(22)) {
            GENERAL_OP_IMPL(KernelDeepNormGradLargeNSmallD, bfloat16_t);
        }
#endif
    }
}
