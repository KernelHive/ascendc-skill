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
 * \file layer_norm_grad_v3.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "layer_norm_grad_v3_workspace.h"
#include "layer_norm_grad_v3_single_read.h"
#include "layer_norm_grad_v3_transpose.h"
#include "layer_norm_grad_v3_common.h"

using namespace LayerNormGradV3;

#define SINGLE_READ_FLOAT_FLOAT 101
#define SINGLE_READ_HALF_HALF 102
#define SINGLE_READ_HALF_FLOAT 103
#define SINGLE_READ_BF16_BF16 104
#define SINGLE_READ_BF16_FLOAT 105
#define SINGLE_READ_FLOAT_FLOAT_DETERMINISTIC 111
#define SINGLE_READ_HALF_HALF_DETERMINISTIC 112
#define SINGLE_READ_HALF_FLOAT_DETERMINISTIC 113
#define SINGLE_READ_BF16_BF16_DETERMINISTIC 114
#define SINGLE_READ_BF16_FLOAT_DETERMINISTIC 115

#define WORKSPACE_FLOAT_FLOAT 201
#define WORKSPACE_HALF_HALF 202
#define WORKSPACE_HALF_FLOAT 203
#define WORKSPACE_BF16_BF16 204
#define WORKSPACE_BF16_FLOAT 205
#define WORKSPACE_FLOAT_FLOAT_DETERMINISTIC 211
#define WORKSPACE_HALF_HALF_DETERMINISTIC 212
#define WORKSPACE_HALF_FLOAT_DETERMINISTIC 213
#define WORKSPACE_BF16_BF16_DETERMINISTIC 214
#define WORKSPACE_BF16_FLOAT_DETERMINISTIC 215

#define TRANSPOSE_FLOAT_FLOAT 301
#define TRANSPOSE_HALF_HALF 302
#define TRANSPOSE_HALF_FLOAT 303
#define TRANSPOSE_BF16_BF16 304
#define TRANSPOSE_BF16_FLOAT 305
#define TRANSPOSE_FLOAT_FLOAT_DETERMINISTIC 311
#define TRANSPOSE_HALF_HALF_DETERMINISTIC 312
#define TRANSPOSE_HALF_FLOAT_DETERMINISTIC 313
#define TRANSPOSE_BF16_BF16_DETERMINISTIC 314
#define TRANSPOSE_BF16_FLOAT_DETERMINISTIC 315

#define COMMON_FLOAT_FLOAT 401
#define COMMON_HALF_HALF 402
#define COMMON_HALF_FLOAT 403
#define COMMON_BFLOAT16_BFLOAT16 404
#define COMMON_BFLOAT16_FLOAT 405
#define COMMON_FLOAT_FLOAT_DETERMINISTIC 411
#define COMMON_HALF_HALF_DETERMINISTIC 412
#define COMMON_HALF_FLOAT_DETERMINISTIC 413
#define COMMON_BFLOAT16_BFLOAT16_DETERMINISTIC 414
#define COMMON_BFLOAT16_FLOAT_DETERMINISTIC 415

#define INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(Tdy, Tgamma, Isdeterministic)               \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataSingleRead, tilingData, tiling);  \
        LayerNormGradV3SingleRead<Tdy, Tgamma, Isdeterministic> op;                            \
        op.Init(dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, &tilingData); \
        op.Process(&tilingData);                                                               \
    } while (0)

#define INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(Tdy, Tgamma, Isdeterministic)                 \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataWorkspace, tilingData, tiling);   \
        LayerNormGradV3Workspace<Tdy, Tgamma, Isdeterministic> op;                             \
        op.Init(dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, &tilingData); \
        op.Process(&tilingData);                                                               \
    } while (0)

#define INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(Tdy, Tgamma, Isdeterministic)                 \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataTranspose, tilingData, tiling);   \
        LayerNormGradV3Transpose<Tdy, Tgamma, Isdeterministic> op;                             \
        op.Init(dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, &tilingData); \
        op.Process();                                                                          \
    } while (0)

#define INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(Tdy, Tgamma, Isdeterministic)                    \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataCommon, tilingData, tiling);      \
        LayerNormGradV3Common<Tdy, Tgamma, Isdeterministic> op;                                \
        op.Init(dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, &tilingData); \
        op.Process(&tilingData);                                                               \
    } while (0)

extern "C" __global__ __aicore__ void layer_norm_grad_v3(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean,
    GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(SINGLE_READ_FLOAT_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_HALF_HALF)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_HALF_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_BF16_BF16)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_BF16_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_HALF_HALF_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_BF16_BF16_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(SINGLE_READ_BF16_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_SINGLE_READ_IMPL(bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_FLOAT_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_HALF)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_BF16)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_HALF_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_BF16_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_WORKSPACE_IMPL(bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_FLOAT_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_HALF_HALF)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_HALF_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_BF16_BF16)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_BF16_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_HALF_HALF_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_BF16_BF16_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(TRANSPOSE_BF16_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_TRANSPOSE_IMPL(bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(COMMON_FLOAT_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_HALF)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_BFLOAT16)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_FLOAT)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_HALF_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_BFLOAT16_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_FLOAT_DETERMINISTIC)) {
        INVOKE_LAYER_NORM_GRAD_V3_COMMON_IMPL(bfloat16_t, float, true);
        return;
    }
    return;
}
