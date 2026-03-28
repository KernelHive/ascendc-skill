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
 * \file layer_norm_v4.cpp
 * \brief
 */

#include "kernel_operator.h"
#if __CCE_AICORE__ == 200
#include "layer_norm_v4_transpose_310p.h"
#else
#include "layer_norm_v4_single_read.h"
#include "layer_norm_v4_transpose.h"
#endif

using namespace LayerNormV4;

#define LNV4_TRANSPOSE_FLOAT_FLOAT 200
#define LNV4_TRANSPOSE_HALF_FLOAT 210
#define LNV4_TRANSPOSE_HALF_HALF 211
#define LNV4_TRANSPOSE_BF16_FLOAT 220
#define LNV4_TRANSPOSE_BF16_BF16 222

#define INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(Tfm, Tweight)                               \
    do {                                                                                  \
        GET_TILING_DATA_WITH_STRUCT(LayerNormV4TilingDataSingleRead, tilingData, tiling); \
        LayerNormV4SingleRead<Tfm, Tweight> op;                                           \
        auto t = &tilingData;                                                             \
        op.Init(x, gamma, beta, y, mean, rstd, workspace, t);                             \
        op.Process();                                                                     \
    } while (0)

#define INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(Tfm, Tweight)                                \
    do {                                                                                 \
        GET_TILING_DATA_WITH_STRUCT(LayerNormV4TilingDataTranspose, tilingData, tiling); \
        LayerNormV4Transpose<Tfm, Tweight> op;                                           \
        auto t = &tilingData;                                                            \
        op.Init(x, gamma, beta, y, mean, rstd, workspace, t);                            \
        op.Process();                                                                    \
    } while (0)

extern "C" __global__ __aicore__ void layer_norm_v4(GM_ADDR x, GM_ADDR normalized_shape, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(211)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(half, half);
        return;
    } else if (TILING_KEY_IS(210)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(half, float);
        return;
    }
#else
    if (g_coreType == AIC) {
        return;
    }
    if (TILING_KEY_IS(100)) {
        INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(float, float);
        return;
    } else if (TILING_KEY_IS(110)) {
        INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(half, float);
        return;
    } else if (TILING_KEY_IS(111)) {
        INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(half, half);
        return;
    } else if (TILING_KEY_IS(120)) {
        INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(122)) {
        INVOKE_LAYER_NORM_V4_SINGLE_READ_IMPL(bfloat16_t, bfloat16_t);
        return;
    } else if (TILING_KEY_IS(LNV4_TRANSPOSE_FLOAT_FLOAT)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(float, float);
        return;
    } else if (TILING_KEY_IS(LNV4_TRANSPOSE_HALF_FLOAT)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(half, float);
        return;
    } else if (TILING_KEY_IS(LNV4_TRANSPOSE_HALF_HALF)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(half, half);
        return;
    } else if (TILING_KEY_IS(LNV4_TRANSPOSE_BF16_FLOAT)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(LNV4_TRANSPOSE_BF16_BF16)) {
        INVOKE_LAYER_NORM_V4_TRANSPOSE_IMPL(bfloat16_t, bfloat16_t);
        return;
    }
#endif
}
