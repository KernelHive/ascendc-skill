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
 * \file add_layer_norm_quant.cpp
 * \brief
 */
#include "add_layer_norm_dynamic_quant_normal_kernel.h"
#include "add_layer_norm_dynamic_quant_single_row_kernel.h"
#include "add_layer_norm_static_quant_single_row_kernel.h"
#include "add_layer_norm_static_quant_normal_kernel.h"

extern "C" __global__ __aicore__ void add_layer_norm_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1, GM_ADDR y2,
    GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS \
    op.Init(x1,          \
        x2,              \
        gamma,           \
        beta,            \
        bias,            \
        scales1,         \
        scales2,         \
        zeroOffset1,     \
        zeroOffset2,     \
        y1,              \
        y2,              \
        x,               \
        outScale1,       \
        outScale2,       \
        usrWorkspace,    \
        &tilingData);    \
    op.Process()

    // SingleRowDynamic
    if (TILING_KEY_IS(1120)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1121)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1122)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1122> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2120)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2121)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2122)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2122> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3120)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3121)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3122)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3122> op(&pipe);
        INIT_AND_PROCESS;
    }

    // SingleRowStatic
    else if (TILING_KEY_IS(1020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        }
    }

    // NormalDynamic
    else if (TILING_KEY_IS(1100)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1101)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1102)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1102> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2100)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2101)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2102)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2102> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3100)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3101)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3102)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3102> op(&pipe);
        INIT_AND_PROCESS;
    }

    // NormalStatic
    else if (TILING_KEY_IS(1000)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1001)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1002)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1002> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2001)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2002)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2002> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3000)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3001)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3002)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3002> op(&pipe);
        INIT_AND_PROCESS;
    }
}