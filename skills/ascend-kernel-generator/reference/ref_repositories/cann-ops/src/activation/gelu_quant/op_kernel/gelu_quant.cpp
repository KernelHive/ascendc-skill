/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "gelu_quant_base.h"
#include "gelu_static_quant.h"
#include "gelu_static_quant_block.h"
#include "gelu_static_quant_per_tensor.h"
#include "gelu_dynamic_quant.h"
#include "gelu_dynamic_quant_workspace.h"

using namespace GeluQuantALL;
#define STATIC_PER_TENSOR_TEMPLATE_HALF_HALF 1001
#define STATIC_PER_TENSOR_TEMPLATE_HALF_FLOAT 1002
#define STATIC_PER_TENSOR_TEMPLATE_FLOAT_FLOAT 1003
#define STATIC_PER_TENSOR_TEMPLATE_BF16_BF16 1004
#define STATIC_PER_TENSOR_TEMPLATE_BF16_FLOAT 1005

#define STATIC_FUNCTION_TEMPLATE_HALF_HALF 1011
#define STATIC_FUNCTION_TEMPLATE_HALF_FLOAT 1012
#define STATIC_FUNCTION_TEMPLATE_FLOAT_FLOAT 1013
#define STATIC_FUNCTION_TEMPLATE_BF16_BF16 1014
#define STATIC_FUNCTION_TEMPLATE_BF16_FLOAT 1015

#define STATIC_PERFORMANCE_TEMPLATE_HALF_HALF 1021
#define STATIC_PERFORMANCE_TEMPLATE_HALF_FLOAT 1022
#define STATIC_PERFORMANCE_TEMPLATE_FLOAT_FLOAT 1023
#define STATIC_PERFORMANCE_TEMPLATE_BF16_BF16 1024
#define STATIC_PERFORMANCE_TEMPLATE_BF16_FLOAT 1025

#define DYNAMIC_NORMAL_TEMPLATE_HALF_HALF 1031
#define DYNAMIC_NORMAL_TEMPLATE_HALF_FLOAT 1032
#define DYNAMIC_NORMAL_TEMPLATE_FLOAT_FLOAT 1033
#define DYNAMIC_NORMAL_TEMPLATE_BF16_BF16 1034
#define DYNAMIC_NORMAL_TEMPLATE_BF16_FLOAT 1035

#define DYNAMIC_WORKSPACE_TEMPLATE_HALF_HALF 1041
#define DYNAMIC_WORKSPACE_TEMPLATE_HALF_FLOAT 1042
#define DYNAMIC_WORKSPACE_TEMPLATE_FLOAT_FLOAT 1043
#define DYNAMIC_WORKSPACE_TEMPLATE_BF16_BF16 1044
#define DYNAMIC_WORKSPACE_TEMPLATE_BF16_FLOAT 1045


template <typename T1, typename T2>
__aicore__ inline void invokeTemplateGeluQuant(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y,
    GM_ADDR out_scale, GM_ADDR userWS, const GeluQuantTilingData &tilingData)
{
    GeluQuant<T1, T2> op;
    op.Init(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    op.Process();
}

template <typename T1, typename T2>
__aicore__ inline void invokeTemplateStaticQuantPerTensor(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset,
    GM_ADDR y, GM_ADDR out_scale, GM_ADDR userWS, const GeluQuantTilingData &tilingData)
{
    StaticQuantPerTensor<T1, T2> op;
    op.Init(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    op.Process();
}

template <typename T1, typename T2>
__aicore__ inline void invokeTemplateStaticQuantBlock(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y,
    GM_ADDR out_scale, GM_ADDR userWS, const GeluQuantTilingData &tilingData)
{
    StaticQuantBlock<T1, T2> op;
    op.Init(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    op.Process();
}

template <typename T1, typename T2>
__aicore__ inline void invokeTemplateGeluDynamicQuant(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y,
    GM_ADDR out_scale, GM_ADDR userWS, const GeluQuantTilingData &tilingData)
{
    GeluDynamicQuant<T1, T2> op;
    op.Init(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    op.Process();
}

template <typename T1, typename T2>
__aicore__ inline void invokeTemplateGeluDynamicQuantWorkspace(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset,
    GM_ADDR y, GM_ADDR out_scale, GM_ADDR userWS, const GeluQuantTilingData &tilingData)
{
    GeluDynamicQuantWorkspace<T1, T2> op;
    op.Init(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void gelu_quant(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y,
    GM_ADDR out_scale, GM_ADDR workspace, GM_ADDR tiling_data)
{
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling_data);
#if (ORIG_DTYPE_X == DT_FLOAT)
    if (TILING_KEY_IS(STATIC_PER_TENSOR_TEMPLATE_FLOAT_FLOAT)) {
        invokeTemplateStaticQuantPerTensor<float, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(STATIC_FUNCTION_TEMPLATE_FLOAT_FLOAT)) {
        invokeTemplateGeluQuant<float, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_PERFORMANCE_TEMPLATE_FLOAT_FLOAT)) {
        invokeTemplateStaticQuantBlock<float, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_NORMAL_TEMPLATE_FLOAT_FLOAT)) {
        invokeTemplateGeluDynamicQuant<float, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_WORKSPACE_TEMPLATE_FLOAT_FLOAT)) {
        invokeTemplateGeluDynamicQuantWorkspace<float, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    }
#endif

#if (ORIG_DTYPE_X == DT_FLOAT16)
    if (TILING_KEY_IS(STATIC_PER_TENSOR_TEMPLATE_HALF_HALF)) {
        invokeTemplateStaticQuantPerTensor<half, half>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_PER_TENSOR_TEMPLATE_HALF_FLOAT)) {
        invokeTemplateStaticQuantPerTensor<half, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_FUNCTION_TEMPLATE_HALF_HALF)) {
        invokeTemplateGeluQuant<half, half>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_FUNCTION_TEMPLATE_HALF_FLOAT)) {
        invokeTemplateGeluQuant<half, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_PERFORMANCE_TEMPLATE_HALF_HALF)) {
        invokeTemplateStaticQuantBlock<half, half>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_PERFORMANCE_TEMPLATE_HALF_FLOAT)) {
        invokeTemplateStaticQuantBlock<half, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_NORMAL_TEMPLATE_HALF_HALF)) {
        invokeTemplateGeluDynamicQuant<half, half>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_NORMAL_TEMPLATE_HALF_FLOAT)) {
        invokeTemplateGeluDynamicQuant<half, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_WORKSPACE_TEMPLATE_HALF_HALF)) {
        invokeTemplateGeluDynamicQuantWorkspace<half, half>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_WORKSPACE_TEMPLATE_HALF_FLOAT)) {
        invokeTemplateGeluDynamicQuantWorkspace<half, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    }
#endif

#if (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(STATIC_PER_TENSOR_TEMPLATE_BF16_BF16)) {
        invokeTemplateStaticQuantPerTensor<bfloat16_t, bfloat16_t>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(STATIC_PER_TENSOR_TEMPLATE_BF16_FLOAT)) {
        invokeTemplateStaticQuantPerTensor<bfloat16_t, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(STATIC_FUNCTION_TEMPLATE_BF16_BF16)) {
        invokeTemplateGeluQuant<bfloat16_t, bfloat16_t>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_FUNCTION_TEMPLATE_BF16_FLOAT)) {
        invokeTemplateGeluQuant<bfloat16_t, float>(x, input_scale, input_offset, y, out_scale, userWS, tilingData);
    } else if (TILING_KEY_IS(STATIC_PERFORMANCE_TEMPLATE_BF16_BF16)) {
        invokeTemplateStaticQuantBlock<bfloat16_t, bfloat16_t>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(STATIC_PERFORMANCE_TEMPLATE_BF16_FLOAT)) {
        invokeTemplateStaticQuantBlock<bfloat16_t, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_NORMAL_TEMPLATE_BF16_BF16)) {
        invokeTemplateGeluDynamicQuant<bfloat16_t, bfloat16_t>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_NORMAL_TEMPLATE_BF16_FLOAT)) {
        invokeTemplateGeluDynamicQuant<bfloat16_t, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_WORKSPACE_TEMPLATE_BF16_BF16)) {
        invokeTemplateGeluDynamicQuantWorkspace<bfloat16_t, bfloat16_t>(x, input_scale, input_offset, y, out_scale,
            userWS, tilingData);
    } else if (TILING_KEY_IS(DYNAMIC_WORKSPACE_TEMPLATE_BF16_FLOAT)) {
        invokeTemplateGeluDynamicQuantWorkspace<bfloat16_t, float>(x, input_scale, input_offset, y, out_scale, userWS,
            tilingData);
    }
#endif

}