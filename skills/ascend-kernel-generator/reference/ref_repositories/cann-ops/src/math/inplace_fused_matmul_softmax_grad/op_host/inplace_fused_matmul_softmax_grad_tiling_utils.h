/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_UTILS_H
#define INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_UTILS_H

#include "inplace_fused_matmul_softmax_grad_tiling.h"

namespace optiling {

constexpr uint32_t SOFTMAX_OUTPUT_INDEX = 0;
constexpr uint32_t GRAD_OUTPUT_INDEX = 1;
constexpr uint32_t VALUES_INDEX = 2;
constexpr uint32_t ONE = 1;
constexpr uint32_t COMPARE_INT = 255;
constexpr uint32_t FLOAT16_BASE_TILING_KEY = 10;
constexpr uint32_t BFLOAT16_BASE_TILING_KEY = 20;
constexpr uint32_t FLOAT_BASE_TILING_KEY = 30;
constexpr uint32_t BIG_SHAPE_BASE_TILING_KEY = 100;
constexpr uint32_t MIN_DIM_NUM = 2;
constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint32_t K_MIN_VALUE = 1;
constexpr uint32_t K_MAX_VALUE = 65535;

template <typename T>
T AlignUp(T num, T div)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline T FloorDIV(T num, T div)
{
    return div == 0 ? 0 : (num + div) / div;
}

template <typename T>
inline T CeilDiv(T num, T div)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

template <typename T>
inline T Min(T num, T div)
{
    return num < div ? num : div;
}

template <typename T>
inline T AlignDown(T a, T base) {
  return base == 0 ? 0 : a / base * base;
}


ge::graphStatus CheckInputGradDtype(gert::TilingContext *context)
{
    auto softmaxOutputDtype = context->GetInputDesc(SOFTMAX_OUTPUT_INDEX)->GetDataType();
    if (softmaxOutputDtype != ge::DT_FLOAT16 && softmaxOutputDtype != ge::DT_BF16 && softmaxOutputDtype != ge::DT_FLOAT) {
        printf("input softmaxOutput dtype is only support fp16/bf16/fp32.\n");
        return ge::GRAPH_FAILED;
    }
    auto gradOutputDtype = context->GetInputDesc(GRAD_OUTPUT_INDEX)->GetDataType();
    if (gradOutputDtype != ge::DT_FLOAT16 && gradOutputDtype != ge::DT_BF16 && gradOutputDtype != ge::DT_FLOAT) {
        printf("input gradOutput dtype is only support fp16/bf16/fp32.\n");
        return ge::GRAPH_FAILED;
    }
    auto valuesDtype = context->GetInputDesc(VALUES_INDEX)->GetDataType();
    if (valuesDtype != ge::DT_FLOAT16 && valuesDtype != ge::DT_BF16 && valuesDtype != ge::DT_FLOAT) {
        printf("input values dtype is only support fp16/bf16/fp32.\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus CheckOpInputGradShape(gert::TilingContext *context)
{
    auto softmaxOutputShape = context->GetInputShape(SOFTMAX_OUTPUT_INDEX);
    size_t softmaxOutputDimNum = softmaxOutputShape->GetStorageShape().GetDimNum();
    if (softmaxOutputDimNum < MIN_DIM_NUM || softmaxOutputDimNum > MAX_DIM_NUM) {
        printf("input softmaxOutput dimension should be in [2, 8]!\n");
        return ge::GRAPH_FAILED;
    }
    auto gradOutputShape = context->GetInputShape(GRAD_OUTPUT_INDEX);
    size_t gradOutputDimNum = gradOutputShape->GetStorageShape().GetDimNum();
    if (gradOutputDimNum < MIN_DIM_NUM  || gradOutputDimNum > MAX_DIM_NUM) {
        printf("input gradOutput dimension should be in [2, 8]!\n");
        return ge::GRAPH_FAILED;
    }

    size_t gradOutputDimValue = gradOutputShape->GetStorageShape().GetDim(1);
    if (gradOutputDimValue > K_MAX_VALUE || gradOutputDimValue < K_MIN_VALUE ){
        printf("input gradOutput ka value should be in [1, 65535]!\n");
        return ge::GRAPH_FAILED;
    }
    auto valuesShape = context->GetInputShape(VALUES_INDEX);
    size_t valuesDimNum = valuesShape->GetStorageShape().GetDimNum();
    if (valuesDimNum < MIN_DIM_NUM || valuesDimNum > MAX_DIM_NUM) {
        printf("input values dimension should be in [2, 8]!\n");
        return ge::GRAPH_FAILED;
    }

    size_t valuesDimValue = valuesShape->GetStorageShape().GetDim(1);
    if (valuesDimValue > K_MAX_VALUE || valuesDimValue < K_MIN_VALUE ){
        printf("input values kb value should be in [1, 65535]!\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckGradOpParams(gert::TilingContext *context)
{
    if (CheckInputGradDtype(context) != ge::GRAPH_SUCCESS) {
        printf("op input dtype is invalid!\n");
        return ge::GRAPH_FAILED;
    }

    if (CheckOpInputGradShape(context) != ge::GRAPH_SUCCESS) {
        printf("input or output shape is invalid.\n");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

#endif  // INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_TILING_UTILS_H