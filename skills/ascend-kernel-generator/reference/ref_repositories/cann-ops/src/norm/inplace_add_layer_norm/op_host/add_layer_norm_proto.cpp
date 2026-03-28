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
 * \file add_layer_norm.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

}

static constexpr int INPUT_NODE_NUM = 4;
static constexpr int OUTPUT_NODE_NUM = 4;
static constexpr int INPUT_NODE_OPTIONAL_NUM = 5;
static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int Y_IDX = 0;
static constexpr int MEAN_IDX = 1;
static constexpr int RSTD_IDX = 2;
static constexpr int X_IDX = 3;

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4AddLayerNorm(gert::InferShapeContext *context)
{
    if ((!(context->GetComputeNodeInputNum() == INPUT_NODE_NUM ||
            context->GetComputeNodeInputNum() == INPUT_NODE_OPTIONAL_NUM)) ||
        context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }
    const gert::Shape *x1_shape = context->GetInputShape(X1_IDX);
    const gert::Shape *gamma_shape = context->GetInputShape(GAMMA_IDX);
    const gert::Shape *beta_shape = context->GetInputShape(BETA_IDX);
    gert::Shape *y_shape = context->GetOutputShape(Y_IDX);
    gert::Shape *mean_shape = context->GetOutputShape(MEAN_IDX);
    gert::Shape *rstd_shape = context->GetOutputShape(RSTD_IDX);
    gert::Shape *x_shape = context->GetOutputShape(X_IDX);
    if (*gamma_shape != *beta_shape) {
        return GRAPH_FAILED;
    }
    *y_shape = *x1_shape;
    *x_shape = *x1_shape;
    auto shape(*x1_shape);
    shape.SetDim(shape.GetDimNum() - 1, 1);
    *mean_shape = shape;
    *rstd_shape = shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddLayerNorm(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4AddLayerNorm");
    if (context->GetInputDataType(X1_IDX) == context->GetInputDataType(X2_IDX)) {
        context->SetOutputDataType(Y_IDX, context->GetInputDataType(X1_IDX));
        context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    } else {
        context->SetOutputDataType(Y_IDX, DT_FLOAT);
        context->SetOutputDataType(X_IDX, DT_FLOAT);
    }
    context->SetOutputDataType(MEAN_IDX, DT_FLOAT);
    context->SetOutputDataType(RSTD_IDX, DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4AddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceAddLayerNorm).InferShape(InferShape4AddLayerNorm).InferDataType(InferDataType4AddLayerNorm);
;
}  // namespace ops