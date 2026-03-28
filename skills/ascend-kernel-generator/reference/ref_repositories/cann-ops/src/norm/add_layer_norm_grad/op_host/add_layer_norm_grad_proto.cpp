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
 * \file add_layer_norm_grad_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
}

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static constexpr int REDUCE_AXIS_INDEX = 2;
static constexpr int DY_INPUT_INDEX = 0;
static constexpr int GAMMA_INPUT_INDEX = 5;
static constexpr int OUTPUT_DX_INDEX = 0;
static constexpr int OUTPUT_DGAMMA_INDEX = 1;
static constexpr int OUTPUT_DBETA_INDEX = 2;
using namespace ge;
namespace ge {
static ge::graphStatus InferShape4AddLayerNormGrad(gert::InferShapeContext *context)
{
    // infer shape
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4AddLayerNormGrad.");
    const gert::Shape *dy_shape = context->GetInputShape(DY_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dy_shape);

    const gert::Shape *data_gamma_shape = context->GetInputShape(GAMMA_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, data_gamma_shape);

    for (size_t i = 0; i < data_gamma_shape->GetDimNum(); i++) {
        OP_LOGD(context->GetNodeName(), " %ld", data_gamma_shape->GetDim(i));
    }

    gert::Shape *dx_shape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    gert::Shape *dgamma_shape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
    gert::Shape *dbeta_shape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dbeta_shape);

    *dx_shape = *dy_shape;
    dgamma_shape->SetDimNum(data_gamma_shape->GetDimNum());
    dbeta_shape->SetDimNum(data_gamma_shape->GetDimNum());
    for (uint32_t i = 0; i < data_gamma_shape->GetDimNum(); i++) {
        dgamma_shape->SetDim(i, data_gamma_shape->GetDim(i));
        dbeta_shape->SetDim(i, data_gamma_shape->GetDim(i));
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShape4AddLayerNormGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddLayerNormGrad(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4AddLayerNormGrad");
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(DY_INPUT_INDEX));
    context->SetOutputDataType(OUTPUT_DGAMMA_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_DBETA_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4AddLayerNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddLayerNormGrad)
    .InferShape(InferShape4AddLayerNormGrad)
    .InferDataType(InferDataType4AddLayerNormGrad);
}  // namespace ge