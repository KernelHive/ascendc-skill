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
 * \file group_norm_grad_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static constexpr size_t GROUPNORMGRAD_IDX_IN_DY = 0;
static constexpr size_t GROUPNORMGRAD_IDX_IN_GAMMA = 4;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DX = 0;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DGAMMA = 1;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DBETA = 2;

static ge::graphStatus GroupNormGradInferShape(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GroupNormGradInferShape");

    // get input shapes
    const gert::Shape *dy_shape = context->GetInputShape(GROUPNORMGRAD_IDX_IN_DY);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dy_shape);
    const gert::Shape *gamma_shape = context->GetInputShape(GROUPNORMGRAD_IDX_IN_GAMMA);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    // get output shapes
    gert::Shape *dx_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    gert::Shape *dgamma_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DGAMMA);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
    gert::Shape *dbeta_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DBETA);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dbeta_shape);

    *dx_shape = *dy_shape;
    *dgamma_shape = *gamma_shape;
    *dbeta_shape = *gamma_shape;

    OP_LOGD(context->GetNodeName(), "End to do GroupNormGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormGradInferDtype(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "GroupNormGradInferDtype enter");
    // Get input tout
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputDtype = context->GetInputDataType(GROUPNORMGRAD_IDX_IN_DY);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DX, inputDtype);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DGAMMA, inputDtype);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DBETA, inputDtype);

    OP_LOGD(context->GetNodeName(), "GroupNormGradInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormGrad).InferShape(GroupNormGradInferShape).InferDataType(GroupNormGradInferDtype);
}  // namespace ops