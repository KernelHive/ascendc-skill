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
 * \file deep_norm_grad_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
}

static constexpr int INPUT_X_INDEX = 1;
static constexpr int INPUT_GX_INDEX = 2;
static constexpr int INPUT_GAMMA_INDEX = 3;
static constexpr int OUTPUT_DX_INDEX = 0;
static constexpr int OUTPUT_DGX_INDEX = 1;
static constexpr int OUTPUT_DBETA_INDEX = 2;
static constexpr int OUTPUT_DGAMMA_INDEX = 3;
using namespace ge;
namespace ge {
static ge::graphStatus InferShape4DeepNormGrad(gert::InferShapeContext *context)
{
    // infer shape
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4DeepNormGrad.");

    const gert::Shape *x_shape = context->GetInputShape(INPUT_X_INDEX);
    const gert::Shape *gx_shape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::Shape *gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gx_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    gert::Shape *dx_shape = context->GetOutputShape(OUTPUT_DX_INDEX);
    gert::Shape *dgx_shape = context->GetOutputShape(OUTPUT_DGX_INDEX);
    gert::Shape *dbeta_shape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    gert::Shape *dgamma_shape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgx_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dbeta_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);

    *dx_shape = *x_shape;
    *dgx_shape = *gx_shape;
    *dbeta_shape = *gamma_shape;
    *dgamma_shape = *gamma_shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4DeepNormGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4DeepNormGrad(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DeepNormGrad");
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_DGX_INDEX, context->GetInputDataType(INPUT_GX_INDEX));
    context->SetOutputDataType(OUTPUT_DBETA_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_DGAMMA_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4DeepNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeepNormGrad).InferShape(InferShape4DeepNormGrad).InferDataType(InferDataType4DeepNormGrad);
}  // namespace ge
