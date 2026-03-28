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
 * \file dua_quantize_add_layer_norm_proto.cpp
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

static constexpr int IDX_X1 = 0;
static constexpr int IDX_X2 = 1;
static constexpr int IDX_GAMMA = 2;
static constexpr int IDX_BETA = 3;
static constexpr int IDX_BIAS = 4;
static constexpr int IDX_Y1 = 0;
static constexpr int IDX_Y2 = 1;
static constexpr int IDX_X = 2;

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4DuaQuantizeAddLayerNorm(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4DuaQuantizeAddLayerNorm");

    // get input shapes
    const gert::Shape *x1Shape = context->GetInputShape(IDX_X1);
    const gert::Shape *x2Shape = context->GetInputShape(IDX_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape *gammaShape = context->GetInputShape(IDX_GAMMA);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shape
    gert::Shape *y1Shape = context->GetOutputShape(IDX_Y1);
    gert::Shape *y2Shape = context->GetOutputShape(IDX_Y2);
    gert::Shape *xShape = context->GetOutputShape(IDX_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *y1Shape = *x1Shape;
    *y2Shape = *x2Shape;
    *xShape = *x1Shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4DuaQuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DuaQuantizeAddLayerNorm(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DuaQuantizeAddLayerNorm");
    context->SetOutputDataType(IDX_Y1, DT_INT8);
    context->SetOutputDataType(IDX_Y2, DT_INT8);
    context->SetOutputDataType(IDX_X, context->GetInputDataType(IDX_X1));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4DuaQuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DuaQuantizeAddLayerNorm)
    .InferShape(InferShape4DuaQuantizeAddLayerNorm)
    .InferDataType(InferDataType4DuaQuantizeAddLayerNorm);

}  // namespace ops
