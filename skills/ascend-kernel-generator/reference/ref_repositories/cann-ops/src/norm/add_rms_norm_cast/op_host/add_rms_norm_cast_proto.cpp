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
 * \file add_rms_norm_cast_proto.cpp
 * \brief
 */
#include <vector>
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;

namespace ops {

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static ge::graphStatus InferShape4AddRmsNormCast(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4AddRmsNormCast");

    // get input shapes
    const gert::Shape *x1Shape = context->GetInputShape(IDX_0);
    const gert::Shape *x2Shape = context->GetInputShape(IDX_1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape *gammaShape = context->GetInputShape(IDX_2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape *y1Shape = context->GetOutputShape(IDX_0);
    gert::Shape *y2Shape = context->GetOutputShape(IDX_1);
    gert::Shape *xShape = context->GetOutputShape(IDX_3);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *y1Shape = *x1Shape;
    *y2Shape = *x1Shape;
    *xShape = *x1Shape;

    size_t xDimNum = x1Shape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();

    gert::Shape *rstdShape = context->GetOutputShape(IDX_2);
    rstdShape->SetDimNum(xDimNum);
    for (size_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - gammaDimNum) {
            rstdShape->SetDim(i, x1Shape->GetDim(i));
        } else {
            rstdShape->SetDim(i, 1);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4AddRmsNormCast");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormCast(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4AddRmsNormCast");
    context->SetOutputDataType(IDX_0, DT_FLOAT);
    context->SetOutputDataType(IDX_1, context->GetInputDataType(IDX_0));
    context->SetOutputDataType(IDX_2, DT_FLOAT);
    context->SetOutputDataType(IDX_3, context->GetInputDataType(IDX_0));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4AddRmsNormCast");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormCast).InferShape(InferShape4AddRmsNormCast).InferDataType(InferDataType4AddRmsNormCast);
}  // namespace ops
