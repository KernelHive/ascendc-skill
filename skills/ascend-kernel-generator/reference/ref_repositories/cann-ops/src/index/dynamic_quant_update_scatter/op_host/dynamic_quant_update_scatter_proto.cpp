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
 * \file dynamic_quant_update_scatter_proto.cpp
 * \brief
 */
#include <cstdint>
#include <string>
#include "register/op_def_registry.h"
using namespace ge;
namespace ops{
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)        \
if ((ptr) == nullptr)                                    \
{                                                        \
    std::printf("nullptr error!");                       \
    return ge::GRAPH_SUCCESS;                            \
}                                                        \

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
    do {                                                      \
        std::printf("op[%s], %s", op_name, err_msg);          \
    } while (0)                                               

#define OP_CHECK(cond, log_func, return_expr) \
    do {                                      \
        if (!(cond)) {                        \
            log_func;                         \
            return_expr;                      \
        }                                     \
    } while (false)

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)

constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
static inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline ge::graphStatus SetUnknownRank(gert::Shape* outShape) {
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
std::string ToString(const gert::Shape& shape);

static ge::graphStatus InferShape4DynamicQuantUpdateScatter(gert::InferShapeContext* context) {
    OP_LOGD(context->GetNodeName(), "Begin to do Infershape of InferShape4DynamicQuantUpdateScatter.");
    const gert::Shape* varShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* varScaleShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varScaleShape);
    
    gert::Shape* varOutputShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varOutputShape);
    gert::Shape* varScaleOutputShape = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varScaleOutputShape);
    
    if (IsUnknownRank(varShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        SetUnknownRank(varOutputShape);
    } else {
        *varOutputShape = *varShape;
    }
    
    if (IsUnknownRank(varScaleShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        SetUnknownRank(varScaleOutputShape);
    } else {
        *varScaleOutputShape = *varScaleShape;
    }
    
    OP_LOGD(context->GetNodeName(), "varOutputShape = %s.", ToString(*varOutputShape).c_str());
    OP_LOGD(context->GetNodeName(), "varScaleOutputShape = %s.", ToString(*varScaleOutputShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShape4DynamicQuantUpdateScatter.");
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantUpdateScatterInferDataType(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    auto input_var_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_var_dtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicQuantUpdateScatter)
    .InferShape(InferShape4DynamicQuantUpdateScatter)
    .InferDataType(DynamicQuantUpdateScatterInferDataType);
} // namespace ops