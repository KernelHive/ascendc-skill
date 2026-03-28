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
 * \file feeds_repeat_proto.cc
 * \brief
 */
#include <base/err_msg.h>
#include "register/op_def_registry.h"
#include "aclnn/opdev/op_log.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
  }
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                                \
  do {                                                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", err_msg);                                          \
    REPORT_INNER_ERR_MSG("89999", "%s",                                                        \
                       err_msg);                                                             \
  } while (0)

namespace {
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
inline bool IsUnknownRank(const gert::Shape* check_shape) {
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}
inline ge::graphStatus SetUnknownRank(gert::Shape* outShape) {
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}
}  // namespace

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4FeedsRepeat(gert::InferShapeContext* context) {
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4FeedsRepeat");
    const gert::Shape* feeds_shape = context->GetInputShape(0);
    OP_CHECK(feeds_shape == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get feedsrepeat feeds_shape failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, feeds_shape);
    const gert::Shape* repeat_times_shape = context->GetInputShape(1);
    OP_CHECK(repeat_times_shape == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get feedsrepeat repeat_times_shape failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, repeat_times_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK(y_shape == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get feedsrepeat y_shape failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK(attrs == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get feedsrepeat runtime attrs failed"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    if (IsUnknownRank(y_shape)) { // [-2]输入
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to -2");
        return SetUnknownRank(y_shape);
    }
    else{
        *y_shape = *feeds_shape;
        const int* output_feeds_size = attrs->GetAttrPointer<int>(0);
        y_shape->SetDim(0, *output_feeds_size);
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShape4FeedsRepeat");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FeedsRepeat(gert::InferDataTypeContext* context) {
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4FeedsRepeat");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4FeedsRepeat");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FeedsRepeat)
    .InferShape(InferShape4FeedsRepeat)
    .InferDataType(InferDataType4FeedsRepeat);
}  // namespace ops