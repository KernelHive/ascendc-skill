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
 * \file mse_loss_v2_proto.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

using namespace ge;

namespace {
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

#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
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

using namespace ge;

namespace ops {
static constexpr size_t INPUT_INDEX = 0;
static constexpr size_t TARGET_INDEX = 1;
static constexpr size_t OUTPUT_INDEX = 0;

graphStatus InferShapeForMSELossV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForMSELossV2.");
  const gert::Shape* input = context->GetInputShape(INPUT_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input);
  const gert::Shape* target = context->GetInputShape(TARGET_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, target);
  gert::Shape* output = context->GetOutputShape(OUTPUT_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output);
  std::string reduction = context->GetAttrs()->GetAttrPointer<char>(0);
  if (reduction == "none") {
    *output = *input;
  } else {
    context->GetOutputShape(OUTPUT_INDEX)->SetDimNum(0);
  }
  OP_LOGD(context->GetNodeName(), "End to do InferShapeForMSELossV2.");
  return GRAPH_SUCCESS;
}

graphStatus InferDataTypeForMSELossV2(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(OUTPUT_INDEX, context->GetInputDataType(INPUT_INDEX));
  return GRAPH_SUCCESS;
}

IMPL_OP(MSELossV2).InferShape(InferShapeForMSELossV2).InferDataType(InferDataTypeForMSELossV2);
}  // namespace ops