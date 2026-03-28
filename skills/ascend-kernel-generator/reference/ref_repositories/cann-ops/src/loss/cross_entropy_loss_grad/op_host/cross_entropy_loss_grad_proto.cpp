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
 * \file cross_entropy_loss_grad.cc
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
constexpr uint64_t INPUT_Y_GRAD_IDX = 0;
constexpr uint64_t INPUT_LOG_PROB_IDX = 1;
constexpr uint64_t INPUT_TARGET_IDX = 2;
constexpr uint64_t INPUT_WEIGHT_IDX = 3;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint64_t OUTPUT_X_GRAD_IDX = 0;

static graphStatus InferShape4CrossEntropyLossGrad(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShape4CrossEntropyLossGrad.");
  const gert::Shape* yGradShape = context->GetInputShape(INPUT_Y_GRAD_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, yGradShape);
  const gert::Shape* logProbShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, logProbShape);
  const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, targetShape);
  const gert::Shape* weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_IDX);

  gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xGradShape);

  if (IsUnknownRank(logProbShape)) { // [-2]输入
    OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2)");
    return SetUnknownRank(xGradShape);
  } else {
    OP_CHECK(logProbShape->GetDimNum() != DIM_NUM_2,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "logProb dim must be 2."),
            return ge::GRAPH_FAILED);

    OP_CHECK(targetShape->GetDimNum() != DIM_NUM_1,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "target dim must be 1."),
            return ge::GRAPH_FAILED);

    OP_CHECK(logProbShape->GetDim(DIM_0) != UNKNOWN_DIM && targetShape->GetDim(DIM_0) != UNKNOWN_DIM &&
            logProbShape->GetDim(DIM_0) != targetShape->GetDim(DIM_0),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "logProb dim 0 should be equal to target dim 0."),
            return ge::GRAPH_FAILED);
  }

  if (weightShape != nullptr) {
    OP_LOGD(context->GetNodeName(), "InferShape4CrossEntropyLossGrad: weightShape is not null");
    OP_CHECK(weightShape->GetDimNum() != DIM_NUM_1,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "weight dim must be 1."),
            return ge::GRAPH_FAILED);

    OP_CHECK(logProbShape->GetDim(DIM_1) != UNKNOWN_DIM && weightShape->GetDim(DIM_0) != UNKNOWN_DIM &&
            logProbShape->GetDim(DIM_1) != weightShape->GetDim(DIM_0),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "logProb dim 1 should be equal to weight dim 0."),
            return ge::GRAPH_FAILED);
  } else {
    OP_LOGD(context->GetNodeName(), "InferShape4CrossEntropyLossGrad: weightShape is null.");
  }  

  *xGradShape = *logProbShape;
  OP_LOGD(context->GetNodeName(), "End to do InferShape4CrossEntropyLossGrad.");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForCrossEntropyLossGrad(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferDataTypeForCrossEntropyLossGrad Begin.");
  context->SetOutputDataType(OUTPUT_X_GRAD_IDX, context->GetInputDataType(INPUT_LOG_PROB_IDX));
  OP_LOGD(context->GetNodeName(), "InferDataTypeForCrossEntropyLossGrad End.");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CrossEntropyLossGrad)
  .InferShape(InferShape4CrossEntropyLossGrad)
  .InferDataType(InferDataTypeForCrossEntropyLossGrad);
}  // namespace ops
