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
 * \file cross_entropy_loss.cc
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
namespace {
    constexpr uint32_t INPUT_DATA_IDX = 0;
    constexpr uint32_t INPUT_TARGET_IDX = 1;
    constexpr uint32_t INPUT_WEIGHT_IDX = 2;
    constexpr uint32_t OUTPUT_LOSS_IDX = 0;
    constexpr uint32_t OUTPUT_LOGPROB_IDX = 1;
    constexpr uint32_t ATTR_REDUCTION_IDX = 0;
    constexpr uint32_t DIM_0 = 0;
    constexpr uint32_t DIM_1 = 1;
    constexpr uint32_t DIM_NUM_1 = 1;
    constexpr uint32_t DIM_NUM_2 = 2;
    constexpr uint32_t LOSS_SHAPE = 1;
}

namespace ops {
static ge::graphStatus InferShapeForCrossEntropyLoss(gert::InferShapeContext *context) 
{
    // input shape
    OP_LOGD(context->GetNodeName(), "InferShapeForCrossEntropyLoss Begin.");
    const gert::Shape* inputShape = context->GetInputShape(INPUT_DATA_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, targetShape);

    // output shape
    gert::Shape* lossShape = context->GetOutputShape(OUTPUT_LOSS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    gert::Shape* logprobShape = context->GetOutputShape(OUTPUT_LOGPROB_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, logprobShape); 

    if (IsUnknownRank(inputShape)) { // -2
        SetUnknownRank(lossShape);
        SetUnknownRank(logprobShape);
    } else {
        OP_CHECK(inputShape->GetDimNum() != DIM_NUM_2, 
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Input dim must be 2."), 
                 return ge::GRAPH_FAILED);

        OP_CHECK(targetShape->GetDimNum() != DIM_NUM_1, 
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "target dim must be 1."), 
                 return ge::GRAPH_FAILED);

        OP_CHECK(inputShape->GetDim(DIM_0) != UNKNOWN_DIM && targetShape->GetDim(DIM_0) != UNKNOWN_DIM &&
                 inputShape->GetDim(DIM_0) != targetShape->GetDim(DIM_0),
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), 
                 "Input dim 0 should be equal to target dim 0."), 
                 return ge::GRAPH_FAILED);    

        const gert::Shape* weightShape = context->GetInputShape(INPUT_WEIGHT_IDX); // optional input
        if (weightShape != nullptr) {
            OP_CHECK(weightShape->GetDimNum() != DIM_NUM_1, 
                     VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "weight dim must be 1."), 
                     return ge::GRAPH_FAILED);

            OP_CHECK(inputShape->GetDim(DIM_1) != UNKNOWN_DIM && weightShape->GetDim(DIM_0) != UNKNOWN_DIM &&
                     inputShape->GetDim(DIM_1) != weightShape->GetDim(DIM_0),
                     VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), 
                     "Input dim 1 should be equal to weight dim 0."), 
                     return ge::GRAPH_FAILED);
        }
        const gert::RuntimeAttrs* attrs = context->GetAttrs();
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
        if (reduction != nullptr && strcmp(reduction, "none") == 0) {
            lossShape->SetDimNum(DIM_NUM_1);
            lossShape->SetDim(DIM_0, inputShape->GetDim(DIM_0));
        } else {
            lossShape->SetDimNum(DIM_NUM_1);
            lossShape->SetDim(DIM_0, LOSS_SHAPE);
        }
        *logprobShape = *inputShape;
    }
    OP_LOGD(context->GetNodeName(), "InferShapeForCrossEntropyLoss End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForCrossEntropyLoss(gert::InferDataTypeContext *context) {
    OP_LOGD(context->GetNodeName(), "InferDataTypeForCrossEntropyLoss Begin.");
    context->SetOutputDataType(OUTPUT_LOSS_IDX, context->GetInputDataType(INPUT_DATA_IDX));
    context->SetOutputDataType(OUTPUT_LOGPROB_IDX, context->GetInputDataType(INPUT_DATA_IDX));
    OP_LOGD(context->GetNodeName(), "InferDataTypeForCrossEntropyLoss End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CrossEntropyLoss).InferShape(InferShapeForCrossEntropyLoss)
                                    .InferDataType(InferDataTypeForCrossEntropyLoss);
}
