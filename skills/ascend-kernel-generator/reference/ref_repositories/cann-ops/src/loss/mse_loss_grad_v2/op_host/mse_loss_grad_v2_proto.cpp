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
 * \file mse_loss_grad_v2_proto.cpp
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

namespace {
constexpr uint32_t INPUT_PREDICT_IDX = 0;
constexpr uint32_t INPUT_LABEL_IDX = 1;
constexpr uint32_t INPUT_DOUT_IDX = 2;
constexpr uint32_t OUTPUT_LOSS_GRAD_IDX = 0;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
}

namespace ops {
static ge::graphStatus InferShapeForMseLossGrad(gert::InferShapeContext *context) 
{
    // input shape
    OP_LOGD(context->GetNodeName(), "MseLossGrad Begin.");
    const gert::Shape* predictShape = context->GetInputShape(INPUT_PREDICT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(INPUT_LABEL_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    const gert::Shape* doutShape = context->GetInputShape(INPUT_DOUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, doutShape);
        
    // output shape
    gert::Shape* lossGradShape = context->GetOutputShape(OUTPUT_LOSS_GRAD_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, lossGradShape);
    if (IsUnknownRank(predictShape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2,)");
        lossGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_);
        return ge::GRAPH_FAILED;
    }
    *lossGradShape = *predictShape;
    OP_LOGD(context->GetNodeName(), "InferShapeForMseLossGrad End.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMseLossGrad(gert::InferDataTypeContext *context)
{
    const ge::DataType predictDtype = context->GetInputDataType(INPUT_PREDICT_IDX);
    context->SetOutputDataType(OUTPUT_LOSS_GRAD_IDX, predictDtype);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MseLossGradV2).InferShape(InferShapeForMseLossGrad).InferDataType(InferDataTypeForMseLossGrad);
}  // namespace ops