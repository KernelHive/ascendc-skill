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
 * \file batch_norm_v3_proto.cc
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

using namespace ge;
namespace ops {
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t WEIGHT_INPUT_IDX = 1;
static constexpr int64_t Y_OUTPUT_IDX = 0;
static constexpr int64_t MEAN_OUTPUT_IDX = 1;
static constexpr int64_t VAR_OUTPUT_IDX = 2;
static constexpr int64_t SAVE_MEAN_OUTPUT_IDX = 3;
static constexpr int64_t SAVE_INVSTD_OUTPUT_IDX = 4;

static ge::graphStatus BatchNormV3InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *weightShape = context->GetInputShape(WEIGHT_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    gert::Shape *yShape = context->GetOutputShape(Y_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape *meanShape = context->GetOutputShape(MEAN_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    gert::Shape *varianceShape = context->GetOutputShape(VAR_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, varianceShape);
    gert::Shape *saveMeanShape = context->GetOutputShape(SAVE_MEAN_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, saveMeanShape);
    gert::Shape *saveInvstdShape = context->GetOutputShape(SAVE_INVSTD_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, saveInvstdShape);

    *yShape = *xShape;
    *meanShape = *weightShape;
    *varianceShape = *weightShape;
    *saveMeanShape = *weightShape;
    *saveInvstdShape = *weightShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormV3InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(X_INPUT_IDX);
    context->SetOutputDataType(Y_OUTPUT_IDX, xDtype);
    // other output is float32
    context->SetOutputDataType(MEAN_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(VAR_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(SAVE_MEAN_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(SAVE_INVSTD_OUTPUT_IDX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNormV3).InferShape(BatchNormV3InferShape).InferDataType(BatchNormV3InferDataType);
}  // namespace ops