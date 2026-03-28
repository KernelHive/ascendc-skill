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
 * \file diag_flat_ops.cc
 * \brief
 */
#include <cmath>
#include "register/op_def_registry.h"

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define OP_CHECK(cond, log_func, ...) \
    do {                              \
        if (cond) {                   \
            log_func;                 \
            return ge::GRAPH_FAILED;  \
        }                             \
    } while (0)

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

inline bool IsUnknownRank(const gert::Shape *check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline bool IsUnknownShape(const gert::Shape *check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

using namespace ge;

namespace ops {
// ------------------- Diagflat Ops START---------------------  1->2
static constexpr size_t DIAGFLAT_IN_X_IDX = 0;
static constexpr size_t DIAGFLAT_OUT_Y_IDX = 0;

template <typename T>
std::string ToString(const T *value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ", ";
    }
    r = r + "]";
    return r;
}

inline ge::graphStatus SetUnknownRank(gert::Shape *output_shape)
{
    OP_CHECK(output_shape == nullptr,
        OP_LOGD("SetUnknownRank", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InfershapeForStridedSliceAssignV2(gert::InferShapeContext *context)
{
    auto in_shape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    if (IsUnknownRank(in_shape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        return SetUnknownRank(out_shape);
    }

    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForStridedSliceAssignV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InfershapeForStridedSliceAssignV2 enter");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InfershapeForStridedSliceAssignV2 end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceAssignV2)
    .InferShape(InfershapeForStridedSliceAssignV2)
    .InferDataType(InferDataTypeForStridedSliceAssignV2)
    .InputsDataDependency({2, 3, 4, 5});

// -------------------StridedSliceAssignV2 Ops END---------------------

}  // namespace ops
