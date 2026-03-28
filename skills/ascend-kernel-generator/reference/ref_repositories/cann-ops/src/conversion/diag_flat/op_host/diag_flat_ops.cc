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

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline bool IsUnknownShape(const gert::Shape* check_shape)
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
constexpr size_t K_AXIS_ATTR_IDX = 0U;
static constexpr size_t INT_DATA_2 = 2;


static ge::graphStatus InfershapeForDiagFlat(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do DiagflatInfershape.");
    // 获取输入值shape
    const gert::Shape* inputShape = context->GetInputShape(DIAGFLAT_IN_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    // 获取属性值
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t diagonal = *(attrs->GetInt(K_AXIS_ATTR_IDX));

    // 获取输出值shape
    gert::Shape* output_y_shape = context->GetOutputShape(DIAGFLAT_OUT_Y_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    const size_t inputDimNum = 2;
    if (IsUnknownRank(inputShape)) {
        output_y_shape->SetDimNum(inputDimNum);
        output_y_shape->SetDim(0, -1);
        output_y_shape->SetDim(1, -1);
    } else if (IsUnknownShape(inputShape)) {
        output_y_shape->SetDimNum(inputDimNum);
        output_y_shape->SetDim(0, -1);
        output_y_shape->SetDim(1, -1);
    } else {
        // 获取元素element的个数, 2D->1D的场景
        output_y_shape->SetDimNum(inputDimNum);
        auto total_element_num = inputShape->GetShapeSize();
        auto output_width = total_element_num + std::abs(diagonal);
        output_y_shape->SetDim(0, output_width);
        output_y_shape->SetDim(1, output_width);
    }

    OP_LOGD(context->GetNodeName(), "End to do DiagFlatInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DiagFlat).InferShape(InfershapeForDiagFlat);
// -------------------DiagFlat Ops END---------------------

}  // namespace ops

