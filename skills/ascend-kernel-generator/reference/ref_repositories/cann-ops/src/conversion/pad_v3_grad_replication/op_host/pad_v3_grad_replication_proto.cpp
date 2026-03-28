/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file pad_v3_grad_replication_proto.cpp
 * \brief
 */
#include "register/op_def_registry.h"

// tools api
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
}

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


namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_PADDINGS = 1;
constexpr size_t INDEX_Y = 0;
constexpr size_t INDEX_PADDINGS_CONTIGUOUS = 1;
constexpr size_t PAIR = 2;
}  // namespace

// proto
using namespace ge;
namespace ops {
template <typename T>
static ge::graphStatus Pad3dGradReplicationInfershape(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                        const gert::Tensor* paddings_tensor, gert::Shape* y_shape)
{
    const T* paddings_value = paddings_tensor->GetData<T>();
    const size_t paddings_num = static_cast<size_t>(paddings_tensor->GetShapeSize());
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* paddings_contiguous = attrs->GetAttrPointer<bool>(INDEX_PADDINGS_CONTIGUOUS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, paddings_contiguous);
    size_t input_dim_size = static_cast<size_t>(x_shape->GetDimNum());
    // pad size check
    if (input_dim_size * PAIR != paddings_num) {
        std::string err_msg = "the paddings num must be twice of the input x rank";
        return ge::GRAPH_FAILED;
    }
    // infer by paddings_contiguous
    y_shape->SetDimNum(input_dim_size);
    int64_t index_cof = 1;
    int64_t index_offset = input_dim_size;
    if (*paddings_contiguous) {
        index_cof = PAIR;
        index_offset = 1;
    }
    for (size_t i = 0; i < input_dim_size; ++i) {
        auto pad_front = paddings_value[index_cof * i];
        auto pad_end = paddings_value[index_cof * i + index_offset];
        y_shape->SetDim(i, x_shape->GetDim(i) - pad_front - pad_end);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePadV3GradReplication(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(INDEX_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(INDEX_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::Tensor* paddings_tensor = context->GetInputTensor(INDEX_PADDINGS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);
    ge::DataType paddings_dtype = paddings_tensor->GetDataType();
    switch (paddings_dtype) {
        case ge::DT_INT32: {
            return Pad3dGradReplicationInfershape<int32_t>(context, x_shape, paddings_tensor, y_shape);
        }
        case ge::DT_INT64: {
            return Pad3dGradReplicationInfershape<int64_t>(context, x_shape, paddings_tensor, y_shape);
        }
        default:
            return ge::GRAPH_FAILED;
    }
}

IMPL_OP_INFERSHAPE(PadV3GradReplication)
    .InferShape(InferShapePadV3GradReplication)
    .InputsDataDependency({INDEX_PADDINGS});
} // namespace ops