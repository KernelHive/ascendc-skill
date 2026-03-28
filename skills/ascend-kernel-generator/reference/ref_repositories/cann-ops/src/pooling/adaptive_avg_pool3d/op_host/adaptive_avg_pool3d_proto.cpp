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
 * \file adaptive_avg_pool3d_proto.cpp
 * \brief
 */
#include <base/err_msg.h>
#include "register/op_def_registry.h"
#include "aclnn/opdev/op_log.h"

// tools api
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


namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t OUTPUT_SIZE_INDEX = 0;

constexpr size_t X_DIMS = 5;
constexpr size_t OUTPUT_SIZE_DIMS = 3;
}  // namespace

// proto
using namespace ge;
namespace ops {
static ge::graphStatus InferShape4AdaptiveAvgPool3d(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attr_ptr = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    auto output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);
    auto output_size = reinterpret_cast<const int64_t*>(output_size_ptr->GetData());

    if (IsUnknownRank(x_shape)) {
      return SetUnknownRank(y_shape);
    }

    size_t input_dim_num = x_shape->GetDimNum();
    OP_CHECK(input_dim_num != X_DIMS,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "The dims of x not equal 5."),
             return GRAPH_FAILED);
    y_shape->SetDimNum(input_dim_num);

    size_t output_size_len = output_size_ptr->GetSize();
    OP_CHECK(output_size_len != OUTPUT_SIZE_DIMS,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "The size of output_size not equal 3."),
             return GRAPH_FAILED);

    y_shape->SetDim(0, x_shape->GetDim(0));
    for (size_t i = 0; i < output_size_len; ++i) {
        y_shape->SetDim(i + 1, output_size[i]);
    }
    y_shape->SetDim(output_size_len + 1, x_shape->GetDim(output_size_len + 1));

    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4AdaptiveAvgPool3d(gert::InferDataTypeContext* context) {
  auto x_dtype = context->GetInputDataType(X_INDEX);
  context->SetOutputDataType(Y_INDEX, x_dtype);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool3d)
  .InferShape(InferShape4AdaptiveAvgPool3d)
  .InferDataType(InferDtype4AdaptiveAvgPool3d);

} // namespace ops
