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
 * \file reduce_sum_v2_proto.cpp
 * \brief
 */
#include "reduce_sum_v2_proto.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ReduceCommon(gert::InferShapeContext* context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto axes_tensor = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const bool* keep_dims = attrs->GetAttrPointer<bool>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, keep_dims);

  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());

  OP_CHECK(axes_size < 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "axes num cannot be less than 0!"),
           return ge::GRAPH_FAILED);

  if (axes_size == 0) {
    *out_shape = *in_shape;
    OP_LOGD(context->GetNodeName(), "axes is empty tensor, will ignore infer, set output shape = input shape");
    return ge::GRAPH_SUCCESS;
  }

  auto dtype = axes_tensor->GetDataType();
  OP_CHECK(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), "axes datatype must in (int32, int64)"),
           return ge::GRAPH_FAILED);
  if (dtype == ge::DT_INT32) {
    return ReduceDims<int32_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
  }
  return ReduceDims<int64_t>(in_shape, axes_tensor, axes_size, *keep_dims, out_shape);
}

IMPL_OP_INFERSHAPE(ReduceSumV2).InferShape(InferShape4ReduceCommon).InputsDataDependency({1});
}  // namespace ops
