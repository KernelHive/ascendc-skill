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
 * \file group_norm_swish_proto.cpp
 * \brief
 */

#include "error/ops_error.h"
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMSWISH_IDX_IN_X = 0;
static constexpr size_t GROUPNORMSWISH_IDX_IN_GAMMA = 1;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_Y = 0;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_MEAN = 1;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_VAR = 2;
static constexpr size_t NUMGROUPS_IDX = 0;
static constexpr size_t N_IDX = 0;

static ge::graphStatus
GroupNormSwishInferShape(gert::InferShapeContext *context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do GroupNormSwishInferShape");

  // get input shapes
  const gert::Shape *x_shape = context->GetInputShape(GROUPNORMSWISH_IDX_IN_X);
  OPS_LOG_E_IF_NULL(context, x_shape, return ge::GRAPH_FAILED);

  // get output shapes
  gert::Shape *y_shape = context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_Y);
  OPS_LOG_E_IF_NULL(context, y_shape, return ge::GRAPH_FAILED);
  gert::Shape *mean_shape =
      context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_MEAN);
  OPS_LOG_E_IF_NULL(context, mean_shape, return ge::GRAPH_FAILED);
  gert::Shape *var_shape = context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_VAR);
  OPS_LOG_E_IF_NULL(context, var_shape, return ge::GRAPH_FAILED);

  *y_shape = *x_shape;
  mean_shape->SetDimNum(0);

  // process attr
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t *num_groups = attrs->GetAttrPointer<int64_t>(NUMGROUPS_IDX);
  OPS_LOG_E_IF_NULL(context, num_groups, return ge::GRAPH_FAILED);

  // update mean and var shape
  const int64_t n_dim = x_shape->GetDim(N_IDX);
  mean_shape->AppendDim(n_dim);
  mean_shape->AppendDim(*num_groups);
  *var_shape = *mean_shape;

  OPS_LOG_D(context->GetNodeName(), "End to do GroupNormSwishInferShape");
  return ge::GRAPH_SUCCESS;
}

static graphStatus
GroupNormSwishInferDtype(gert::InferDataTypeContext *context) {
  OPS_LOG_D(context->GetNodeName(), "GroupNormSwishInferDtype enter");

  // Get input tout
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  auto inputXDtype = context->GetInputDataType(GROUPNORMSWISH_IDX_IN_X);
  auto inputGammaDtype = context->GetInputDataType(GROUPNORMSWISH_IDX_IN_GAMMA);
  context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_Y, inputXDtype);
  context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_MEAN, inputGammaDtype);
  context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_VAR, inputGammaDtype);

  OPS_LOG_D(context->GetNodeName(), "GroupNormSwishInferDtype end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSwish)
    .InferShape(GroupNormSwishInferShape)
    .InferDataType(GroupNormSwishInferDtype);
} // namespace ops