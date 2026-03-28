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
 * \file group_norm_swish_grad_proto.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
  #define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
  #define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
  #define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
      std::printf("nullptr error!");                                                               \
      return ge::GRAPH_FAILED;                                                                     \
    }
  
  #define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
    if ((ptr) == nullptr) {                                                                        \
      std::printf("nullptr error!");                                                               \
      return ret;                                                                                  \
    }
  #define OP_TILING_CHECK(cond, log_func, expr)  \
    do {                                         \
      if (cond) {                                \
        std::printf(log_func);                     \
        expr;                                      \
      }                                          \
    } while (0)
}  // namespace ops

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMSWISHGRAD_IDX_IN_DY = 0;
static constexpr size_t GROUPNORMSWISHGRAD_IDX_IN_GAMMA = 4;
static constexpr size_t GROUPNORMSWISHGRAD_IDX_OUT_DX = 0;
static constexpr size_t GROUPNORMSWISHGRAD_IDX_OUT_DGAMMA = 1;
static constexpr size_t GROUPNORMSWISHGRAD_IDX_OUT_DBETA = 2;

static ge::graphStatus GroupNormSwishGradInferShape(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do GroupNormSwishGradInferShape");

  // get input shapes
  const gert::Shape* dy_shape = context->GetInputShape(GROUPNORMSWISHGRAD_IDX_IN_DY);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dy_shape);
  const gert::Shape* gamma_shape = context->GetInputShape(GROUPNORMSWISHGRAD_IDX_IN_GAMMA);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
  // get output shapes
  gert::Shape* dx_shape = context->GetOutputShape(GROUPNORMSWISHGRAD_IDX_OUT_DX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
  gert::Shape* dgamma_shape = context->GetOutputShape(GROUPNORMSWISHGRAD_IDX_OUT_DGAMMA);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
  gert::Shape* dbeta_shape = context->GetOutputShape(GROUPNORMSWISHGRAD_IDX_OUT_DBETA);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dbeta_shape);

  *dx_shape = *dy_shape;
  *dgamma_shape = *gamma_shape;
  *dbeta_shape = *gamma_shape;

  OP_LOGD(context->GetNodeName(), "End to do GroupNormSwishGradInferShape");
  return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormSwishGradInferDtype (gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "GroupNormSwishGradInferDtype enter");
  // Get input tout
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto inputDtype = context->GetInputDataType(GROUPNORMSWISHGRAD_IDX_IN_DY);
  context->SetOutputDataType(GROUPNORMSWISHGRAD_IDX_OUT_DX, inputDtype);
  context->SetOutputDataType(GROUPNORMSWISHGRAD_IDX_OUT_DGAMMA, inputDtype);
  context->SetOutputDataType(GROUPNORMSWISHGRAD_IDX_OUT_DBETA, inputDtype);

  OP_LOGD(context->GetNodeName(), "GroupNormSwishGradInferDtype end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSwishGrad)
  .InferShape(GroupNormSwishGradInferShape)
  .InferDataType(GroupNormSwishGradInferDtype);
}  // namespace ops
