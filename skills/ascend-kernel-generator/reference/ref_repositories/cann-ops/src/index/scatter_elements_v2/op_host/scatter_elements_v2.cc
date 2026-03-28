/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_elements_v2.cc
 * \brief
 */
#include "runtime_util.h"
#include "op_util.h"

using namespace ge;
namespace ops {
static graphStatus InferDataType4ScatterElementsV2(gert::InferDataTypeContext *context) {
  auto var_dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, var_dtype);

  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterElementsV2(gert::InferShapeContext* context) {
  const gert::Shape* var_in_shape = context->GetInputShape(0);
  gert::Shape* var_out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, var_in_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, var_out_shape);

  *var_out_shape = *var_in_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterElementsV2).InferShape(InferShape4ScatterElementsV2).InferDataType(InferDataType4ScatterElementsV2);
}  // namespace ops
