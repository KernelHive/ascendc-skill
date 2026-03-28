/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file multi_scale_deformable_attn_function_l0.cpp
 * \brief
 */

#include "multi_scale_deformable_attn_function_l0.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
using namespace op;
namespace l0op {

OP_TYPE_REGISTER(MultiScaleDeformableAttnFunction);
static const int64_t VALUE_INDEX_0 = 0;
static const int64_t VALUE_INDEX_3 = 3;
static const int64_t LOCATION_INDEX_1 = 1;
static const int64_t LOCATION_INDEX_5 = 5;
static const int64_t VALUE_INDEX_1 = 1;
static const int64_t VALUE_INDEX_2 = 2;
static const int64_t NUMQUERIES_MIN = 32;

const std::tuple<aclTensor*> MultiScaleDeformableAttnFunction(const aclTensor *value,
                                                              const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                                                              const aclTensor *location, const aclTensor *attnWeight,
                                                              aclOpExecutor *executor) {
  L0_DFX(MultiScaleDeformableAttnFunction, value, spatialShape, levelStartIndex, location, attnWeight);
  // shape推导
  Shape outputShape;
  auto valueShape = value->GetViewShape();
  auto locationShape = location->GetViewShape();

  uint64_t numQueries = locationShape.GetDim(1);
  uint64_t nqIndex = LOCATION_INDEX_1;
  uint64_t nhIndex = VALUE_INDEX_2;
  if (numQueries < NUMQUERIES_MIN) {
    nqIndex = LOCATION_INDEX_5;
    nhIndex = VALUE_INDEX_1;
  } 
  outputShape.AppendDim(valueShape.GetDim(VALUE_INDEX_0));
  outputShape.AppendDim(locationShape.GetDim(nqIndex));
  outputShape.AppendDim(valueShape.GetDim(nhIndex) * valueShape.GetDim(VALUE_INDEX_3));

  // 创建输出Tensor
  auto output = executor->AllocTensor(outputShape, value->GetDataType());

  // 调用device的MultiScaleDeformableAttnFunction算子
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MultiScaleDeformableAttnFunction,
                                         OP_INPUT(value, spatialShape, levelStartIndex, location, attnWeight),
                                         OP_OUTPUT(output));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MultiScaleDeformableAttnFunctionAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    std::tuple<aclTensor*>(nullptr));
  return std::tie(output);
}
} // l0op
