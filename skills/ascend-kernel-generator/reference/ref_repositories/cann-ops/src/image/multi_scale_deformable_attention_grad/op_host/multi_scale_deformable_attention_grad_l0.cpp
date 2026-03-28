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
 * \file multi_scale_deformable_attention_grad_l0.cpp
 * \brief
 */

#include "multi_scale_deformable_attention_grad_l0.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

namespace l0op {

OP_TYPE_REGISTER(MultiScaleDeformableAttentionGrad);

const std::tuple<aclTensor*, aclTensor*, aclTensor*> MultiScaleDeformableAttentionGrad(const aclTensor *value,
                                                                                       const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                                                                                       const aclTensor *location, const aclTensor *attnWeight,
                                                                                       const aclTensor *gradOutput, aclOpExecutor *executor) {
  L0_DFX(MultiScaleDeformableAttentionGrad, value, spatialShape, levelStartIndex, location, attnWeight, gradOutput);
  // shape推导
  auto valueShape = value->GetViewShape();
  auto locationShape = location->GetViewShape();
  auto attnWeightShape = attnWeight->GetViewShape();

  // 创建输出Tensor
  auto gradValue = executor->AllocTensor(valueShape, value->GetDataType());
  auto gradLocation = executor->AllocTensor(locationShape, location->GetDataType());
  auto gradAttnWeight = executor->AllocTensor(attnWeightShape, attnWeight->GetDataType());

  // 调用device的MultiScaleDeformableAttentionGrad算子
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MultiScaleDeformableAttentionGrad,
                                         OP_INPUT(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput),
                                         OP_OUTPUT(gradValue, gradLocation, gradAttnWeight));
  if (ret !=  ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MultiScaleDeformableAttentionGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr);
  }
  return std::tie(gradValue, gradLocation, gradAttnWeight);
}
} // l0op
