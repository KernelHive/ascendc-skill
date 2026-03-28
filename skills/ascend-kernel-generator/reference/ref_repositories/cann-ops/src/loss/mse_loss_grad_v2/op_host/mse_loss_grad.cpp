/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file mse_loss_grad.cpp
 * \brief
 */

#include "level0/mse_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
 
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MseLossGradV2);

const aclTensor *MseLossGradV2(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out) {
  OP_LOGD("Entering L0 MseLossGradV2");
  L0_DFX(MseLossGradV2, gradOutput, self, target, reduction)
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MseLossGradV2, OP_INPUT(self, target, gradOutput), OP_OUTPUT(out), OP_ATTR(reduction));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MseLossGradV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op