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
 * \file mse_loss.cpp
 * \brief
 */
#include "level0/mse_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MSELossV2);

static const std::string REDUCTION_NONE = "none";

const aclTensor* MSELossV2(const aclTensor *self, const aclTensor *target, const std::string &reduction,
                         aclOpExecutor *executor) {
  L0_DFX(MSELossV2, self, target)
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and target tensor shape:%s can't broadcast.",
            ToString(self->GetViewShape()).GetString(), ToString(target->GetViewShape()).GetString());
    return nullptr;
  }
  aclTensor *lossOut = nullptr;
  if (reduction == REDUCTION_NONE) {
    lossOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  } else {
    lossOut = executor->AllocTensor({}, self->GetDataType());
  }
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MSELossV2, OP_INPUT(self, target), OP_OUTPUT(lossOut), OP_ATTR(reduction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MSELossV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return lossOut;
}
}  // namespace l0op
