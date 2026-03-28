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
 * \file pows.cpp
 * \brief
 */

#include "pows_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Pows);
 
// AICORE算子kernel
static const aclTensor *PowsAiCore(const aclTensor *self, const aclTensor *exponent,
                                  aclTensor *powsOut, aclOpExecutor *executor) {
  L0_DFX(PowsAiCore, self, exponent, powsOut);

  ADD_TO_LAUNCHER_LIST_AICORE(Pows,
                              OP_INPUT(self, exponent),
                              OP_OUTPUT(powsOut));
  return powsOut;
}
 
const aclTensor *Pows(const aclTensor *self, const aclTensor *exponent, aclOpExecutor *executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), exponent->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(exponent->GetViewShape()).GetString());
    return nullptr;
  }

  auto powsOut = executor->AllocTensor(broadcastShape, self->GetDataType());
  CHECK_RET(powsOut != nullptr, nullptr);
  return PowsAiCore(self, exponent, powsOut, executor);
}
}
