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

#include "diag_v2_l0.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

namespace l0op {
OP_TYPE_REGISTER(DiagV2);

const aclTensor *DiagV2(const aclTensor *self, int64_t diagonal, aclOpExecutor *executor) {
  L0_DFX(DiagV2, self);
  // 固定写法，创建OpExecutor
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  auto ret = INFER_SHAPE(DiagV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(diagonal));
  if (ret != ACLNN_SUCCESS) {
      return nullptr;
  }
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(DiagV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(diagonal));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "DiagV2 add to aicore launch list failed.");
  return out;
}
}  // namespace l0op
