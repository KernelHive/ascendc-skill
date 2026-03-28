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

#include "diag_flat_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include <cmath>
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

// *******************************************************************************************************
// ******************************** [AI CORE] DiagFlat 算子入口 ***************************************
// *******************************************************************************************************

namespace l0op {

OP_TYPE_REGISTER(DiagFlat);

// AICORE算子kernel
static const aclTensor *DiagFlatAiCore(const aclTensor *self, int64_t diagonal, aclTensor *diagFlatOut,
                                       aclOpExecutor *executor) {
  L0_DFX(DiagFlatAiCore, self, diagonal, diagFlatOut);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore DiagFlat算子加入任务队列
  // self是算子的输入，diagonal是算子的属性，out是算子的输出
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
    DiagFlat,
    OP_INPUT(self),
    OP_OUTPUT(diagFlatOut),
    OP_ATTR(diagonal)
  );
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "DiagFlat ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return diagFlatOut;
}

const aclTensor *DiagFlat(const aclTensor *self, int64_t diagonal, aclOpExecutor *executor) {
  auto outputLength = self->Numel() + std::abs(diagonal);
  op::Shape diagflatOutShape = self->GetViewShape();
  
  diagflatOutShape.SetDimNum(2);
  diagflatOutShape.SetDim(0, outputLength);
  diagflatOutShape.SetDim(1, outputLength);

  auto diagFlatOut = executor->AllocTensor(diagflatOutShape, self->GetDataType(), op::Format::FORMAT_ND);
  return DiagFlatAiCore(self, diagonal, diagFlatOut, executor);
}

}  // namespace l0op
