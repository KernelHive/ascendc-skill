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

#include "masked_select_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MaskedSelectV3);

// AICORE算子kernel
static const aclTensor* MaskedSelectAiCore(const aclTensor* self, const aclTensor* mask, aclTensor* out, aclOpExecutor* executor) {
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetStorageShape(), mask->GetStorageShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(mask->GetViewShape()).GetString());
    return nullptr;
  }

  L0_DFX(MaskedSelectAiCore, self, mask, out);
  // outShapeTensor用于执行期放置实际shape，并用于刷新out的大小。
  Shape outShapeShape{2};
  auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
  CHECK_RET(outShapeTensor != nullptr, nullptr);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将MaskedSelect算子加入任务队列
  // MaskedSelect是算子的OpType，self是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MaskedSelectV3,
                                         OP_INPUT(self, mask),
                                         OP_OUTPUT(out),
                                         OP_OUTSHAPE({outShapeTensor, 0}));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedSelectV3AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor* MaskedSelectV3(const aclTensor* self, const aclTensor* mask, aclTensor* out, aclOpExecutor* executor) {
  return MaskedSelectAiCore(self, mask, out, executor);
}
}  // namespace l0op
