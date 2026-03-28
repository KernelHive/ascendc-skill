/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lin_space_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LinSpace);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor *start) {
  // 获取芯片类型
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return CheckType(start->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(start->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* LinspaceAiCore(const aclTensor* start, const aclTensor* end, const aclTensor* steps,
                                       aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LinspaceAiCore, start, end, steps, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Linspace算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LinSpace, OP_INPUT(start, end, steps), OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinspaceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor* Linspace(const aclTensor* start, const aclTensor* end, int64_t steps,
                          aclOpExecutor* executor) {
  auto out = executor->AllocTensor({op::Shape({steps})}, start->GetDataType());
  auto stepsTensor = executor->ConvertToTensor(executor->AllocScalar(steps), op::DataType::DT_INT64);
  if (IsAiCoreSupport(start)) {
    return LinspaceAiCore(start, end, stepsTensor, out, executor);
  } else {
    return nullptr;
  }
}
}  // namespace l0op
