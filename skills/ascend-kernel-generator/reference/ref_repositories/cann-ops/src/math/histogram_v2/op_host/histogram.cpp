/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file histogram.cpp
 * \brief
 */
#include "histogram.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Histogram);
OP_TYPE_REGISTER(HistogramV2);
// AiCore支持的类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_SELF = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
  op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8};
  
static bool IsAiCoreSupport(const aclTensor* self, const aclTensor *out) {
  if (self == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "self is nullptr, check AiCoreSupport failed.");
    return false;
  }
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr, check AiCoreSupport failed.");
    return false;
  }
  auto checkSelfType = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_SELF);

  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
      GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
    return false;
  }
  return checkSelfType;
}

// AiCore的执行逻辑
inline const aclTensor *HistogramAiCore(const aclTensor *self, const aclTensor *min, const aclTensor *max,
                                        const aclTensor *out, int64_t bins, aclOpExecutor *executor) {
  L0_DFX(HistogramAiCore, self, min, max, out, bins);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HistogramV2, OP_INPUT(self, min, max), OP_OUTPUT(out), OP_ATTR(bins));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
           "HistogramAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor *Histogram(const aclTensor *self, const aclTensor *min, const aclTensor *max, const aclTensor *out,
                           int64_t bins, float minValue, float maxValue, aclOpExecutor *executor) {
  OP_CHECK(IsAiCoreSupport(self, out), OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HistogramAiCore not support."),
           return nullptr);
  auto outAiCore = executor->AllocTensor(out->GetViewShape(), op::DataType::DT_INT32);
  return HistogramAiCore(self, min, max, outAiCore, bins, executor);
}
}  // namespace l0op
