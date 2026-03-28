/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "top_kv3_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TopKV2);
OP_TYPE_REGISTER(TopKV3);
OP_TYPE_REGISTER(TopK);

const int64_t MAX_AICORE_CALC_INPUTSIZE = 32768;
const int64_t MAX_AICORE_CALC_DIM = 8;
const int64_t K_LIMIT = 16;

// 根据芯片类型、dtype判断算子是否支持走TopKV3
static bool IsAscendCSupport(const aclTensor *self, int64_t k) {
  SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
  if (version == SocVersion::ASCEND310P && CheckType(self->GetDataType(), {op::DataType::DT_FLOAT16}) && k < K_LIMIT) {
    return true;
  }
  return false;
}
// AICORE算子kernel
std::tuple<aclTensor*, aclTensor*> TopkV2(const aclTensor *self, const aclTensor *k, int64_t dim,
                                              bool largest, bool sorted, aclTensor *values, aclTensor *indices,
                                              aclOpExecutor *executor) {
  L0_DFX(TopkV2, self, k, dim, largest, sorted, values, indices);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV2算子加入任务队列
  ADD_TO_LAUNCHER_LIST_AICORE(TopKV2,
                              OP_INPUT(self, k),
                              OP_OUTPUT(values, indices),
                              OP_ATTR(sorted, dim, largest));
  return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> TopkV3(const aclTensor *self, const aclTensor *k, int64_t dim,
                                              bool largest, bool sorted, aclTensor *values, aclTensor *indices,
                                              aclOpExecutor *executor) {
  L0_DFX(TopkV3, self, k, dim, largest, sorted, values, indices);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV3算子加入任务队列
  ADD_TO_LAUNCHER_LIST_AICORE(TopKV3,
                              OP_INPUT(self, k),
                              OP_OUTPUT(values, indices),
                              OP_ATTR(sorted, dim, largest));
  return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> Topk(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted,
                                        aclOpExecutor *executor) {
  op::Shape outShape = self->GetStorageShape();
  outShape.SetDim(dim, k);

  const aclScalar *kScalar = executor->AllocScalar(k);
  if (kScalar == nullptr) {
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }

  const aclTensor *kTensor =  executor->ConvertToTensor(kScalar, op::ToOpDataType(ACL_INT32));
  auto valuesOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
  auto indicesOut = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());

  if (IsAscendCSupport(self, k)) {
    return TopkV3(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
  } else {
    return TopkV2(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
  }
}
}
