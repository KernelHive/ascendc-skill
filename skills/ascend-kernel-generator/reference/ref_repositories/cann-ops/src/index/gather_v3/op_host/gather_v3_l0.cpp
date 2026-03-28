/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_v3_l0.h"
#include "gather_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GatherV3);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_SELF = {
    op::DataType::DT_BF16,  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, 
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,   op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_UINT16,  op::DataType::DT_UINT32,  op::DataType::DT_UINT64, 
    op::DataType::DT_BOOL
};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_INDICES = {
    op::DataType::DT_INT32, op::DataType::DT_INT64
};

static bool IsAiCoreSupport(const aclTensor *self,
                            const aclTensor *indices) {
  auto checkSoc = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                   GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);

  auto checkSelfType = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_SELF);
  auto checkIndicesType = CheckType(indices->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_INDICES);

  return (checkSoc && checkSelfType && checkIndicesType);
}

static void GatherV3AiCore(const aclTensor *self, const aclTensor *indices, const aclTensor *axis,
                           aclTensor *gatherV3Out, int batchDims, bool negativeIndexSupport, aclOpExecutor *executor) {
  L0_DFX(GatherV3AiCore, self, indices, axis, gatherV3Out, batchDims, negativeIndexSupport);

  ADD_TO_LAUNCHER_LIST_AICORE(GatherV3,
                              OP_INPUT(self, indices, axis),
                              OP_OUTPUT(gatherV3Out),
                              OP_ATTR(batchDims, negativeIndexSupport));
}

const aclTensor *GatherV3(const aclTensor *self, int64_t axis, const aclTensor *indices, aclOpExecutor *executor,
                          int batchDims, bool negativeIndexSupport) {
  if (!IsAiCoreSupport(self, indices)) {
    return GatherV2(self, axis, indices, executor, batchDims, negativeIndexSupport);
  }

  int64_t selfDim = self->GetViewShape().GetDimNum() > 0 ? self->GetViewShape().GetDimNum() : 1;
  if (axis < 0) {
    axis += selfDim;
  }

  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  for (int64_t i = 0; i < axis; i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }
  for (size_t i = batchDims; i < indices->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (size_t i = axis + 1; i < self->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 当self是零维tensor时，上述推导公式不再适用，不管一维indices中有多少个0，out始终是零维tensor
  if (self->GetViewShape().GetDimNum() == 0) {
    outShape = self->GetViewShape();
  }

  // 根据推导出的输出shape申请输出tensor
  auto gatherV3Out = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  const aclTensor *axisTensor = executor->ConvertToTensor(&axis, 1, op::DataType::DT_INT64);
  GatherV3AiCore(self, indices, axisTensor, gatherV3Out, batchDims, negativeIndexSupport, executor);
  
  return gatherV3Out;
}
} // l0op
