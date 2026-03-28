/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_sum_v2.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceSumV2);

static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  bool isSocSupport = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                       GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
  return isSocSupport && CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *ReduceSumV2AiCore(const aclTensor *x, const aclTensor *axes, bool keepDim,
                                          bool noopWithEmptyAxes, const aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(ReduceSumV2AiCore, x, axes, keepDim, noopWithEmptyAxes, out);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ReduceSumV2, OP_INPUT(x, axes), OP_OUTPUT(out), OP_ATTR(keepDim, noopWithEmptyAxes));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "ReduceSumOp ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}

const aclTensor *ReduceSumV2(const aclTensor *x, const aclIntArray *axes, bool keepDim, aclOpExecutor *executor) {
  auto axesTensor = executor->ConvertToTensor(axes, op::ToOpDataType(ACL_INT64));
  auto out = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);

  // dim为空时，默认保留所有轴
  bool noopWithEmptyAxes = true;
  INFER_SHAPE(ReduceSumV2, OP_INPUT(x, axesTensor), OP_OUTPUT(out), OP_ATTR(keepDim, noopWithEmptyAxes));
  op::Shape outShape = x->GetViewShape();
  auto count = axes->Size();
  size_t dimNum = outShape.GetDimNum();
  if (keepDim) {
    for (uint64_t i = 0; i < count; i++) {
      int64_t dimIndex = static_cast<int64_t>((*axes)[i]);
      int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + dimNum;
      outShape.SetDim(dimNew, 1);
    }
    out->SetViewShape(outShape);
  }

  if (IsAiCoreSupport(x)) {
    return ReduceSumV2AiCore(x, axesTensor, keepDim, noopWithEmptyAxes, out, executor);
  } else {
    return nullptr;
  }
}
} // namespace l0op
