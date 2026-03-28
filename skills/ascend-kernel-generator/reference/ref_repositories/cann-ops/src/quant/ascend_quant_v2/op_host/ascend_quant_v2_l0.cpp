/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_quant_v2_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AscendQuantV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
              op::DataType::DT_FLOAT16,
              op::DataType::DT_FLOAT,
              op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
  // AscendQuantV2只需要判断dtype
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor *AscendQuantV2Aicore(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
                                     bool sqrtMode, const char *roundMode, int32_t dstType, int32_t axis,
                                     aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(AscendQuantV2Aicore, x, scale, offset, sqrtMode, roundMode, dstType, axis, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AscendQuantV2,
                                         OP_INPUT(x, scale, offset),
                                         OP_OUTPUT(out),
                                         OP_ATTR(sqrtMode, roundMode, dstType, axis));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                       "AscendQuantV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}

const aclTensor *AscendQuantV2(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
                               bool sqrtMode, const char *roundMode, int32_t dstType, int32_t axis,
                               aclOpExecutor *executor) {
  op::Shape outShape = x->GetViewShape();
  auto out = executor->AllocTensor(outShape, op::DataType(dstType));

  if (!IsAiCoreSupport(x)) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "dtype of input x is not supported");
    return nullptr;
  }

  return AscendQuantV2Aicore(x, scale, offset, sqrtMode, roundMode, dstType, axis, out, executor);
}
} // l0op
