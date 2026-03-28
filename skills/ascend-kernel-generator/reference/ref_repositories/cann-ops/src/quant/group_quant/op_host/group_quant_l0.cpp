/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "group_quant_l0.h"
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

OP_TYPE_REGISTER(GroupQuant);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
              op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion() {
  const std::map<op::SocVersion, const std::initializer_list<op::DataType>*> socSupportDtypes = {
    {SocVersion::ASCEND910B, &AICORE_DTYPE_SUPPORT_LIST}
  };
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto found = socSupportDtypes.find(socVersion);
  if (found != socSupportDtypes.end()) {
    return *(found->second);
  }
  return AICORE_DTYPE_SUPPORT_LIST;
}

const aclTensor *GroupQuantAiCore(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                                  const aclTensor *offsetOptional, int32_t dstType, aclTensor *out,
                                  aclOpExecutor *executor) {
  L0_DFX(GroupQuantAiCore, x, scale, groupIndex, offsetOptional, dstType, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupQuant,
                                         OP_INPUT(x, scale, groupIndex, offsetOptional),
                                         OP_OUTPUT(out),
                                         OP_ATTR(dstType));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                       "GroupQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}

const aclTensor *GroupQuant(const aclTensor *x, const aclTensor *scale, const aclTensor *groupIndex,
                            const aclTensor *offsetOptional, int32_t dstType, aclOpExecutor *executor) {
  op::Shape outShape = x->GetViewShape();
  auto out = executor->AllocTensor(outShape, op::DataType(dstType));

  if (!CheckType(x->GetDataType(), GetDtypeSupportListBySocVersion())) {
    OP_LOGE(ACL_ERROR_INVALID_PARAM, "dtype of input x is not supported");
    return nullptr;
  }

  return GroupQuantAiCore(x, scale, groupIndex, offsetOptional, dstType, out, executor);
}
} // namespace l0op
