/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file fake_quant_affine_cachemask_l0.cpp
 * \brief
 */

#include "fake_quant_affine_cachemask_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FakeQuantAffineCachemask);

std::tuple<aclTensor*, aclTensor*> FakeQuantAffineCachemask(const aclTensor *self, const aclTensor *scale,
    const aclTensor *zeroPoint, int64_t quantMin, int64_t quantMax, aclOpExecutor *executor) {
  L0_DFX(FakeQuantAffineCachemask, self, scale, zeroPoint, quantMin, quantMax);

  auto fakeOut = executor->AllocTensor(self->GetStorageShape(), self->GetDataType(), self->GetStorageFormat());
  auto maskOut = executor->AllocTensor(self->GetStorageShape(), op::DataType::DT_BOOL, self->GetStorageFormat());

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FakeQuantAffineCachemask,
                                         OP_INPUT(self, scale, zeroPoint),
                                         OP_OUTPUT(fakeOut, maskOut),
                                         OP_ATTR(0, quantMin, quantMax));
  if (ret !=  ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FakeQuantAffineCachemaskAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return std::tuple<aclTensor*, aclTensor*>(fakeOut, maskOut);
}
}  // namespace l0op
