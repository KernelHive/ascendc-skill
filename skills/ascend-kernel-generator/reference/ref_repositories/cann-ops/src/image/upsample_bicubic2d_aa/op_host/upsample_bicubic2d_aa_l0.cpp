/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bicubic2d_aa_l0.cpp
 * \brief
 */
#include "upsample_bicubic2d_aa_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBicubic2dAA);

const aclTensor *UpsampleBicubic2dAA(const aclTensor *x, const aclIntArray *size, const bool alignCorners,
                                     const aclTensor *y, const float scalesH, const float scalesW, aclOpExecutor *executor) {
  L0_DFX(UpsampleBicubic2dAA, x, size, alignCorners, scalesH, scalesH);

  aclTensor* out = executor->AllocTensor(y->GetViewShape(), op::DataType::DT_FLOAT, y->GetViewFormat());
  CHECK_RET(out != nullptr, nullptr);
  
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBicubic2dAA,
                                         OP_INPUT(x),
                                         OP_OUTPUT(out),
                                         OP_ATTR(size, alignCorners, scalesH, scalesW));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleBicubic2dAAAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op
