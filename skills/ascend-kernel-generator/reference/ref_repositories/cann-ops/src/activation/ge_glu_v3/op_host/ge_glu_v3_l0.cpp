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
 * \file ge_glu_v3_l0.cpp
 * \brief
 */
#include "level0/geglu_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_def.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GeGluV3);

std::tuple<aclTensor*, aclTensor*> GeGluV3(const aclTensor* self, int64_t dim, int64_t approximate,
                                           bool activateLeft, aclOpExecutor* executor) {
  L0_DFX(GeGluV3, self, dim, approximate, activateLeft);

  op::Shape outShape = self->GetViewShape();
  size_t dimNum = outShape.GetDimNum();
  auto sliceDim = dim < 0 ? dimNum + dim : dim;
  const int64_t SLICE_NUM = 2;
  outShape.SetDim(sliceDim, outShape.GetDim(sliceDim) / SLICE_NUM);

  auto out = executor->AllocTensor(outShape, self->GetDataType(), Format::FORMAT_ND);
  auto outGelu = executor->AllocTensor(outShape, self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr || outGelu == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out or outGelu tensor failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }

  ADD_TO_LAUNCHER_LIST_AICORE(
    GeGluV3, OP_INPUT(self), OP_OUTPUT(out, outGelu), OP_ATTR(dim, approximate, activateLeft));
  return std::tuple<aclTensor*, aclTensor*>(out, outGelu);
}

}  // namespace l0op
