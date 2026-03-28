/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_index_add_with_sorted_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(InplaceIndexAdd);
OP_TYPE_REGISTER(InplaceIndexAddWithSorted);
// AICORE算子kernel
const aclTensor *InplaceIndexAddAiCore(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                       const aclTensor *source, const aclTensor *alphaTensor,
                                       aclOpExecutor *executor) {
  L0_DFX(InplaceIndexAddAiCore, self, dim, index, source, alphaTensor);
  auto indexAddOut = const_cast<aclTensor*>(self);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceIndexAdd,
                                         OP_INPUT(self, index, source, alphaTensor),
                                         OP_OUTPUT(indexAddOut),
                                         OP_ATTR(dim));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "InplaceIndexAddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return indexAddOut;
}

// 排序后AICORE算子kernel
const aclTensor *InplaceIndexAddWithSorted(const aclTensor *self, const int64_t dim, const aclTensor *sortedIndices,
                                           const aclTensor *pos, const aclTensor *value, const aclTensor *alphaTensor,
                                           aclOpExecutor *executor) {
    L0_DFX(InplaceIndexAddWithSorted, self, dim, value, sortedIndices, pos, alphaTensor);
    auto indexAddOut = const_cast<aclTensor*>(self);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceIndexAddWithSorted,
                                           OP_INPUT(self, value, sortedIndices, pos, alphaTensor),
                                           OP_OUTPUT(indexAddOut),
                                           OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
             "InplaceIndexAddWithSortedAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return indexAddOut;
}
}  // namespace l0op
