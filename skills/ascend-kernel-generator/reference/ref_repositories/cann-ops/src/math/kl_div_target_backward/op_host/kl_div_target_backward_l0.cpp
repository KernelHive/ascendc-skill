/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "kl_div_target_backward_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(KlDivTargetBackward);

// AICORE算子kernel
static const aclTensor* KlDivTargetBackwardAiCore(const aclTensor* gradOutput, const aclTensor* self,
    const aclTensor* target, int64_t reduction, bool logTarget, aclOpExecutor* executor) {
    L0_DFX(KlDivTargetBackwardAiCore, gradOutput, self, target, reduction, logTarget);
    auto out = executor->AllocTensor(target->GetViewShape(), target->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(KlDivTargetBackward,
                                           OP_INPUT(gradOutput, self, target),
                                           OP_OUTPUT(out),
                                           OP_ATTR(reduction, logTarget));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
        "KlDivTargetBackwardAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* KlDivTargetBackward(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    int64_t reduction, bool logTarget, aclOpExecutor* executor) {
        return KlDivTargetBackwardAiCore(gradOutput, self, target, reduction, logTarget, executor);
    }
}  // namespace l0op
