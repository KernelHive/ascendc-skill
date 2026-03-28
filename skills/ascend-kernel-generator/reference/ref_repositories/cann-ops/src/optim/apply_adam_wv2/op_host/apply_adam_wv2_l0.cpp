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
 * @file apply_adam_wv2_l0.cpp
 */

#include "apply_adam_wv2_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ApplyAdamWV2);

// AICORE算子kernel
void ApplyAdamWV2(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* maxGradNormOptionalRef,
                  const aclTensor* grad, const aclTensor* step, float lr, float beta1, float beta2,
                  float weightDecay, float eps, bool amsgrad, bool maximize, aclOpExecutor *executor) {
    L0_DFX(ApplyAdamWV2, varRef, mRef, vRef, maxGradNormOptionalRef, grad, step, lr, beta1, beta2, weightDecay, eps,
           amsgrad, maximize);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdamWV2,
                                                 OP_INPUT(varRef, mRef, vRef, grad, step, maxGradNormOptionalRef),
                                                 OP_ATTR(lr, beta1, beta2, weightDecay, eps, amsgrad, maximize));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdamWV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
    return;
}

}  // namespace l0op
