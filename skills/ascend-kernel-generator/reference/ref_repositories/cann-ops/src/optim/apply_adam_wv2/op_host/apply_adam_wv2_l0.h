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
 * @file apply_adam_wv2_l0.h
 */
#ifndef OP_API_OPEN_LEVEL0_OP_APPLY_ADAM_W_V2_H_
#define OP_API_OPEN_LEVEL0_OP_APPLY_ADAM_W_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
void ApplyAdamWV2(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* maxGradNormOptionalRef,
                  const aclTensor* grad, const aclTensor* step, float lr, float beta1, float beta2,
                  float weightDecay, float eps, bool amsgrad, bool maximize, aclOpExecutor *executor);
}

#endif
