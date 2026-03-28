/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<const aclTensor*, const aclTensor*> AdaptiveMaxPool3d(const aclTensor* self, const aclIntArray* outputSize,
                                   aclOpExecutor* executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_ADAPTIVE_MAX_POOL3D_OP_H_