/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ADAPTIVE_MAX_POOL3D_H_
#define OP_API_INC_ADAPTIVE_MAX_POOL3D_H_
#include <array>
#include "aclnn/aclnn_base.h"
#define ACLNN_API __attribute__((visibility("default")))


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaptiveMaxPool3d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                             aclTensor* outputOut, aclTensor* indicesOut,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAdaptiveMaxPool3d的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnAdaptiveMaxPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
