/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_BICUBIC_2D_BACKWARD_H_
#define OP_API_INC_UNAMPLE_BICUBIC_2D_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBicubic2dBackward的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);

/**
 * @brief aclnnUpsampleBicubic2dBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, const bool alignCorners,
    double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_Bicubic_2D_BACKWARD_H_