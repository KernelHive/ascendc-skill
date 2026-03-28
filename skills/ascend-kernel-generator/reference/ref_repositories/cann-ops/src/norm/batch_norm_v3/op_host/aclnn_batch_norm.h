/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BATCH_NORM_H_
#define OP_API_INC_BATCH_NORM_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBatchNorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input,
    const aclTensor *weight, const aclTensor *bias, aclTensor *runningMean, aclTensor *runningVar, bool training,
    double momentum, double eps, aclTensor *output, aclTensor *saveMean, aclTensor *saveInvstd, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnBatchNorm的第二段接口，用于执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBatchNorm(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BATCH_NORM_H_
