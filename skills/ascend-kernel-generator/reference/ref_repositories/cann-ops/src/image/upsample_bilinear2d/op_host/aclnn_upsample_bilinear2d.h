/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_UNAMPLE_BILINEAR_H_
#define OP_API_INC_UNAMPLE_BILINEAR_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBilinear2d的第二段接口，用于执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnUpsampleBilinear2d(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief aclnnUpsampleBilinear2d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self,
    const aclIntArray *outputSize, const bool alignCorners, const double scalesH, const double scalesW, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_BILINEAR_H_