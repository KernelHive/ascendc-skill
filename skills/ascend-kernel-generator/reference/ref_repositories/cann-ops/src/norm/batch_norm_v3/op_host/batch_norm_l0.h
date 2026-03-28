/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *BNInfer(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    const aclTensor *runningMean, const aclTensor *runningVar, float eps, aclOpExecutor *executor);

const std::array<aclTensor *, 2> BNTrainingReduce(
    const aclTensor *x, const op::Shape &outShape, aclOpExecutor *executor);
const std::array<aclTensor *, 2> BN3DTrainingReduce(
    const aclTensor *x, const op::Shape &outShape, aclOpExecutor *executor);

const std::array<aclTensor *, 3> BNTrainingUpdate(const aclTensor *x, const aclTensor *sum, const aclTensor *squareSum,
    const aclTensor *scale, const aclTensor *offset, aclTensor *mean, aclTensor *var, float factor, float eps,
    aclOpExecutor *executor);
const std::array<aclTensor *, 3> BN3DTrainingUpdate(const aclTensor *x, const aclTensor *sum,
    const aclTensor *squareSum, const aclTensor *scale, const aclTensor *offset, aclTensor *mean, aclTensor *var,
    float factor, float eps, aclOpExecutor *executor);
const std::array<aclTensor *, 3> BatchNormV3(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    aclTensor *running_mean, aclTensor *running_var, float momentum, float eps, aclOpExecutor *executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_OP_H_
