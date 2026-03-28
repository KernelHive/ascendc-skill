/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_norm_l0.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BNInfer);

const aclTensor *BNInfer(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    const aclTensor *runningMean, const aclTensor *runningVar, float eps, aclOpExecutor *executor)
{
    L0_DFX(BNInfer, input, weight, bias, runningMean, runningVar, eps);

    auto out = executor->AllocTensor(input->GetStorageShape(), input->GetDataType(), input->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BNInfer, OP_INPUT(input, weight, bias, runningMean, runningVar), OP_OUTPUT(out), OP_ATTR(eps));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNInferAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

OP_TYPE_REGISTER(BNTrainingReduce);

const std::array<aclTensor *, 2> BNTrainingReduce(
    const aclTensor *x, const op::Shape &outShape, aclOpExecutor *executor)
{
    L0_DFX(BNTrainingReduce, x);

    auto sum = executor->AllocTensor(outShape, DataType::DT_FLOAT, x->GetViewFormat());
    auto squareSum = executor->AllocTensor(outShape, DataType::DT_FLOAT, x->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BNTrainingReduce, OP_INPUT(x), OP_OUTPUT(sum, squareSum));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNTrainingReduceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {sum, squareSum};
}

OP_TYPE_REGISTER(BN3DTrainingReduce);

const std::array<aclTensor *, 2> BN3DTrainingReduce(
    const aclTensor *x, const op::Shape &outShape, aclOpExecutor *executor)
{
    L0_DFX(BN3DTrainingReduce, x);

    auto sum = executor->AllocTensor(outShape, DataType::DT_FLOAT, x->GetStorageFormat());
    auto squareSum = executor->AllocTensor(outShape, DataType::DT_FLOAT, x->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BN3DTrainingReduce, OP_INPUT(x), OP_OUTPUT(sum, squareSum));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BN3DTrainingReduceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {sum, squareSum};
}

OP_TYPE_REGISTER(BNTrainingUpdate);

const std::array<aclTensor *, 3> BNTrainingUpdate(const aclTensor *x, const aclTensor *sum, const aclTensor *squareSum,
    const aclTensor *scale, const aclTensor *offset, aclTensor *mean, aclTensor *var, float factor, float eps,
    aclOpExecutor *executor)
{
    L0_DFX(BNTrainingUpdate, x, sum, squareSum, scale, offset, mean, var, factor, eps);

    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    auto batchMean = executor->AllocTensor(sum->GetViewShape(), DataType::DT_FLOAT, sum->GetViewFormat());
    auto batchVar = executor->AllocTensor(sum->GetViewShape(), DataType::DT_FLOAT, sum->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BNTrainingUpdate,
        OP_INPUT(x, sum, squareSum, scale, offset, mean, var),
        OP_OUTPUT(y, mean, var, batchMean, batchVar),
        OP_ATTR(factor, eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNTrainingUpdateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {y, batchMean, batchVar};
}

OP_TYPE_REGISTER(BN3DTrainingUpdate);

const std::array<aclTensor *, 3> BN3DTrainingUpdate(const aclTensor *x, const aclTensor *sum,
    const aclTensor *squareSum, const aclTensor *scale, const aclTensor *offset, aclTensor *mean, aclTensor *var,
    float factor, float eps, aclOpExecutor *executor)
{
    L0_DFX(BN3DTrainingUpdate, x, sum, squareSum, scale, offset, mean, var, factor, eps);

    auto y = executor->AllocTensor(
        x->GetStorageShape(), x->GetOriginalShape(), x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    auto batchMean = executor->AllocTensor(sum->GetStorageShape(),
        mean->GetOriginalShape(),
        DataType::DT_FLOAT,
        sum->GetStorageFormat(),
        mean->GetOriginalFormat());
    auto batchVar = executor->AllocTensor(sum->GetStorageShape(),
        var->GetOriginalShape(),
        DataType::DT_FLOAT,
        sum->GetStorageFormat(),
        var->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BN3DTrainingUpdate,
        OP_INPUT(x, sum, squareSum, scale, offset, mean, var),
        OP_OUTPUT(y, mean, var, batchMean, batchVar),
        OP_ATTR(factor, eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BN3DTrainingUpdateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {y, batchMean, batchVar};
}

OP_TYPE_REGISTER(BatchNormV3);

const std::array<aclTensor *, 3> BatchNormV3(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    aclTensor *running_mean, aclTensor *running_var, float momentum, float eps, aclOpExecutor *executor)
{
    L0_DFX(BatchNormV3, x, weight, bias, running_mean, running_var, momentum, eps);

    auto y = executor->AllocTensor(
        x->GetStorageShape(), x->GetOriginalShape(), x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    auto batchMean = executor->AllocTensor(running_mean->GetStorageShape(),
        running_mean->GetOriginalShape(),
        DataType::DT_FLOAT,
        running_mean->GetStorageFormat(),
        running_mean->GetOriginalFormat());
    auto batchVar = executor->AllocTensor(running_var->GetStorageShape(),
        running_var->GetOriginalShape(),
        DataType::DT_FLOAT,
        running_var->GetStorageFormat(),
        running_var->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BatchNormV3,
        OP_INPUT(x, weight, bias, running_mean, running_var),
        OP_OUTPUT(y, running_mean, running_var, batchMean, batchVar),
        OP_ATTR(eps, momentum, true));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BatchNormV3 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {y, batchMean, batchVar};
}
}  // namespace l0op
