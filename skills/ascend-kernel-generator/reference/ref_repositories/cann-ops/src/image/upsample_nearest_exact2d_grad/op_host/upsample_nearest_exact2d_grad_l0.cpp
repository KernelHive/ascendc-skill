/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest_exact2d_grad_l0.cpp
 * \brief
 */
#include "upsample_nearest_exact2d_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearestExact2dGrad);
OP_TYPE_REGISTER(UpsampleNearest2dGrad);

const aclTensor *UpsampleNearestExact2dGrad(const aclTensor *gradOutput, const aclIntArray *outputSize,
    const aclIntArray *inputSize, aclTensor *output, float scales_h, float scales_w, bool isExact,
    aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearestExact2dGrad, gradOutput, outputSize, inputSize, output, scales_h, scales_w);
    auto dataType = gradOutput->GetDataType();
    if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
        gradOutput = l0op::Cast(gradOutput, op::DataType::DT_FLOAT, executor);
    }

    const aclTensor *out =
        executor->AllocTensor(output->GetViewShape(), gradOutput->GetDataType(), output->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    if (isExact) {
        ADD_TO_LAUNCHER_LIST_AICORE(UpsampleNearestExact2dGrad,
            OP_INPUT(gradOutput),
            OP_OUTPUT(out),
            OP_ATTR(outputSize, inputSize, scales_h, scales_w));
    } else {
        ADD_TO_LAUNCHER_LIST_AICORE(UpsampleNearest2dGrad,
            OP_INPUT(gradOutput),
            OP_OUTPUT(out),
            OP_ATTR(outputSize, inputSize, scales_h, scales_w));
    }

    if (op::DataType::DT_BF16 == dataType) {
        out = l0op::Cast(out, op::DataType::DT_BF16, executor);
    } else if (op::DataType::DT_FLOAT16 == dataType) {
        out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
    }
    return out;
}
}  // namespace l0op
