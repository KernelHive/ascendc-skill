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
 * \file upsample_bicubic2d_aa_grad_l0.cpp
 * \brief
 */
#include "upsample_bicubic2d_aa_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBicubic2dAAGrad);

const aclTensor *UpsampleBicubic2dAAGrad(const aclTensor *gradOutput, const aclIntArray *outputSize,
    const aclIntArray *inputSize, aclTensor *output, bool alignCorners, float scales_h, float scales_w,
    aclOpExecutor *executor)
{
    L0_DFX(UpsampleBicubic2dAAGrad, gradOutput, outputSize, inputSize, output, alignCorners, scales_h, scales_w);

    auto out = executor->AllocTensor(output->GetViewShape(), output->GetDataType(), output->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBicubic2dAAGrad,
        OP_INPUT(gradOutput),
        OP_OUTPUT(out),
        OP_ATTR(outputSize, inputSize, alignCorners, scales_h, scales_w));

    return out;
}
}  // namespace l0op
