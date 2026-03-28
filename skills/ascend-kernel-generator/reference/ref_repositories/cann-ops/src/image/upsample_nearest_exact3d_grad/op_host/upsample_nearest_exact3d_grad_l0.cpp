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
 * \file upsample_nearest_exact3d_grad_l0.cpp
 * \brief
 */
#include "upsample_nearest_exact3d_grad_l0.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearestExact3dGrad);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

const aclTensor *UpsampleNearestExact3dGradNcdhw(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, const aclFloatArray *scales, aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearestExact3dGradNcdhw, gradOut, outputSize, inputSize, scales);

    // 获取DHW维度Size D = inputSize[2], H = inputSize[3], W = inputSize[4]
    auto dataType = gradOut->GetDataType();
    const int64_t sizeD = (*inputSize)[DIM_TWO];
    const int64_t sizeH = (*inputSize)[DIM_THREE];
    const int64_t sizeW = (*inputSize)[DIM_FOUR];

    // 生成out shape为 (N，C，inputSize[2], inputSize[3], inputSize[4])
    op::Shape gradInputStorageShape = gradOut->GetStorageShape();
    op::Shape gradInputOriginalShape = gradOut->GetOriginalShape();
    gradInputStorageShape.SetDim(DIM_TWO, sizeD);
    gradInputStorageShape.SetDim(DIM_THREE, sizeH);
    gradInputStorageShape.SetDim(DIM_FOUR, sizeW);
    gradInputOriginalShape.SetDim(DIM_TWO, sizeD);
    gradInputOriginalShape.SetDim(DIM_THREE, sizeH);
    gradInputOriginalShape.SetDim(DIM_FOUR, sizeW);

    if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
        gradOut = l0op::Cast(gradOut, op::DataType::DT_FLOAT, executor);
    }
    const aclTensor *gradInput = executor->AllocTensor(gradInputStorageShape,
        gradInputOriginalShape,
        gradOut->GetDataType(),
        gradOut->GetStorageFormat(),
        gradOut->GetOriginalFormat());
    CHECK_RET(gradInput != nullptr, nullptr);

    ADD_TO_LAUNCHER_LIST_AICORE(
        UpsampleNearestExact3dGrad, OP_INPUT(gradOut), OP_OUTPUT(gradInput), OP_ATTR(inputSize, outputSize, scales));
    if (op::DataType::DT_BF16 == dataType) {
        gradInput = l0op::Cast(gradInput, op::DataType::DT_BF16, executor);
    } else if (op::DataType::DT_FLOAT16 == dataType) {
        gradInput = l0op::Cast(gradInput, op::DataType::DT_FLOAT16, executor);
    }
    return gradInput;
}
}  // namespace l0op
