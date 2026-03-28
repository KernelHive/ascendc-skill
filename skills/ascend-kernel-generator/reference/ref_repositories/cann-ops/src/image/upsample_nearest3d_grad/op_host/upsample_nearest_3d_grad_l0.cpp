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
 * \file upsample_nearest_3d_grad_l0.cpp
 * \brief
 */
#include "upsample_nearest_3d_grad_l0.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleNearest3dGrad);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr float MAX_SUPPORT_SCALE = 50;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

float ComputeNearest3dGradScales(int64_t input_size, int64_t output_size, float scale)
{
    auto zero = static_cast<float>(0.);
    if (scale != zero && scale > zero) {
        return scale;
    } else {
        return input_size != 0 ? (static_cast<float>(output_size) / input_size) : zero;
    }
}

bool CheckNearest3dGradScales(const aclTensor *gradOut, const aclIntArray *inputSize, const aclFloatArray *castScales)
{
    float scales_d = 0.0;
    float scales_h = 0.0;
    float scales_w = 0.0;
    if (castScales->Size() == DIM_THREE) {
        scales_d = (*castScales)[DIM_ZERO];
        scales_h = (*castScales)[DIM_ONE];
        scales_w = (*castScales)[DIM_TWO];
    }
    const int64_t sizeD = (*inputSize)[DIM_TWO];
    const int64_t sizeH = (*inputSize)[DIM_THREE];
    const int64_t sizeW = (*inputSize)[DIM_FOUR];
    auto inputShape = gradOut->GetViewShape();
    float scaleW = ComputeNearest3dGradScales(sizeW, inputShape.GetDim(DIM_FOUR), scales_w);
    float scaleH = ComputeNearest3dGradScales(sizeH, inputShape.GetDim(DIM_THREE), scales_h);
    float scaleD = ComputeNearest3dGradScales(sizeD, inputShape.GetDim(DIM_TWO), scales_d);
    return (scaleW <= MAX_SUPPORT_SCALE && scaleH <= MAX_SUPPORT_SCALE && scaleD <= MAX_SUPPORT_SCALE);
}

const aclTensor *UpsampleNearest3dGradNcdhw(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, const aclFloatArray *scales, const aclFloatArray *castScales, aclOpExecutor *executor)
{
    L0_DFX(UpsampleNearest3dGradNcdhw, gradOut, outputSize, inputSize, scales);

    // 获取DHW维度Size D = inputSize[2], H = inputSize[3], W = inputSize[4]
    const int64_t sizeD = (*inputSize)[DIM_TWO];
    const int64_t sizeH = (*inputSize)[DIM_THREE];
    const int64_t sizeW = (*inputSize)[DIM_FOUR];
    // 生成out shape为 (N，C，inputSize[2], inputSize[3], inputSize[4])
    op::Shape gradInputStorageShape = gradOut->GetStorageShape();
    gradInputStorageShape.SetDim(DIM_TWO, sizeD);
    gradInputStorageShape.SetDim(DIM_THREE, sizeH);
    gradInputStorageShape.SetDim(DIM_FOUR, sizeW);
    op::Shape gradInputOriginalShape = gradOut->GetOriginalShape();
    gradInputOriginalShape.SetDim(DIM_TWO, sizeD);
    gradInputOriginalShape.SetDim(DIM_THREE, sizeH);
    gradInputOriginalShape.SetDim(DIM_FOUR, sizeW);

    auto dataType = gradOut->GetDataType();
    // npu实现
    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    if (CheckNearest3dGradScales(gradOut, inputSize, castScales) &&
        (socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) &&
        CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST)) {
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
            UpsampleNearest3dGrad, OP_INPUT(gradOut), OP_OUTPUT(gradInput), OP_ATTR(inputSize, outputSize, castScales));
        if (op::DataType::DT_BF16 == dataType) {
            gradInput = l0op::Cast(gradInput, op::DataType::DT_BF16, executor);
        } else if (op::DataType::DT_FLOAT16 == dataType) {
            gradInput = l0op::Cast(gradInput, op::DataType::DT_FLOAT16, executor);
        }
        return gradInput;
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleNearest3dGrad ADD_TO_LAUNCHER_LIST_AICPU not supported.");
        return nullptr;
    }
}
}  // namespace l0op
