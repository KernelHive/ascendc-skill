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
 * \file upsample_trilinear_3d_grad_l0.cpp
 * \brief
 */
#include "upsample_trilinear_3d_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleTrilinear3dBackward);

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr float MIN_SUPPORT_SCALE = 0.02;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

float ComputeScalesGrad(float scale, uint32_t input_size, uint32_t output_size)
{
    auto zero = static_cast<float>(0.);
    if (scale > zero) {
        return scale;
    } else {
        return output_size != 0 ? (static_cast<float>(input_size) / output_size) : zero;
    }
}

bool CheckScalesGrad(float scaleW, float scaleH, float scaleD)
{
    return (scaleW >= MIN_SUPPORT_SCALE && scaleH >= MIN_SUPPORT_SCALE && scaleD >= MIN_SUPPORT_SCALE);
}

const aclTensor *UpsampleTrilinear3dGradNcdhw(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, bool alignCorners, const aclFloatArray *scales, const aclFloatArray *castScales,
    aclOpExecutor *executor)
{
    L0_DFX(UpsampleTrilinear3dGradNcdhw, gradOut, outputSize, inputSize, alignCorners, scales);

    auto gradOutShape = op::ToShapeVector(gradOut->GetViewShape());
    gradOutShape[DIM_TWO] = (*inputSize)[DIM_TWO];
    gradOutShape[DIM_THREE] = (*inputSize)[DIM_THREE];
    gradOutShape[DIM_FOUR] = (*inputSize)[DIM_FOUR];

    float scales_d = (*castScales)[DIM_ZERO];
    float scales_h = (*castScales)[DIM_ONE];
    float scales_w = (*castScales)[DIM_TWO];

    op::Shape outShape;
    op::ToShape(gradOutShape.data(), gradOutShape.size(), outShape);

    auto inputShape = gradOut->GetViewShape();
    // if scale is smaller than 0.02,back to AICPU
    float scaleW = ComputeScalesGrad(scales_w, outShape.GetDim(DIM_FOUR), inputShape.GetDim(DIM_FOUR));
    float scaleH = ComputeScalesGrad(scales_h, outShape.GetDim(DIM_THREE), inputShape.GetDim(DIM_THREE));
    float scaleD = ComputeScalesGrad(scales_d, outShape.GetDim(DIM_TWO), inputShape.GetDim(DIM_TWO));

    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    auto dataType = gradOut->GetDataType();
    if ((socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) &&
        CheckType(gradOut->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) && CheckScalesGrad(scaleW, scaleH, scaleD)) {
        if (op::DataType::DT_FLOAT16 == dataType || op::DataType::DT_BF16 == dataType) {
            gradOut = l0op::Cast(gradOut, op::DataType::DT_FLOAT, executor);
        }
        const aclTensor *out = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, gradOut->GetStorageFormat());
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleTrilinear3dBackward,
            OP_INPUT(gradOut),
            OP_OUTPUT(out),
            OP_ATTR(outputSize, inputSize, alignCorners, scales_d, scales_h, scales_w));
        OP_CHECK(ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleTrilinear3dBackwardAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        if (op::DataType::DT_FLOAT16 == dataType) {
            out = l0op::Cast(out, op::DataType::DT_FLOAT16, executor);
        } else if (op::DataType::DT_BF16 == dataType) {
            // cast back to bf16
            out = l0op::Cast(out, op::DataType::DT_BF16, executor);
        }
        return out;
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleTrilinear3dGrad ADD_TO_LAUNCHER_LIST_AICPU not supported.");
        return nullptr;
    }
}
}  // namespace l0op