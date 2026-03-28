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
 * \file grid_sampler2d_grad_l0.cpp
 * \brief
 */

#include "grid_sampler2d_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GridSampler2DGrad);

static const string INTERPOLATION_BILINEAR = "bilinear";
static const string INTERPOLATION_NEAREST = "nearest";

static const string PADDING_ZEROS = "zeros";
static const string PADDING_BORDER = "border";
static const string PADDING_REFLECTION = "reflection";

inline const string &GetInterpolationModeStr(int64_t interpolationMode)
{
    if (interpolationMode == 0) {
        return INTERPOLATION_BILINEAR;
    }
    return INTERPOLATION_NEAREST;
}

inline const string &GetPaddingModeStr(int64_t paddingMode)
{
    if (paddingMode == 0) {
        return PADDING_ZEROS;
    }
    if (paddingMode == 1) {
        return PADDING_BORDER;
    }
    return PADDING_REFLECTION;
}

const std::tuple<aclTensor *, aclTensor *> GridSampler2DGrad(const aclTensor *gradOutput, const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclOpExecutor *executor)
{
    L0_DFX(GridSampler2DGrad, gradOutput, input, grid, interpolationMode, paddingMode, alignCorners);

    auto dataType = input->GetDataType();
    if (dataType == op::DataType::DT_BF16) {
        gradOutput = l0op::Cast(gradOutput, op::DataType::DT_FLOAT, executor);
        input = l0op::Cast(input, op::DataType::DT_FLOAT, executor);
        grid = l0op::Cast(grid, op::DataType::DT_FLOAT, executor);
    }

    auto inputGrad = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetStorageFormat());
    auto gridGrad = executor->AllocTensor(grid->GetViewShape(), grid->GetDataType(), grid->GetStorageFormat());
    if (inputGrad == nullptr || gridGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gridGrad or inputGrad tensor failed.");
        return std::tie(inputGrad, gridGrad);
    }

    if (dataType == op::DataType::DT_BF16) {
        inputGrad = const_cast<aclTensor *>(l0op::Cast(inputGrad, op::DataType::DT_BF16, executor));
        gridGrad = const_cast<aclTensor *>(l0op::Cast(gridGrad, op::DataType::DT_BF16, executor));
    }

    return std::tie(inputGrad, gridGrad);
}

const std::tuple<aclTensor *, aclTensor *> GridSamplerGrad(const aclTensor *gradOutput, const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclOpExecutor *executor)
{
    L0_DFX(GridSamplerGrad, gradOutput, input, grid, interpolationMode, paddingMode, alignCorners);
    auto inputGrad = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetStorageFormat());
    auto gridGrad = executor->AllocTensor(grid->GetViewShape(), grid->GetDataType(), grid->GetStorageFormat());
    if (inputGrad == nullptr || gridGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gridGrad or inputGrad tensor failed.");
        return std::tie(inputGrad, gridGrad);
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GridSampler2DGrad,
        OP_INPUT(gradOutput, input, grid),
        OP_OUTPUT(inputGrad, gridGrad),
        OP_ATTR(GetInterpolationModeStr(interpolationMode), GetPaddingModeStr(paddingMode), alignCorners));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GridSamplerGrad AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor *, aclTensor *>(nullptr, nullptr);
    }
    return std::tie(inputGrad, gridGrad);
}
}  // namespace l0op