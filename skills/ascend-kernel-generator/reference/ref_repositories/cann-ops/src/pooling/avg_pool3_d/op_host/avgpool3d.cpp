/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "avgpool3d.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AvgPool3D);

constexpr int64_t DIM_D = 1;
constexpr int64_t DIM_H = 2;
constexpr int64_t DIM_W = 3;

static const std::initializer_list<DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                               DataType::DT_FLOAT16,
                                                                               DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return CheckType(inputDtype, AICORE_910B_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICORE算子kernel
static inline const aclTensor* AvgPool3DAiCore(const aclTensor* input, aclTensor* output, const aclIntArray *kernelSize,
                                               const aclIntArray *stride, const aclIntArray *pad, const bool ceilMode,
                                               const bool countIncludePad, const int64_t divisorOverride,
                                               const std::string &dataFormat, aclOpExecutor* executor)
{
    L0_DFX(AvgPool3DAiCore, input, output, kernelSize, stride, pad, ceilMode,
        countIncludePad, divisorOverride, dataFormat);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore AvgPool3D算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPool3D, OP_INPUT(input), OP_OUTPUT(output),
                                OP_ATTR(kernelSize, stride, pad, ceilMode, countIncludePad,
                                divisorOverride, dataFormat));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "AvgPool3D ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}

static inline int64_t AvgPool3DOutputShape(const int64_t inputLen, const int64_t kernelLen,
    const int64_t padLen, const int64_t strideLen, const bool ceilMode)
{
    float outputTempSize = 1.0f;
    if (strideLen > 0) {
        outputTempSize = 1.0f * (inputLen + padLen * 2 - kernelLen + strideLen) / strideLen; // 2: double
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of strides should be greater than 0, but got strideLen:%ld", strideLen);
        return -1;
    }
    if (ceilMode) {
        // The sliding window starting from the right filled area will be ignored
        int64_t outHeight = static_cast<int64_t>(std::ceil(outputTempSize));
        if ((outHeight - 1) * strideLen >= inputLen + padLen) {
            --outHeight;
        }
        return outHeight;
    } else {
        return static_cast<int64_t>(std::floor(outputTempSize));
    }
}

const aclTensor* AvgPool3D(const aclTensor* input, const aclIntArray *kernelSize,
                           const aclIntArray *stride, const aclIntArray *pad, const bool ceilMode,
                           const bool countIncludePad, const int64_t divisorOverride,
                           const std::string &dataFormat, aclOpExecutor* executor)
{
    int64_t dimD = dataFormat == "NCDHW" ? DIM_D + 1 : DIM_D;
    int64_t dimH = dataFormat == "NCDHW" ? DIM_H + 1 : DIM_H;
    int64_t dimW = dataFormat == "NCDHW" ? DIM_W + 1 : DIM_W;

    const int64_t outDepth = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimD), (*kernelSize)[0], (*pad)[0], (*stride)[0], ceilMode);
    const int64_t outHeight = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimH), (*kernelSize)[1], (*pad)[1], (*stride)[1], ceilMode);
    const int64_t outWidth = AvgPool3DOutputShape(
        input->GetViewShape().GetDim(dimW), (*kernelSize)[2], (*pad)[2], (*stride)[2], ceilMode); // 2: index

    auto outputShape = input->GetViewShape();
    outputShape.SetDim(dimD, outDepth);
    outputShape.SetDim(dimH, outHeight);
    outputShape.SetDim(dimW, outWidth);
    auto output = executor->AllocTensor(outputShape, input->GetDataType(), input->GetStorageFormat());
    CHECK_RET(output != nullptr, nullptr);

    if (IsAiCoreSupport(input->GetDataType())) {
        return AvgPool3DAiCore(input, output,
            kernelSize, stride, pad, ceilMode, countIncludePad, divisorOverride, dataFormat, executor);
    }

    return nullptr;
}
}  // namespace l0op
