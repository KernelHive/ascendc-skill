/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "padv3grad.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PadV3Grad);
OP_TYPE_REGISTER(PadV4Grad);
OP_TYPE_REGISTER(PadV3GradReplicate);
OP_TYPE_REGISTER(ReflectionPad3dGrad);
OP_TYPE_REGISTER(PadV3GradReplication);
OP_TYPE_REGISTER(CircularPadGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PAD_V4_GRAD_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> REPLICATION_2D_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REFLECT_3D_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> REPLICATION_3D_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> CIRCULAR_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const size_t DIM_3D_SHAPE = 5;
static const size_t DIM_3D_H_DIM_INDEX = 3;
static const size_t DIM_3D_W_DIM_INDEX = 4;
static const int64_t DIM_3D_REFLECTION_FLOAT32_MAX= 8960;
static const int64_t DIM_3D_REFLECTION_FLOAT16_MAX= 17920;
static const int64_t DIM_3D_REFLECTION_BF16_MAX= 11136;
static const size_t AI_CORE_DIM_BOUND = 4;
static const size_t AI_CORE_DIM_3D = 5;
static const size_t H_DIM_INDEX = 2;
static const size_t W_DIM_INDEX = 3;
static const int64_t FLOAT32_MAX_H = 800;
static const int64_t FLOAT32_MAX_W = 1500;
static const int64_t FLOAT16_MAX_H = 950;
static const int64_t FLOAT16_MAX_W = 3000;
static const string REPLICATION_PAD_MODE = "edge";

inline static bool IsTbeSupport(const aclTensor *gradOutput)
{
    // Tbe仅支持Pad1d、Pad2d, 如果是Pad3d场景，需要走AscendC或AiCpu分支
    if (gradOutput->GetViewShape().GetDimNum() > AI_CORE_DIM_BOUND) {
      return false;
    }
    // AiCore仅支持一定范围内的shape
    if (gradOutput->GetDataType() == op::DataType::DT_FLOAT) {
      return gradOutput->GetViewShape().GetDim(H_DIM_INDEX) <= FLOAT32_MAX_H &&
               gradOutput->GetViewShape().GetDim(W_DIM_INDEX) <= FLOAT32_MAX_W;
    } else if (gradOutput->GetDataType() == op::DataType::DT_FLOAT16) {
      return gradOutput->GetViewShape().GetDim(H_DIM_INDEX) <= FLOAT16_MAX_H &&
             gradOutput->GetViewShape().GetDim(W_DIM_INDEX) <= FLOAT16_MAX_W;
    } else {
      return CheckType(gradOutput->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }
}

inline static bool IsPadV4GradAicoreSupport(const aclTensor *gradOutput, const std::string &mode) {
    // padv4grad 支持类型和平台
    if (gradOutput->GetViewShape().GetDimNum() > AI_CORE_DIM_BOUND) {
        return false;
    }
    return mode == "reflect" && CheckType(gradOutput->GetDataType(), PAD_V4_GRAD_AICORE_DTYPE_SUPPORT_LIST) &&
           (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B);
}

inline static bool IsPadV3GradReplicateAicoreSupport(const aclTensor *gradOutput, const std::string &mode) {
    // PadV3GradReplicate 支持类型和平台
    if (gradOutput->GetViewShape().GetDimNum() > AI_CORE_DIM_BOUND) {
        return false;
    }
    return mode == REPLICATION_PAD_MODE &&
            CheckType(gradOutput->GetDataType(), REPLICATION_2D_AICORE_DTYPE_SUPPORT_LIST) &&
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B );
}

inline static bool IsAiCoreSupportReplication3D(const aclTensor *gradOutput, const std::string &mode)
{
    if (gradOutput->GetViewShape().GetDimNum() == AI_CORE_DIM_3D && \
        mode == REPLICATION_PAD_MODE && \
        CheckType(gradOutput->GetDataType(), REPLICATION_3D_AICORE_DTYPE_SUPPORT_LIST) && \
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B)
    ) {
        OP_LOGD("[PadV3Grad] IsAiCoreSupportReplication3D: true");
        return true;
    }
    OP_LOGD("[PadV3Grad] IsAiCoreSupportReplication3D: false");
    return false;
}

inline static bool IsAiCoreSupportReflection3D(const aclTensor *gradOutput, const std::string &mode)
{
    // AscendC仅支持Pad3d的部分场景，其余需要走AiCpu分支
    if (gradOutput->GetViewShape().GetDimNum() != AI_CORE_DIM_3D || mode != "reflect") {
        OP_LOGD("[PadV3Grad] IsAiCoreSupportReflection3D: false, The input dimension is not a 3d scene or the mode is not reflect");
        return false;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        OP_LOGD("[PadV3Grad] IsAiCoreSupportReflection3D: false, The device is not support");
        return false;
    }
    if (CheckType(gradOutput->GetDataType(), REFLECT_3D_AICORE_DTYPE_SUPPORT_LIST) == false) {
        OP_LOGD("[PadV3Grad] IsAiCoreSupportReflection3D: false, The data type does not support aicore");
        return false;
    } 
    OP_LOGD("[PadV3Grad] IsAiCoreSupportReflection3D: ture");
    return true;
}

inline const aclTensor *PadV3GradAiCore(const aclTensor *gradOutput, const aclTensor *paddings,
    const std::string& mode, const bool paddingsContiguous, aclTensor *padV3GradOut, aclOpExecutor *executor)
{
    L0_DFX(PadV3GradAiCore, gradOutput, paddings, mode, paddingsContiguous, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PadV3Grad, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut),
                                           OP_ATTR(mode, paddingsContiguous));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV3GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return padV3GradOut;
}

inline const aclTensor *PadV4GradAiCore(const aclTensor *gradOutput, const aclTensor *paddings,
                                        const std::string &mode, const bool paddingsContiguous,
                                        aclTensor *padV3GradOut, aclOpExecutor *executor) {
    L0_DFX(PadV4GradAiCore, gradOutput, paddings, mode, paddingsContiguous, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PadV4Grad, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut),
                                           OP_ATTR(mode, paddingsContiguous));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV4GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return padV3GradOut;
}

inline const aclTensor *PadV3GradReplicateAiCore(const aclTensor *gradOutput, const aclTensor *paddings,
                                        const std::string &mode, const bool paddingsContiguous,
                                        aclTensor *padV3GradOut, aclOpExecutor *executor) {
    L0_DFX(PadV3GradReplicateAiCore, gradOutput, paddings, mode, paddingsContiguous, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PadV3GradReplicate, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut),
                                           OP_ATTR(mode, paddingsContiguous));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV3GradReplicateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return padV3GradOut;
}

inline const aclTensor *PadV3Grad3DAiCoreReflection(const aclTensor *gradOutput, const aclTensor *paddings, 
             aclTensor *padV3GradOut, aclOpExecutor *executor)
{
    L0_DFX(PadV3Grad3DAiCoreReflection, gradOutput, paddings, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReflectionPad3dGrad, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV3Grad3DAiCoreReflection ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return padV3GradOut;
}

inline const aclTensor *PadV3GradReplicationAiCore(const aclTensor *gradOutput, const aclTensor *paddings,
    aclTensor *padV3GradOut, aclOpExecutor *executor)
{
    L0_DFX(PadV3GradReplicationAiCore, gradOutput, paddings, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PadV3GradReplication, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadV3GradReplicationAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return padV3GradOut;
}

inline const aclTensor *CircularPadGradAiCore(const aclTensor *gradOutput, const aclTensor *paddings, 
             aclTensor *padV3GradOut, aclOpExecutor *executor)
{
    L0_DFX(CircularPadGradAiCore, gradOutput, paddings, padV3GradOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CircularPadGrad, OP_INPUT(gradOutput, paddings), OP_OUTPUT(padV3GradOut));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CircularPadGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return padV3GradOut;
}

const aclTensor *PadV3Grad(const aclTensor *gradOutput,
                           const aclTensor *paddings,
                           const std::string& mode,
                           const bool paddingsContiguous,
                           const bool padFlag,
                           aclOpExecutor *executor)
{
    auto padV3GradOut = executor->AllocTensor(gradOutput->GetDataType(), gradOutput->GetViewFormat(),
        gradOutput->GetViewFormat());
    INFER_SHAPE(PadV3Grad,
                OP_INPUT(gradOutput, paddings),
                OP_OUTPUT(padV3GradOut),
                OP_ATTR(mode, paddingsContiguous));

    if (mode == "circular") {
        return CircularPadGradAiCore(gradOutput, paddings, padV3GradOut, executor);
    }
    if (padFlag && IsPadV4GradAicoreSupport(gradOutput, mode)) {
        return PadV4GradAiCore(gradOutput, paddings, mode, paddingsContiguous, padV3GradOut, executor);
    } else if (padFlag && IsPadV3GradReplicateAicoreSupport(gradOutput, mode)) {
        return PadV3GradReplicateAiCore(gradOutput, paddings, mode, paddingsContiguous, padV3GradOut, executor);
    } else if (IsAiCoreSupportReplication3D(gradOutput, mode)) {
        return PadV3GradReplicationAiCore(gradOutput, paddings, padV3GradOut, executor);
    } else if (padFlag && IsAiCoreSupportReflection3D(gradOutput, mode)) {
        return PadV3Grad3DAiCoreReflection(gradOutput, paddings, padV3GradOut, executor);
    } else if (padFlag && IsTbeSupport(gradOutput)) {
        return PadV3GradAiCore(gradOutput, paddings, mode, paddingsContiguous, padV3GradOut, executor);
    }
}

}
