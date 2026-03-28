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
 * \file ctc_loss_v2_grad_l0.cpp
 * \brief
 */
#include "ctc_loss_v2_grad_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr size_t SYMBOL_SET_INDEX = 2;
constexpr size_t BATCH_INDEX = 1;
constexpr size_t TIME_INDEX = 0;
constexpr size_t LABEL_INDEX = 2;
constexpr int64_t MAX_SYMBOL_SET_LEN_V3 = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 10000;
constexpr int64_t COEF = 2;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t FOUR_BYTE = 4;
constexpr int64_t EIGHT_BYTE = 8;
constexpr int64_t SMALL_UB_SIZE = 192 * 1024;
constexpr int64_t UB_SIZE = 256 * 1024;
constexpr int64_t RESERVED_UB_SIZE = 2 * 1024;
constexpr int64_t C_LOOP_BLOCK = 64;
constexpr int64_t C_LOOP_SUM_BLOCK = 8;
constexpr int64_t C_UB_NUM = 5;
constexpr int64_t BATCH_UB_NUM = 2;
constexpr int64_t TARGETS_UB_NUM = 24;
constexpr float ONE_FLOAT = 1.0f;
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
}

namespace l0op {
OP_TYPE_REGISTER(CTCLossV2Grad);
OP_TYPE_REGISTER(CTCLossV3Grad);

static const std::string REDUCTION_MEAN = "mean";
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> V3_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FOUR_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> EIGHT_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT64};

static int64_t GetDtypeSize(const op::DataType dtype) {
    if (op::CheckType(dtype, FOUR_BYTE_DTYPE_LIST)) {
        return FOUR_BYTE;
    }
    if (op::CheckType(dtype, EIGHT_BYTE_DTYPE_LIST)) {
        return EIGHT_BYTE;
    }
    return 0;
}

// 根据size判断算子是否支持走aicore CTCLossV2Grad
static bool IsV2AiCoreSupport(const aclTensor *logProbs, const aclTensor *logAlpha,
                              const aclTensor *targets, const aclTensor *inputLengthsTensor, bool isA2SocVersion) {
    if (targets->IsEmpty()) {
        return false;
    }
    auto logAlphaShape = logAlpha->GetViewShape();
    int64_t maxLabel = (logAlphaShape.GetDim(LABEL_INDEX) - 1) / COEF;
    if (maxLabel > MAX_LABEL_LEN) {
        return false;
    }
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto logProbsShape = logProbs->GetViewShape();
    int64_t targetsDsize = GetDtypeSize(targets->GetDataType());
    int64_t inputLengthsDsize = GetDtypeSize(inputLengthsTensor->GetDataType());
    int64_t symbleSet = logProbsShape.GetDim(SYMBOL_SET_INDEX);
    int64_t batchSize = logProbsShape.GetDim(BATCH_INDEX);
    int64_t timeStep = logProbsShape.GetDim(TIME_INDEX);
    int64_t allDataSize = symbleSet * (C_UB_NUM + ONE_FLOAT / C_LOOP_BLOCK + ONE_FLOAT / C_LOOP_SUM_BLOCK) * FP32_BYTES + 
                                        timeStep * inputLengthsDsize + maxLabel * (TARGETS_UB_NUM * FP32_BYTES + targetsDsize);
    int64_t ubSize = (socVersion >= SocVersion::ASCEND910B) ? SMALL_UB_SIZE : UB_SIZE;
    int64_t availableUbSize = ubSize - RESERVED_UB_SIZE;
    if (batchSize * (targetsDsize + FP32_BYTES) >= availableUbSize) {
        return false;
    }
    if (allDataSize >= availableUbSize) {
        return false;
    }
    return CheckType(logProbs->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// 根据size判断算子是否支持走aicore CTCLossV3Grad
static bool IsV3AiCoreSupport(const aclTensor *logProbs, const aclTensor *logAlpha,
                              const aclTensor *targets, const aclTensor *inputLengthsTensor, bool isA2SocVersion) {
    if (targets->IsEmpty()) {
        return false;
    }
    if (!isA2SocVersion) {
        return false;
    }
    auto logAlphaShape = logAlpha->GetViewShape();
    int64_t maxLabel = (logAlphaShape.GetDim(LABEL_INDEX) - 1) / COEF;
    if (maxLabel > MAX_LABEL_LEN) {
        return false;
    }
    auto logProbsShape = logProbs->GetViewShape();
    int64_t maxSymbol = logProbsShape.GetDim(SYMBOL_SET_INDEX);
    if (maxSymbol > MAX_SYMBOL_SET_LEN_V3) {
        return false;
    }
    int64_t batchSize = logProbsShape.GetDim(BATCH_INDEX);
    if (batchSize > MAX_BATCH) {
        return false;
    }
    auto targetsDtype = targets->GetDataType();
    if ((targetsDtype != op::DataType::DT_INT32) && (targetsDtype != op::DataType::DT_INT64)) {
        return false;
    }
    return CheckType(logProbs->GetDataType(), V3_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
const aclTensor *CtcLossV2GradAiCore(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclTensor *inputLengthsTensor, const aclTensor *targetLengthsTensor, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, int64_t blank, bool zeroInfinity, aclTensor *result, aclOpExecutor *executor) {
    L0_DFX(CtcLossV2GradAiCore, gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
           negLogLikelihood, logAlpha, blank, zeroInfinity, result);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CTCLossV2Grad,
                                           OP_INPUT(gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
                                                    negLogLikelihood, logAlpha),
                                           OP_OUTPUT(result),
                                           OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGD(ACLNN_ERR_INNER_NULLPTR, "CtcLossV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return result;
}

// CTCLossV3Grad AICORE算子kernel
const aclTensor *CtcLossV3GradAiCore(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclTensor *inputLengthsTensor, const aclTensor *targetLengthsTensor, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, int64_t blank, bool zeroInfinity, aclTensor *result, aclOpExecutor *executor) {
    L0_DFX(CtcLossV3GradAiCore, gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
           negLogLikelihood, logAlpha, blank, zeroInfinity, result);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CTCLossV3Grad,
                                           OP_INPUT(gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
                                                    negLogLikelihood, logAlpha),
                                           OP_OUTPUT(result),
                                           OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGD(ACLNN_ERR_INNER_NULLPTR, "CtcLossV3GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return result;
}

const op::Shape CtcLossV2GradNpuOutputShape(const aclTensor *logProbs) {
    return logProbs->GetViewShape();
}

const aclTensor *CtcLossV2Grad(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclTensor *inputLengthsTensor, const aclTensor *targetLengthsTensor, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, int64_t blank, bool zeroInfinity, aclOpExecutor *executor) {
    // 计算输出Tensor的Shape
    auto outputShape = CtcLossV2GradNpuOutputShape(logProbs);

    // 申请输出tensor的空间
    auto result = executor->AllocTensor(outputShape, logProbs->GetDataType(), op::Format::FORMAT_ND);
    bool isA2SocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B);
    if (IsV2AiCoreSupport(logProbs, logAlpha, targets, inputLengthsTensor, isA2SocVersion)) {
        return CtcLossV2GradAiCore(gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
                                   negLogLikelihood, logAlpha, blank, zeroInfinity, result, executor);
    } else if (IsV3AiCoreSupport(logProbs, logAlpha, targets, inputLengthsTensor, isA2SocVersion)) {
        return CtcLossV3GradAiCore(gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor,
                                   negLogLikelihood, logAlpha, blank, zeroInfinity, result, executor);
    } else {
        printf("CTCLossBackward ERROR, AICPU not supported");
        return nullptr;
    }
}
}  // namespace l0op
