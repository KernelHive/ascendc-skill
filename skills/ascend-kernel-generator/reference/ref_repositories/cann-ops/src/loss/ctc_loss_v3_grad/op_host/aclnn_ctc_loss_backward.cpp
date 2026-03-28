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
 * \file aclnn_ctc_loss_backward.cpp
 * \brief
 */
#include "aclnn_ctc_loss_backward.h"
#include "ctc_loss_v2_grad_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include"aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t TARGET_MAX_DIM_NUM = 2;
constexpr int64_t LOP_PROB_DIM_NUM = 3;
constexpr int64_t MAX_SYMBOL_SET_LEN = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 10000;
constexpr int64_t DOUBLE = 2;
constexpr int64_t CDIM = 2;
constexpr int64_t SDIM = 2;
constexpr int64_t T_INDEX = 0;
constexpr int64_t N_INDEX = 1;
constexpr int64_t C_INDEX = 2;
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
// 根据API定义,需要列出log_prob&out参数所能支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LOGPROB_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_A2_LOGPROB_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_DOUBLE};

// 根据API定义,需要列出target参数所能支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_TARGET_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16};
}

static bool CheckNotNull(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclIntArray *inputLengths, const aclIntArray *targetLengths, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, const aclTensor *out) {
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(logProbs, return false);
    OP_CHECK_NULL(targets, return false);
    OP_CHECK_NULL(inputLengths, return false);
    OP_CHECK_NULL(targetLengths, return false);
    OP_CHECK_NULL(negLogLikelihood, return false);
    OP_CHECK_NULL(logAlpha, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclTensor *negLogLikelihood, const aclTensor *logAlpha, const aclTensor *out) {
    bool isA2SocVersion = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    int64_t inputN = logProbs->GetViewShape().GetDim(1);
    int64_t inputC = logProbs->GetViewShape().GetDim(CDIM);
    int64_t inputS = (logAlpha->GetViewShape().GetDim(SDIM) - 1) / DOUBLE;
    bool shapeCheck = inputC > MAX_SYMBOL_SET_LEN || inputN > MAX_BATCH || inputS > MAX_LABEL_LEN;
    bool dtypeCheck = logProbs->GetDataType() == op::DataType::DT_BF16 ||
                      logProbs->GetDataType() == op::DataType::DT_FLOAT16;
    // 检查logProbs的数据类型是否在算子的支持列表内
    if (!isA2SocVersion) {
        OP_CHECK_DTYPE_NOT_SUPPORT(logProbs, DTYPE_SUPPORT_LOGPROB_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(logProbs, DTYPE_SUPPORT_A2_LOGPROB_LIST, return false);
        if (shapeCheck && dtypeCheck) {
            OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Not supported now, we will support it later.");
            return false;
        }
    }

    // gradOut的数据类型必须与logProbs相同
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, logProbs->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_MATCH(negLogLikelihood, logProbs->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_MATCH(logAlpha, logProbs->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_MATCH(out, logProbs->GetDataType(), return false);

    // 检查targets的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(targets, DTYPE_SUPPORT_TARGET_LIST, return false);

    return true;
}

static int64_t GetIntArraySum(const aclIntArray *intArrayValue) {
    // 获取intArrayValue的和
    int64_t sum = 0;
    for (size_t i = 0; i < intArrayValue->Size(); i++) {
        sum += static_cast<int64_t>((*intArrayValue)[i]);
    }
    return sum;
}

static int64_t GetIntArrayMaxValue(const aclIntArray *intArrayValue) {
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    for (size_t i = 0; i < intArrayValue->Size(); i++) {
        if (static_cast<int64_t>((*intArrayValue)[i]) > maxLength) {
            maxLength = static_cast<int64_t>((*intArrayValue)[i]);
        }
    }
    return maxLength;
}

static bool CheckTargetAndOutShapeParameterValue(const aclTensor *logProbs, const aclTensor *targets,
    const aclIntArray *targetLengths, const aclTensor *out) {
    // 当target的shape为2维,则必须满足shape(N,S),S >=targetMaxLength
    if (targets->GetViewShape().GetDimNum() == TARGET_MAX_DIM_NUM) {
        int64_t targetMaxLength = GetIntArrayMaxValue(targetLengths);
        int64_t inputN = logProbs->GetViewShape().GetDim(1);
        if (targets->GetViewShape().GetDim(0) != inputN || targets->GetViewShape().GetDim(1) < targetMaxLength) {
            OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Target's shape expect [%ld, >=%ld], but get current shape %s.",
                    inputN, targetMaxLength, op::ToString(targets->GetViewShape()).GetString());
            return false;
        }
    } else {
        // 当target的shape为1维,则必须满足shape(sum(targetLengths))
        int64_t targetLengthsSum = GetIntArraySum(targetLengths);
        if (targets->GetViewShape().GetDim(0) != targetLengthsSum) {
            OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Target's shape expect [%ld], but get current shape %s.",
                    targetLengthsSum, op::ToString(targets->GetViewShape()).GetString());
            return false;
        }
    }

    // 检查 out的shape
    OP_CHECK_SHAPE_NOT_EQUAL(out, logProbs, return false);
    return true;
}

static bool CheckShapeDimNum(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclTensor *negLogLikelihood, const aclTensor *logAlpha, const int64_t blank, const aclTensor *out)
{
    if (logProbs->GetViewShape().GetDim(0) == 0) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "start (0) + length (1) exceeds dimension size (0).");
        return false;
    }

    // gradOut 必须为1维Tensor
    if (gradOut->GetViewShape().GetDimNum() != 1) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of gradOut [%zu] must equal [1].",
                gradOut->GetViewShape().GetDimNum());
        return false;
    }

    // negLogLikelihood 必须为1维Tensor
    if (negLogLikelihood->GetViewShape().GetDimNum() != 1) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of negLogLikelihood [%zu] must equal [1].",
                negLogLikelihood->GetViewShape().GetDimNum());
        return false;
    }

    // logProbs 必须为3维Tensor
    if (logProbs->GetViewShape().GetDimNum() != LOP_PROB_DIM_NUM) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of logProbs [%zu] must equal 3.",
                logProbs->GetViewShape().GetDimNum());
        return false;
    }

    // logAlpha 必须为3维Tensor
    if (logAlpha->GetViewShape().GetDimNum() != LOP_PROB_DIM_NUM) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of logAlpha [%zu] must equal 3.",
                logAlpha->GetViewShape().GetDimNum());
        return false;
    }

    // targets 必须为1维或2维Tensor
    if (targets->GetViewShape().GetDimNum() != 1 && targets->GetViewShape().GetDimNum() != TARGET_MAX_DIM_NUM) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of targets [%zu] must equal 1 or 2.",
                targets->GetViewShape().GetDimNum());
        return false;
    }

    // out必须为3维tensor
    if (out->GetViewShape().GetDimNum() != LOP_PROB_DIM_NUM) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "DimNum of out [%zu] must equal 3.",
                out->GetViewShape().GetDimNum());
        return false;
    }

    // 检查blank参数范围
    int64_t inputC = logProbs->GetViewShape().GetDim(2);
    if (blank < 0 || blank >= inputC) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID,
                "Attr parameter blank %ld must be greater than or equal to 0 and less than %ld.", blank, inputC);
        return false;
    }
    return true;
}

static bool CheckShapeAndParameterValue(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclIntArray *inputLengths, const aclIntArray *targetLengths, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, int64_t blank, const aclTensor *out) {
    if (!CheckShapeDimNum(gradOut, logProbs, targets, negLogLikelihood, logAlpha, blank, out)) {
        return false;
    }
    //对N，T，S，C，Alpha校验
    int64_t inputT = logProbs->GetViewShape().GetDim(T_INDEX);
    int64_t inputN = logProbs->GetViewShape().GetDim(N_INDEX);
    int64_t inputC = logProbs->GetViewShape().GetDim(C_INDEX);
    int64_t outT = out->GetViewShape().GetDim(T_INDEX);
    int64_t outN = out->GetViewShape().GetDim(N_INDEX);
    int64_t outC = out->GetViewShape().GetDim(C_INDEX);
    int64_t gradOutN = gradOut->GetViewShape().GetDim(0);
    int64_t negLogLikelihoodN = negLogLikelihood->GetViewShape().GetDim(0);
    int64_t logAlphaN = logAlpha->GetViewShape().GetDim(T_INDEX);
    int64_t logAlphaT = logAlpha->GetViewShape().GetDim(N_INDEX);
    int64_t logAlphaLength = logAlpha->GetViewShape().GetDim(C_INDEX);
    int64_t inputLengthN = static_cast<int64_t>(inputLengths->Size());
    int64_t targetsLengthN = static_cast<int64_t>(targetLengths->Size());
    const int64_t maxInputLength = GetIntArrayMaxValue(inputLengths);
    const int64_t maxTargetLength = GetIntArrayMaxValue(targetLengths);
    if (maxTargetLength > (logAlphaLength - 1) / DOUBLE) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "S of targets [%zu] must be smaller than half of alphaLength [%zu].",
                maxTargetLength, logAlphaLength);
        return false;
    }
    bool NCheck = inputN == gradOutN && gradOutN == outN && outN == negLogLikelihoodN &&
                  negLogLikelihoodN == logAlphaN && logAlphaN == inputLengthN && inputLengthN == targetsLengthN;

    if (targets->GetViewShape().GetDimNum() > 1) {
        int64_t targetsN = targets->GetViewShape().GetDim(0);
        int64_t targetsS = targets->GetViewShape().GetDim(1);
        if (targetsS < maxTargetLength) {
            OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Targets tensor's size at dim 1 is at least%ld", maxTargetLength);
            return false;
        }
        NCheck = NCheck && targetsN == targetsLengthN;
    }
    if (!NCheck) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Check batchSize failed.");
        return false;
    }
    bool TCheck = outT == inputT && inputT == logAlphaT && maxInputLength <= inputT;
    if (!TCheck) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Check max time failed.");
        return false;
    }
    bool CCheck = inputC == outC;
    if (!CCheck) {
        OP_LOGD(ACLNN_ERR_PARAM_INVALID, "Check symbolSet failed.");
        return false;
    }
    return CheckTargetAndOutShapeParameterValue(logProbs, targets, targetLengths, out);
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets,
    const aclIntArray *inputLengths, const aclIntArray *targetLengths, const aclTensor *negLogLikelihood,
    const aclTensor *logAlpha, int64_t blank, const aclTensor *out) {
    // 1.检查输入shape是否满足要求
    CHECK_RET(CheckShapeAndParameterValue(gradOut, logProbs, targets, inputLengths, targetLengths,
        negLogLikelihood, logAlpha, blank, out), ACLNN_ERR_PARAM_INVALID);
    // 2.检查输入的数据类型是否在API支持的数据类型范围之内,需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, logProbs, targets, negLogLikelihood, logAlpha, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *logProbs,
    const aclTensor *targets, const aclIntArray *inputLengths, const aclIntArray *targetLengths,
    const aclTensor *negLogLikelihood, const aclTensor *logAlpha, int64_t blank, bool zeroInfinity,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnCtcLossBackward, DFX_IN(gradOut, logProbs, targets, inputLengths, targetLengths,
                                                negLogLikelihood, logAlpha, blank, zeroInfinity), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1、检查入参参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, out),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2、固定写法，参数检查
    auto ret = CheckParams(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha,
                           blank, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果inputN为0，空Tensor，则直接返回空或者nan
    if (logProbs->GetViewShape().GetDim(1) == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入 gradOut 转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入logProbs转换成连续的tensor
    auto logProbsContiguous = l0op::Contiguous(logProbs, uniqueExecutor.get());
    CHECK_RET(logProbsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入 negLogLikelihood 转换成连续的tensor
    auto negLogLikelihoodContiguous = l0op::Contiguous(negLogLikelihood, uniqueExecutor.get());
    CHECK_RET(negLogLikelihoodContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入 logAlpha 转换成连续的tensor
    auto logAlphaContiguous = l0op::Contiguous(logAlpha, uniqueExecutor.get());
    CHECK_RET(logAlphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入targets转换成连续的tensor
    auto targetsContiguous = targets;
    if (!targets->IsEmpty()) {
        targetsContiguous = l0op::Contiguous(targets, uniqueExecutor.get());
        CHECK_RET(targetsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 如果targetsContiguous为Bool,float,float16类型，则转成int64
    auto targetsContiguousCasted = targetsContiguous;
    if (targetsContiguous->GetDataType() != op::DataType::DT_INT32 &&
        targetsContiguous->GetDataType() != op::DataType::DT_INT64) {
        targetsContiguousCasted = l0op::Cast(targetsContiguous, op::DataType::DT_INT64, uniqueExecutor.get());
        CHECK_RET(targetsContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将inputLengths和targetLengths转化成Tensor
    auto inputLengthsTensor = uniqueExecutor.get()->ConvertToTensor(inputLengths,
                                                                    targetsContiguousCasted->GetDataType());
    auto targetLengthsTensor = uniqueExecutor.get()->ConvertToTensor(targetLengths,
                                                                     targetsContiguousCasted->GetDataType());

    // 调用CTCLossV2Grad算子kernel
    auto ctcLossV2GradOpOut = l0op::CtcLossV2Grad(gradOutContiguous, logProbsContiguous, targetsContiguousCasted,
        inputLengthsTensor, targetLengthsTensor, negLogLikelihoodContiguous, logAlphaContiguous, blank, zeroInfinity,
        uniqueExecutor.get());
    CHECK_RET(ctcLossV2GradOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(ctcLossV2GradOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCtcLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnCtcLossBackward);

    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
