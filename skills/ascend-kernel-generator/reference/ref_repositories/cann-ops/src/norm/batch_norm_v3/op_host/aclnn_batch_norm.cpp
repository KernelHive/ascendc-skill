/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_batch_norm.h"
#include "tensor_util.h"
#include "batch_norm_l0.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "fill_l0.h"
#include "aclnn_kernels/reshape.h"
#include "squeeze_l0.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "unsqueeze_l0.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t VAR_INDEX = 2;
constexpr size_t MEAN_INDEX = 1;
constexpr size_t MIN_BN_DIMS = 2;
constexpr size_t MAX_BN_DIMS = 5;
constexpr size_t BN2D_INPUT_DIMS = 4;
constexpr size_t REDUCE_RESULT_CNT = 2;
constexpr size_t UPDATE_RESULT_CNT = 3;
constexpr size_t DIM_C_IDX = 1;
constexpr int64_t PATTERN_A_MIN = 64;
constexpr int64_t PATTERN_R_MIN = 8192;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> &GetSupportDtypeList(SocVersion socVersion)
{
    static const std::initializer_list<op::DataType> emptyDtypes = {};
    static const std::map<SocVersion, std::initializer_list<op::DataType>> dataTypeSupportedMap = {
        {SocVersion::ASCEND310P, {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16}},
        {SocVersion::ASCEND910, {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16}},
        {SocVersion::ASCEND910B, {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16}},
        {SocVersion::ASCEND910_93, {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16}}};

    auto found = dataTypeSupportedMap.find(socVersion);
    if (found == dataTypeSupportedMap.end()) {
        return emptyDtypes;
    }
    return found->second;
}

static bool CheckNotNull(
    const aclTensor *input, const aclTensor *out, const aclTensor *saveMean, const aclTensor *saveInvstd, bool training)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(out, return false);

    if (training) {
        OP_CHECK_NULL(saveMean, return false);
        OP_CHECK_NULL(saveInvstd, return false);
    }
    return true;
}

static inline bool IsBatchNormSupportNcdhw(void)
{
    return ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) ||
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P));
}

static bool isBNV3Supported(const aclTensor *input)
{
    auto inputShape = input->GetViewShape();
    auto inputFormat = input->GetStorageFormat();
    const size_t inputDim = inputShape.GetDimNum();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool isSocSupport = ((socVersion == SocVersion::ASCEND910B) || (socVersion == SocVersion::ASCEND910_93));
    bool isFormatSupport = ((inputFormat == Format::FORMAT_NCDHW) || (inputFormat == Format::FORMAT_NCHW));
    int64_t patternR1 = 1;
    int64_t patternA = 1;
    int64_t patternR0 = 1;
    for (size_t i = 0; i < inputDim; i++) {
        int64_t nowDim = inputShape.GetDim(i);
        if (i < DIM_C_IDX) {
            patternR1 = patternR1 * nowDim;
        } else if (i == DIM_C_IDX) {
            patternA = nowDim;
        } else {
            patternR0 = patternR0 * nowDim;
        }
    }
    bool isShapeSupport = (patternA >= PATTERN_A_MIN);
    OP_LOGD("isBNV3Supported, isSocSupport: %d, isFormatSupport: %d, isShapeSupport: %d",
        isSocSupport,
        isFormatSupport,
        isShapeSupport);

    return (isSocSupport && isFormatSupport && isShapeSupport);
}

static bool CheckDtypeValid(
    const aclTensor *input, const aclTensor *out, const aclTensor *saveMean, const aclTensor *saveInvstd, bool training)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto &DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(socVersion);
    if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return false;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST_CURRENT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_CURRENT, return false);

    if (training) {
        OP_CHECK_DTYPE_NOT_SUPPORT(saveMean, DTYPE_SUPPORT_LIST_CURRENT, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(saveInvstd, DTYPE_SUPPORT_LIST_CURRENT, return false);
    }
    return true;
}

static bool CheckOtherDtypeValid(
    const aclTensor *weight, const aclTensor *bias, const aclTensor *runningMean, const aclTensor *runningVar)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_LOGD("socVersion is : %s", op::ToString(socVersion).GetString());
    const auto &DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(socVersion);
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST_CURRENT, return false);
    }
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_CURRENT, return false);
    }
    if (runningMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningMean, DTYPE_SUPPORT_LIST_CURRENT, return false);
    }
    if (runningVar != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningVar, DTYPE_SUPPORT_LIST_CURRENT, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor *input, const aclTensor *out)
{
    if (input->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be equal, input [%s], output [%s].",
            op::ToString(input->GetStorageFormat()).GetString(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if ((input->GetViewShape().GetDimNum() == MAX_BN_DIMS) && (input->GetStorageFormat() != Format::FORMAT_NCDHW)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input should be NCDWH, when input dim is 5.");
        return false;
    }

    if ((out->GetViewShape().GetDimNum() == MAX_BN_DIMS) && (out->GetStorageFormat() != Format::FORMAT_NCDHW)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of output should be NCDWH, when input dim is 5.");
        return false;
    }

    if (op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCL、NCHW、NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *input, const aclTensor *out)
{
    const int maxCheckNums = 8;
    OP_CHECK_MAX_DIM(input, maxCheckNums, return false);
    OP_CHECK_MIN_DIM(input, MIN_BN_DIMS, return false);
    OP_CHECK_MAX_DIM(out, maxCheckNums, return false);
    OP_CHECK_MIN_DIM(out, MIN_BN_DIMS, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(out, input, return false);
    return true;
}

static bool CheckOtherShape(
    int dimC, const aclTensor *weight, const aclTensor *bias, const aclTensor *runningMean, const aclTensor *runningVar)
{
    if (weight != nullptr && (weight->GetViewShape().GetDimNum() != 1 || weight->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of weight should be one and shape is channel num of input.");
        return false;
    }
    if (bias != nullptr && (bias->GetViewShape().GetDimNum() != 1 || bias->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of bias should be one and shape is channel num of input.");
        return false;
    }
    if (runningMean != nullptr &&
        (runningMean->GetViewShape().GetDimNum() != 1 || runningMean->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of runningMean should be one and shape is channel num of input.");
        return false;
    }
    if (runningVar != nullptr &&
        (runningVar->GetViewShape().GetDimNum() != 1 || runningVar->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of runningVar should be one and shape is channel num of input.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    const aclTensor *runningMean, const aclTensor *runningVar, const aclTensor *output, const aclTensor *saveMean,
    const aclTensor *saveInvstd, bool training)
{
    CHECK_RET(CheckDtypeValid(input, output, saveMean, saveInvstd, training), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOtherDtypeValid(weight, bias, runningMean, runningVar), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckFormat(input, output), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(input, output), ACLNN_ERR_PARAM_INVALID);
    int dimC = input->GetViewShape()[1];
    CHECK_RET(CheckOtherShape(dimC, weight, bias, runningMean, runningVar), ACLNN_ERR_PARAM_INVALID);

    if (training) {
        const op::Shape expectWeightShape = op::Shape{dimC};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(saveMean, expectWeightShape, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(saveInvstd, expectWeightShape, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static bool isEvalAndNotSupportNcdhw(bool training, size_t dimNum)
{
    return (!training && dimNum == MAX_BN_DIMS && !IsBatchNormSupportNcdhw());
}

aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    aclTensor *runningMean, aclTensor *runningVar, bool training, double momentum, double eps, aclTensor *output,
    aclTensor *saveMean, aclTensor *saveInvstd, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBatchNorm,
        DFX_IN(input, weight, bias, runningMean, runningVar, training, momentum, eps),
        DFX_OUT(output, saveMean, saveInvstd));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(input, output, saveMean, saveInvstd, training), ACLNN_ERR_PARAM_NULLPTR);

    if (input->IsEmpty()) {
        auto ret = op::ProcessEmptyTensorWithValue(saveMean, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret =
            op::ProcessEmptyTensorWithValue(saveInvstd, std::numeric_limits<float>::quiet_NaN(), uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = CheckParams(input, weight, bias, runningMean, runningVar, output, saveMean, saveInvstd, training);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputShape = input->GetViewShape();
    auto inputDims = inputShape.GetDimNum();
    if (inputDims > MAX_BN_DIMS) {
        const int64_t shapes[5] = {inputShape[0], inputShape[1], inputShape[2], inputShape[3], -1};
        aclIntArray *shapeArray = uniqueExecutor.get()->AllocIntArray(shapes, 5);
        inputContiguous = l0op::Reshape(inputContiguous, shapeArray, uniqueExecutor.get());
        inputContiguous = l0op::ReFormat(inputContiguous, Format::FORMAT_NCDHW);
    }

    aclTensor *bnOutput = nullptr;
    auto bnResult = BatchNorm(inputContiguous,
        weight,
        bias,
        runningMean,
        runningVar,
        training,
        momentum,
        eps,
        &bnOutput,
        saveMean,
        saveInvstd,
        uniqueExecutor.get());
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    if (inputDims > MAX_BN_DIMS) {
        int64_t originShapes[inputDims];
        for (size_t i = 0; i < inputDims; ++i) {
            originShapes[i] = inputShape[i];
        }
        aclIntArray *originShapeArray = uniqueExecutor.get()->AllocIntArray(originShapes, inputDims);
        auto bnOutputReshape = l0op::Reshape(bnOutput, originShapeArray, uniqueExecutor.get());
        auto bnOutputReformat = l0op::ReFormat(bnOutputReshape, Format::FORMAT_ND);
        auto viewCopyResult = l0op::ViewCopy(bnOutputReformat, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto viewCopyResult = l0op::ViewCopy(bnOutput, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus DoBatchNormProc(const aclTensor *input, const aclTensor *weightResize, const aclTensor *biasResize,
    aclTensor *runningMeanOut, aclTensor *runningVarOut, float momentum, float eps, aclTensor **output,
    aclTensor *saveMean, aclTensor *saveInvstd, aclOpExecutor *executor)
{
    size_t dimNum = input->GetViewShape().GetDimNum();
    std::array<aclTensor *, UPDATE_RESULT_CNT> outTensor;
    bool isSupportNcdhw = IsBatchNormSupportNcdhw();
    if (dimNum == MAX_BN_DIMS && !isSupportNcdhw) {
        auto input6hd = l0op::TransDataSpecial(input, Format::FORMAT_NDC1HWC0, 0, executor);
        CHECK_RET(input6hd != nullptr, ACLNN_ERR_INNER_NULLPTR);

        std::array<aclTensor *, REDUCE_RESULT_CNT> sumTensor =
            l0op::BN3DTrainingReduce(input6hd, weightResize->GetStorageShape(), executor);
        outTensor = l0op::BN3DTrainingUpdate(input6hd,
            sumTensor[0],
            sumTensor[1],
            weightResize,
            biasResize,
            runningMeanOut,
            runningVarOut,
            momentum,
            eps,
            executor);
        CHECK_RET(outTensor[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resultNcdhw = l0op::TransDataSpecial(outTensor[0], Format::FORMAT_NCDHW, 0, executor);
        CHECK_RET(resultNcdhw != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *output = const_cast<aclTensor *>(resultNcdhw);
    } else if (dimNum == MAX_BN_DIMS) {
        std::array<aclTensor *, REDUCE_RESULT_CNT> sumTensor =
            l0op::BN3DTrainingReduce(input, weightResize->GetStorageShape(), executor);
        outTensor = l0op::BN3DTrainingUpdate(input,
            sumTensor[0],
            sumTensor[1],
            weightResize,
            biasResize,
            runningMeanOut,
            runningVarOut,
            momentum,
            eps,
            executor);
        CHECK_RET(outTensor[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *output = const_cast<aclTensor *>(outTensor[0]);
    } else {
        std::array<aclTensor *, REDUCE_RESULT_CNT> sumTensor =
            l0op::BNTrainingReduce(input, weightResize->GetViewShape(), executor);
        outTensor = l0op::BNTrainingUpdate(input,
            sumTensor[0],
            sumTensor[1],
            weightResize,
            biasResize,
            runningMeanOut,
            runningVarOut,
            momentum,
            eps,
            executor);
        CHECK_RET(outTensor[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *output = outTensor[0];
    }

    auto viewCopyMean = op::ResizeTo1D(outTensor[MEAN_INDEX], saveMean, isSupportNcdhw, executor);
    CHECK_RET(viewCopyMean != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyVar = op::ResizeTo1D(outTensor[VAR_INDEX], saveInvstd, isSupportNcdhw, executor);
    CHECK_RET(viewCopyVar != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormProc(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    aclTensor *runningMean, aclTensor *runningVar, bool training, float momentum, float eps, aclTensor **output,
    aclTensor *saveMean, aclTensor *saveInvstd, aclOpExecutor *executor)
{
    bool isSupportNcdhw = IsBatchNormSupportNcdhw();
    auto weightResize = op::ResizeFrom1D(weight, input, isSupportNcdhw, executor);
    CHECK_RET(weightResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasResize = op::ResizeFrom1D(bias, input, isSupportNcdhw, executor);
    CHECK_RET(biasResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanResize = op::ResizeFrom1D(runningMean, input, isSupportNcdhw, executor);
    CHECK_RET(runningMeanResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarResize = op::ResizeFrom1D(runningVar, input, isSupportNcdhw, executor);
    CHECK_RET(runningVarResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!training) {
        auto inferResult =
            l0op::BNInfer(input, weightResize, biasResize, runningMeanResize, runningVarResize, eps, executor);
        CHECK_RET(inferResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *output = const_cast<aclTensor *>(inferResult);
    } else {
        auto runningMeanOut = const_cast<aclTensor *>(runningMeanResize);
        auto runningVarOut = const_cast<aclTensor *>(runningVarResize);

        CHECK_RET(DoBatchNormProc(input, weightResize, biasResize, runningMeanOut, runningVarOut, momentum, eps, output,
            saveMean, saveInvstd, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

        if (!runningMean->IsFromWorkspace()) {
            auto viewCopyRunningMean = op::ResizeTo1D(runningMeanOut, runningMean, isSupportNcdhw, executor);
            CHECK_RET(viewCopyRunningMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if (!runningVar->IsFromWorkspace()) {
            auto viewCopyRunningVar = op::ResizeTo1D(runningVarOut, runningVar, isSupportNcdhw, executor);
            CHECK_RET(viewCopyRunningVar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormV3Proc(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
    aclTensor *runningMean, aclTensor *runningVar, float momentum, float eps, aclTensor **output, aclTensor *saveMean,
    aclTensor *saveInvstd, aclOpExecutor *executor)
{
    op::DataType weightBiasPromoteType = op::PromoteType(weight->GetDataType(), bias->GetDataType());
    op::DataType weightBiasDstType =
        (weightBiasPromoteType == input->GetDataType()) ? input->GetDataType() : DataType::DT_FLOAT;

    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightCast = l0op::Cast(weightContiguous, weightBiasDstType, executor);
    CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasContiguous = l0op::Contiguous(bias, executor);
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto biasCast = l0op::Cast(biasContiguous, weightBiasDstType, executor);
    CHECK_RET(biasCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanContiguous = l0op::Contiguous(runningMean, executor);
    CHECK_RET(runningMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto runningMeanCast = l0op::Cast(runningMeanContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(runningMeanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarContiguous = l0op::Contiguous(runningVar, executor);
    CHECK_RET(runningVarContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto runningVarCast = l0op::Cast(runningVarContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(runningVarCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanOut = const_cast<aclTensor *>(runningMeanCast);
    auto runningVarOut = const_cast<aclTensor *>(runningVarCast);
    std::array<aclTensor *, UPDATE_RESULT_CNT> outTensor =
        l0op::BatchNormV3(input, weightCast, biasCast, runningMeanOut, runningVarOut, momentum, eps, executor);
    *output = outTensor[0];
    CHECK_RET(outTensor[MEAN_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outTensor[VAR_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!runningMean->IsFromWorkspace()) {
        auto runningMeanResCast = l0op::Cast(runningMeanOut, runningMean->GetDataType(), executor);
        CHECK_RET(runningMeanResCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto runningMeanResViewCopy = l0op::ViewCopy(runningMeanResCast, runningMean, executor);
        CHECK_RET(runningMeanResViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (!runningVar->IsFromWorkspace()) {
        auto runningVarResCast = l0op::Cast(runningVarOut, runningVar->GetDataType(), executor);
        CHECK_RET(runningVarResCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto runningVarResViewCopy = l0op::ViewCopy(runningVarResCast, runningVar, executor);
        CHECK_RET(runningVarResViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto saveMeanResCast = l0op::Cast(outTensor[MEAN_INDEX], saveMean->GetDataType(), executor);
    CHECK_RET(saveMeanResCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto saveMeanResViewCopy = l0op::ViewCopy(saveMeanResCast, saveMean, executor);
    CHECK_RET(saveMeanResViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto saveInvstdResCast = l0op::Cast(outTensor[VAR_INDEX], saveInvstd->GetDataType(), executor);
    CHECK_RET(saveInvstdResCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto saveInvstdResViewCopy = l0op::ViewCopy(saveInvstdResCast, saveInvstd, executor);
    CHECK_RET(saveInvstdResViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus BatchNorm(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *runningMean,
    aclTensor *runningVar, bool training, float momentum, float eps, aclTensor **output, aclTensor *saveMean,
    aclTensor *saveInvstd, aclOpExecutor *executor)
{
    size_t dimC = input->GetViewShape()[1];
    if (runningMean == nullptr) {
        runningMean = op::FillScalar(dimC, 0, executor);
        CHECK_RET(runningMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (runningVar == nullptr) {
        runningVar = op::FillScalar(dimC, 1, executor);
        CHECK_RET(runningVar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (weight == nullptr) {
        weight = op::FillScalar(dimC, 1, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (bias == nullptr) {
        bias = op::FillScalar(dimC, 0, executor);
        CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    size_t dimNum = input->GetViewShape().GetDimNum();
    auto inputPre = input;
    if (dimNum < BN2D_INPUT_DIMS) {
        inputPre = op::ResizeFromND(input, executor);
    } else if (isEvalAndNotSupportNcdhw(training, dimNum)) {
        inputPre = op::ResizeFrom5D(input, executor);
    }
    CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclTensor *result = nullptr;
    if (isBNV3Supported(inputPre) && training) {
        CHECK_RET(BatchNormV3Proc(inputPre, weight, bias, runningMean, runningVar, momentum, eps, &result, saveMean,
            saveInvstd, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(BatchNormProc(inputPre, weight, bias, runningMean, runningVar, training, momentum, eps, &result,
            saveMean, saveInvstd, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
    }

    *output = result;
    if (dimNum < BN2D_INPUT_DIMS) {
        auto outputFormat = op::ResizeToND(result, input, executor);
        CHECK_RET(outputFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *output = const_cast<aclTensor *>(outputFormat);
    } else if (isEvalAndNotSupportNcdhw(training, dimNum)) {
        auto outputTranspose = op::ResizeTo5D(result, input, executor);
        CHECK_RET(outputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *output = const_cast<aclTensor *>(outputTranspose);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
