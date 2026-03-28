/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "mul_l0.h"
#include "div_l0.h"
#include "ones_like_l0.h"
#include "sub_l0.h"
#include "squeeze_l0.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "unsqueeze_l0.h"
#include "batch_norm_backward_l0.h"
#include "group_norm_grad_l0.h"
#include "aclnn_kernels/cast.h"
#include "expand.h"
#include "fill_l0.h"
#include "zero_op_l0.h"
#include "reduce_sum_op_l0.h"
#include "tensor_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api_def.h"
#include "opdev/platform.h"
#include "level2_base.h"
#include "aclnn_group_norm_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t SHAPE_DIM = 3;
const int64_t AXIS_ZERO = 0;
const int64_t AXIS_ONE = 1;
const int64_t AXIS_TWO = 2;
constexpr size_t MAX_BN_DIMS = 5;
constexpr size_t BN2D_INPUT_DIMS = 4;
constexpr size_t UPDATE_GRAD_RESULT_CNT = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *gradOut, const aclTensor *input, const aclTensor *mean, const aclTensor *rstd)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(rstd, return false);
    return true;
}

static inline bool IsBatchNormSupportNcdhw(void)
{
    return ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) ||
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P));
}

static bool CheckMaskNotNull(const aclTensor *gradInput, const aclTensor *gradGammaOut, const aclTensor *gradBetaOut,
    const aclBoolArray *outputMask)
{
    if ((*outputMask)[AXIS_ZERO]) {
        OP_CHECK_NULL(gradInput, return false);
    }
    if ((*outputMask)[AXIS_ONE]) {
        OP_CHECK_NULL(gradGammaOut, return false);
    }
    if ((*outputMask)[AXIS_TWO]) {
        OP_CHECK_NULL(gradBetaOut, return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *input, const aclTensor *mean,
    const aclTensor *rstd, const aclTensor *gamma, const aclTensor *gradInput, const aclTensor *gradGammaOut,
    const aclTensor *gradBetaOut, const aclBoolArray *outputMask)
{
    // 检查self的数据类型是否在支持列表内
    auto supportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(input, gradOut->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(mean, gradOut->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(rstd, gradOut->GetDataType(), return false);
    if (gamma != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(gamma, gradOut->GetDataType(), return false);
    }
    if ((*outputMask)[AXIS_ZERO]) {
        OP_CHECK_DTYPE_NOT_MATCH(gradInput, gradOut->GetDataType(), return false);
    }
    if ((*outputMask)[AXIS_ONE]) {
        OP_CHECK_DTYPE_NOT_MATCH(gradGammaOut, gradOut->GetDataType(), return false);
    }
    if ((*outputMask)[AXIS_TWO]) {
        OP_CHECK_DTYPE_NOT_MATCH(gradBetaOut, gradOut->GetDataType(), return false);
    }
    return true;
}

static int64_t GetTensorNumel(const aclTensor *x)
{
    size_t xShapeDim = x->GetViewShape().GetDimNum();
    int64_t xShapeSize = 1;
    for (int64_t i = 0; i < xShapeDim; i++) {
        xShapeSize *= x->GetViewShape().GetDim(i);
    }
    return xShapeSize;
}

static int64_t GetTensorHxW(const aclTensor *x)
{
    size_t xShapeDim = x->GetViewShape().GetDimNum();
    int64_t xShapeSize = 1;
    for (int64_t i = 2; i < xShapeDim; i++) {
        xShapeSize *= x->GetViewShape().GetDim(i);
    }
    return xShapeSize;
}

static bool CheckAttr(int64_t C, int64_t group)
{
    // 检查group是否大于0
    OP_CHECK(group > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected group to be greater than 0, got %ld.", group),
        return false);
    // 检查C是否能被group整除
    OP_CHECK(C % group == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected C can be divided by group, but got false."),
        return false);
    return true;
}

static bool CheckShape(const aclTensor *gradOut, const aclTensor *input, const aclTensor *mean, const aclTensor *rstd,
    const aclTensor *gamma, int64_t N, int64_t C, int64_t HxW, int64_t group, const aclBoolArray *outputMask,
    const aclTensor *gradInput, const aclTensor *gradGammaOut, const aclTensor *gradBetaOut)
{
    // 检查gradOut维度是否小于等于8
    OP_CHECK_MAX_DIM(gradOut, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查gradOut的元素数量是否等于 N * C * HxW
    OP_CHECK(GetTensorNumel(gradOut) == N * C * HxW,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected dY.numel() == N * C * HxW to be true, but got false."),
        return false);
    // 检查input的元素数量是否等于 N * C * HxW
    OP_CHECK(GetTensorNumel(input) == N * C * HxW,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected X.numel() == N * C * HxW to be true, but got false."),
        return false);
    // 检查mean的元素数量是否等于 N * group
    OP_CHECK(GetTensorNumel(mean) == N * group,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected mean.numel() == N * group to be true, but got false."),
        return false);
    // 检查rstd的元素数量是否等于 N * group
    OP_CHECK(GetTensorNumel(rstd) == N * group,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected rstd.numel() == N * group to be true, but got false."),
        return false);
    // 获取原始输入shape
    auto orginInputShape = input->GetViewShape();
    auto orginGammaShape = op::Shape({C});
    if (gamma != nullptr) {
        orginGammaShape = gamma->GetViewShape();
    }
    // 检查(gamma是否为空指针)或者(元素数量是否等于C)
    if (gamma != nullptr && (GetTensorNumel(gamma) != C)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected gamma.numel() == C to be true, but got false.");
        return false;
    }
    // 检查gradInput的shape是否与gradOut不同。
    if ((*outputMask)[AXIS_ZERO]) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, orginInputShape, return false);
    }
    // 检查gradGammaOut, gradBetaOut的shape是否与gamma的shape相同
    if ((*outputMask)[AXIS_ONE]) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradGammaOut, orginGammaShape, return false);
    }
    if ((*outputMask)[AXIS_TWO]) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradBetaOut, orginGammaShape, return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclTensor *input, const aclTensor *mean,
    const aclTensor *rstd, const aclTensor *gamma, int64_t N, int64_t C, int64_t HxW, int64_t group,
    const aclBoolArray *outputMask, const aclTensor *gradInput, const aclTensor *gradGammaOut,
    const aclTensor *gradBetaOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, input, mean, rstd), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查对应输出元素是否为空
    CHECK_RET(CheckMaskNotNull(gradInput, gradGammaOut, gradBetaOut, outputMask), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, input, mean, rstd, gamma, gradInput, gradGammaOut, gradBetaOut, outputMask),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查对应属性
    CHECK_RET(CheckAttr(C, group), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查是否shape一致
    CHECK_RET(
        CheckShape(
            gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormBackwardProcInner(const aclTensor *gradOut, const aclTensor *input, const aclTensor *weight,
    const aclTensor *runningMean, const aclTensor *runningVar, const aclTensor *saveMean, const aclTensor *saveInvstd,
    bool training, float eps, aclTensor **gradInput, aclTensor **gradWeight, aclTensor **gradBias,
    aclOpExecutor *executor)
{
    bool isSupportNcdhw = IsBatchNormSupportNcdhw();
    auto weightResize = op::ResizeFrom1D(weight, input, isSupportNcdhw, executor);
    CHECK_RET(weightResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    size_t dimNum = input->GetViewShape().GetDimNum();
    std::array<aclTensor *, UPDATE_GRAD_RESULT_CNT> grad;
    if (!training) {
        auto runningMeanResize = op::ResizeFrom1D(runningMean, input, isSupportNcdhw, executor);
        CHECK_RET(runningMeanResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto runningVarResize = op::ResizeFrom1D(runningVar, input, isSupportNcdhw, executor);
        CHECK_RET(runningVarResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        grad = l0op::BNTrainingUpdateGrad(gradOut, input, runningMeanResize, runningVarResize, eps, executor);
        CHECK_RET(grad[0] != nullptr && grad[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto reduceGrad = l0op::BNInferGrad(gradOut, weightResize, runningVarResize, eps, executor);
        CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor *>(reduceGrad);
    } else {
        auto saveMeanResize = op::ResizeFrom1D(saveMean, input, isSupportNcdhw, executor);
        CHECK_RET(saveMeanResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto saveInvstdResize = op::ResizeFrom1D(saveInvstd, input, isSupportNcdhw, executor);
        CHECK_RET(saveInvstdResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (dimNum == MAX_BN_DIMS) {
            grad = l0op::BN3DTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(grad[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(grad[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto reduceGrad = l0op::BN3DTrainingReduceGrad(
                gradOut, input, grad[0], grad[1], weightResize, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            *gradInput = const_cast<aclTensor *>(reduceGrad);
        } else {
            grad = l0op::BNTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(grad[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(grad[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto reduceGrad = l0op::BNTrainingReduceGrad(
                gradOut, input, grad[0], grad[1], weightResize, saveMeanResize, saveInvstdResize, eps, executor);
            *gradInput = const_cast<aclTensor *>(reduceGrad);
        }
    }
    *gradWeight = grad[0];
    *gradBias = grad[1];
    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormBackwardInner(const aclTensor *gradOut, const aclTensor *input, const aclTensor *weight,
    const aclTensor *runningMean, const aclTensor *runningVar, const aclTensor *saveMean, const aclTensor *saveInvstd,
    bool training, float eps, aclTensor **gradInput, aclTensor **gradWeight, aclTensor **gradBias,
    aclOpExecutor *executor)
{
    if (runningMean == nullptr) {
        runningMean = op::FillScalar(input->GetViewShape()[1], 0, executor);
        CHECK_RET(runningMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (runningVar == nullptr) {
        runningVar = op::FillScalar(input->GetViewShape()[1], 1, executor);
        CHECK_RET(runningVar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (weight == nullptr) {
        weight = op::FillScalar(input->GetViewShape()[1], 1, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (saveMean == nullptr) {
        saveMean = op::FillScalar(input->GetViewShape()[1], 0, executor);
        CHECK_RET(saveMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (saveInvstd == nullptr) {
        saveInvstd = op::FillScalar(input->GetViewShape()[1], 1, executor);
        CHECK_RET(saveInvstd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto gradOutPre = gradOut;
    auto inputPre = input;
    if (input->GetViewShape().GetDimNum() < BN2D_INPUT_DIMS) {
        gradOutPre = op::ResizeFromND(gradOut, executor);
        CHECK_RET(gradOutPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        inputPre = op::ResizeFromND(input, executor);
        CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor *result = nullptr;
    auto bnResult = BatchNormBackwardProcInner(gradOutPre,
        inputPre,
        weight,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        &result,
        gradWeight,
        gradBias,
        executor);
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    *gradInput = result;
    if (input->GetViewShape().GetDimNum() < BN2D_INPUT_DIMS) {
        auto outputFormat = op::ResizeToND(result, input, executor);
        CHECK_RET(outputFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor *>(outputFormat);
    }

    return ACLNN_SUCCESS;
}

static const aclTensor *getBatchNormBackwardGradInput(const aclTensor *gradOut, const aclTensor *input,
    const aclTensor *weight, const aclTensor *runningMean, const aclTensor *runningVar, const aclTensor *saveMean,
    const aclTensor *saveInvstd, bool training, double eps, const aclBoolArray *outputMask, aclOpExecutor *executor)
{
    auto inputShape = input->GetViewShape();
    aclTensor *bnGradInput = nullptr;
    aclTensor *bnGradWeight = nullptr;
    aclTensor *bnGradBias = nullptr;
    auto bnResult = BatchNormBackwardInner(gradOut,
        input,
        weight,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        &bnGradInput,
        &bnGradWeight,
        &bnGradBias,
        executor);
    CHECK_RET(bnResult == ACLNN_SUCCESS, nullptr);

    return bnGradInput;
}

static const aclTensor *Reshape1DProcess(const aclTensor *self, int64_t N, aclOpExecutor *executor)
{
    int64_t shapeValue[1] = {N};
    aclIntArray *shapeArray = executor->AllocIntArray(shapeValue, 1);
    CHECK_RET(shapeArray != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, shapeArray, executor);
    return selfReshape;
}

static const aclTensor *Reshape3DProcess(
    const aclTensor *self, int64_t N, int64_t C, int64_t HxW, aclOpExecutor *executor)
{
    int64_t shapeValue[3] = {N, C, HxW};
    aclIntArray *shapeArray = executor->AllocIntArray(shapeValue, 3);
    CHECK_RET(shapeArray != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, shapeArray, executor);
    return selfReshape;
}

static const aclTensor *FillValue(const aclTensor *out, float val, aclOpExecutor *executor)
{
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    CHECK_RET(dims != nullptr, nullptr);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    CHECK_RET(shapeArray != nullptr, nullptr);

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    CHECK_RET(valTensor != nullptr, nullptr);
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, nullptr);
    return fillOut;
}

// 获得tensor的维度数
static inline int64_t GetTensorDim(const aclTensor *self)
{
    return static_cast<int64_t>(self->GetViewShape().GetDimNum());
}

static aclIntArray *GetShape(const aclTensor *self, aclOpExecutor *executor)
{
    int64_t dimSize = GetTensorDim(self);
    int64_t selfShapeValue[dimSize];
    for (int64_t i = 0; i < dimSize; i++) {
        selfShapeValue[i] = self->GetViewShape().GetDim(i);
    }
    aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, dimSize);
    CHECK_RET(selfShape != nullptr, nullptr);
    return selfShape;
}

static const aclTensor *GetvarianceBatchNorm(
    const aclTensor *rstdContiguous, int64_t N, int64_t group, double eps, aclOpExecutor *executor)
{
    auto variance = l0op::Mul(rstdContiguous, rstdContiguous, executor);
    CHECK_RET(variance != nullptr, nullptr);
    auto rstdOne = l0op::OnesLike(variance, executor);
    CHECK_RET(rstdOne != nullptr, nullptr);
    variance = l0op::Div(rstdOne, variance, executor);
    CHECK_RET(variance != nullptr, nullptr);
    auto epsScalar = executor->AllocScalar(static_cast<float>(eps));
    auto epsTensor = executor->ConvertToTensor(epsScalar, variance->GetDataType());
    CHECK_RET(epsTensor != nullptr, nullptr);
    variance = l0op::Sub(variance, epsTensor, executor);
    CHECK_RET(variance != nullptr, nullptr);
    auto varianceBatchNorm = Reshape1DProcess(variance, N * group, executor);
    CHECK_RET(varianceBatchNorm != nullptr, nullptr);
    return varianceBatchNorm;
}

static aclnnStatus GradInputProcess(const aclTensor *gradOutReshape, const aclTensor *meanContiguous,
    const aclTensor *rstdContiguous, const aclTensor *gammaContiguous, const aclTensor *inputReshape, int64_t N,
    int64_t C, int64_t group, const aclBoolArray *outputMask, aclTensor *gradInput, aclOpExecutor *executor)
{
    auto gradOutMulGamma = gradOutReshape;
    if (gamma != nullptr) {
        auto gammaReshape = Reshape3DProcess(gammaContiguous, 1, C, 1, executor);
        CHECK_RET(gammaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutMulGamma = l0op::Mul(gradOutReshape, gammaReshape, executor);
    }
    CHECK_RET(gradOutMulGamma != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputBatchNorm = Reshape3DProcess(inputReshape, 1, N * group, N ? -1 : 1, executor);
    CHECK_RET(inputBatchNorm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutBatchNorm = Reshape3DProcess(gradOutMulGamma, 1, N * group, N ? -1 : 1, executor);
    CHECK_RET(gradOutBatchNorm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanBatchNorm = Reshape1DProcess(meanContiguous, N * group, executor);
    CHECK_RET(meanBatchNorm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const float eps = 1e-5;
    auto varianceBatchNorm = GetvarianceBatchNorm(rstdContiguous, N, group, eps, executor);
    CHECK_RET(varianceBatchNorm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightTmp = executor->AllocTensor(meanBatchNorm->GetViewShape(), meanContiguous->GetDataType());
    CHECK_RET(weightTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weight = FillValue(weightTmp, 1.0, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputBatch = getBatchNormBackwardGradInput(gradOutBatchNorm,
        inputBatchNorm,
        weight,
        nullptr,
        nullptr,
        meanBatchNorm,
        varianceBatchNorm,
        true,
        eps,
        outputMask,
        executor);
    CHECK_RET(gradInputBatch != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradInputReshape = l0op::Reshape(gradInputBatch, gradInput->GetViewShape(), executor);
    CHECK_RET(gradInputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto inputViewCopyResult = l0op::ViewCopy(gradInputReshape, gradInput, executor);
    CHECK_RET(inputViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor *CalDgamma(const aclTensor *gradOutReshape, const aclTensor *inputReshape, const aclTensor *mean,
    const aclTensor *rstd, aclIntArray *AxisZeroShape, aclIntArray *AxisTwoShape, int64_t N, int64_t C, int64_t group,
    aclTensor *gradGammaOut, aclOpExecutor *executor)
{
    int64_t cDivGroup = group > 0 ? C / group : C;
    int64_t expandShapeValue[3] = {N, group, cDivGroup};
    aclIntArray *expandShape = executor->AllocIntArray(expandShapeValue, 3);
    CHECK_RET(expandShape != nullptr, nullptr);
    auto meanBroadcast = l0op::Expand(mean, expandShape, executor);
    CHECK_RET(meanBroadcast != nullptr, nullptr);
    meanBroadcast = Reshape3DProcess(meanBroadcast, N, C, 1, executor);
    CHECK_RET(meanBroadcast != nullptr, nullptr);

    auto rstdBroadcast = l0op::Expand(rstd, expandShape, executor);
    CHECK_RET(rstdBroadcast != nullptr, nullptr);
    rstdBroadcast = Reshape3DProcess(rstdBroadcast, N, C, 1, executor);
    CHECK_RET(rstdBroadcast != nullptr, nullptr);

    auto xHat = l0op::Sub(inputReshape, meanBroadcast, executor);
    CHECK_RET(xHat != nullptr, nullptr);
    auto xHatMul = l0op::Mul(xHat, rstdBroadcast, executor);
    CHECK_RET(xHatMul != nullptr, nullptr);
    auto dgammaMul = l0op::Mul(gradOutReshape, xHatMul, executor);
    CHECK_RET(dgammaMul != nullptr, nullptr);

    auto dgammaSum = l0op::ReduceSumOp(dgammaMul, AxisTwoShape, false, executor);
    CHECK_RET(dgammaSum != nullptr, nullptr);

    auto dgamma = l0op::ReduceSumOp(dgammaSum, AxisZeroShape, false, executor);
    return dgamma;
}

static aclnnStatus GetDgamma(const aclTensor *gradOutReshape, const aclTensor *inputReshape,
    const aclTensor *meanContiguous, const aclTensor *rstdContiguous, int64_t N, int64_t C, int64_t group,
    const aclBoolArray *outputMask, aclTensor *gradGammaOut, aclOpExecutor *executor)
{
    int64_t AxisTwoShapeValue[1] = {AXIS_TWO};
    aclIntArray *AxisTwoShape = executor->AllocIntArray(AxisTwoShapeValue, 1);
    CHECK_RET(AxisTwoShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t AxisZeroShapeValue[1] = {AXIS_ZERO};
    aclIntArray *AxisZeroShape = executor->AllocIntArray(AxisZeroShapeValue, 1);
    CHECK_RET(AxisZeroShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *meanBroadcast;
    const aclTensor *rstdBroadcast;
    if (GetTensorDim(meanContiguous) == 1) {
        int64_t broadcastShapeValue[2] = {N, group};
        aclIntArray *broadcastShape = executor->AllocIntArray(broadcastShapeValue, 2);
        CHECK_RET(broadcastShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanTmp = l0op::Reshape(meanContiguous, broadcastShape, executor);
        CHECK_RET(meanTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        meanBroadcast = l0op::UnsqueezeNd(meanTmp, AXIS_TWO, executor);
        auto rstdTmp = l0op::Reshape(rstdContiguous, broadcastShape, executor);
        CHECK_RET(meanTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        rstdBroadcast = l0op::UnsqueezeNd(rstdTmp, AXIS_TWO, executor);
    } else {
        meanBroadcast = l0op::UnsqueezeNd(meanContiguous, AXIS_TWO, executor);
        rstdBroadcast = l0op::UnsqueezeNd(rstdContiguous, AXIS_TWO, executor);
    }
    CHECK_RET(meanBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(rstdBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dgamma = CalDgamma(gradOutReshape,
        inputReshape,
        meanBroadcast,
        rstdBroadcast,
        AxisZeroShape,
        AxisTwoShape,
        N,
        C,
        group,
        gradGammaOut,
        executor);
    CHECK_RET(dgamma != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dgammaReshape = l0op::Reshape(dgamma, gradGammaOut->GetViewShape(), executor);
    CHECK_RET(dgammaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出gradGammaOut上，gradGammaOut可能是非连续的tensor
    auto gammaViewCopyResult = l0op::ViewCopy(dgammaReshape, gradGammaOut, executor);
    CHECK_RET(gammaViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus GetDbeta(
    const aclTensor *gradOutReshape, const aclBoolArray *outputMask, aclTensor *gradBetaOut, aclOpExecutor *executor)
{
    int64_t AxisTwoShapeValue[1] = {AXIS_TWO};
    aclIntArray *AxisTwoShape = executor->AllocIntArray(AxisTwoShapeValue, 1);
    CHECK_RET(AxisTwoShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t AxisZeroShapeValue[1] = {AXIS_ZERO};
    aclIntArray *AxisZeroShape = executor->AllocIntArray(AxisZeroShapeValue, 1);
    CHECK_RET(AxisZeroShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dbetaSum = l0op::ReduceSumOp(gradOutReshape, AxisTwoShape, false, executor);
    CHECK_RET(dbetaSum != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dbeta = l0op::ReduceSumOp(dbetaSum, AxisZeroShape, false, executor);
    CHECK_RET(dbeta != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dbetaReshape = l0op::Reshape(dbeta, gradBetaOut->GetViewShape(), executor);
    CHECK_RET(dbetaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出gradBetaOut上，gradBetaOut可能是非连续的tensor
    auto betaViewCopyResult = l0op::ViewCopy(dbetaReshape, gradBetaOut, executor);
    CHECK_RET(betaViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor *GetgammaContiguous(
    const aclTensor *gradOut, const aclTensor *gamma, int64_t C, aclOpExecutor *executor)
{
    auto gammaContiguous = gamma;
    if (gamma == nullptr) {
        auto gammaTmp = executor->AllocTensor(op::Shape({C}), gradOut->GetDataType());
        CHECK_RET(gammaTmp != nullptr, nullptr);
        gammaContiguous = FillValue(gammaTmp, 1.0, executor);
    } else {
        gammaContiguous = l0op::Contiguous(gamma, executor);
    }
    return gammaContiguous;
}

aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
    const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma, int64_t N, int64_t C, int64_t HxW,
    int64_t group, const aclBoolArray *outputMask, aclTensor *gradInput, aclTensor *gradGammaOut,
    aclTensor *gradBetaOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnGroupNormBackward,
        DFX_IN(gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask),
        DFX_OUT(gradInput, gradGammaOut, gradBetaOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(
        gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持
    if (gradOut->IsEmpty() || input->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto rstdContiguous = l0op::Contiguous(rstd, uniqueExecutor.get());
    CHECK_RET(rstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gammaContiguous = GetgammaContiguous(gradOut, gamma, C, uniqueExecutor.get());
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // reshape为 N C HxW 的维度
    auto gradOutReshape = l0op::Reshape(gradOutContiguous, {N, C, HxW}, uniqueExecutor.get());
    auto inputReshape = l0op::Reshape(inputContiguous, {N, C, HxW}, uniqueExecutor.get());
    auto meanReshape = l0op::Reshape(meanContiguous, {N, group}, uniqueExecutor.get());
    auto rstdReshape = l0op::Reshape(rstdContiguous, {N, group}, uniqueExecutor.get());
    auto gammaReshape = l0op::Reshape(gammaContiguous,
        {
            C,
        },
        uniqueExecutor.get());
    CHECK_RET(gradOutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(inputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(meanReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(rstdReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::string data_format = "NCHW";
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        // 创建输出Tensor
        aclTensor *dx_output = uniqueExecutor.get()->AllocTensor(
            gradOutReshape->GetViewShape(), gradOutReshape->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(dx_output != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclTensor *dgamma_output = uniqueExecutor.get()->AllocTensor(
            gammaReshape->GetViewShape(), gammaReshape->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(dgamma_output != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclTensor *dbeta_output = uniqueExecutor.get()->AllocTensor(
            gammaReshape->GetViewShape(), gammaReshape->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(dbeta_output != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用l0接口GroupNormGrad
        auto GroupNormGradOut = l0op::GroupNormGrad(gradOutReshape,
            meanReshape,
            rstdReshape,
            inputReshape,
            gammaReshape,
            group,
            data_format,
            (*outputMask)[AXIS_ZERO],
            (*outputMask)[AXIS_ONE],
            (*outputMask)[AXIS_TWO],
            dx_output,
            dgamma_output,
            dbeta_output,
            uniqueExecutor.get());
        if ((*outputMask)[AXIS_ZERO]) {
            auto dx = std::get<0>(GroupNormGradOut);
            CHECK_RET(dx != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dxReshape = l0op::Reshape(dx, gradInput->GetViewShape(), uniqueExecutor.get());
            auto dxViewCopyResult = l0op::ViewCopy(dxReshape, gradInput, uniqueExecutor.get());
            CHECK_RET(dxViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if ((*outputMask)[AXIS_ONE]) {
            auto dgamma = std::get<1>(GroupNormGradOut);
            CHECK_RET(dgamma != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dgammaReshape = l0op::Reshape(dgamma, gradGammaOut->GetViewShape(), uniqueExecutor.get());
            auto dgammaViewCopyResult = l0op::ViewCopy(dgammaReshape, gradGammaOut, uniqueExecutor.get());
            CHECK_RET(dgammaViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if ((*outputMask)[AXIS_TWO]) {
            auto dbeta = std::get<2>(GroupNormGradOut);
            CHECK_RET(dbeta != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dbetaReshape = l0op::Reshape(dbeta, gradGammaOut->GetViewShape(), uniqueExecutor.get());
            auto dbetaViewCopyResult = l0op::ViewCopy(dbetaReshape, gradBetaOut, uniqueExecutor.get());
            CHECK_RET(dbetaViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        // 计算并将结果拷贝到输出gradInput
        if ((*outputMask)[AXIS_ZERO]) {
            ret = GradInputProcess(gradOutReshape,
                meanContiguous,
                rstdContiguous,
                gammaContiguous,
                inputReshape,
                N,
                C,
                group,
                outputMask,
                gradInput,
                uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
        if ((*outputMask)[AXIS_ONE]) {
            // 计算并将结果拷贝到输出gradGammaOut
            ret = GetDgamma(gradOutReshape,
                inputReshape,
                meanContiguous,
                rstdContiguous,
                N,
                C,
                group,
                outputMask,
                gradGammaOut,
                uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
        if ((*outputMask)[AXIS_TWO]) {
            // 计算并将结果拷贝到输出gradBetaOut
            ret = GetDbeta(gradOutReshape, outputMask, gradBetaOut, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupNormBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupNormBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
