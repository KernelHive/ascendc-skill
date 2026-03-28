/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_layer_norm_backward.h"
#include "tensor_util.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "fill_l0.h"
#include "layer_norm_beta_gamma_backprop_v2_l0.h"
#include "layer_norm_x_backprop_v3_l0.h"
#include "layer_norm_grad_v3_l0.h"
#include "ones_like_l0.h"
#include "squeeze_l0.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t NORMALIZE_LEN = 3;
constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t GRAD_INPUT_INDEX = 0;
constexpr size_t GRAD_WEIGHT_INDEX = 1;
constexpr size_t GRAD_BIAS_INDEX = 2;
constexpr size_t LEAST_NORMALIZED_SHAPE_LEN = 1;
constexpr size_t GRAD_OUT_NUM = 3;
constexpr size_t X_OUT_NUM = 2;
constexpr size_t BETA_GAMMAX_NUM = 2;
constexpr int64_t MIN_GRAD_REDUCE_AXIS = 1024;
constexpr int64_t MAX_GRAD_REDUCE_AXIS = 4096;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor *gradOut, const aclTensor *input, const aclIntArray *normalizedShape,
    const aclTensor *mean, const aclTensor *rstd, const aclBoolArray *outputMask)
{
    // 校验输入是否为空指针
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(normalizedShape, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(rstd, return false);
    OP_CHECK_NULL(outputMask, return false);
    return true;
}

inline static bool CheckOptionalDtype(const aclTensor *tensorOptional)
{
    if (tensorOptional && !CheckType(tensorOptional->GetDataType(), DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward tensorOptional dtype [%s] to be in support list [%s] but check failed.",
            ToString(tensorOptional->GetDataType()).GetString(),
            ToString(DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    return true;
}

static bool CheckDtype(const aclTensor *gradOut, const aclTensor *input, const aclTensor *mean, const aclTensor *rstd,
    const aclTensor *weightOptional, const aclTensor *biasOptional, const aclTensor *gradInputOut,
    const aclTensor *gradWeightOut, const aclTensor *gradBiasOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(rstd, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradOut, input, return false);
    if (weightOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightOptional, DTYPE_SUPPORT_LIST, return false);
        if (weightOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(weightOptional, input, return false);
        }
    }
    if (biasOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, DTYPE_SUPPORT_LIST, return false);
    }
    if (gradInputOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradInputOut, DTYPE_SUPPORT_LIST, return false);
    }
    if (gradWeightOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradWeightOut, DTYPE_SUPPORT_LIST, return false);
    }
    if (gradBiasOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradBiasOut, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckShapeEqual(const aclTensor *input, const aclIntArray *normalizedShape)
{
    auto inputShape = input->GetViewShape();
    size_t inputLen = inputShape.GetDimNum();
    size_t normLen = normalizedShape->Size();
    size_t axis = inputLen - normLen;
    for (size_t i = axis; i < normLen; i++) {
        int64_t normDim = *(normalizedShape->GetData() + i);
        int64_t inputDim = inputShape.GetDim(i + axis);
        if (normDim != inputDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected normalized index [%zu] shape [%ld] be equal to tensor index [%zu] shape [%ld], but failed.",
                i,
                normDim,
                i + axis,
                inputDim);
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor *gradOut, const aclTensor *input, const aclIntArray *normalizedShape,
    const aclTensor *mean, const aclTensor *rstd, const aclTensor *weightOptional, const aclTensor *biasOptional,
    const aclBoolArray *outputMask, const aclTensor *gradInputOut, const aclTensor *gradWeightOut,
    const aclTensor *gradBiasOut, int64_t M, int64_t N)
{
    // 1.检查输入维度是否小于8维
    OP_CHECK_MAX_DIM(gradOut, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(mean, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(rstd, MAX_DIM_LEN, return false);
    if (normalizedShape->Size() > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward normalizedShape len [%zu] to not be greater than [%zu] but check failed.",
            normalizedShape->Size(),
            MAX_DIM_LEN);
        return false;
    }
    if ((*outputMask)[GRAD_INPUT_INDEX]) {
        OP_CHECK_MAX_DIM(gradInputOut, MAX_DIM_LEN, return false);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK_MAX_DIM(gradWeightOut, MAX_DIM_LEN, return false);
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK_MAX_DIM(gradBiasOut, MAX_DIM_LEN, return false);
    }

    // 2.检查normalizedShape的长度是否大于等于1
    if (normalizedShape->Size() < LEAST_NORMALIZED_SHAPE_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNorm normalizedShape len [%zu] to be greater than [%zu] but check failed.",
            normalizedShape->Size(),
            LEAST_NORMALIZED_SHAPE_LEN);
        return false;
    }
    // 3.检查input维度是否不小于normalizedShape的长度
    OP_CHECK_MIN_DIM(input, normalizedShape->Size(), return false);
    // 4.检查input和normalizedShape间的约束关系
    if (!CheckShapeEqual(input, normalizedShape)) {
        return false;
    }

    // 5.校验weight存在时与normalizedShape间的关系
    if (weightOptional) {
        OP_CHECK_MAX_DIM(weightOptional, MAX_DIM_LEN, return false);
        OP_CHECK_WRONG_DIMENSION(weightOptional, normalizedShape->Size(), return false);
        if (!CheckShapeEqual(weightOptional, normalizedShape)) {
            return false;
        }
    }
    // 6.校验bias存在时与normalizedShape间的关系
    if (biasOptional) {
        OP_CHECK_MAX_DIM(biasOptional, MAX_DIM_LEN, return false);
        OP_CHECK_WRONG_DIMENSION(biasOptional, normalizedShape->Size(), return false);
        if (!CheckShapeEqual(biasOptional, normalizedShape)) {
            return false;
        }
    }

    // 7.校验gradOut与input输入是否相等
    OP_CHECK_SHAPE_NOT_EQUAL(gradOut, input, return false);
    // 8.校验mean和rstd的shape是否相等
    OP_CHECK_SHAPE_NOT_EQUAL(mean, rstd, return false);
    if (mean->GetViewShape().GetShapeSize() != M) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward mean shape size %s to be equal to %ld but check failed.",
            ToString(mean->GetViewShape()).GetString(),
            M);
        return false;
    }
    // 9.校验输出shape间的关系
    if ((*outputMask)[GRAD_INPUT_INDEX]) {
        OP_CHECK_SHAPE_NOT_EQUAL(gradInputOut, input, return false);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX] && gradWeightOut->GetViewShape().GetShapeSize() != N) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward gradWeightOut shape size [%s] to be equal to [%ld] but check failed.",
            ToString(gradWeightOut->GetViewShape()).GetString(),
            N);
        return false;
    }
    if ((*outputMask)[GRAD_BIAS_INDEX] && gradBiasOut->GetViewShape().GetShapeSize() != N) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward gradBiasOut shape size [%s] to be equal to [%ld] but check failed.",
            ToString(gradBiasOut->GetViewShape()).GetString(),
            N);
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclTensor *input, const aclIntArray *normalizedShape,
    const aclTensor *mean, const aclTensor *rstd, const aclTensor *weightOptional, const aclTensor *biasOptional,
    const aclBoolArray *outputMask, aclTensor *gradInputOut, aclTensor *gradWeightOut, aclTensor *gradBiasOut,
    int64_t M, int64_t N)
{
    // 1. 检查数据类型是否在API支持的数据类型范围之内
    CHECK_RET(
        CheckDtype(gradOut, input, mean, rstd, weightOptional, biasOptional, gradInputOut, gradWeightOut, gradBiasOut),
        ACLNN_ERR_PARAM_INVALID);
    // 2. 检查入参间的shape关系
    CHECK_RET(CheckShape(gradOut, input, normalizedShape, mean, rstd, weightOptional, biasOptional, outputMask,
        gradInputOut, gradWeightOut, gradBiasOut, M, N),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GenOutWithMask(const aclTensor *tempOut, const aclTensor *output, bool mask, aclOpExecutor *executor)
{
    if (mask) {
        OP_LOGD("Entering into GenOutWithMask Func.");
        CHECK_RET(tempOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto tempCast = l0op::Cast(tempOut, output->GetDataType(), executor);
        CHECK_RET(tempCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto outputViewCopy = l0op::ViewCopy(tempCast, output, executor);
        CHECK_RET(outputViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
    const aclIntArray *normalizedShape, const aclTensor *mean, const aclTensor *rstd, const aclTensor *weightOptional,
    const aclTensor *biasOptional, const aclBoolArray *outputMask, aclTensor *gradInputOut, aclTensor *gradWeightOut,
    aclTensor *gradBiasOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLayerNormBackward,
        DFX_IN(gradOut, input, normalizedShape, mean, rstd, weightOptional, biasOptional, outputMask),
        DFX_OUT(gradInputOut, gradWeightOut, gradBiasOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 提前检查参数是否为空指针，防止input为空指针时获取input的shape报错
    CHECK_RET(CheckNotNull(gradOut, input, normalizedShape, mean, rstd, outputMask), ACLNN_ERR_PARAM_NULLPTR);
    // outputMask长度只能为3
    if (outputMask->Size() != NORMALIZE_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNormBackward outputMask len to be 3, but got [%zu].",
            normalizedShape->Size());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*outputMask)[GRAD_INPUT_INDEX]) {
        OP_CHECK_NULL(gradInputOut, return ACLNN_ERR_PARAM_NULLPTR);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK_NULL(gradWeightOut, return ACLNN_ERR_PARAM_NULLPTR);
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK_NULL(gradBiasOut, return ACLNN_ERR_PARAM_NULLPTR);
    }
    // 根据input_shape和normalizedShape的关系进行校验和后处理
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();
    int64_t normDim = normalizedShape->Size();
    int64_t beginAxis = inputDim - normDim;
    int64_t M = 1;
    int64_t N = 1;
    for (int64_t index = 0; index < beginAxis; index++) {
        M *= inputShape.GetDim(index);
    }
    for (int64_t index = beginAxis; index < inputDim; index++) {
        N *= inputShape.GetDim(index);
    }
    // 固定写法，参数检查
    auto ret = CheckParams(gradOut,
        input,
        normalizedShape,
        mean,
        rstd,
        weightOptional,
        biasOptional,
        outputMask,
        gradInputOut,
        gradWeightOut,
        gradBiasOut,
        M,
        N);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor场景处理
    if (M <= 0 || N <= 0) {
        if ((*outputMask)[GRAD_WEIGHT_INDEX] && M <= 0) {
            ret = ProcessEmptyTensorWithValue(gradWeightOut, 0, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        if ((*outputMask)[GRAD_BIAS_INDEX] && M <= 0) {
            ret = ProcessEmptyTensorWithValue(gradBiasOut, 0, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto rstdContiguous = l0op::Contiguous(rstd, uniqueExecutor.get());
    CHECK_RET(rstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 构造新的weightContiguous
    const aclTensor *weightContiguous = nullptr;
    if (weightOptional) {
        weightContiguous = l0op::Contiguous(weightOptional, uniqueExecutor.get());
    } else {
        auto weightTensor = uniqueExecutor.get()->ConvertToTensor(normalizedShape, DataType::DT_INT64);
        aclScalar *scalarOne = uniqueExecutor.get()->AllocScalar(1);
        auto oneTensor = uniqueExecutor.get()->ConvertToTensor(scalarOne, inputContiguous->GetDataType());
        weightContiguous = l0op::Fill(weightTensor, oneTensor, normalizedShape, uniqueExecutor.get());
    }
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行LayerNorm反向计算，根据平台决定使用合一算子或拆分算子
    bool gradV3Compute = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;
    if (gradV3Compute) {
        OP_LOGD("Entering into layer_norm_grad Func.");
        // LayerNormGradV3只支持fp32 rstd mean输入，如果不是fp32先转fp32
        auto rstdContiguousFp32 = l0op::Cast(rstdContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(rstdContiguousFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanContiguousFp32 = l0op::Cast(meanContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(meanContiguousFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        std::array<aclTensor *, GRAD_OUT_NUM> gradRes = l0op::LayerNormGradV3(gradOutContiguous,
            inputContiguous,
            rstdContiguousFp32,
            meanContiguousFp32,
            weightContiguous,
            uniqueExecutor.get());
        // 根据mask处理输出
        GenOutWithMask(gradRes[GRAD_INPUT_INDEX], gradInputOut, (*outputMask)[GRAD_INPUT_INDEX], uniqueExecutor.get());
        GenOutWithMask(
            gradRes[GRAD_WEIGHT_INDEX], gradWeightOut, (*outputMask)[GRAD_WEIGHT_INDEX], uniqueExecutor.get());
        GenOutWithMask(gradRes[GRAD_BIAS_INDEX], gradBiasOut, (*outputMask)[GRAD_BIAS_INDEX], uniqueExecutor.get());
    } else {
        OP_LOGD("Entering into layer_norm_backward split ops Func.");
        // 进行layer_norm反向的首个输出计算，调用LayerNormXBackpropV3算子
        const aclTensor *meanCastTemp = nullptr;
        const aclTensor *rstdCastTemp = nullptr;
        if (mean->GetDataType() != rstd->GetDataType()) {
            meanCastTemp = l0op::Cast(meanContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(meanCastTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
            rstdCastTemp = l0op::Cast(rstdContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(rstdCastTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            meanCastTemp = meanContiguous;
            rstdCastTemp = rstdContiguous;
        }
        const aclTensor *meanCast = nullptr;
        const aclTensor *rstdCast = nullptr;
        bool isMeanCastTempB16 =
            (meanCastTemp->GetDataType() == DataType::DT_FLOAT16 || meanCastTemp->GetDataType() == DataType::DT_BF16);
        if (weightContiguous->GetDataType() == DataType::DT_FLOAT && isMeanCastTempB16) {
            meanCast = l0op::Cast(meanCastTemp, DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            rstdCast = l0op::Cast(rstdCastTemp, DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(rstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            meanCast = meanCastTemp;
            rstdCast = rstdCastTemp;
        }

        std::array<aclTensor *, X_OUT_NUM> xBackpropV3Res = l0op::LayerNormXBackpropV3(
            gradOutContiguous, inputContiguous, rstdCast, meanCast, weightContiguous, uniqueExecutor.get());
        auto TempWeightRes = xBackpropV3Res[1];
        CHECK_RET(TempWeightRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用layer_norm_beta_gamma_backprop_v2算子进行计算
        std::array<aclTensor *, BETA_GAMMAX_NUM> betaGammaBackpropV2Res =
            l0op::LayerNormBetaGammaBackpropV2(gradOutContiguous, TempWeightRes, normalizedShape, uniqueExecutor.get());
        CHECK_RET(betaGammaBackpropV2Res[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(betaGammaBackpropV2Res[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用SqueezeNd将reduce的输出shape前面的1轴删除
        FVector<int64_t> dim(beginAxis);
        for (size_t index = 0; index < beginAxis; index++) {
            dim[index] = index;
        }
        aclIntArray *resArray = uniqueExecutor.get()->AllocIntArray(dim.data(), dim.size());
        // 根据BeginAxis决定是否进行squeeze操作
        auto dWeightTemp = beginAxis != 0 ? l0op::SqueezeNd(betaGammaBackpropV2Res[0], resArray, uniqueExecutor.get())
                                          : betaGammaBackpropV2Res[0];
        auto dBiasTemp = beginAxis != 0 ? l0op::SqueezeNd(betaGammaBackpropV2Res[1], resArray, uniqueExecutor.get())
                                        : betaGammaBackpropV2Res[1];

        // 根据mask处理输出
        GenOutWithMask(xBackpropV3Res[0], gradInputOut, (*outputMask)[GRAD_INPUT_INDEX], uniqueExecutor.get());
        GenOutWithMask(dWeightTemp, gradWeightOut, (*outputMask)[GRAD_WEIGHT_INDEX], uniqueExecutor.get());
        GenOutWithMask(dBiasTemp, gradBiasOut, (*outputMask)[GRAD_BIAS_INDEX], uniqueExecutor.get());
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLayerNormBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
