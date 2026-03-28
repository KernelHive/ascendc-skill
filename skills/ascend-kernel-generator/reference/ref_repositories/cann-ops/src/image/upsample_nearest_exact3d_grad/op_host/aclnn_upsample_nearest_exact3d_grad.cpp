/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "upsample_nearest_exact3d_grad_l0.h"
#include "aclnn_upsample_nearest_exact3d_grad.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 5;
static const double MAX_SUPPORT_SCALE = 50;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static const int64_t EXPECT_SIZE = 3;

static bool CheckNotNull(
    const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *gradInput)
{
    // 检查gradOut的数据类型是否在ResizeNearestNeighborV2Grad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    // 检查gradInput的数据类型是否与gradOut一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
    return true;
}

static double ComputeNearestExact3dGradScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return scale;
    } else {
        return input_size != 0 ? (static_cast<double>(output_size) / input_size) : 0.0;
    }
}

static bool CheckShape(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 3, but got size %zu", outputSizeNum),
        return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 5, but got size %zu", inputSizeNum),
        return false);
    return true;
}

static bool CheckInputElement(const aclTensor *gradOut, const aclTensor *gradInput, const aclIntArray *outputSize,
    const aclIntArray *inputSize, double scalesD, double scalesH, double scalesW)
{
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputD = (*inputSize)[DIM_TWO];
    int64_t inputH = (*inputSize)[DIM_THREE];
    int64_t inputW = (*inputSize)[DIM_FOUR];
    FVector<int64_t> fullOutputSize = {batch, channels, outD, outH, outW};
    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        fullOutputSize = {batch, outD, outH, outW, channels};
    }
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();

    OP_CHECK(inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (H: %ld,"
            " W: %ld) output (H: %ld, W: %ld)",
            inputH,
            inputW,
            outH,
            outW),
        return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected grad_output to have the same shape as output;"
                " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                i,
                fullOutputSize[i],
                i,
                gradOutShape.GetDim(i));
            return false;
        }
    }

    op::Shape expectShape = op::Shape{batch, channels, inputD, inputH, inputW};
    if (gradInput->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        expectShape = op::Shape{batch, inputD, inputH, inputW, channels};
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, expectShape, return false);

    double realScalesD = ComputeNearestExact3dGradScales(inputD, outD, scalesD);
    double realScalesH = ComputeNearestExact3dGradScales(inputH, outH, scalesH);
    double realScalesW = ComputeNearestExact3dGradScales(inputW, outW, scalesW);
    OP_CHECK(realScalesD <= MAX_SUPPORT_SCALE && realScalesH <= MAX_SUPPORT_SCALE && realScalesW <= MAX_SUPPORT_SCALE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "scalesD, scalesH and scalesW are too large, scalesD [%f], scalesH [%f], scalesW [%f].",
            realScalesD,
            realScalesH,
            realScalesW),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize,
    double scalesD, double scalesH, double scalesW, const aclTensor *gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOut, gradInput, outputSize, inputSize, scalesD, scalesH, scalesW),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, double scalesD, double scalesH, double scalesW, aclTensor *gradInput,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact3dBackward,
        DFX_IN(gradOut, outputSize, inputSize, scalesD, scalesH, scalesW),
        DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, outputSize, inputSize, scalesD, scalesH, scalesW, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor支持
    if (gradOut->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOut转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vector<float> scalesList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesList.push_back(static_cast<float>(scalesD));
        scalesList.push_back(static_cast<float>(scalesH));
        scalesList.push_back(static_cast<float>(scalesW));
        outputSize = uniqueExecutor.get()->AllocIntArray({}, 0);
    } else {
        scalesList.push_back(0.0);
        scalesList.push_back(0.0);
        scalesList.push_back(0.0);
    }
    const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutTranspose = gradOutContiguous;
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {DIM_ZERO, DIM_FOUR, DIM_ONE, DIM_TWO, DIM_THREE};
        auto permuteNCDHWArray = uniqueExecutor->AllocIntArray(permuteNCDHWList, DIM_LIMIT);
        CHECK_RET(permuteNCDHWArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutTranspose = l0op::Transpose(gradOutContiguous, permuteNCDHWArray, uniqueExecutor.get());
        CHECK_RET(gradOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用UpsampleNearestExact3dGradNcdhw算子kernel, inputSize对应[N, C, D, H, W]
    auto result =
        l0op::UpsampleNearestExact3dGradNcdhw(gradOutTranspose, outputSize, inputSize, scales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resultTranspose = result;
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNDHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ONE};
        auto permuteNDHWCArray = uniqueExecutor->AllocIntArray(permuteNDHWCList, DIM_LIMIT);
        CHECK_RET(permuteNDHWCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        resultTranspose = l0op::Transpose(result, permuteNDHWCArray, uniqueExecutor.get());
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(resultTranspose, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact3dBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact3dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
