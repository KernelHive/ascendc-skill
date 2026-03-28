/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grid_sampler2d_backward.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "grid_sampler2d_grad_l0.h"
#include "aclnn_kernels/transpose.h"
#include "zero_op_l0.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/cast.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;

static const int64_t INTERPOLATION_MODE_MIN_VALUE = 0;
static const int64_t INTERPOLATION_MODE_MAX_VALUE = 1;
static const int64_t PADDING_MODE_MIN_VALUE = 0;
static const int64_t PADDING_MODE_MAX_VALUE = 2;
static const int64_t SPATIAL_GRID_LAST_DIM_SIZE = 2;
static const int64_t SPATIAL_DIM_NUM = 4;
static const int64_t GRAD_RESULT_SIZE = 2;
static const int64_t MAX_CHANNEL_SIZE = 2048;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static inline bool CheckSocVersionGe910B(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline const std::initializer_list<op::DataType> &GetDtypeSupportList()
{
    if (CheckSocVersionGe910B()) {
        return DTYPE_SUPPORT_LIST_910B;
    } else {
        return DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *grid,
    const aclTensor *inputGrad, const aclTensor *gridGrad)
{
    // 检查输入输出数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(grid, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputGrad, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gridGrad, input->GetDataType(), return false);

    // 检查input的数据类型是否在gridsampler2dgrad算子的支持列表内
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, dtypeSupportList, return false);
    return true;
}

static bool CheckAttrValid(int64_t interpolationMode, int64_t paddingMode)
{
    // 检查interpolationMode 、paddingMode是否在支持范围内
    if (interpolationMode < INTERPOLATION_MODE_MIN_VALUE || interpolationMode > INTERPOLATION_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "interpolationMode %ld should be in range [%ld, %ld].",
            interpolationMode,
            INTERPOLATION_MODE_MIN_VALUE,
            INTERPOLATION_MODE_MAX_VALUE);
        return false;
    }

    if (paddingMode < PADDING_MODE_MIN_VALUE || paddingMode > PADDING_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "paddingMode %ld should be in range [%ld, %ld].",
            paddingMode,
            PADDING_MODE_MIN_VALUE,
            PADDING_MODE_MAX_VALUE);
        return false;
    }
    return true;
}

// 检查tuple <values, indices>里的元素是否为非null, true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<aclTensor *, aclTensor *> tensorTuple)
{
    if (std::tuple_size<decltype(tensorTuple)>::value != GRAD_RESULT_SIZE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "the length of tuple returned by GridSampler2DGrad is not %ld.", GRAD_RESULT_SIZE);
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *grid,
    const aclTensor *inputGrad, const aclTensor *gridGrad)
{
    const auto &gradOutputShape = gradOutput->GetViewShape();
    const auto &inputShape = input->GetViewShape();
    const auto &gridShape = grid->GetViewShape();
    const auto &inputGradShape = inputGrad->GetViewShape();
    const auto &gridGradShape = gridGrad->GetViewShape();
    OP_CHECK_WRONG_DIMENSION(input, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(grid, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(gradOutput, SPATIAL_DIM_NUM, return false);
    if (inputShape.GetDim(FIRST_DIM) != gridShape.GetDim(FIRST_DIM) ||
        inputShape.GetDim(FIRST_DIM) != gradOutputShape.GetDim(FIRST_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect input grid and gridOutput to have same batch size, but got input with \
            shape [%s] grid with shape [%s] and gridOutput with shape [%s]",
            op::ToString(inputShape).GetString(),
            op::ToString(gridShape).GetString(),
            op::ToString(gradOutputShape).GetString());
        return false;
    }
    if (inputShape.GetDim(SECOND_DIM) != gradOutputShape.GetDim(SECOND_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect input and gridOutput to have same channel size, but got input with shape \
            [%s] and gridOutput with shape [%s]",
            op::ToString(inputShape).GetString(),
            op::ToString(gradOutputShape).GetString());
        return false;
    }
    if (gridShape.GetDim(SECOND_DIM) != gradOutputShape.GetDim(THIRD_DIM) ||
        gridShape.GetDim(THIRD_DIM) != gradOutputShape.GetDim(FOURTH_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect grid and gridOutput to have same H and W size, but got grid with shape \
            [%s] and gridOutput with shape [%s]",
            op::ToString(gridShape).GetString(),
            op::ToString(gradOutputShape).GetString());
        return false;
    }
    if (inputShape.GetDim(THIRD_DIM) == 0 || inputShape.GetDim(FOURTH_DIM) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect input to have non-empty spatial dimensions, but got input with shape [%s]",
            op::ToString(inputShape).GetString());
        return false;
    }
    if (gridShape.GetDim(FOURTH_DIM) != SPATIAL_GRID_LAST_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect grid to have size %ld in last dimension, but got grid with shape [%s]",
            SPATIAL_GRID_LAST_DIM_SIZE,
            op::ToString(gridShape).GetString());
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(inputGrad, input, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gridGrad, grid, return false);
    return true;
}

static bool CheckDtypeAndChannelCanTranspose(const aclTensor *input, int64_t interpolationMode, int64_t paddingMode)
{
    return input->GetDataType() != op::DataType::DT_DOUBLE && (interpolationMode == 0 || interpolationMode == 1) &&
           (paddingMode == 0 || paddingMode == 1) &&
           GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           input->GetViewShape().GetDim(SECOND_DIM) <= MAX_CHANNEL_SIZE;
}

static bool CheckNotNull2In1Out(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *grid,
    const aclTensor *inputGrad, const aclTensor *gridGrad)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(grid, return false);
    OP_CHECK_NULL(inputGrad, return false);
    OP_CHECK_NULL(gridGrad, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *grid,
    int64_t interpolationMode, int64_t paddingMode, const aclTensor *inputGrad, const aclTensor *gridGrad)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2In1Out(gradOutput, input, grid, inputGrad, gridGrad), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, input, grid, inputGrad, gridGrad), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性参数是否在支持范围内
    CHECK_RET(CheckAttrValid(interpolationMode, paddingMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入、输出的shape匹配关系
    CHECK_RET(CheckShape(gradOutput, input, grid, inputGrad, gridGrad), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input,
    const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
    const aclBoolArray *outputMask, aclTensor *inputGrad, aclTensor *gridGrad, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnGridSampler2DBackward,
        DFX_IN(gradOutput, input, grid, interpolationMode, paddingMode, alignCorners, outputMask),
        DFX_OUT(inputGrad, gridGrad));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输出掩码校验
    if (outputMask == nullptr || ((*outputMask)[0] == false && (*outputMask)[1] == false)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, input, grid, interpolationMode, paddingMode, inputGrad, gridGrad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // gridsampler2d算子的空tensor在kernel中支持
    if (gradOutput->IsEmpty() || input->IsEmpty() || grid->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入grid转换成连续的tensor
    auto gridContiguous = l0op::Contiguous(grid, uniqueExecutor.get());
    CHECK_RET(gridContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t dimSize = static_cast<int64_t>(inputContiguous->GetViewShape().GetDimNum());
    bool transposeFlag = CheckDtypeAndChannelCanTranspose(input, interpolationMode, paddingMode);
    bool castFlag = input->GetDataType() == op::DataType::DT_FLOAT16 &&
                    GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910;
    std::tuple<aclTensor *, aclTensor *> gridSampler2DBackwardOut;
    if (transposeFlag) {
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), dimSize);
        aclTensor *inputTranspose =
            const_cast<aclTensor *>(l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get()));
        CHECK_RET(inputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclTensor *gradOutputTranspose =
            const_cast<aclTensor *>(l0op::Transpose(gradOutputContiguous, valuePerm, uniqueExecutor.get()));
        CHECK_RET(gradOutputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutputTranspose->SetStorageFormat(op::Format::FORMAT_NHWC);
        inputTranspose->SetStorageFormat(op::Format::FORMAT_NHWC);
        // 调用AICROE算子
        gridSampler2DBackwardOut = l0op::GridSamplerGrad(gradOutputTranspose,
            inputTranspose,
            gridContiguous,
            interpolationMode,
            paddingMode,
            alignCorners,
            uniqueExecutor.get());
    } else {
        if (castFlag) {
            gradOutputContiguous = l0op::Cast(gradOutputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            inputContiguous = l0op::Cast(inputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            gridContiguous = l0op::Cast(gridContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gridContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // 进行计算
        gridSampler2DBackwardOut = l0op::GridSampler2DGrad(gradOutputContiguous,
            inputContiguous,
            gridContiguous,
            interpolationMode,
            paddingMode,
            alignCorners,
            uniqueExecutor.get());
    }

    CHECK_RET(CheckTupleNullptr(gridSampler2DBackwardOut), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出inputGrad上
    if ((*outputMask)[0]) {
        if (transposeFlag) {
            std::vector<int64_t> outPerm = {0, 3, 1, 2};
            auto outValuePerm = uniqueExecutor.get()->AllocIntArray(outPerm.data(), dimSize);
            aclTensor *transOutput0 = const_cast<aclTensor *>(
                l0op::Transpose(std::get<0>(gridSampler2DBackwardOut), outValuePerm, uniqueExecutor.get()));
            CHECK_RET(transOutput0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            transOutput0->SetStorageFormat(inputGrad->GetStorageFormat());
            auto inputGradViewCopyResult = l0op::ViewCopy(transOutput0, inputGrad, uniqueExecutor.get());
            CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            if (castFlag) {
                // 固定写法，将计算结果转换成输出out的数据类型
                auto inputGradCal = std::get<0>(gridSampler2DBackwardOut);
                CHECK_RET(inputGradCal != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto inputGradCast = l0op::Cast(inputGradCal, input->GetDataType(), uniqueExecutor.get());
                CHECK_RET(inputGradCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto inputGradViewCopyResult = l0op::ViewCopy(inputGradCast, inputGrad, uniqueExecutor.get());
                CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            } else {
                auto inputGradViewCopyResult =
                    l0op::ViewCopy(std::get<0>(gridSampler2DBackwardOut), inputGrad, uniqueExecutor.get());
                CHECK_RET(inputGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
    }
    // 固定写法，将计算结果拷贝到输出gridGrad上
    if ((*outputMask)[1]) {
        if (castFlag) {
            auto gridGradCal = std::get<1>(gridSampler2DBackwardOut);
            CHECK_RET(gridGradCal != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gridGradCast = l0op::Cast(gridGradCal, grid->GetDataType(), uniqueExecutor.get());
            CHECK_RET(gridGradCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gridGradViewCopyResult = l0op::ViewCopy(gridGradCast, gridGrad, uniqueExecutor.get());
            CHECK_RET(gridGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto gridGradViewCopyResult =
                l0op::ViewCopy(std::get<1>(gridSampler2DBackwardOut), gridGrad, uniqueExecutor.get());
            CHECK_RET(gridGradViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGridSampler2DBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGridSampler2DBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif