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
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

#include "resize_nearest_neighbor_v2_grad_l0.h"
#include "upsample_nearest_exact2d_grad_l0.h"
#include "upsample_util.h"
#include "aclnn_upsample_nearest_2d_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static constexpr int64_t DIM_LIMIT = 4;
static constexpr int64_t EXPECT_SIZE = 2;

static bool CheckNotNull(
    const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static const std::initializer_list<DataType> &GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *gradInput)
{
    const auto &supportList = GetDtypeSupportList();
    // 检查gradOut的数据类型是否在ResizeNearestNeighborV2Grad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, supportList, return false);
    // 检查gradInput的数据类型是否与gradOut一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 2, but got size %zu", outputSizeNum),
        return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 4, but got size %zu", inputSizeNum),
        return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, const aclTensor *gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查gradOut和gradInput N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(gradInput, gradOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool isAiCoreSupport(
    const aclTensor *gradOutContiguous, const aclIntArray *inputSize, double scalesH, double scalesW)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
        return false;
    }
    if (scalesH <= 0 || scalesW <= 0) {
        return false;
    }

    const int64_t inputC = (*inputSize)[1];
    const int64_t inputH = (*inputSize)[2];
    const int64_t inputW = (*inputSize)[3];
    const int64_t gradOutH = gradOutContiguous->GetViewShape().GetDim(2);
    const int64_t gradOutW = gradOutContiguous->GetViewShape().GetDim(3);
    const int64_t MAX_C = 6;
    const int64_t MIN_H = 8;
    const int64_t MIN_W = 16;
    if (inputC > MAX_C * inputH * inputW || inputC > MAX_C * gradOutH * gradOutW) {
        return false;
    }
    if (inputH < MIN_H || gradOutH < MIN_H || inputW < MIN_W || gradOutW < MIN_W) {
        return false;
    }
    if (static_cast<int64_t>(inputH * scalesH) != gradOutH || static_cast<int64_t>(inputW * scalesW) != gradOutW) {
        return false;
    }
    return true;
}

aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, double scalesH, double scalesW, aclTensor *gradInput, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnUpsampleNearest2dBackward, DFX_IN(gradOut, outputSize, inputSize, scalesH, scalesW), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, outputSize, inputSize, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // resize_nearest_neighbor_v2_grad算子的空tensor在kernel中支持
    if (gradOut->IsEmpty() || gradInput->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOut转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *resizeNearestGradOutCast = nullptr;
    if (isAiCoreSupport(gradOutContiguous, inputSize, scalesH, scalesW)) {
        auto gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
        CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const float realScalesH = static_cast<float>(scalesH);
        const float realScalesW = static_cast<float>(scalesW);
        resizeNearestGradOutCast = l0op::UpsampleNearestExact2dGrad(
            gradOutContiguous, outputSize, inputSize, gradInput, realScalesH, realScalesW, false, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // FLOAT16转FLOAT计算
        auto gradOutCast = gradOutContiguous;
        auto gradOutDtype = gradOut->GetDataType();
        if (gradOutDtype == op::DataType::DT_FLOAT16) {
            gradOutCast = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 将输入gradOut格式转换成NC1HWC0
        auto gradOutTransdata =
            l0op::TransDataSpecial(gradOutCast, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
        CHECK_RET(gradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用ResizeNearestNeighborV2Grad算子kernel, inputSize对应[N, C, H, W]或者[N, H, W, C]
        bool alignCorners = false;
        bool halfPixelCenters = false;
        auto resizeNearestGradOut = l0op::ResizeNearestNeighborV2Grad5Hd(
            gradOutTransdata, inputSize, alignCorners, halfPixelCenters, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto resizeNearestGradOutTransdata =
            l0op::TransData(resizeNearestGradOut, gradInput->GetStorageFormat(), 0, uniqueExecutor.get());
        CHECK_RET(resizeNearestGradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // FLOAT16转FLOAT计算后转回FLOAT
        resizeNearestGradOutCast = resizeNearestGradOutTransdata;
        if (gradOutDtype == op::DataType::DT_FLOAT16) {
            resizeNearestGradOutCast =
                l0op::Cast(resizeNearestGradOutTransdata, op::DataType::DT_FLOAT16, uniqueExecutor.get());
            CHECK_RET(resizeNearestGradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(CheckReduceOutShape(resizeNearestGradOutCast, gradInput), ACLNN_ERR_PARAM_INVALID);
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(resizeNearestGradOutCast, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest2dBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest2dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
