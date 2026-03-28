/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn/aclnn_base.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "squeeze_l0.h"
#include "unsqueeze_l0.h"
#include "upsample_nearest_exact2d_grad_l0.h"
#include "aclnn_upsample_nearest_exact1d_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 3;
static const double MIN_SUPPORT_SCALE = 0.02;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 3;
static constexpr size_t EXPECT_OUTPUTSIZE = 1;

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, out->GetDataType(), return false);
    return true;
}

static bool CheckNotNull(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclTensor *out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(
    const aclTensor *gradOutput, const aclTensor *out, const aclIntArray *outputSize, const aclIntArray *inputSize)
{
    const op::Format gradOutputFormat1D = gradOutput->GetStorageFormat();
    if (gradOutputFormat1D != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be equal, gradOutput [%s], out [%s].",
            op::ToString(gradOutputFormat1D).GetString(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 3, but got size %zu", inputSizeNum),
        return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 1, but got size %zu", outputSizeNum),
        return false);
    return true;
}

static bool CheckScalesValid(const double weight)
{
    if (weight < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales cannot be negative , scales [%f].", weight);
        return false;
    }
    return true;
}

static double ComputeNearestExact1dBackwardScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static bool CheckInputElement(
    const aclTensor *gradOutput, const aclIntArray *inputSize, const aclTensor *out, double scales)
{
    auto gradOutputShape = gradOutput->GetViewShape();
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t outL = 0;
    int64_t inputL = (*inputSize)[DIM_TWO];

    outN = gradOutputShape.GetDim(DIM_ZERO);
    outC = gradOutputShape.GetDim(DIM_ONE);
    outL = gradOutputShape.GetDim(DIM_TWO);

    double realScales = ComputeNearestExact1dBackwardScales(inputL, outL, scales);
    OP_CHECK(realScales >= MIN_SUPPORT_SCALE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales is too large, scales [%f].", realScales),
        return false);

    OP_CHECK(outN > 0 && inputL > 0 && outC > 0 && outL > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (N: %ld, C: %ld,"
            " L: %ld) output (L: %ld)",
            outN,
            outC,
            inputL,
            outL),
        return false);
    return true;
}

static bool CheckNCDimEqual(const aclTensor *self, const aclTensor *out)
{
    int64_t selfDimN = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t selfDimC = self->GetViewShape().GetDim(DIM_ONE);
    int64_t outDimN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outDimC = out->GetViewShape().GetDim(DIM_ONE);
    if ((outDimC != selfDimC) || (outDimN != selfDimN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "selfDimC[%ld]/outDimC[%ld] or selfDimN[%ld]/outDimN[%ld] not equal .",
            selfDimC,
            outDimC,
            selfDimN,
            outDimN);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize,
    double scales, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOutput, out, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4.检查gradOutput和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimEqual(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 5.检验scales_w/scales_h，与资料保持一致
    CHECK_RET(CheckScalesValid(scales), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOutput, inputSize, out, scales), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *View4dAs3d(const aclTensor *input, const aclTensor *out, aclOpExecutor *executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {2};
    aclIntArray *dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    auto reformatInput = l0op::ReFormat(squeezedInput, out->GetStorageFormat());
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor *View3dAs4d(const aclTensor *input, aclOpExecutor *executor)
{
    // NCL -> contigious -> unsqueeze(2) -> reformat -> NCHW
    // contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    const int64_t appendDim[] = {2};
    aclIntArray *dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

aclnnStatus aclnnUpsampleNearestExact1dBackwardGetWorkspaceSize(const aclTensor *gradOutput,
    const aclIntArray *outputSize, const aclIntArray *inputSize, double scales, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnUpsampleNearestExact1dBackward, DFX_IN(gradOutput, outputSize, inputSize, scales), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, outputSize, inputSize, scales, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor，3维视作4维
    auto selfContiguous = View3dAs4d(gradOutput, uniqueExecutor.get());

    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = View3dAs4d(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dtype = gradOutput->GetDataType();

    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t inputSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], 1, (*inputSize)[DIM_TWO]};
    auto inputSizeArray = uniqueExecutor.get()->AllocIntArray(inputSizeList, 4);
    CHECK_RET(inputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t outputSizeList[] = {1, (*outputSize)[DIM_ZERO]};
    auto outputSizeArray = uniqueExecutor.get()->AllocIntArray(outputSizeList, 2);
    CHECK_RET(outputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
    const float realScales_w = scales > 0 ? static_cast<float>(scales) : 0;
    const float realScales_h = static_cast<float>(1.0);
    // 调用算子计算
    const aclTensor *upsampleOut = l0op::UpsampleNearestExact2dGrad(selfContiguous,
        outputSizeArray,
        inputSizeArray,
        const_cast<aclTensor *>(outContiguous),
        realScales_h,
        realScales_w,
        true,
        uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *resizeNearestOut = nullptr;
    resizeNearestOut = l0op::TransData(upsampleOut, outContiguous->GetStorageFormat(), 0, uniqueExecutor.get());
    CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *out3d = nullptr;
    out3d = View4dAs3d(resizeNearestOut, out, uniqueExecutor.get());
    auto viewCopyResult = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact1dBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact1dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
