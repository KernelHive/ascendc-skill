/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_bilinear2d_aa_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_upsample_bilinear2d_aa.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t EXPECT_SIZE = 2;
static constexpr double MAX_SUPPORT_SCALE = 50.0;

static bool CheckNotNull(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out, const aclIntArray *outputSize)
{
    const op::Format selfFormat = self->GetStorageFormat();
    if (selfFormat != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format of input and output should be equal, self [%s], out [%s].",
            op::ToString(selfFormat).GetString(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 2, but got size %zu", outputSizeNum),
        return false);
    OP_CHECK(selfFormat == op::Format::FORMAT_ND || selfFormat == op::Format::FORMAT_NCHW,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input storage format only support NCHW, but got %s.",
            op::ToString(selfFormat).GetString()),
        return false);
    return true;
}

static bool CheckScalesValid(const double weight, const double high)
{
    if ((weight < 0) || (high < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "scales_w and scales_h cannot be negative , scales_w [%f], scales_h [%f].",
            weight,
            high);
        return false;
    }
    return true;
}

static bool CheckInputElement(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    auto selfShape = self->GetViewShape();
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];

    outN = selfShape.GetDim(DIM_ZERO);
    outC = selfShape.GetDim(DIM_ONE);
    inputH = selfShape.GetDim(DIM_TWO);
    inputW = selfShape.GetDim(DIM_THREE);

    OP_CHECK(outN > 0 && inputH > 0 && inputW > 0 && outC > 0 && outH > 0 && outW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (N: %ld, C: %ld,"
            " H: %ld, W: %ld) output (H: %ld, W: %ld)",
            outN,
            outC,
            inputH,
            inputW,
            outH,
            outW),
        return false);
    return true;
}

static bool CheckMaxScaleSupport(
    const aclTensor *input, const aclIntArray *outputSize, const double scalesH, const double scalesW)
{
    auto selfShape = input->GetViewShape();
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : static_cast<float>(inputH / outputH);
    const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : static_cast<float>(inputW / outputW);
    if (realScalesH > MAX_SUPPORT_SCALE || realScalesW > MAX_SUPPORT_SCALE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Scales should not exceed 50, but got scale (scales_w: %f, scales_h: %f).",
            realScalesW,
            realScalesH);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *input, const aclIntArray *outputSize, double scalesH, double scalesW, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(input, out, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 4.检查input和out N/C轴的大小是否一致
    if (input->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        CHECK_RET(CheckNCDimValid(input, out), ACLNN_ERR_PARAM_INVALID);
    }

    // 6.检验scales_w/scales_h，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(input, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckMaxScaleSupport(input, outputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor *input, const aclIntArray *outputSize,
    bool alignCorners, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleBilinear2dAA, DFX_IN(input, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, outputSize, scalesH, scalesW, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入out转换成连续的tensor
    auto out_contiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(out_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dtype = input->GetDataType();
    // 将fp16/bf16类型转为fp32处理，保证精度
    if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    }
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
    const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
    const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

    // 调用算子计算
    const aclTensor *upsampleOut = l0op::UpsampleBilinear2dAA(
        selfContiguous, outputSize, out, alignCorners, realScalesH, realScalesW, uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dtype == op::DataType::DT_BF16) {
        upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_BF16, uniqueExecutor.get());
    } else if (dtype == op::DataType::DT_FLOAT16) {
        upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
    }
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dAA(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2dAA);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif