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
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "upsample_nearest_l0.h"
#include "aclnn_upsample_nearest_exact2d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 2;

static bool CheckNotNull(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 检查self的数据类型是否在UpsampleNearestExact2d算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    // 检查self的数据类型是否与out一致
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 2, but got size %zu", outputSizeNum),
        return false);

    return true;
}

static bool CheckInputElement(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    auto selfShape = self->GetViewShape();
    auto outShape = out->GetViewShape();
    size_t dimNum = selfShape.GetDimNum();
    int64_t inputN = selfShape.GetDim(DIM_ZERO);
    int64_t inputC = selfShape.GetDim(DIM_ONE);
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputN = outShape.GetDim(DIM_ZERO);
    int64_t outputC = outShape.GetDim(DIM_ONE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    if (self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputC = selfShape.GetDim(DIM_THREE);
        outputC = outShape.GetDim(DIM_THREE);
        inputH = selfShape.GetDim(DIM_ONE);
        inputW = selfShape.GetDim(DIM_TWO);
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outputH > 0 && outputW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input ("
            "H: %ld, W: %ld) output (H: %ld, W: %ld)",
            inputH,
            inputW,
            outputH,
            outputW),
        return false);
    if ((inputN != outputN) || (inputC != outputC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "inputN[%ld]/outputN[%ld] or inputC[%ld]/outputC[%ld] not equal .",
            inputN,
            outputN,
            inputC,
            outputC);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *upsampleNearestExact2dCompute(const aclTensor *selfContiguous, const aclIntArray *outputSize,
    const aclTensor *output, const aclFloatArray *scales, aclOpExecutor *executor)
{
    float scalesH = (*scales)[DIM_ZERO];
    float scalesW = (*scales)[DIM_ONE];
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCHW ||
        selfContiguous->GetStorageFormat() == op::Format::FORMAT_ND) {
        const int64_t permuteNCHWList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_ONE};
        auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, DIM_LIMIT);
        CHECK_RET(permuteNCHWArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCHWArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto self = l0op::ReFormat(selfTranspose, op::Format::FORMAT_NHWC);
        CHECK_RET(self != nullptr, nullptr);

        auto selfUpsampleNearestExact =
            l0op::UpsampleNearestExact2d(self, outputSize, output, scalesH, scalesW, true, executor);
        CHECK_RET(selfUpsampleNearestExact != nullptr, nullptr);

        const int64_t permuteNHWCList[] = {DIM_ZERO, DIM_THREE, DIM_ONE, DIM_TWO};
        auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, DIM_LIMIT);
        CHECK_RET(permuteNHWCArray != nullptr, nullptr);

        auto outReformat = l0op::ReFormat(selfUpsampleNearestExact, op::Format::FORMAT_NCHW);
        CHECK_RET(outReformat != nullptr, nullptr);

        return l0op::Transpose(outReformat, permuteNHWCArray, executor);
    }
    // NHWC
    return l0op::UpsampleNearestExact2d(selfContiguous, outputSize, output, scalesH, scalesW, true, executor);
}

aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact2d, DFX_IN(self, outputSize, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor支持
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
    float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
    float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

    // 调用upsampleNearestExact2dCompute计算
    vector<float> scalesList{};
    scalesList.push_back(realScalesH);
    scalesList.push_back(realScalesW);
    const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto result = upsampleNearestExact2dCompute(selfContiguous, outputSize, out, scales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact2d(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact2d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
