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
#include "upsample_nearest_l0.h"
#include "aclnn_upsample_nearest_exact1d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 3;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t EXPECT_SIZE = 1;

static bool CheckNotNull(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 检查self的数据类型是否在UpsampleNearestExact1d算子的支持列表内
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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 1, but got size %zu", outputSizeNum),
        return false);

    return true;
}

static bool CheckInputElement(const aclTensor *self, const aclIntArray *outputSize)
{
    auto selfShape = self->GetViewShape();
    size_t dimNum = selfShape.GetDimNum();
    int64_t outN = selfShape.GetDim(DIM_ZERO);
    int64_t outC = selfShape.GetDim(DIM_ONE);
    int64_t inputL = selfShape.GetDim(DIM_TWO);
    int64_t outL = (*outputSize)[DIM_ZERO];

    OP_CHECK(outN > 0 && outC > 0 && inputL > 0 && outL > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (N: %ld,"
            " C: %ld, L: %ld) output (N: %ld, C: %ld, L: %ld)",
            outN,
            outC,
            inputL,
            outN,
            outC,
            outL),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(self, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *upsampleNearestExact1dCompute(const aclTensor *selfContiguous, const aclIntArray *outputSize,
    const aclTensor *output, float scales, aclOpExecutor *executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCL ||
        selfContiguous->GetStorageFormat() == op::Format::FORMAT_ND) {
        const int64_t permuteNCLList[] = {DIM_ZERO, DIM_TWO, DIM_ONE};
        auto permuteNCLArray = executor->AllocIntArray(permuteNCLList, DIM_LIMIT);
        CHECK_RET(permuteNCLArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCLArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto selfUpsampleNearestExact =
            l0op::UpsampleNearestExact2d(selfTranspose, outputSize, output, scales, scales, true, executor);
        CHECK_RET(selfUpsampleNearestExact != nullptr, nullptr);

        const int64_t permuteNLCList[] = {DIM_ZERO, DIM_TWO, DIM_ONE};
        auto permuteNLCArray = executor->AllocIntArray(permuteNLCList, DIM_LIMIT);
        CHECK_RET(permuteNLCArray != nullptr, nullptr);

        return l0op::Transpose(selfUpsampleNearestExact, permuteNLCArray, executor);
    }
    // NLC
    return l0op::UpsampleNearestExact2d(selfContiguous, outputSize, output, scales, scales, true, executor);
}

aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact1d, DFX_IN(self, outputSize, scales), DFX_OUT(out));
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
    float realScales = scales > 0 ? static_cast<float>(1.0 / scales) : 0;

    // 调用upsampleNearestExact1dCompute计算
    auto result = upsampleNearestExact1dCompute(selfContiguous, outputSize, out, realScales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact1d(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact1d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
