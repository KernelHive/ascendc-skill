/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_nearest_3d_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_upsample_nearest_3d.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == op::SocVersion::ASCEND310P) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, UPSAMPLE_DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == UPSAMPLE_EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 3, but got size %zu", outputSizeNum),
        return false);

    // out的shape应该等于推导后的shape
    op::Shape selfShape = self->GetViewShape();
    op::Shape expectShape;
    if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        expectShape.AppendDim(selfShape.GetDim(DIM_ZERO));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
        expectShape.AppendDim((*outputSize)[DIM_TWO]);
        expectShape.AppendDim(selfShape.GetDim(DIM_FOUR));
    } else {
        expectShape.AppendDim(selfShape.GetDim(DIM_ZERO));
        expectShape.AppendDim(selfShape.GetDim(DIM_ONE));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
        expectShape.AppendDim((*outputSize)[DIM_TWO]);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

static bool CheckInputElement(const aclTensor *self, const aclIntArray *outputSize)
{
    auto selfShape = self->GetViewShape();
    size_t dimNum = selfShape.GetDimNum();
    int64_t outN = selfShape.GetDim(DIM_ZERO);
    int64_t outC = selfShape.GetDim(DIM_ONE);
    int64_t inputD = selfShape.GetDim(DIM_TWO);
    int64_t inputH = selfShape.GetDim(DIM_THREE);
    int64_t inputW = selfShape.GetDim(DIM_FOUR);
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    if (self->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        outC = selfShape.GetDim(DIM_FOUR);
        inputD = selfShape.GetDim(DIM_ONE);
        inputH = selfShape.GetDim(DIM_TWO);
        inputW = selfShape.GetDim(DIM_THREE);
    }

    OP_CHECK(inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should greater than 0, bug got input (D: %ld,"
            " H: %ld, W: %ld) output (D: %ld, H: %ld, W: %ld)",
            inputD,
            inputH,
            inputW,
            outD,
            outH,
            outW),
        return false);

    OP_CHECK(outC > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Non-empty 5D data tensor expected but got a tensor with sizes %s.",
            op::ToString(self->GetViewShape()).GetString()),
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

const aclTensor *upsampleNearest3dCompute(const aclTensor *selfContiguous, const aclIntArray *outputSize,
    const aclFloatArray *scales, const aclFloatArray *castScales, aclOpExecutor *executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {DIM_ZERO, DIM_FOUR, DIM_ONE, DIM_TWO, DIM_THREE};
        auto permuteNCDHWArray = executor->AllocIntArray(permuteNCDHWList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNCDHWArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCDHWArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto selfUpsampleNearest =
            l0op::UpsampleNearest3dNcdhw(selfTranspose, outputSize, scales, castScales, executor);
        CHECK_RET(selfUpsampleNearest != nullptr, nullptr);

        const int64_t permuteNDHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ONE};
        auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWCList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNDHWCArray != nullptr, nullptr);

        return l0op::Transpose(selfUpsampleNearest, permuteNDHWCArray, executor);
    }
    // NCDHW
    return l0op::UpsampleNearest3dNcdhw(selfContiguous, outputSize, scales, castScales, executor);
}

aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, double scalesD,
    double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearest3d, DFX_IN(self, outputSize, scalesD, scalesH, scalesW), DFX_OUT(out));
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

    // 调用upsampleNearest3dCompute计算
    const float scalesList[] = {float(scalesD), float(scalesH), float(scalesW)};
    const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList, UPSAMPLE_EXPECT_SIZE);
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vector<float> scalesCastList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesCastList.push_back(static_cast<float>(1.0 / scalesD));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesH));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesW));
    }
    const aclFloatArray *castScales = uniqueExecutor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = upsampleNearest3dCompute(selfContiguous, outputSize, scales, castScales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest3d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
