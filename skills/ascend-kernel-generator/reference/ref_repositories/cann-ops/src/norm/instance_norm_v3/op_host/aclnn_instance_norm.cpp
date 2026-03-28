/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_instance_norm.h"
#include "instance_norm_v3_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

constexpr int DIM_NUM = 4;
constexpr int DIM_NUM_GAMMA = 1;

static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, const aclTensor *y,
    const aclTensor *mean, const aclTensor *variance, const char *dataFormat)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(variance, return false);
    CHECK_RET(dataFormat != nullptr, false);
    return true;
}

static bool CheckParamValid(const char *dataFormat)
{
    // 检查参数 dataFormat 是否合法
    OP_LOGD("CheckParamValid: dataFormat: %s", dataFormat);
    OP_LOGD("CheckParamValid: nhwc: %d, nchw: %d",
        strncmp(dataFormat, "NHWC", strlen("NHWC")),
        strncmp(dataFormat, "NCHW", strlen("NCHW")));
    bool isValidDataFormat =
        strncmp(dataFormat, "NHWC", strlen("NHWC")) == 0 || strncmp(dataFormat, "NCHW", strlen("NCHW")) == 0;
    return isValidDataFormat;
}

static aclnnStatus CheckPlatform()
{
    CHECK_RET(GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, const aclTensor *y,
    const aclTensor *mean, const aclTensor *variance)
{
    // 检查 x 的数据类型是否在 InstanceNorm 算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gamma, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(mean, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(variance, x->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *x)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckInputAndWeightShape(
    const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, bool needTranspose)
{
    int64_t xDimNums = x->GetViewShape().GetDimNum();
    int64_t gammaDimNums = gamma->GetViewShape().GetDimNum();
    int64_t betaDimNums = beta->GetViewShape().GetDimNum();
    CHECK_RET((xDimNums == DIM_NUM) && (gammaDimNums == DIM_NUM_GAMMA) && (betaDimNums == DIM_NUM_GAMMA),
        ACLNN_ERR_PARAM_INVALID);

    int64_t gammaDimValue = gamma->GetViewShape().GetDim(IDX_0);
    int64_t xCAxisValue = (needTranspose) ? x->GetViewShape().GetDim(IDX_3) : x->GetViewShape().GetDim(IDX_1);
    OP_LOGD("aclnnInstanceNormGetWorkspaceSize xCAxisValue: %ld, gammaDimValue: %ld", xCAxisValue, gammaDimValue);
    CHECK_RET(xCAxisValue == gammaDimValue, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta,
    const char *dataFormat, const aclTensor *y, const aclTensor *mean, const aclTensor *variance)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, gamma, beta, y, mean, variance, dataFormat), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数 dataFormat 是否合法
    CHECK_RET(CheckParamValid(dataFormat), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查 x, gamma, beta, y, mean, variance 的数据类型是否合法
    CHECK_RET(CheckDtypeValid(x, gamma, beta, y, mean, variance), ACLNN_ERR_PARAM_INVALID);

    // 4. 查输入tensor的shape是否为异常
    CHECK_RET(CheckShape(x), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta,
    const char *dataFormat, double eps, aclTensor *y, aclTensor *mean, aclTensor *variance, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_LOGD("Enter aclnnInstanceNormGetWorkspaceSize");

    L2_DFX_PHASE_1(aclnnInstanceNorm, DFX_IN(x, gamma, beta, dataFormat, eps), DFX_OUT(y, mean, variance));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(x, gamma, beta, dataFormat, y, mean, variance);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 当前仅支持310P
    ret = CheckPlatform();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool needTranspose = (strncmp(dataFormat, "NHWC", strlen("NHWC")) == 0);
    OP_LOGD("aclnnInstanceNormGetWorkspaceSize needTranspose: %d", needTranspose);
    ret = CheckInputAndWeightShape(x, gamma, beta, needTranspose);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool hasEmptyTensor =
        x->IsEmpty() || gamma->IsEmpty() || beta->IsEmpty() || y->IsEmpty() || mean->IsEmpty() || variance->IsEmpty();
    if (hasEmptyTensor) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    auto gammaContiguous = l0op::Contiguous(gamma, uniqueExecutor.get());
    auto betaContiguous = l0op::Contiguous(beta, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclTensor *yComputeOut = nullptr;
    aclTensor *meanOut = nullptr;
    aclTensor *varianceOut = nullptr;

    if (needTranspose) {
        std::vector<int64_t> permNHWC2NCHW = {IDX_0, IDX_3, IDX_1, IDX_2};
        std::vector<int64_t> permNCHW2NHWC = {IDX_0, IDX_2, IDX_3, IDX_1};
        aclIntArray *axesNHWC2NCHW = uniqueExecutor.get()->AllocIntArray(permNHWC2NCHW.data(), DIM_NUM);
        aclIntArray *axesNCHW2NHWC = uniqueExecutor.get()->AllocIntArray(permNCHW2NHWC.data(), DIM_NUM);
        CHECK_RET((axesNCHW2NHWC != nullptr && axesNHWC2NCHW != nullptr), ACLNN_ERR_INNER_NULLPTR);

        // x 做 transpose 到 NCHW
        auto xTranspose = l0op::Transpose(xContiguous, axesNHWC2NCHW, uniqueExecutor.get());
        CHECK_RET(xTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 进行 instanceNorm 计算
        const char *realDataFormat = "NCHW";
        auto instanceNormOut = l0op::InstanceNormV3(
            xTranspose, gammaContiguous, betaContiguous, realDataFormat, eps, uniqueExecutor.get());
        yComputeOut = std::get<IDX_0>(instanceNormOut);
        meanOut = std::get<IDX_1>(instanceNormOut);
        varianceOut = std::get<IDX_2>(instanceNormOut);
        CHECK_RET(yComputeOut != nullptr && meanOut != nullptr && varianceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将结果 yComputeOut 进行 transpose，转换成正确的 NHWC
        yComputeOut = const_cast<aclTensor *>(l0op::Transpose(yComputeOut, axesNCHW2NHWC, uniqueExecutor.get()));

        int64_t nhwcReduceShapeValue[DIM_NUM] = {
            meanOut->GetViewShape().GetDim(0), 1, 1, meanOut->GetViewShape().GetDim(1)};
        aclIntArray *nhwcReduceShape = uniqueExecutor.get()->AllocIntArray(nhwcReduceShapeValue, DIM_NUM);
        CHECK_RET(nhwcReduceShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        meanOut = const_cast<aclTensor *>(l0op::Reshape(meanOut, nhwcReduceShape, uniqueExecutor.get()));
        varianceOut = const_cast<aclTensor *>(l0op::Reshape(varianceOut, nhwcReduceShape, uniqueExecutor.get()));
    } else {
        auto instanceNormOut =
            l0op::InstanceNormV3(xContiguous, gammaContiguous, betaContiguous, dataFormat, eps, uniqueExecutor.get());
        yComputeOut = std::get<IDX_0>(instanceNormOut);
        meanOut = std::get<IDX_1>(instanceNormOut);
        varianceOut = std::get<IDX_2>(instanceNormOut);
    }
    CHECK_RET(yComputeOut != nullptr && meanOut != nullptr && varianceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 yComputeOut 结果拷贝到 y 上
    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 meanOut 结果拷贝到 mean 上
    auto viewCopyMeanResult = l0op::ViewCopy(meanOut, mean, uniqueExecutor.get());
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 meanOut 结果拷贝到 mean 上
    auto viewCopyVarResult = l0op::ViewCopy(varianceOut, variance, uniqueExecutor.get());
    CHECK_RET(viewCopyVarResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnInstanceNormGetWorkspaceSize");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInstanceNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInstanceNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
