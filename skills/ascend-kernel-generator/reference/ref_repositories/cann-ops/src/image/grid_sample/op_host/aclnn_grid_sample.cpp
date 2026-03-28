/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grid_sample.h"
#include "aclnn_kernels/contiguous.h"
#include "grid_sample_l0.h"
#include "grid_sampler2d_l0.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;

static const int64_t INTERPOLATION_MODE_MIN_VALUE = 0;
static const int64_t INTERPOLATION_MODE_MAX_VALUE = 2;
static const int64_t INTERPOLATION_MODE_BILINEAR_VALUE = 0;
static const int64_t INTERPOLATION_MODE_NEAREST_VALUE = 1;
static const int64_t INTERPOLATION_MODE_BICUBIC_VALUE = 2;
static const int64_t PADDING_MODE_MIN_VALUE = 0;
static const int64_t PADDING_MODE_MAX_VALUE = 2;
static const int64_t SPATIAL_GRID_LAST_DIM_SIZE = 2;
static const int64_t SPATIAL_DIM_NUM = 4;
static const int64_t AICORE_MAX_SIZE_310P = 20480;
static const int64_t SUPPORT_CHANNEL_310P = 32;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static bool CheckNotNull(const aclTensor *input, const aclTensor *grid, const aclTensor *out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(grid, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *input, const aclTensor *grid, const aclTensor *out)
{
    // 检查input、grid、out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(grid, input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, input->GetDataType(), return false);

    // 检查input的数据类型是否在gridsampler2d算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckAttrValid(int64_t interpolationMode, int64_t paddingMode)
{
    // 检查interpolationMode 、paddingMode是否在支持范围内
    if (interpolationMode < INTERPOLATION_MODE_MIN_VALUE || interpolationMode > INTERPOLATION_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "interpolationMode %ld should be in support list {0(bilinear), 1(nearest), 2(bicubic)}.",
            interpolationMode);
        return false;
    }

    if (paddingMode < PADDING_MODE_MIN_VALUE || paddingMode > PADDING_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "paddingMode %ld should be in support list {0(zeros), 1(border), 2(reflection)}.",
            paddingMode);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *input, const aclTensor *grid, const aclTensor *out)
{
    const auto &inputShape = input->GetViewShape();
    const auto &gridShape = grid->GetViewShape();
    const auto &outShape = out->GetViewShape();

    OP_CHECK_WRONG_DIMENSION(input, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(grid, SPATIAL_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(out, SPATIAL_DIM_NUM, return false);

    if (inputShape.GetDim(FIRST_DIM) != gridShape.GetDim(FIRST_DIM) ||
        inputShape.GetDim(FIRST_DIM) != outShape.GetDim(FIRST_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect input, grid and out to have same batch size, but got input with shape [%s] \
            grid with shape [%s] and out with shape [%s]",
            op::ToString(inputShape).GetString(),
            op::ToString(gridShape).GetString(),
            op::ToString(outShape).GetString());
        return false;
    }
    if (inputShape.GetDim(SECOND_DIM) != outShape.GetDim(SECOND_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect input and out to have same channel size, but got input with shape [%s] \
            and out with shape [%s]",
            op::ToString(inputShape).GetString(),
            op::ToString(outShape).GetString());
        return false;
    }
    if (gridShape.GetDim(SECOND_DIM) != outShape.GetDim(THIRD_DIM) ||
        gridShape.GetDim(THIRD_DIM) != outShape.GetDim(FOURTH_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expect grid and out to have same H and W size, but got grid with shape [%s] \
            and out with shape [%s]",
            op::ToString(gridShape).GetString(),
            op::ToString(outShape).GetString());
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
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *input, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, grid, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, grid, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性参数是否在支持范围内
    CHECK_RET(CheckAttrValid(interpolationMode, paddingMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入、输出的shape匹配关系
    CHECK_RET(CheckShape(input, grid, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckAiCpuSupport(int64_t interpolationMode)
{
    if (interpolationMode == INTERPOLATION_MODE_BICUBIC_VALUE) {
        OP_LOGD("interpolation mode bicubic is not support in AICPU.");
        return false;
    }
    return true;
}

static bool CheckAiCoreSuppport(const aclTensor *input, int64_t interpolationMode, int64_t paddingMode)
{
    const auto &inputShape = input->GetViewShape();
    if (input->GetDataType() != op::DataType::DT_FLOAT && input->GetDataType() != op::DataType::DT_FLOAT16) {
        OP_LOGD("Only support float16 or float32 on AICore, but got data type is %s",
            op::ToString(input->GetDataType()).GetString());
        return false;
    }
    bool is910bSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                             GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    if (is910bSocVersion) {
        return true;
    }

    bool is310PSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
            input->GetDataType() == op::DataType::DT_FLOAT && interpolationMode == INTERPOLATION_MODE_BILINEAR_VALUE &&
            inputShape.GetDim(SECOND_DIM) == SUPPORT_CHANNEL_310P && paddingMode == PADDING_MODE_MIN_VALUE);

    bool is310BSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310B &&
            input->GetDataType() == op::DataType::DT_FLOAT16 &&
            interpolationMode == INTERPOLATION_MODE_BILINEAR_VALUE &&
            inputShape.GetDim(SECOND_DIM) == SUPPORT_CHANNEL_310P && paddingMode == PADDING_MODE_MIN_VALUE);
    if (is310PSocVersion || is310BSocVersion) {
        return true;
    }
    return false;
}

aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor *input, const aclTensor *grid, int64_t interpolationMode,
    int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGridSampler2D, DFX_IN(input, grid, interpolationMode, paddingMode, alignCorners), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, grid, interpolationMode, paddingMode, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // gridsampler2d算子的空tensor在kernel中支持
    if (input->IsEmpty() || grid->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入grid转换成连续的tensor
    auto gridContiguous = l0op::Contiguous(grid, uniqueExecutor.get());
    CHECK_RET(gridContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行计算
    bool supportAiCore = CheckAiCoreSuppport(input, interpolationMode, paddingMode);
    bool supportAicpu = CheckAiCpuSupport(interpolationMode);

    const aclTensor *gridSampler2DOut = nullptr;
    if (supportAiCore) {
        // transpose NCHW to NHWC
        int64_t schedulerMode = 1;
        int64_t perm[4] = {0, 2, 3, 1};
        bool channelLast = true;
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm, 4);
        inputContiguous = l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get());
        OP_LOGD("Lanuch GridSample in AICore. Attrs: [%ld], [%ld], [%d], [%d], [%ld]",
            interpolationMode,
            paddingMode,
            alignCorners,
            channelLast,
            schedulerMode);
        gridSampler2DOut = l0op::GridSample(inputContiguous,
            gridContiguous,
            interpolationMode,
            paddingMode,
            alignCorners,
            channelLast,
            schedulerMode,
            uniqueExecutor.get());
    } else if (supportAicpu) {
        OP_LOGD(
            "Lanuch GridSampler2D in AICPU. Attrs: [%ld], [%ld], [%d]", interpolationMode, paddingMode, alignCorners);
        gridSampler2DOut = l0op::GridSampler2D(
            inputContiguous, gridContiguous, interpolationMode, paddingMode, alignCorners, uniqueExecutor.get());
    } else {
        std::string alignCornerStr = alignCorners ? "true" : "false";
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The op info is not supported. Plsease check op info! DataType support list is %s, got data type is %s. \
            interpolationMode support 0(bilinear) , 1(nearest) or 2(bicubic), got interpolationMode is %ld. \
            paddingMode support 0(zeros) , 1(border) or 2(reflection), got paddingMode is %ld. \
            alignCorners support false and true, got alignCorners is %s. \
            Notice that when data type is double, no support interpolation mode is bicubic.",
            op::ToString(DTYPE_SUPPORT_LIST).GetString(),
            op::ToString(input->GetDataType()).GetString(),
            interpolationMode,
            paddingMode,
            alignCornerStr.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(gridSampler2DOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(gridSampler2DOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGridSampler2D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGridSampler2D);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif