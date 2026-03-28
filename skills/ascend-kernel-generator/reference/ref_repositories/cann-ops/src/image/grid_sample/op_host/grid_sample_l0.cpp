/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_sample_l0.cpp
 * \brief
 */

#include "grid_sample_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GridSample);  // AscendC
static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;
static const size_t FIFTH_DIM = 4;
static const string INTERPOLATION_BILINEAR = "bilinear";
static const string INTERPOLATION_NEAREST = "nearest";
static const string INTERPOLATION_BICUBIC = "bicubic";

static const string PADDING_ZEROS = "zeros";
static const string PADDING_BORDER = "border";
static const string PADDING_REFLECTION = "reflection";

static const int64_t SPATIAL_DIM_NUM = 4;
static const int64_t INTERPOLATION_MODE_MIN_VALUE = 0;
static const int64_t INTERPOLATION_MODE_MAX_VALUE = 2;
static const int64_t INTERPOLATION_MODE_BILINEAR_VALUE = 0;
static const int64_t INTERPOLATION_MODE_NEAREST_VALUE = 1;
static const int64_t INTERPOLATION_MODE_BICUBIC_VALUE = 2;
static const int64_t PADDING_MODE_MIN_VALUE = 0;
static const int64_t PADDING_MODE_MAX_VALUE = 2;
static const int64_t SCHEDULER_MODE_MIN_VALUE = 0;
static const int64_t SCHEDULER_MODE_MAX_VALUE = 1;

static bool CheckAttrValid(int64_t interpolationMode, int64_t paddingMode, int64_t schedulerMode)
{
    // 检查interpolationMode 、paddingMode 、 schedulerMode 是否在支持范围内
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

    if (schedulerMode < SCHEDULER_MODE_MIN_VALUE || schedulerMode > SCHEDULER_MODE_MAX_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "schedulerMode %ld should be in range [%ld, %ld].",
            schedulerMode,
            SCHEDULER_MODE_MIN_VALUE,
            SCHEDULER_MODE_MAX_VALUE);
        return false;
    }
    return true;
}

inline const string &GetInterpolationModeStr(int64_t interpolationMode)
{
    if (interpolationMode == 0) {
        return INTERPOLATION_BILINEAR;
    }
    if (interpolationMode == 1) {
        return INTERPOLATION_NEAREST;
    }
    return INTERPOLATION_BICUBIC;
}

inline const string &GetPaddingModeStr(int64_t paddingMode)
{
    if (paddingMode == 0) {
        return PADDING_ZEROS;
    }
    if (paddingMode == 1) {
        return PADDING_BORDER;
    }
    return PADDING_REFLECTION;
}

const aclTensor *GridSample(const aclTensor *x, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode,
    bool alignCorners, bool channelLast, int64_t schedulerMode, aclOpExecutor *executor)
{
    L0_DFX(GridSample, x, grid, interpolationMode, paddingMode, alignCorners, channelLast, schedulerMode);
    op::Shape yShape;
    if (x == nullptr || x->GetViewShape().GetDimNum() != SPATIAL_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "input tensor x is nullptr or its dimension is not equal to %ld.",
            SPATIAL_DIM_NUM);
        return nullptr;
    }
    yShape.AppendDim(x->GetViewShape().GetDim(FIRST_DIM));
    // 'C' dim of output shape
    if (channelLast) {
        yShape.AppendDim(x->GetViewShape().GetDim(FOURTH_DIM));
    } else {
        yShape.AppendDim(x->GetViewShape().GetDim(SECOND_DIM));
    }
    yShape.AppendDim(grid->GetViewShape().GetDim(SECOND_DIM));
    yShape.AppendDim(grid->GetViewShape().GetDim(THIRD_DIM));

    auto y = executor->AllocTensor(yShape, x->GetDataType(), op::Format::FORMAT_ND);
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc y tensor failed.");
        return nullptr;
    }

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将GridSample算子加入任务队列
    if (CheckAttrValid(interpolationMode, paddingMode, schedulerMode)) {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GridSample,
            OP_ATTR_NAMES({"interpolation_mode", "padding_mode", "align_corners", "channel_last", "scheduler_mode"}),
            OP_INPUT(x, grid),
            OP_OUTPUT(y),
            OP_ATTR(GetInterpolationModeStr(interpolationMode),
                GetPaddingModeStr(paddingMode),
                alignCorners,
                channelLast,
                schedulerMode));
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to launch GridSample kernel");
            return nullptr;
        }
    } else {
        return nullptr;
    }
    return y;
}

const aclTensor *GridSample3D(const aclTensor *x, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode,
    bool alignCorners, bool channelLast, aclOpExecutor *executor)
{
    L0_DFX(GridSample3D, x, grid, interpolationMode, paddingMode, alignCorners, channelLast);
    op::Shape yShape;
    yShape.AppendDim(x->GetViewShape().GetDim(FIRST_DIM));
    // 'C' dim of output shape
    if (channelLast) {
        yShape.AppendDim(x->GetViewShape().GetDim(FIFTH_DIM));
    } else {
        yShape.AppendDim(x->GetViewShape().GetDim(SECOND_DIM));
    }
    yShape.AppendDim(grid->GetViewShape().GetDim(SECOND_DIM));
    yShape.AppendDim(grid->GetViewShape().GetDim(THIRD_DIM));
    yShape.AppendDim(grid->GetViewShape().GetDim(FOURTH_DIM));

    auto y = executor->AllocTensor(yShape, x->GetDataType(), op::Format::FORMAT_ND);
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc y tensor failed.");
        return nullptr;
    }

    const op::Format inputFormat = x->GetStorageFormat();
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GridSample,
        OP_ATTR_NAMES({"interpolation_mode", "padding_mode", "align_corners", "channel_last"}),
        OP_INPUT(x, grid),
        OP_OUTPUT(y),
        OP_ATTR(
            GetInterpolationModeStr(interpolationMode), GetPaddingModeStr(paddingMode), alignCorners, channelLast, 0));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GridSample AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}
}  // namespace l0op