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
 * \file batch_norm_v3_tiling_base.cpp
 * \brief
 */

#include "batch_norm_v3_tiling.h"

static constexpr int64_t NCHW_DIM_NUM = 4;
static constexpr int64_t NCDHW_DIM_NUM = 5;
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t WEIGHT_INPUT_IDX = 1;
static constexpr int64_t BIAS_INPUT_IDX = 2;
static constexpr int64_t MEAN_INPUT_IDX = 3;
static constexpr int64_t VAR_INPUT_IDX = 4;
static constexpr int64_t EPS_ATTR_IDX = 0;
static constexpr int64_t MOMENTUM_ATTR_IDX = 1;
static constexpr int64_t IS_TRAINING_ATTR_IDX = 2;
static constexpr int64_t DIM_0 = 0;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;
static constexpr int64_t DIM_4 = 4;
static constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
static constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
static constexpr int64_t B32_BLOCK_ALIGN_NUM = 8;

namespace optiling {
static inline bool IsDtypeSupported(const ge::DataType dtype)
{
    return ((dtype == ge::DT_FLOAT16) || (dtype == ge::DT_BF16) || (dtype == ge::DT_FLOAT));
}

bool BatchNormV3TilingBase::CheckInputDtype()
{
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, context_->GetInputDesc(X_INPUT_IDX), false);
    auto xDtype = context_->GetInputDesc(X_INPUT_IDX)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, context_->GetInputDesc(WEIGHT_INPUT_IDX), false);
    auto weightDtype = context_->GetInputDesc(WEIGHT_INPUT_IDX)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, context_->GetInputDesc(BIAS_INPUT_IDX), false);
    auto biasDtype = context_->GetInputDesc(BIAS_INPUT_IDX)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, context_->GetInputDesc(MEAN_INPUT_IDX), false);
    auto meanDtype = context_->GetInputDesc(MEAN_INPUT_IDX)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, context_->GetInputDesc(VAR_INPUT_IDX), false);
    auto varDtype = context_->GetInputDesc(VAR_INPUT_IDX)->GetDataType();
    OP_TILING_CHECK(!IsDtypeSupported(xDtype),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x dtype must in [DT_FLOAT, DT_FLOAT16, DT_BF16]."),
        return false);
    OP_TILING_CHECK((!IsDtypeSupported(weightDtype)),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "weight dtype must in [DT_FLOAT, DT_FLOAT16, DT_BF16]."),
        return false);
    OP_TILING_CHECK((xDtype != weightDtype) && (weightDtype != ge::DT_FLOAT),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "when weight dtype not same as x dtype, weight dtype must be DT_FLOAT."),
        return false);
    OP_TILING_CHECK((weightDtype != biasDtype),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "bias dtype must be same as weight dtype."),
        return false);
    OP_TILING_CHECK((meanDtype != ge::DT_FLOAT),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "running_mean dtype should be DT_FLOAT."),
        return false);
    OP_TILING_CHECK((varDtype != ge::DT_FLOAT),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "running_var dtype should be DT_FLOAT."),
        return false);
    commonParams.xDtype = xDtype;
    return true;
}

bool BatchNormV3TilingBase::CheckInputShape()
{
    auto xShape = context_->GetInputShape(X_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, xShape, false);
    auto xStorageShape = xShape->GetStorageShape();
    auto weightShape = context_->GetInputShape(WEIGHT_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, weightShape, false);
    auto weightStorageShape = weightShape->GetStorageShape();
    auto biasShape = context_->GetInputShape(BIAS_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, biasShape, false);
    auto biasStorageShape = biasShape->GetStorageShape();
    auto meanShape = context_->GetInputShape(MEAN_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, meanShape, false);
    auto meanStorageShape = meanShape->GetStorageShape();
    auto varShape = context_->GetInputShape(VAR_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, varShape, false);
    auto varStorageShape = varShape->GetStorageShape();
    auto xDesc = context_->GetInputDesc(X_INPUT_IDX);
    auto format = xDesc->GetFormat().GetStorageFormat();
    if (format == FORMAT_NCHW) {
        OP_TILING_CHECK(xStorageShape.GetDimNum() != NCHW_DIM_NUM,
            VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x shape dims should be 4 with NCHW format."),
            return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_TILING_CHECK(xStorageShape.GetDimNum() != NCDHW_DIM_NUM,
            VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x shape dims should be 5 with NCDHW format."),
            return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 =
            xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(
            commonParams.nodeName, "Not supported x format.");
        return false;
    }
    OP_TILING_CHECK(commonParams.patternR1 <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x shape dim 0 should be more than zero."),
        return false);
    OP_TILING_CHECK(commonParams.patternA <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x shape dim 1 should be more than zero."),
        return false);
    OP_TILING_CHECK(commonParams.patternR0 <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "x shape dim_2 * dim_3 should be more than zero."),
        return false);
    OP_TILING_CHECK(weightStorageShape.GetShapeSize() != commonParams.patternA,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "weight ShapeSize: %ld should equal x shape C dim: %ld",
            weightStorageShape.GetShapeSize(),
            commonParams.patternA),
        return false);
    OP_TILING_CHECK(biasStorageShape.GetShapeSize() != commonParams.patternA,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "bias ShapeSize: %ld should equal x shape C dim: %ld",
            biasStorageShape.GetShapeSize(),
            commonParams.patternA),
        return false);
    OP_TILING_CHECK(meanStorageShape.GetShapeSize() != commonParams.patternA,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "running_mean ShapeSize: %ld should equal x shape C dim: %ld",
            meanStorageShape.GetShapeSize(),
            commonParams.patternA),
        return false);
    OP_TILING_CHECK(varStorageShape.GetShapeSize() != commonParams.patternA,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "running_var ShapeSize: %ld should equal x shape C dim: %ld",
            varStorageShape.GetShapeSize(),
            commonParams.patternA),
        return false);
    return true;
}

ge::graphStatus BatchNormV3TilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        commonParams.coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        commonParams.ubSizePlatForm = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const BatchNormV3CompileInfo *>(context_->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr,
            VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "compile info is null"),
            return ge::GRAPH_FAILED);
        commonParams.coreNum = compileInfoPtr->coreNum;
        commonParams.ubSizePlatForm = compileInfoPtr->ubSize;
    }
    OP_TILING_CHECK(commonParams.coreNum == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "blockDim should not be equal to zero."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commonParams.ubSizePlatForm == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "ubSize should not be equal to zero."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3TilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGD("BatchNormV3", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    commonParams.nodeName = context_->GetNodeName();
    // 获取attr
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float *epsilon = attrs->GetFloat(EPS_ATTR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, epsilon);
    commonParams.epsilon = *epsilon;
    const float *momentum = attrs->GetFloat(MOMENTUM_ATTR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, momentum);
    const bool *isTraining = attrs->GetBool(IS_TRAINING_ATTR_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, isTraining);
    commonParams.momentum = *momentum;
    commonParams.momentumReverse = 1 - *momentum;
    OP_TILING_CHECK(!*isTraining,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "Attr is_training false is not supported."),
        return ge::GRAPH_FAILED);
    // check输入dtype
    OP_TILING_CHECK(!BatchNormV3TilingBase::CheckInputDtype(),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "CheckInputDtype failed."),
        return ge::GRAPH_FAILED);
    // check输入shape
    OP_TILING_CHECK(!BatchNormV3TilingBase::CheckInputShape(),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "CheckInputShape failed."),
        return ge::GRAPH_FAILED);
    commonParams.patternR0Align = (commonParams.xDtype == ge::DT_FLOAT)
                                      ? CeilAlign(commonParams.patternR0, B32_BLOCK_ALIGN_NUM)
                                      : CeilAlign(commonParams.patternR0, B16_BLOCK_ALIGN_NUM);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3TilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
