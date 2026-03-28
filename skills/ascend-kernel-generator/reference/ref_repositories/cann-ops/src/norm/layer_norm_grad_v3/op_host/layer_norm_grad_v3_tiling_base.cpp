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
 * \file layer_norm_grad_v3_tiling_base.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
static const size_t INPUT_IDX_ZERO = 0;
static const size_t INPUT_IDX_ONE = 1;
static const size_t INPUT_IDX_TWO = 2;
static const size_t INPUT_IDX_THREE = 3;
static const size_t INPUT_IDX_FOUR = 4;
static const size_t OUTPUT_IDX_ZERO = 0;
static const size_t OUTPUT_IDX_ONE = 1;
static const size_t OUTPUT_IDX_TWO = 2;
static const size_t BASE_WSP_SIZE = 0;

bool LayerNormGradV3TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormGradV3TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3TilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus LayerNormGradV3TilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const LayerNormGradV3CompileInfo *>(context_->GetCompileInfo());
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

bool CheckShapeSame(const gert::TilingContext *context_, const size_t leftIndex, const size_t rightIndex,
    const bool isLeftInput, const bool isRightInput)
{
    const gert::StorageShape *leftShape = nullptr;
    const gert::StorageShape *rightShape = nullptr;
    if (isLeftInput) {
        leftShape = context_->GetInputShape(leftIndex);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    } else {
        leftShape = context_->GetOutputShape(leftIndex);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, leftShape);
    }
    if (isRightInput) {
        rightShape = context_->GetInputShape(rightIndex);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    } else {
        rightShape = context_->GetOutputShape(rightIndex);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, rightShape);
    }

    // get storage shape
    gert::Shape leftShapeVal = leftShape->GetStorageShape();
    gert::Shape rightShapeVal = rightShape->GetStorageShape();

    // check the leftIndex shape and rightIndex shape are the same
    OP_TILING_CHECK((leftShapeVal != rightShapeVal),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "The shape of leftIndex(%zu) is not equal to"
            "the shape of rightIndex(%zu), please check.",
            leftIndex,
            rightIndex),
        return false);
    return true;
}

ge::graphStatus LayerNormGradV3TilingBase::GetShapeAttrsInfo()
{
    // check dy and x and pdx shape must be the same
    CheckShapeSame(context_, INPUT_IDX_ZERO, INPUT_IDX_ONE, true, true);
    CheckShapeSame(context_, INPUT_IDX_ZERO, OUTPUT_IDX_ZERO, true, false);
    // check rstd and mean shape must be the same
    CheckShapeSame(context_, INPUT_IDX_TWO, INPUT_IDX_THREE, true, true);
    // check gamma and pdbeta and pdgamma shape must be the same
    CheckShapeSame(context_, INPUT_IDX_FOUR, OUTPUT_IDX_ONE, true, false);
    CheckShapeSame(context_, INPUT_IDX_FOUR, OUTPUT_IDX_TWO, true, false);

    auto dy = context_->GetInputShape(INPUT_IDX_ZERO);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dy);
    commonParams.dyDtype = context_->GetInputDesc(INPUT_IDX_ZERO)->GetDataType();
    auto dyShape = dy->GetStorageShape();
    auto dyDimNum = dyShape.GetDimNum();

    auto gamma = context_->GetInputShape(INPUT_IDX_FOUR);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, gamma);
    commonParams.gammaDtype = context_->GetInputDesc(INPUT_IDX_FOUR)->GetDataType();
    auto gammaShape = gamma->GetStorageShape();
    auto gammaDimNum = gammaShape.GetDimNum();

    OP_TILING_CHECK((dyDimNum < gammaDimNum),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "TilingForLayerNormGradV3: dy dim num(=%zu) is less than"
            "gamma dim num(=%zu), please check.",
            dyDimNum,
            gammaDimNum),
        return ge::GRAPH_FAILED);
    // fuse dims
    uint64_t row = 1;
    uint64_t col = 1;
    for (size_t i = 0; i < dyDimNum; i++) {
        OP_TILING_CHECK((dyShape.GetDim(i) == 0),
            VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "Tiling4LayerNormGradV3: dy shape contain zero, please check."),
            return ge::GRAPH_FAILED);
        if (i < dyDimNum - gammaDimNum) {
            row *= dyShape.GetDim(i);
        } else {
            OP_TILING_CHECK((dyShape.GetDim(i) != gammaShape.GetDim(i - dyDimNum + gammaDimNum)),
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                    "TilingForLayerNormGradV3: dy dim value(=%ld) is not equal to"
                    "gamma dim value(=%ld), please check.",
                    dyShape.GetDim(i),
                    gammaShape.GetDim(i - dyDimNum + gammaDimNum)),
                return ge::GRAPH_FAILED);
            col *= dyShape.GetDim(i);
        }
    }
    commonParams.colSize = col;
    commonParams.rowSize = row;
    commonParams.colAlign =
        (commonParams.colSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
    commonParams.isDeterministicKey = context_->GetDeterministic() == 1 ? 1 : 0;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        commonParams.colAlign =
            (commonParams.colSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
        commonParams.dtypeKey = LNGDtypeKey::FLOAT_FLOAT;
    } else if (commonParams.dyDtype == ge::DataType::DT_FLOAT16) {
        if (commonParams.gammaDtype == ge::DataType::DT_FLOAT16) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT;
        }
    } else if (commonParams.dyDtype == ge::DataType::DT_BF16) {
        if (commonParams.gammaDtype == ge::DataType::DT_BF16) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_BFLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_FLOAT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
