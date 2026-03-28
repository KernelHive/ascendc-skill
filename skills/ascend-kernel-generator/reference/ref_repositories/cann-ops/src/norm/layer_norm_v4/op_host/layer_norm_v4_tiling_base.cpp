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
 * \file layer_norm_v4_tiling_base.cc
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
constexpr size_t K_INPUT_IDX_X = 0;
constexpr size_t K_INPUT_IDX_NORM_SHAPE = 1;
constexpr size_t K_INPUT_IDX_GAMMA = 2;
constexpr size_t K_INPUT_IDX_BETA = 3;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t SIZE_OF_FLOAT = 4;
constexpr uint64_t SIZE_OF_HALF = 2;
constexpr uint64_t BASE_WSP_SIZE = 32;

bool LayerNormV4TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormV4TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormV4TilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus LayerNormV4TilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const LayerNormV4CompileInfo *>(context_->GetCompileInfo());
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.isAscend310P = compileInfo->isAscend310P;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetShapeAttrsInfo()
{
    commonParams.tensorDtype = context_->GetInputDesc(K_INPUT_IDX_X)->GetDataType();
    commonParams.paramDtype = ge::DT_FLOAT;
    auto gammaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_GAMMA);
    auto betaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_BETA);
    if (gammaDesc != nullptr) {
        commonParams.paramDtype = gammaDesc->GetDataType();
    } else if (betaDesc != nullptr) {
        commonParams.paramDtype = betaDesc->GetDataType();
    }
    commonParams.gammaNullPtr = (gammaDesc == nullptr ? 1 : 0);
    commonParams.betaNullPtr = (betaDesc == nullptr ? 1 : 0);

    uint64_t normalizedShapeLen;
    const gert::Shape xShape = context_->GetInputShape(K_INPUT_IDX_X)->GetStorageShape();
    const gert::Shape normalizedShape = context_->GetInputShape(K_INPUT_IDX_NORM_SHAPE)->GetStorageShape();
    OP_CHECK(normalizedShape.GetDimNum() > 1,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "normalizedShape dim num must be 1, dim num: %u",
            static_cast<uint32_t>(normalizedShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    normalizedShapeLen = normalizedShape.IsScalar() ? 1 : normalizedShape.GetDim(0);
    OP_CHECK(static_cast<uint64_t>(xShape.GetDimNum()) < normalizedShapeLen,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "normalizedShape dim num must be less than xShape dim num, xShape dim num: %u, "
            "normalizedShape dim num: %u",
            static_cast<uint32_t>(xShape.GetDimNum()),
            static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);
    OP_CHECK(normalizedShapeLen < 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "normalizedShapeLen must be greater than 0, normalizedShapeLen: %u",
            static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);

    // fuse axis
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (i < xShape.GetDimNum() - normalizedShapeLen) {
            colSize *= xShape.GetDim(i);
        } else {
            rowSize *= xShape.GetDim(i);
        }
    }
    commonParams.colSize = colSize;
    commonParams.rowSize = rowSize;
    OP_CHECK(commonParams.colSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "colSize must be greater than 0, colSize: %u",
            static_cast<uint32_t>(commonParams.colSize)),
        return ge::GRAPH_FAILED);
    OP_CHECK(commonParams.rowSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "rowSize must be greater than 0, rowSize: %u",
            static_cast<uint32_t>(commonParams.rowSize)),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    commonParams.eps = *(attrs->GetFloat(0));
    commonParams.coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    uint64_t alignment = 16;
    if (commonParams.tensorDtype == ge::DT_FLOAT) {
        alignment = (BLOCK_SIZE / SIZE_OF_FLOAT);
    } else {
        alignment = (BLOCK_SIZE / SIZE_OF_HALF);
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
