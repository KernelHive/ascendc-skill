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
 * \file layer_norm_v4_single_read_tiling.cc
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t NUM_EIGHT = 8;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t N_ROW_LIMIT = 32;

bool LayerNormV4SingleReadTiling::IsCapable()
{
    uint32_t rowMax = 0;
    rowMax = ((commonParams.ubSizePlatForm - NUM_EIGHT * BLOCK_SIZE) / FLOAT_SIZE) / NUM_TWO;
    if (commonParams.rowAlign > rowMax) {
        OPS_LOG_I(context_->GetNodeName(),
            "LayerNormV4SingleRead Template only support rowAlign <= rowMax, rowAlign: %u, rowMax: %u",
            static_cast<uint32_t>(commonParams.rowAlign),
            static_cast<uint32_t>(rowMax));
        return false;
    }
    if (LayerNormV4SingleReadTiling::GetTilingKey() == 0) {
        OPS_LOG_I(context_->GetNodeName(),
            "LayerNormV4SingleRead Template Unsupported dtype, tensorDtype: %d, paramDtype: %d",
            commonParams.tensorDtype,
            commonParams.paramDtype);
        return false;
    }
    return true;
}

uint64_t LayerNormV4SingleReadTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (commonParams.tensorDtype == ge::DT_FLOAT && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = LayerNormV4TilingKey::LAYER_NORM_SINGLE_READ_FLOAT32_FLOAT32;
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = LayerNormV4TilingKey::LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT32;
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT16) {
        tilingKey = LayerNormV4TilingKey::LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT16;
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = LayerNormV4TilingKey::LAYER_NORM_SINGLE_READ_BFLOAT16_FLOAT32;
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_BF16) {
        tilingKey = LayerNormV4TilingKey::LAYER_NORM_SINGLE_READ_BFLOAT16_BFLOAT16;
    }
    return tilingKey;
}

ge::graphStatus LayerNormV4SingleReadTiling::DoOpTiling()
{
    uint32_t blockDim = 0;
    uint32_t nRow = 0;
    uint32_t tailNRow = 0;
    uint32_t blockCount = 0;
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t rowMax = 0;
    rowMax = ((commonParams.ubSizePlatForm - NUM_EIGHT * BLOCK_SIZE) / FLOAT_SIZE) / NUM_TWO;
    if (commonParams.colSize <= commonParams.coreNum) {
        nRow = 1;
        blockDim = static_cast<uint32_t>(commonParams.colSize);
    } else {
        nRow = rowMax / commonParams.rowAlign;
        if (nRow > N_ROW_LIMIT) {
            nRow = N_ROW_LIMIT;
        }
        blockDim = static_cast<uint32_t>(commonParams.coreNum);
    }

    blockCount = commonParams.colSize / nRow;
    tailNRow = commonParams.colSize - blockCount * nRow;
    loopCount = blockCount / blockDim;
    tailLoop = blockCount - loopCount * blockDim;

    td_.set_blockDim(blockDim);
    td_.set_colSize(commonParams.colSize);
    td_.set_rowSize(commonParams.rowSize);
    td_.set_eps(commonParams.eps);
    td_.set_coefficient(commonParams.coefficient);
    td_.set_rowAlign(commonParams.rowAlign);
    td_.set_nRow(nRow);
    td_.set_tailNRow(tailNRow);
    td_.set_loopCount(loopCount);
    td_.set_tailLoop(tailLoop);
    td_.set_tileLength(nRow * commonParams.rowAlign);
    td_.set_blockLength(nRow * commonParams.rowSize);
    td_.set_nullptrGamma(commonParams.gammaNullPtr);
    td_.set_nullptrBeta(commonParams.betaNullPtr);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4SingleReadTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_blockDim());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormV4", LayerNormV4SingleReadTiling, 1000);
}  // namespace optiling
