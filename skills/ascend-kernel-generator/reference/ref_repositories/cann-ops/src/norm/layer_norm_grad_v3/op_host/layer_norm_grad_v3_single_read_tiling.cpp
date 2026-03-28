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
 * \file layer_norm_grad_v3_single_read_tiling.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
static const int64_t LNG_TMP_BUFFER_SIZE_0 = 64;
static const int64_t LNG_TMP_BUFFER_SIZE_1 = 512;
static const int64_t LNG_B32_DTYPE_SIZE = 4;
static const int64_t LNG_B16_ALIGN_FACTOR = 16;
static const int64_t LNG_B32_ALIGN_FACTOR = 8;
static const int64_t LNG_MAX_BUFFER_NUM = 4;
static const int64_t LNG_CONSTANT_TWO = 2;
static const int64_t LNG_CONSTANT_FIVE = 5;
static const size_t LNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
static const uint64_t LNG_WEIGHT = 10;
static const int64_t LNG_MIN_COL = 768;
static const int64_t LNG_MAX_COL = 12176;

static inline int64_t CeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool LayerNormGradV3SingleReadTiling::IsCapable()
{
    if (LNG_MIN_COL > commonParams.colSize || commonParams.colSize > LNG_MAX_COL) {
        OPS_LOG_I(context_->GetNodeName(),
            "LayerNormGradV3SingleReadTiling: col value(=%ld) is not support now, "
            "please check.",
            commonParams.colSize);
        return false;
    }
    return true;
}

uint64_t LayerNormGradV3SingleReadTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::SINGEL_READ);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + commonParams.isDeterministicKey * LNG_DETERMINISTIC_KEY_WEIGHT +
           static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus LayerNormGradV3SingleReadTiling::DoOpTiling()
{
    colAlignV = CeilDiv(static_cast<int64_t>(commonParams.colSize), LNG_B32_ALIGN_FACTOR) * LNG_B32_ALIGN_FACTOR;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        colAlignM = colAlignV;
    } else {
        colAlignM = CeilDiv(static_cast<int64_t>(commonParams.colSize), LNG_B16_ALIGN_FACTOR) * LNG_B16_ALIGN_FACTOR;
    }

    td_.set_row(commonParams.rowSize);
    td_.set_col(commonParams.colSize);
    td_.set_colAlignV(colAlignV);
    td_.set_colAlignM(colAlignM);
    // calculate block tiling, row split block
    int64_t blockFormer =
        CeilDiv(static_cast<int64_t>(commonParams.rowSize), static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CeilDiv(static_cast<int64_t>(commonParams.rowSize), blockFormer);
    int64_t blockTail = commonParams.rowSize - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);
    // calculate ub tiling, row split block
    int64_t maxBufferSize = (commonParams.ubSizePlatForm - LNG_TMP_BUFFER_SIZE_0 * LNG_CONSTANT_FIVE -
                                LNG_TMP_BUFFER_SIZE_1 * LNG_CONSTANT_TWO) /
                            LNG_B32_DTYPE_SIZE / LNG_MAX_BUFFER_NUM;
    if (maxBufferSize < colAlignM) {
        OPS_LOG_E(context_->GetNodeName(),
            "LayerNormGradV3SingleReadTiling: ubSize (=%ld) and maxBufferSize (=%ld) is less than colAlignM(=%ld) "
            "please check.",
            commonParams.ubSizePlatForm,
            maxBufferSize,
            colAlignM);
        return ge::GRAPH_FAILED;
    }
    int64_t ubFormer = maxBufferSize / colAlignM;
    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer);
    td_.set_ubTailOfTailBlock(blockTail - (ubLoopOfTailBlock - 1) * ubFormer);
    td_.set_bufferElemNums(colAlignM * ubFormer);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3SingleReadTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_blockNum());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3SingleReadTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    // workspace size is coreNum * colAlignV * 4 * 2
    workspaces[0] = td_.get_blockNum() * colAlignV * LNG_B32_DTYPE_SIZE * LNG_CONSTANT_TWO + LNG_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3SingleReadTiling, 1000);
}  // namespace optiling
