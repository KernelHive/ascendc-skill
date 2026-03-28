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
 * \file layer_norm_grad_v3_common_tiling.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
static const int64_t COMMON_LNG_BLOCK_BYTES = 32;
static const int64_t COMMON_LNG_B32_DTYPE_SIZE = 4;
static const int64_t COMMON_LNG_B16_ALIGN_FACTOR = 16;
static const int64_t COMMON_LNG_B32_ALIGN_FACTOR = 8;
static const int64_t COMMON_LNG_WHOLE_BUFFER_NUM = 4;
static const int64_t COMMON_LNG_NLAST_R_BUFFER_NUM = 3;
static const int64_t COMMON_LNG_LAST_R_BUFFER_NUM = 3;
static const int64_t COMMON_LNG_BRCB_ALIGN_FACTOR = 8;
static const int64_t COMMON_LNG_WORKSPACE_NUM = 2;
static const size_t COMMON_LNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
static const int64_t COMMON_LNG_MAX_COL = 768;

static inline int64_t CommonCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool LayerNormGradV3CommonTiling::IsCapable()
{
    if (commonParams.colSize > COMMON_LNG_MAX_COL) {
        OPS_LOG_I(context_->GetNodeName(),
            "LayerNormGradV3CommonTiling: col value(=%ld) is not support now, "
            "please check.",
            commonParams.colSize);
        return false;
    }
    return true;
}

uint64_t LayerNormGradV3CommonTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::COMMON);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + commonParams.isDeterministicKey * LNG_DETERMINISTIC_KEY_WEIGHT +
           static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus LayerNormGradV3CommonTiling::DoOpTiling()
{
    row_ = static_cast<int64_t>(commonParams.rowSize);
    col_ = static_cast<int64_t>(commonParams.colSize);
    colAlignV_ = CommonCeilDiv(col_, COMMON_LNG_B32_ALIGN_FACTOR) * COMMON_LNG_B32_ALIGN_FACTOR;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        colAlignM_ = colAlignV_;
    } else {
        colAlignM_ = CommonCeilDiv(col_, COMMON_LNG_B16_ALIGN_FACTOR) * COMMON_LNG_B16_ALIGN_FACTOR;
    }

    td_.set_row(row_);
    td_.set_col(col_);
    td_.set_colAlignV(colAlignV_);
    td_.set_colAlignM(colAlignM_);
    // calculate block tiling, row split block
    int64_t blockFormer = CommonCeilDiv(row_, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CommonCeilDiv(row_, blockFormer);
    int64_t blockTail = row_ - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);
    td_.set_nlastRBufferBytes(colAlignM_ * COMMON_LNG_B32_DTYPE_SIZE);
    // calculate ub tiling, row split ub
    int64_t ubFormer = CalculateUbFormer();
    OP_TILING_CHECK((ubFormer <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
            "TilingForLayerNormGradV3: ub former(=%ld) is not greater than 0 "
            "in common tiling, please check",
            ubFormer),
        return ge::GRAPH_FAILED);
    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer);
    td_.set_ubTailOfTailBlock(blockTail - (ubLoopOfTailBlock - 1) * ubFormer);
    td_.set_wholeBufferElemNums(colAlignM_ * ubFormer);

    return ge::GRAPH_SUCCESS;
}

int64_t LayerNormGradV3CommonTiling::CalculateUbFormer()
{
    int64_t maxUBSize = commonParams.ubSizePlatForm - td_.get_nlastRBufferBytes() * COMMON_LNG_NLAST_R_BUFFER_NUM;
    int64_t coff = colAlignM_ * COMMON_LNG_B32_DTYPE_SIZE * COMMON_LNG_WHOLE_BUFFER_NUM +
                   COMMON_LNG_B32_DTYPE_SIZE * COMMON_LNG_LAST_R_BUFFER_NUM + COMMON_LNG_BLOCK_BYTES;
    int64_t curUbFormer = maxUBSize / coff;
    for (; curUbFormer >= 0; curUbFormer--) {
        int64_t wholeBufferBytes = curUbFormer * colAlignM_ * COMMON_LNG_B32_DTYPE_SIZE;
        int64_t lastRBufferBytes =
            CommonCeilDiv(curUbFormer * COMMON_LNG_B32_DTYPE_SIZE, COMMON_LNG_BLOCK_BYTES) * COMMON_LNG_BLOCK_BYTES;
        int64_t lastBrcbBufferBytes = CommonCeilDiv(curUbFormer, COMMON_LNG_BRCB_ALIGN_FACTOR) *
                                      COMMON_LNG_BRCB_ALIGN_FACTOR * COMMON_LNG_BLOCK_BYTES;
        int64_t curUBSize = wholeBufferBytes * COMMON_LNG_WHOLE_BUFFER_NUM +
                            lastRBufferBytes * COMMON_LNG_LAST_R_BUFFER_NUM + lastBrcbBufferBytes;
        if (curUBSize <= maxUBSize && lastBrcbBufferBytes <= wholeBufferBytes) {
            td_.set_wholeBufferBytes(wholeBufferBytes);
            td_.set_lastRBufferBytes(lastRBufferBytes);
            td_.set_lastBrcbBufferBytes(lastBrcbBufferBytes);
            break;
        }
    }

    return curUbFormer;
}

ge::graphStatus LayerNormGradV3CommonTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_blockNum());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3CommonTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    // workspace size is coreNum * colAlignV * 4 * 2
    workspaces[0] = td_.get_blockNum() * colAlignV_ * COMMON_LNG_B32_DTYPE_SIZE * COMMON_LNG_WORKSPACE_NUM +
                    COMMON_LNG_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3CommonTiling, 3000);
}  // namespace optiling
