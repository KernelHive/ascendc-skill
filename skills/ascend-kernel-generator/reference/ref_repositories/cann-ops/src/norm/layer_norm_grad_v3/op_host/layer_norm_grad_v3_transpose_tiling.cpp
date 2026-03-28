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
 * \file layer_norm_grad_v3_transpose_tiling.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
static constexpr uint32_t TRANSPOSE_COL_LIMIT = 64;
static constexpr uint32_t TRANSPOSE_C0_SIZE = 16;
static constexpr uint32_t REDUCE_BUF_NUM = 3;
static constexpr uint32_t TRANSPOSE_DY_BUF_NUM = 4;
static constexpr uint32_t TRANSPOSE_MEAN_BUF_NUM = 3;
static constexpr uint32_t TWO_POWER_ONE = 2;
static constexpr uint32_t TWO_POWER_TWO = 4;
static constexpr uint32_t TWO_POWER_THREE = 8;
static constexpr uint32_t TWO_POWER_FOUR = 16;
static constexpr uint32_t TWO = 2;
static constexpr uint32_t GM_ALIGN = 512;
static constexpr uint32_t LNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;

uint64_t LayerNormGradV3TransposeTiling::CalcBorrowFactor(uint64_t oriFactor)
{
    return (oriFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
}

uint32_t LayerNormGradV3TransposeTiling::FindDichotomizeAddDiffSize()
{
    // 找到col与小于col的最近二次幂的差值 eg：colSize = 15，结果为15 - 8 = 7
    if ((commonParams.colSize & (commonParams.colSize - 1)) != 0) {
        uint32_t temp = commonParams.colSize - 1;
        temp |= temp >> 1;
        temp |= temp >> TWO_POWER_ONE;
        temp |= temp >> TWO_POWER_TWO;
        temp |= temp >> TWO_POWER_THREE;
        temp |= temp >> TWO_POWER_FOUR;
        return (commonParams.colSize - ((temp + 1) / TWO));
    } else {
        return 0;
    }
}
bool LayerNormGradV3TransposeTiling::IsCapable()
{
    if ((commonParams.colSize > TRANSPOSE_COL_LIMIT) || (commonParams.colSize == commonParams.colAlign)) {
        OPS_LOG_I(context_->GetNodeName(),
            "LayerNormGradV3Transpose Template only support colSize <= 64 and not 32B align, colSize: %u",
            static_cast<uint32_t>(commonParams.colSize));
        return false;
    }
    return true;
}

uint64_t LayerNormGradV3TransposeTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::TRANSPOSE);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + commonParams.isDeterministicKey * LNG_DETERMINISTIC_KEY_WEIGHT +
           static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus LayerNormGradV3TransposeTiling::DoOpTiling()
{
    // 计算block相关tiling data
    uint64_t blockFormer = (commonParams.rowSize + commonParams.coreNum - 1) / commonParams.coreNum;
    uint64_t blockNum = (commonParams.rowSize + blockFormer - 1) / blockFormer;
    uint64_t blockTail = commonParams.rowSize - (blockNum - 1) * blockFormer;
    // 计算ub相关tiling data
    uint64_t curUbSize = commonParams.ubSizePlatForm - TRANSPOSE_COL_LIMIT * FLOAT_SIZE * REDUCE_BUF_NUM -
                         commonParams.colAlign * BLOCK_SIZE;
    uint64_t maxUbFormer =
        curUbSize / (commonParams.colSize * TRANSPOSE_DY_BUF_NUM * FLOAT_SIZE + TRANSPOSE_MEAN_BUF_NUM * FLOAT_SIZE);
    uint64_t ubFormer = 0;
    uint64_t bFormer = 0;
    uint64_t alignB = 0;
    uint64_t alignBAndCol = 0;
    for (int64_t curUbFormer = std::min(maxUbFormer, blockFormer); curUbFormer > 0; curUbFormer--) {
        ubFormer = curUbFormer;
        bFormer = CalcBorrowFactor(ubFormer);
        alignBAndCol =
            (bFormer * commonParams.colSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
        alignB = (bFormer + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
        if ((alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE * TRANSPOSE_MEAN_BUF_NUM +
                alignBAndCol * TRANSPOSE_C0_SIZE * FLOAT_SIZE * TRANSPOSE_DY_BUF_NUM) <= curUbSize) {
            break;
        }
    }

    uint64_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    uint64_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    uint64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    uint64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    uint32_t dichotomizeAddDiffSize = 0;
    dichotomizeAddDiffSize = FindDichotomizeAddDiffSize();

    float coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.colSize);

    uint32_t colB32AlignSize =
        (commonParams.colSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM * FLOAT_SIZE;
    uint32_t deterministicComputeWspSize = (colB32AlignSize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * blockNum;

    td_.set_row(commonParams.rowSize);
    td_.set_col(commonParams.colSize);
    td_.set_blockDim(blockNum);
    td_.set_blockFormer(blockFormer);
    td_.set_blockTail(blockTail);
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    td_.set_ubTailOfTailBlock(ubTailOfTailBlock);
    td_.set_bFormer(bFormer);
    td_.set_dichotomizeAddDiffSize(dichotomizeAddDiffSize);
    td_.set_coefficient(coefficient);
    td_.set_deterministicComputeWspSize(deterministicComputeWspSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = td_.get_deterministicComputeWspSize() * TWO + LNG_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3TransposeTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_blockDim());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3TransposeTiling, 2000);
}  // namespace optiling
