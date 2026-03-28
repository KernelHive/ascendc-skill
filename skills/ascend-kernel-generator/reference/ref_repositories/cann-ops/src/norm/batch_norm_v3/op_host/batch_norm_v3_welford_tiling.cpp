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
 * \file batch_norm_v3_welford_tiling.cpp
 * \brief
 */

#include "batch_norm_v3_tiling.h"

static constexpr uint64_t BNV3_WELFORD_R0_SPLIT_NOT_ALIGN_TILING_KEY = 1000;
static constexpr uint64_t BNV3_WELFORD_R0_SPLIT_ALIGN_TILING_KEY = 1001;
static constexpr uint64_t BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_TILING_KEY = 1002;
static constexpr uint64_t BNV3_WELFORD_R1_SPLIT_ALIGN_TILING_KEY = 1003;
static constexpr uint64_t R0_ALIGN_TILING_KEY_BIAS = 10;
static constexpr uint32_t TWO_POWER_ONE = 2;
static constexpr uint32_t TWO_POWER_TWO = 4;
static constexpr uint32_t TWO_POWER_THREE = 8;
static constexpr uint32_t TWO_POWER_FOUR = 16;
static constexpr uint64_t FLOAT_SIZE = 4;
static constexpr int64_t A_UB_SIZE_LIMIT = 512;
static constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
static constexpr int64_t A_UB_NUM = 10;
static constexpr int64_t R0_UB_NUM = 5;
static constexpr int64_t TWO_NUM = 2;

namespace optiling {
uint32_t BatchNormV3WelfordTiling::FindDichotomizeAddDiffSize(uint32_t parallelN)
{
    // 找到parallelN与小于parallelN的最近二次幂的差值 例如：parallelN = 15，结果为15 - 8 = 7
    if ((parallelN & (parallelN - 1)) != 0) {
        uint32_t temp = parallelN - 1;
        temp |= temp >> 1;
        temp |= temp >> TWO_POWER_ONE;
        temp |= temp >> TWO_POWER_TWO;
        temp |= temp >> TWO_POWER_THREE;
        temp |= temp >> TWO_POWER_FOUR;
        return (parallelN - ((temp + 1) / TWO_POWER_ONE));
    } else {
        return 0;
    }
}

bool BatchNormV3WelfordTiling::IsCapable()
{
    return true;
}

uint64_t BatchNormV3WelfordTiling::GetTilingKey() const
{
    return welfordTilingkey;
}

void BatchNormV3WelfordTiling::DoUbTiling(int64_t &aUbFactor, int64_t &r0UbFactor)
{
    // 需要16对齐, 使得fp16 Block对齐，方便原地cast处理
    int64_t eleNum = FloorDiv(commonParams.ubSizePlatForm, FLOAT_SIZE);
    aUbFactor = (td_.get_blockFactor() > A_UB_SIZE_LIMIT) ? A_UB_SIZE_LIMIT
                                                          : CeilAlign(td_.get_blockFactor(), B16_BLOCK_ALIGN_NUM);
    r0UbFactor = FloorAlign(FloorDiv(eleNum - aUbFactor * A_UB_NUM, R0_UB_NUM), B16_BLOCK_ALIGN_NUM);
}

ge::graphStatus BatchNormV3WelfordTiling::DoOpTiling()
{
    int64_t blockFactor = CeilDiv(commonParams.patternA, static_cast<int64_t>(commonParams.coreNum));
    usedCoreNum = CeilDiv(commonParams.patternA, blockFactor);
    td_.set_blockFactor(blockFactor);
    td_.set_tailCoreBlockFactor(commonParams.patternA - (usedCoreNum - 1) * blockFactor);
    float batchVarScale =
        (commonParams.patternR0 * commonParams.patternR1 == 1)
            ? 1.0
            : static_cast<float>(static_cast<double>(commonParams.patternR0 * commonParams.patternR1) /
                                 static_cast<double>(commonParams.patternR0 * commonParams.patternR1 - 1));
    td_.set_batchVarScale(batchVarScale);
    int64_t aUbFactor = 1;
    int64_t r0UbFactor = 1;
    DoUbTiling(aUbFactor, r0UbFactor);
    td_.set_aUbFactor(aUbFactor);
    td_.set_r0UbFactor(r0UbFactor);
    td_.set_aUbLoop(CeilDiv(blockFactor, aUbFactor));
    td_.set_aUbTail(blockFactor - (td_.get_aUbLoop() - 1) * aUbFactor);
    td_.set_tailCoreAUbLoop(CeilDiv(td_.get_tailCoreBlockFactor(), aUbFactor));
    td_.set_tailCoreAUbTail(td_.get_tailCoreBlockFactor() - (td_.get_tailCoreAUbLoop() - 1) * aUbFactor);
    td_.set_r0UbLoop(CeilDiv(commonParams.patternR0, r0UbFactor));
    td_.set_r0UbTail(commonParams.patternR0 - (td_.get_r0UbLoop() - 1) * r0UbFactor);
    td_.set_procNR0(1);
    td_.set_nR0Loop(commonParams.patternR1);
    td_.set_lastLoopNR0(1);
    uint32_t parallelN =
        (td_.get_r0UbLoop() == 1) ? static_cast<uint32_t>(commonParams.patternR0) : static_cast<uint32_t>(r0UbFactor);
    if ((td_.get_r0UbLoop() == 1) || (td_.get_r0UbFactor() == td_.get_r0UbTail())) {
        welfordTilingkey = BNV3_WELFORD_R0_SPLIT_ALIGN_TILING_KEY;
    } else {
        welfordTilingkey = BNV3_WELFORD_R0_SPLIT_NOT_ALIGN_TILING_KEY;
    }
    // R0不切分，R1补充切分
    if ((commonParams.patternR0Align <= (r0UbFactor / TWO_NUM)) && commonParams.patternR1 > 1) {
        int64_t procNR0 = FloorDiv(r0UbFactor, commonParams.patternR0Align);
        int64_t nR0Loop = CeilDiv(commonParams.patternR1, procNR0);
        int64_t lastLoopNR0 = commonParams.patternR1 - (nR0Loop - 1) * procNR0;
        td_.set_procNR0(procNR0);
        td_.set_nR0Loop(nR0Loop);
        td_.set_lastLoopNR0(lastLoopNR0);
        parallelN = (nR0Loop == 1) ? lastLoopNR0 * commonParams.patternR0Align : procNR0 * commonParams.patternR0Align;
        uint64_t r0AlignTilingKeyBias =
            (commonParams.patternR0 == commonParams.patternR0Align) ? R0_ALIGN_TILING_KEY_BIAS : 0;
        if ((nR0Loop == 1) || (lastLoopNR0 == procNR0)) {
            welfordTilingkey = BNV3_WELFORD_R1_SPLIT_ALIGN_TILING_KEY + r0AlignTilingKeyBias;
        } else {
            welfordTilingkey = BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_TILING_KEY + r0AlignTilingKeyBias;
        }
    }
    td_.set_dichotomizeAddDiffSize(static_cast<int64_t>(FindDichotomizeAddDiffSize(parallelN)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3WelfordTiling::PostTiling()
{
    td_.set_patternR1(commonParams.patternR1);
    td_.set_patternR0(commonParams.patternR0);
    td_.set_patternA(commonParams.patternA);
    td_.set_patternR0Align(commonParams.patternR0Align);
    td_.set_epsilon(commonParams.epsilon);
    td_.set_momentum(commonParams.momentum);
    td_.set_momentumReverse(commonParams.momentumReverse);
    context_->SetBlockDim(usedCoreNum);
    auto rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK(td_.GetDataSize() > rawTilingData->GetCapacity(),
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName,
            "actual tiling data size %zu > context tiling data size %zu",
            td_.GetDataSize(),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    td_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3WelfordTiling, 2000);
}  // namespace optiling
