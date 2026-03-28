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
 * \file batch_norm_v3_full_reduce_tiling.cpp
 * \brief
 */

#include "batch_norm_v3_tiling.h"

static constexpr uint64_t BNV3_FULL_REDUCE_NOMAL_TILING_KEY = 2000;
static constexpr uint64_t BNV3_FULL_REDUCE_A_PARALLEL_TILING_KEY = 2001;
static constexpr uint32_t TWO_POWER_ONE = 2;
static constexpr uint32_t TWO_POWER_TWO = 4;
static constexpr uint32_t TWO_POWER_THREE = 8;
static constexpr uint32_t TWO_POWER_FOUR = 16;
static constexpr uint64_t HALF_SIZE = 2;
static constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
static constexpr int64_t A_UB_NUM = 20;
static constexpr int64_t Y_HALF_R_UB_NUM = 7;
static constexpr int64_t Y_FLOAT_R_UB_NUM = 8;
static constexpr int64_t FULL_REDUCE_TEMPLATE_R_LIMIT = 8192;

namespace optiling {
static uint32_t FindDichotomizeAddDiffSize(uint32_t parallelN)
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

static uint32_t FindCofFactor(uint32_t n)
{
    // 找到比n大的最邻近的二次幂数, n = 15，结果为16
    if ((n & (n - 1)) != 0) {
        uint32_t temp = n - 1;
        temp |= temp >> 1;
        temp |= temp >> TWO_POWER_ONE;
        temp |= temp >> TWO_POWER_TWO;
        temp |= temp >> TWO_POWER_THREE;
        temp |= temp >> TWO_POWER_FOUR;
        return (temp + 1);
    } else {
        return n;
    }
}

int64_t BatchNormV3FullReduceTiling::DoUbTiling(const int64_t blockFactor, int64_t &aUbSize, int64_t &rUbSize)
{
    /*
    A_UB_NUM和rUbNum表示kernel侧按fp16算需要的两种ub的节点个数
    eleNum为按照ub大小计算出的总共能存放fp16元素的个数
    初始aUbFactor计算方式：切分方式在A上做ub切分，需要计算出最多一次能计算几个A
    计算一个A需要搬入一个完整的R即patternR1*patternR0,eleNum / rUbNum  / patternR1 / patternR0为最大aUbFactor
    */
    int64_t eleNum = FloorDiv(commonParams.ubSizePlatForm, HALF_SIZE);
    int64_t rUbNum = (commonParams.xDtype == ge::DT_FLOAT) ? Y_FLOAT_R_UB_NUM : Y_HALF_R_UB_NUM;
    int64_t aUbFactor = std::min(blockFactor, eleNum / rUbNum / (commonParams.patternR1 * commonParams.patternR0));
    while (aUbFactor > 0) {
        // 需要16对齐, 使得fp16 Block对齐，方便原地cast处理
        aUbSize = CeilAlign(aUbFactor, B16_BLOCK_ALIGN_NUM);
        if (commonParams.patternR0 == 1) {
            rUbSize = CeilAlign(
                commonParams.patternR1 * CeilAlign(aUbFactor, commonParams.patternR0Align), B16_BLOCK_ALIGN_NUM);
        } else {
            rUbSize =
                CeilAlign(aUbFactor * commonParams.patternR1 * commonParams.patternR0Align, B16_BLOCK_ALIGN_NUM);
        }
        if (aUbSize * A_UB_NUM + rUbSize * rUbNum > eleNum) {
            aUbFactor = aUbFactor - 1;
        } else {
            break;
        }
    }
    return aUbFactor;
}

bool BatchNormV3FullReduceTiling::IsCapable()
{
    if (commonParams.patternR1 * commonParams.patternR0 >= FULL_REDUCE_TEMPLATE_R_LIMIT) {
        return false;
    }
    return true;
}

uint64_t BatchNormV3FullReduceTiling::GetTilingKey() const
{
    return fullReduceTilingkey;
}

ge::graphStatus BatchNormV3FullReduceTiling::DoOpTiling()
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
    uint32_t cofFactor = FindCofFactor(commonParams.patternR0 * commonParams.patternR1);
    td_.set_coefficient0(static_cast<float>(1.0 / static_cast<double>(cofFactor)));
    td_.set_coefficient1(static_cast<float>(
        static_cast<double>(cofFactor) / static_cast<double>(commonParams.patternR0 * commonParams.patternR1)));
    int64_t aUbSize = 1;
    int64_t rUbSize = 1;
    int64_t aUbFactor = DoUbTiling(blockFactor, aUbSize, rUbSize);
    OP_TILING_CHECK(aUbFactor == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(commonParams.nodeName, "BatchNormV3FullReduceTiling not supported this case"),
        return ge::GRAPH_PARAM_INVALID);
    td_.set_aUbFactor(aUbFactor);
    td_.set_aUbSize(aUbSize);
    td_.set_rUbSize(rUbSize);
    td_.set_aUbLoop(CeilDiv(blockFactor, aUbFactor));
    td_.set_aUbTail(blockFactor - (td_.get_aUbLoop() - 1) * aUbFactor);
    td_.set_tailCoreAUbLoop(CeilDiv(td_.get_tailCoreBlockFactor(), aUbFactor));
    td_.set_tailCoreAUbTail(td_.get_tailCoreBlockFactor() - (td_.get_tailCoreAUbLoop() - 1) * aUbFactor);
    fullReduceTilingkey = BNV3_FULL_REDUCE_NOMAL_TILING_KEY;
    int64_t parallelN = commonParams.patternR1 * commonParams.patternR0Align;
    if (commonParams.patternR0 == 1) {
        fullReduceTilingkey = BNV3_FULL_REDUCE_A_PARALLEL_TILING_KEY;
        parallelN = commonParams.patternR1;
    }
    td_.set_dichotomizeAddDiffSize(static_cast<int64_t>(FindDichotomizeAddDiffSize(parallelN)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3FullReduceTiling::PostTiling()
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

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3FullReduceTiling, 1000);
}  // namespace optiling
