/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file max_pool3d_grad_with_argmax_normal_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {
constexpr uint64_t DOUBLE_SIZE_NUM = 2;

ge::graphStatus MaxPool3DGradWithArgmaxNormalTiling::GetShapeAttrsInfo()
{
    auto ret = MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxNormalTiling::IsCapable()
{
    if ((maxPoolGradParams.dilationD != 1) || (maxPoolGradParams.dilationH != 1) ||
        (maxPoolGradParams.dilationW != 1)) {
        return false;
    }

    uint64_t normalTensorSizeMin = CalUBTotalSize(1, 1, 1, maxPoolGradParams.isOverLap);
    if (normalTensorSizeMin <= maxPoolGradParams.maxUbSize) {
        return true;
    }
    return false;
}

uint64_t MaxPool3DGradWithArgmaxNormalTiling::CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo,
    bool isCoreOverLap)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t baseDi = (baseDo - 1) * maxPoolGradParams.sd + maxPoolGradParams.kd;
    uint64_t baseHi = (baseHo - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh;
    uint64_t baseWi = (baseWo - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    uint64_t diHiWi = baseDi * baseHi * baseWi;
    uint64_t doHoWo = baseDo * baseHo * baseWo;
    uint64_t b8BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B8;
    uint64_t b32BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B32;
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize;
    uint64_t wiB32Align = vectorutil::CeilDiv(baseWi, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t wiDtypeAlign = vectorutil::CeilDiv(baseWi, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t diHiWiB32Align = baseDi * baseHi * wiB32Align;
    uint64_t diHiWiDtypeAlign = baseDi * baseHi * wiDtypeAlign;
    uint64_t doHoWoB32Align = vectorutil::CeilDiv(doHoWo, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t doHoWoDtypeAlign = vectorutil::CeilDiv(doHoWo, dtypeBlockAlignNum) * dtypeBlockAlignNum;

    uint64_t normalDiHiWiTensorSize;
    if (isCoreOverLap) {
        normalDiHiWiTensorSize = 1 * vl * diHiWi * DTYPE_LEN_B32 + // xIndexBuf
            DOUBLE_SIZE_NUM * vl * diHiWiB32Align * DTYPE_LEN_B32;               // yQue&yTransposeBuf
    } else {
        normalDiHiWiTensorSize = 1 * vl * diHiWi * DTYPE_LEN_B32 +    // xIndexBuf
            1 * vl * diHiWiDtypeAlign * DTYPE_LEN_B32 +               // yTransposeBuf
            1 * vl * diHiWiDtypeAlign * maxPoolGradParams.xDtypeSize; // yQue
    }
    uint64_t normalDoHoWoTensorSize =
        vectorutil::CeilDiv(1 * vl * doHoWo, b8BlockAlignNum) * b8BlockAlignNum * DTYPE_LEN_B8 + // maskBuf
        DOUBLE_SIZE_NUM * vl * doHoWoB32Align * DTYPE_LEN_B32 +                 // argmaxQue&argmaxTransposeBuf
        DOUBLE_SIZE_NUM * vl * doHoWoDtypeAlign * maxPoolGradParams.xDtypeSize; // gradQue&gradTransposeBuf
    return normalDiHiWiTensorSize + normalDoHoWoTensorSize + SELECT_RESERVED_UB_SIZE;
}

void MaxPool3DGradWithArgmaxNormalTiling::CalcBaseNc()
{
    maxPoolGradParams.vl = NUM_PER_REP_B32;
    if (maxPoolGradParams.ncDim >= maxPoolGradParams.vl * maxPoolGradParams.totalCoreNum) {
        if (CalUBTotalSize(maxPoolGradParams.doDim, maxPoolGradParams.hoDim, maxPoolGradParams.woDim,
            maxPoolGradParams.isOverLap) <= maxPoolGradParams.maxUbSize) {
            return;
        }
    } else if (maxPoolGradParams.ncDim >= maxPoolGradParams.vl) {
        if (!maxPoolGradParams.isOverLap) {
            return;
        }
    }
    const uint64_t minVL = 16;
    if (maxPoolGradParams.ncDim <= maxPoolGradParams.vl - minVL) {
        maxPoolGradParams.vl = vectorutil::CeilAlign(maxPoolGradParams.ncDim, minVL);
    }
    if (CalUBTotalSize(maxPoolGradParams.doDim, maxPoolGradParams.hoDim, maxPoolGradParams.woDim,
        maxPoolGradParams.isOverLap) > maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.vl = minVL;
    }
    return;
}

/* *
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
uint64_t MaxPool3DGradWithArgmaxNormalTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd,
    const uint32_t ubCutAxis)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE_SIZE_NUM;
        if (ubCutAxis == TILING_UB_CUT_DO) {
            tmpTotalSize = CalUBTotalSize(baseXoMid, maxPoolGradParams.baseHo, maxPoolGradParams.baseWo,
                maxPoolGradParams.isOverLap);
        } else if (ubCutAxis == TILING_UB_CUT_HO) {
            tmpTotalSize = CalUBTotalSize(maxPoolGradParams.baseDo, baseXoMid, maxPoolGradParams.baseWo,
                maxPoolGradParams.isOverLap);
        } else if (ubCutAxis == TILING_UB_CUT_WO) {
            tmpTotalSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, baseXoMid,
                maxPoolGradParams.isOverLap);
        }
        if (tmpTotalSize <= maxPoolGradParams.maxUbSize) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalParamsNotCutUB(const uint32_t ubCutAxis, bool isCoreOverLap)
{
    uint64_t noCutSize = CalUBTotalSize(maxPoolGradParams.singleCoreDo, maxPoolGradParams.singleCoreHo,
        maxPoolGradParams.singleCoreWo, isCoreOverLap);
    if (noCutSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = maxPoolGradParams.singleCoreDo;
        maxPoolGradParams.baseHo = maxPoolGradParams.singleCoreHo;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        maxPoolGradParams.ubCutAxis = ubCutAxis;
        return true;
    }
    return false;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalParamsCutUB()
{
    // 1. Cut d
    uint64_t perDoSize =
        CalUBTotalSize(1, maxPoolGradParams.singleCoreHo, maxPoolGradParams.singleCoreWo, maxPoolGradParams.isOverLap);
    if (perDoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseHo = maxPoolGradParams.singleCoreHo;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        // Cal best baseDo
        maxPoolGradParams.baseDo = CalBestBaseSize(1, maxPoolGradParams.singleCoreDo, TILING_UB_CUT_DO);
        uint64_t baseDoCnt = vectorutil::CeilDiv(maxPoolGradParams.singleCoreDo, maxPoolGradParams.baseDo);
        maxPoolGradParams.baseDo = vectorutil::CeilDiv(maxPoolGradParams.singleCoreDo, baseDoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_DO;
        return true;
    }

    // 2. Cut d&h
    uint64_t perHoSize = CalUBTotalSize(1, 1, maxPoolGradParams.singleCoreWo, maxPoolGradParams.isOverLap);
    if (perHoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = 1;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        // Cal best baseHo
        maxPoolGradParams.baseHo = CalBestBaseSize(1, maxPoolGradParams.singleCoreHo, TILING_UB_CUT_HO);
        uint64_t baseHoCnt = vectorutil::CeilDiv(maxPoolGradParams.singleCoreHo, maxPoolGradParams.baseHo);
        maxPoolGradParams.baseHo = vectorutil::CeilDiv(maxPoolGradParams.singleCoreHo, baseHoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_HO;
        return true;
    }

    // 3. Cut d&h&w
    uint64_t perWoSize = CalUBTotalSize(1, 1, 1, maxPoolGradParams.isOverLap);
    if (perWoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = 1;
        maxPoolGradParams.baseHo = 1;
        // Cal best baseWo
        maxPoolGradParams.baseWo = CalBestBaseSize(1, maxPoolGradParams.singleCoreWo, TILING_UB_CUT_WO);
        uint64_t baseWoCnt = vectorutil::CeilDiv(maxPoolGradParams.singleCoreWo, maxPoolGradParams.baseWo);
        maxPoolGradParams.baseWo = vectorutil::CeilDiv(maxPoolGradParams.singleCoreWo, baseWoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }

    return false;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalTilingParams()
{
    const uint64_t ncDim = maxPoolGradParams.ncDim;
    const uint64_t doDim = maxPoolGradParams.doDim;
    const uint64_t hoDim = maxPoolGradParams.hoDim;
    const uint64_t woDim = maxPoolGradParams.woDim;
    const uint64_t totalCoreNum = maxPoolGradParams.totalCoreNum;

    CalcBaseNc();
    const uint64_t vl = maxPoolGradParams.vl;
    maxPoolGradParams.singleCoreDo = doDim;
    maxPoolGradParams.singleCoreHo = hoDim;
    maxPoolGradParams.singleCoreWo = woDim;

    bool fixOverlapOutput = (context_->GetDeterministic() == 1) && maxPoolGradParams.isOverLap;
    OP_LOGI(context_->GetNodeName(), "GetDeterministic state: %u", context_->GetDeterministic());
    // Normal tiling cal begin
    // 1. Cut nc between core
    maxPoolGradParams.singleCoreNc = vl;
    maxPoolGradParams.baseNc = vl;
    uint64_t ncCnt = vectorutil::CeilDiv(ncDim, maxPoolGradParams.singleCoreNc);
    if (SetNormalParamsNotCutUB(TILING_UB_NO_CUT, false)) {
        maxPoolGradParams.isOverLap = false;
        return true;
    }
    if (ncCnt >= totalCoreNum) {
        return SetNormalParamsCutUB();
    }

    // 2. Cut nc&do between core
    uint64_t doCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt); // need bigger than this
    if (doCntNeed <= doDim) {
        maxPoolGradParams.singleCoreHo = hoDim;
        maxPoolGradParams.singleCoreWo = woDim;
        // 2.1 Dim no overlap
        if (!fixOverlapOutput && (0 != doCntNeed)) {
            maxPoolGradParams.singleCoreDo = vectorutil::CeilDiv(doDim, doCntNeed);
            return SetNormalParamsCutUB();
        }
        // 2.2 Dim overlap, cut nc
        maxPoolGradParams.singleCoreDo = doDim;
        return SetNormalParamsCutUB();
    }
    maxPoolGradParams.singleCoreDo = 1;
    uint64_t doCnt = vectorutil::CeilDiv(doDim, maxPoolGradParams.singleCoreDo);

    // 3. Cut nc&do&ho between core
    uint64_t hoCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt * doCnt); // Need bigger than this
    if (SetNormalParamsNotCutUB(TILING_UB_CUT_DO, maxPoolGradParams.isOverLap)) {
        return true;
    }
    if (hoCntNeed <= hoDim) {
        maxPoolGradParams.singleCoreWo = woDim;
        // 3.1 Dim no overlap
        if (!fixOverlapOutput && (0 != hoCntNeed)) {
            maxPoolGradParams.singleCoreHo = vectorutil::CeilDiv(hoDim, hoCntNeed);
            return SetNormalParamsCutUB();
        }
        // 3.2 Cut nc
        maxPoolGradParams.singleCoreHo = hoDim;
        return SetNormalParamsCutUB();
    }
    maxPoolGradParams.singleCoreHo = 1;
    uint64_t hoCnt = hoDim;

    // 4. Cut nc&do&ho&wo between core
    uint64_t woCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt * doCnt * hoCnt); // Need bigger than this
    if (SetNormalParamsNotCutUB(TILING_UB_CUT_HO, maxPoolGradParams.isOverLap)) {
        return true;
    }
    if (!fixOverlapOutput) {
        // 4.1 Dim no overlap
        if (woCntNeed <= woDim && (0 != woCntNeed)) {
            maxPoolGradParams.singleCoreWo = vectorutil::CeilDiv(woDim, woCntNeed);
        } else {
            maxPoolGradParams.singleCoreWo = 1;
        }
        return SetNormalParamsCutUB();
    } else {
        // 4.2 Cut nc
        maxPoolGradParams.singleCoreWo = woDim;
        return SetNormalParamsCutUB();
    }

    OP_LOGE(context_->GetNodeName(), "Normal set tiling failed.");
    return false;
}

void MaxPool3DGradWithArgmaxNormalTiling::SetOtherTilingParams()
{
    maxPoolGradParams.ncCnt = vectorutil::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.singleCoreNc);
    maxPoolGradParams.doCnt = vectorutil::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.singleCoreDo);
    maxPoolGradParams.hoCnt = vectorutil::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.singleCoreHo);
    maxPoolGradParams.woCnt = vectorutil::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.singleCoreWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1) * maxPoolGradParams.singleCoreNc;
    maxPoolGradParams.doTail = maxPoolGradParams.doDim - (maxPoolGradParams.doCnt - 1) * maxPoolGradParams.singleCoreDo;
    maxPoolGradParams.hoTail = maxPoolGradParams.hoDim - (maxPoolGradParams.hoCnt - 1) * maxPoolGradParams.singleCoreHo;
    maxPoolGradParams.woTail = maxPoolGradParams.woDim - (maxPoolGradParams.woCnt - 1) * maxPoolGradParams.singleCoreWo;
    maxPoolGradParams.totalCnt =
        maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
    maxPoolGradParams.usedCoreNum = std::min(maxPoolGradParams.totalCnt, maxPoolGradParams.totalCoreNum);
    if (maxPoolGradParams.xDtypeSize != DTYPE_LEN_B32 && maxPoolGradParams.isOverLap) {
        maxPoolGradParams.workspaceSize = maxPoolGradParams.ncDim * maxPoolGradParams.diDim * maxPoolGradParams.hiDim *
            maxPoolGradParams.wiDim * sizeof(float);
    } else {
        maxPoolGradParams.workspaceSize = 0;
    }
}

void MaxPool3DGradWithArgmaxNormalTiling::SetNormalTilingData()
{
    tilingData.set_singleCoreNc(maxPoolGradParams.singleCoreNc);
    tilingData.set_singleCoreDo(maxPoolGradParams.singleCoreDo);
    tilingData.set_singleCoreHo(maxPoolGradParams.singleCoreHo);
    tilingData.set_singleCoreWo(maxPoolGradParams.singleCoreWo);
}

void MaxPool3DGradWithArgmaxNormalTiling::PrintNormalTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData singleCoreNc: %lu, singleCoreDo: %lu, singleCoreHo: %lu, singleCoreWo: %lu.",
        tilingData.get_singleCoreNc(), tilingData.get_singleCoreDo(), tilingData.get_singleCoreHo(),
        tilingData.get_singleCoreWo());
}

ge::graphStatus MaxPool3DGradWithArgmaxNormalTiling::DoOpTiling()
{
    bool res = SetNormalTilingParams();
    OP_TILING_CHECK(!res, OP_LOGE(context_->GetNodeName(), "Normal cal tiling params failed."),
        return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_NORMAL;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetNormalTilingData();
    PrintTilingData();
    PrintNormalTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxNormalTiling, 2);
} // namespace optiling
