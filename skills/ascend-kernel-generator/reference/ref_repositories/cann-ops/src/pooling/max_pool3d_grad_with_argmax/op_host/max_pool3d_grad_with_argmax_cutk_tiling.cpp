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
 * \file max_pool3d_grad_with_argmax_cutk_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {
constexpr uint64_t WHITE_NC = 64 * 48;
constexpr uint64_t WHITE_DO = 1;
constexpr uint64_t WHITE_HO = 6;
constexpr uint64_t WHITE_WO = 8;
constexpr uint64_t WHITE_KD = 5;
constexpr uint64_t WHITE_KH = 1;
constexpr uint64_t WHITE_KW = 6;
constexpr uint64_t WHITE_SD = 8;
constexpr uint64_t WHITE_SH = 13;
constexpr uint64_t WHITE_SW = 18;
// it is the tiling key 31 where cutK's performance is better than normal 
constexpr uint64_t WHITE_TILING_KEY = 31; 
ge::graphStatus MaxPool3DGradWithArgmaxCutKTiling::GetShapeAttrsInfo()
{
    MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxCutKTiling::IsCapable()
{
    if ((maxPoolGradParams.dilationD != 1) ||
        (maxPoolGradParams.dilationH != 1) ||
        (maxPoolGradParams.dilationW != 1)) {
        return false;
    }
    // cutting do&&kd&&ho is supported, while other scenes are blocked
    uint64_t cutKUBSizeMin = CalUBTotalSize(1, 1, 1, TILING_UB_CUT_KW);
    /* Since it is difficult to distinguish between the normal and cutK scenarios,
       we will first implement a whitelist and adjust the logic later.*/
    if (maxPoolGradParams.ncDim <= WHITE_NC &&
        maxPoolGradParams.doDim == WHITE_DO &&
        maxPoolGradParams.hoDim == WHITE_HO &&
        maxPoolGradParams.woDim == WHITE_WO &&
        maxPoolGradParams.kd == WHITE_KD &&
        maxPoolGradParams.kh == WHITE_KH &&
        maxPoolGradParams.kw == WHITE_KW &&
        maxPoolGradParams.sd == WHITE_SD &&
        maxPoolGradParams.sh == WHITE_SH &&
        maxPoolGradParams.sw == WHITE_SW &&
        cutKUBSizeMin <= maxPoolGradParams.maxUbSize) {
        return true;
    }
    return false;
}

void MaxPool3DGradWithArgmaxCutKTiling::CalXp(uint64_t &baseDo, uint64_t &baseHo, uint64_t &baseWo,
                                              uint64_t &baseDp, uint64_t &baseHp, uint64_t &baseWp,
                                              const uint32_t ubCutAxis)
{
    if ((TILING_UB_NO_CUT == ubCutAxis) || (TILING_UB_CUT_DO == ubCutAxis)) {
        baseDp = (baseDo - 1) * maxPoolGradParams.sd + maxPoolGradParams.kd;
        baseHp = (baseHo - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh;
        baseWp = (baseWo - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    } else if ((TILING_UB_CUT_KD == ubCutAxis) || (TILING_UB_CUT_HO == ubCutAxis)) {
        baseDp = baseDo;
        baseHp = (baseHo - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh;
        baseWp = (baseWo - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    } else if ((TILING_UB_CUT_KH == ubCutAxis) || (TILING_UB_CUT_WO == ubCutAxis)) {
        baseDp = baseDo;
        baseHp = baseHo;
        baseWp = (baseWo - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    } else if ((TILING_UB_CUT_KW == ubCutAxis)) {
        baseDp = baseDo;
        baseHp = baseHo;
        baseWp = baseWo;
    }
}

uint64_t MaxPool3DGradWithArgmaxCutKTiling::CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo,
                                                           const uint32_t ubCutAxis)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t baseDp, baseHp, baseWp;
    CalXp(baseDo, baseHo, baseWo, baseDp, baseHp, baseWp, ubCutAxis);

    uint64_t dpHpWp = baseDp * baseHp * baseWp;
    uint64_t doHoWo = baseDo * baseHo * baseWo;
    uint64_t b8BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B8;
    uint64_t b32BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B32;
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize;
    uint64_t wpB32Align = vectorutil::CeilDiv(baseWp, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t dpHpWpB32Align = baseDp * baseHp * wpB32Align;
    if (TILING_UB_CUT_KW == ubCutAxis) {
        dpHpWpB32Align = dpHpWp * b32BlockAlignNum;
    }
    uint64_t doHoWoB32Align = vectorutil::CeilDiv(doHoWo, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t doHoWoDtypeAlign = vectorutil::CeilDiv(doHoWo, dtypeBlockAlignNum) * dtypeBlockAlignNum;

    uint64_t cutKDpHpWpTensorSize =
        1 * vl * dpHpWp * DTYPE_LEN_B32 +                            // indexImgBuf
        3 * vl * dpHpWpB32Align * DTYPE_LEN_B32;                     // yQue&yTransposeBuf&yTranDepadBuf(all fp32)
    uint64_t cutKDoHoWoTensorSize =
        vectorutil::CeilDiv(1 * vl * doHoWo, b8BlockAlignNum) * b8BlockAlignNum * DTYPE_LEN_B8 +    // mask
        2 * vl * doHoWo * DTYPE_LEN_B32 +                            // tmpGradBuf(fp32)&indexColBuf
        2 * vl * doHoWoB32Align * DTYPE_LEN_B32 +                    // argmaxQue&argmaxTransposeBuf
        2 * vl * doHoWoDtypeAlign * maxPoolGradParams.xDtypeSize;    // gradQue&gradTransposeBuf
    return cutKDpHpWpTensorSize + cutKDoHoWoTensorSize + SELECT_RESERVED_UB_SIZE;
}

uint64_t MaxPool3DGradWithArgmaxCutKTiling::CalCloseBaseSize(uint64_t notCutOuterDimsMul, uint64_t notCutProcessOneDim,
                                                             uint64_t wiDim, uint64_t kCutDim, uint64_t sCutDim,
                                                             const uint32_t ubCutAxis)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t b8BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B8;
    uint64_t b32BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B32;
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize;
    uint64_t wiDimB32Align = vectorutil::CeilDiv(wiDim, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t notCutProcessDimsMul = notCutProcessOneDim * wiDim;
    uint64_t notCutProcessDimsMulB32Align = notCutProcessOneDim * wiDimB32Align;
    if (TILING_UB_CUT_KW == ubCutAxis) {
        notCutProcessDimsMulB32Align = notCutProcessDimsMul * b32BlockAlignNum;
    }
    uint64_t notCutOuterDimsMulB32Align =
        vectorutil::CeilDiv(notCutOuterDimsMul, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t notCutOuterDimsMulDtypeAlign =
        vectorutil::CeilDiv(notCutOuterDimsMul, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t cutKDpHpWpTensorSizeWithoutXo =
        1 * vl * notCutProcessDimsMul * (kCutDim - sCutDim) * DTYPE_LEN_B32 +        // indexImgBuf
        3 * vl * notCutProcessDimsMulB32Align * (kCutDim - sCutDim) * DTYPE_LEN_B32; // yQue&yTransposeBuf&yTranDepadBuf
    uint64_t cutKDiHiWiTensorSizePreXo =
        1 * vl * notCutProcessDimsMul * sCutDim * DTYPE_LEN_B32 +                    // indexImgBuf
        3 * vl * notCutProcessDimsMulB32Align * sCutDim * DTYPE_LEN_B32 +            // yQue&yTransposeBuf&yTranDepadBuf
        vectorutil::CeilDiv(1 * vl * notCutOuterDimsMul, b8BlockAlignNum) * b8BlockAlignNum * DTYPE_LEN_B8 +    // mask
        2 * vl * notCutOuterDimsMul * DTYPE_LEN_B32 +                                // tmpGradBuf(fp32)&indexColBuf
        2 * vl * notCutOuterDimsMulB32Align * DTYPE_LEN_B32 +                        // argmaxQue&argmaxTransposeBuf
        2 * vl * notCutOuterDimsMulDtypeAlign * maxPoolGradParams.xDtypeSize;        // gradQue&gradTransposeBuf
    if (maxPoolGradParams.maxUbSize < (SELECT_RESERVED_UB_SIZE + cutKDpHpWpTensorSizeWithoutXo)) {
        return 1;
    }
    uint64_t dimXo = (maxPoolGradParams.maxUbSize - SELECT_RESERVED_UB_SIZE - cutKDpHpWpTensorSizeWithoutXo) /
                     cutKDiHiWiTensorSizePreXo;
    if (dimXo >= dtypeBlockAlignNum) {
        return dimXo / dtypeBlockAlignNum * dtypeBlockAlignNum;
    } else {
        return 1;
    }
}

bool MaxPool3DGradWithArgmaxCutKTiling::SetCutKParamsNotCutUB(const uint32_t ubCutAxis)
{
    uint64_t noCutSize = CalUBTotalSize(maxPoolGradParams.singleCoreDo,
                                        maxPoolGradParams.singleCoreHo,
                                        maxPoolGradParams.singleCoreWo, ubCutAxis);
    if (noCutSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = maxPoolGradParams.singleCoreDo;
        maxPoolGradParams.baseHo = maxPoolGradParams.singleCoreHo;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        maxPoolGradParams.ubCutAxis = ubCutAxis;
        return true;
    }
    return false;
}

bool MaxPool3DGradWithArgmaxCutKTiling::SetCutKParamsCutUB()
{
    // 1. Cut do&kd&ho
    uint64_t perHoSize = CalUBTotalSize(1, 1, maxPoolGradParams.singleCoreWo, TILING_UB_CUT_HO);
    if (perHoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = 1;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        uint64_t baseDp = 1;
        uint64_t baseWp = (maxPoolGradParams.baseWo - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
        maxPoolGradParams.baseHo = CalCloseBaseSize(maxPoolGradParams.baseDo * maxPoolGradParams.baseWo,
            baseDp, baseWp, maxPoolGradParams.kh, maxPoolGradParams.sh, TILING_UB_CUT_HO);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_HO;
        return true;
    }

    return false;
}

bool MaxPool3DGradWithArgmaxCutKTiling::SetCutKTilingParams()
{
    const uint64_t totalCoreNum = maxPoolGradParams.totalCoreNum;
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t doDim = maxPoolGradParams.doDim;
    uint64_t hoDim = maxPoolGradParams.hoDim;
    uint64_t woDim = maxPoolGradParams.woDim;
    maxPoolGradParams.singleCoreDo = doDim;
    maxPoolGradParams.singleCoreHo = hoDim;
    maxPoolGradParams.singleCoreWo = woDim;

    // CutK tiling cal begin
    // 1. Cut nc between core
    maxPoolGradParams.baseNc = vl;
    uint64_t ncCnt = vectorutil::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.baseNc);
    if (SetCutKParamsNotCutUB(TILING_UB_NO_CUT)) {
        return true;
    }
    if (ncCnt >= totalCoreNum) {
        return SetCutKParamsCutUB();
    }

    // 2. Cut nc&do between core
    uint64_t doCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt);
    if (doCntNeed <= doDim) {
        maxPoolGradParams.singleCoreHo = hoDim;
        maxPoolGradParams.singleCoreWo = woDim;
        // 2.1  No dim overlap
        if (!maxPoolGradParams.isOverLap && (0 != doCntNeed)) {
            uint64_t singleCoreDoMax = doDim / doCntNeed;
            maxPoolGradParams.singleCoreDo = singleCoreDoMax;
            return SetCutKParamsCutUB();
        }
        // 2.2 Cut nc
        maxPoolGradParams.singleCoreDo = doDim;
        return SetCutKParamsCutUB();
    }
    maxPoolGradParams.singleCoreDo = 1;
    uint64_t doCnt = vectorutil::CeilDiv(doDim, maxPoolGradParams.singleCoreDo);

    // 3. Cut nc&do&ho between core
    uint64_t hoCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt * doCnt);    // Need bigger than this
    if (SetCutKParamsNotCutUB(TILING_UB_CUT_DO)) {
        return true;
    }
    if (hoCntNeed <= hoDim) {
        maxPoolGradParams.singleCoreWo = woDim;
        // 3.1  No dim overlap
        if (!maxPoolGradParams.isOverLap && (0 != hoCntNeed)) {
            uint64_t singleCoreHoMax = hoDim / hoCntNeed;
            maxPoolGradParams.singleCoreHo = singleCoreHoMax;
            return SetCutKParamsCutUB();
        }
        // 3.2 Cut nc
        maxPoolGradParams.singleCoreHo = hoDim;
        return SetCutKParamsCutUB();
    }
    maxPoolGradParams.singleCoreHo = 1;
    uint64_t hoCnt = hoDim;

    // 4. Cut nc&do&ho&wo between core
    uint64_t woCntNeed = vectorutil::CeilDiv(totalCoreNum, ncCnt * doCnt * hoCnt);     // Need bigger than this
    if (SetCutKParamsNotCutUB(TILING_UB_CUT_HO)) {
        return true;
    }
    if (!maxPoolGradParams.isOverLap) {
        // 4.1  No dim overlap
        if (woCntNeed <= woDim && (0 != woCntNeed)) {
            maxPoolGradParams.singleCoreWo = woDim / woCntNeed;
        } else {
            maxPoolGradParams.singleCoreWo = 1;
        }
        return SetCutKParamsCutUB();
    }

    return false;
}

void MaxPool3DGradWithArgmaxCutKTiling::SetOtherTilingParams()
{
    SetCntTailTilingParams();
    if (maxPoolGradParams.isOverLap) {
        maxPoolGradParams.usedCoreNum = std::min(maxPoolGradParams.ncCnt, maxPoolGradParams.totalCoreNum);
    } else {
        maxPoolGradParams.usedCoreNum = std::min(maxPoolGradParams.totalCnt, maxPoolGradParams.totalCoreNum);
    }
    if (maxPoolGradParams.xDtypeSize != DTYPE_LEN_B32 && maxPoolGradParams.isOverLap) {
        maxPoolGradParams.workspaceSize = maxPoolGradParams.ncDim * maxPoolGradParams.diDim * maxPoolGradParams.hiDim *
            maxPoolGradParams.wiDim * sizeof(float);
    } else {
        maxPoolGradParams.workspaceSize = 0;
    }
}

ge::graphStatus MaxPool3DGradWithArgmaxCutKTiling::DoOpTiling()
{
    bool res = SetCutKTilingParams();
    OP_TILING_CHECK(!res, OP_LOGE(context_->GetNodeName(), "CutK cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_CUTK;
    SetOtherTilingParams();
    SetBaseTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxCutKTiling, 0);
} // namespace optiling
