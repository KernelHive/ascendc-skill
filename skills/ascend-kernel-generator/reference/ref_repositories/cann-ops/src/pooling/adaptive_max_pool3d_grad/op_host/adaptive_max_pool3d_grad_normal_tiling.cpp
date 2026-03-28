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
 * \file adaptive_max_pool3d_grad_normal_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "adaptive_max_pool3d_grad_tiling.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
constexpr int FACTOR_TWO = 2;
constexpr int FACTOR_SEVEN = 7;

namespace optiling {
ge::graphStatus AdaptiveMaxPool3DGradNormalTiling::GetShapeAttrsInfo()
{
    auto ret = AdaptiveMaxPool3DGradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3DGradNormalTiling::IsCapable()
{
    uint64_t normalTensorSizeMin = CalUBTotalSize(1, 1, 1);
    if (normalTensorSizeMin <= maxPoolGradParams.maxUbSize) {
        return true;
    }
    return false;
}

uint64_t AdaptiveMaxPool3DGradNormalTiling::CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t doHoWo = baseDo * baseHo * baseWo;
    uint64_t b32BlockAlignNum = BLOCK_SIZE / DTYPE_LEN_B32;  
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize; 
    uint64_t doHoWoB32Align = CeilDiv(doHoWo, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t doHoWoDtypeAlign = CeilDiv(doHoWo, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t maxkWB32Align = CeilDiv(maxPoolGradParams.kwMax, b32BlockAlignNum) * b32BlockAlignNum; 
    uint64_t maxkWAlignDtype = CeilDiv(maxPoolGradParams.kwMax, dtypeBlockAlignNum) * dtypeBlockAlignNum; 
    uint64_t vlMaskNumAlign = CeilDiv(vl / 8, BLOCK_SIZE) * BLOCK_SIZE;

    uint64_t normalDiHiWiTensorSize = 0;
    uint64_t normalDoHoWoTensorSize = FACTOR_TWO * vl * doHoWoDtypeAlign *  maxPoolGradParams.xDtypeSize;    // gradQue&gradTransposeBuf
    if (maxPoolGradParams.isOverLap) {
        normalDiHiWiTensorSize = FACTOR_TWO * vl * maxPoolGradParams.kdMax * maxPoolGradParams.khMax * 
                                 maxkWB32Align * DTYPE_LEN_B32;    // yQue&yTransposeBuf
    } else {
        normalDiHiWiTensorSize = FACTOR_TWO * vl * maxPoolGradParams.kdMax * maxPoolGradParams.khMax * 
                                 maxkWAlignDtype * maxPoolGradParams.xDtypeSize;    // yQue&yTransposeBuf
    }

    normalDoHoWoTensorSize += FACTOR_SEVEN * vl * doHoWoB32Align * DTYPE_LEN_B32;     // indicesQue&indicesTransposeBuf&indicesFloatBuf
                                                                           // indicesDBuf&indicesHBuf&indicesWBuf
                                                                           // tempBuf
    normalDoHoWoTensorSize += FACTOR_TWO * vl * maxPoolGradParams.kdMax * maxPoolGradParams.khMax * 
                              maxPoolGradParams.kwMax * DTYPE_LEN_B32;   // kernelIdxBuf&tempGradBuf
    
    normalDoHoWoTensorSize += vlMaskNumAlign * maxPoolGradParams.kdMax * maxPoolGradParams.khMax * 
                              maxPoolGradParams.kwMax * DTYPE_LEN_B8;       // maskBuf

    return normalDiHiWiTensorSize + normalDoHoWoTensorSize + SELECT_RESERVED_UB_SIZE;
}

uint64_t AdaptiveMaxPool3DGradNormalTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / FACTOR_TWO;
        tmpTotalSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, baseXoMid);
        if (tmpTotalSize <= maxPoolGradParams.maxUbSize) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

bool AdaptiveMaxPool3DGradNormalTiling::SetNormalParamsUB()
{
    uint64_t noCutSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, maxPoolGradParams.singleCoreWo);
    if (noCutSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        maxPoolGradParams.ubCutAxis = TILING_UB_NO_CUT;
        return true;
    }
    // 3. Cut d&h&w
    uint64_t perWoSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, 1);
    if (perWoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseWo = CalBestBaseSize(1, maxPoolGradParams.singleCoreWo);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }
    OP_LOGE(context_->GetNodeName(), "Normal set tiling failed.");
    return false;
}

bool AdaptiveMaxPool3DGradNormalTiling::SetNormalTilingParams()
{
    const uint64_t ncDim = maxPoolGradParams.ncDim;
    const uint64_t doDim = maxPoolGradParams.doDim;
    const uint64_t hoDim = maxPoolGradParams.hoDim;
    const uint64_t woDim = maxPoolGradParams.woDim;
    const uint64_t totalCoreNum = maxPoolGradParams.totalCoreNum;
    const uint64_t vl = maxPoolGradParams.vl;
    maxPoolGradParams.singleCoreDo = doDim;
    maxPoolGradParams.singleCoreHo = hoDim;
    maxPoolGradParams.singleCoreWo = woDim;
    bool isDOverlap = (maxPoolGradParams.diDim % maxPoolGradParams.doDim) != 0;
    maxPoolGradParams.baseDo = 1;
    maxPoolGradParams.baseHo = 1;

    // Normal tiling cal begin
    // 1. Cut nc between core
    maxPoolGradParams.singleCoreNc = vl;
    maxPoolGradParams.baseNc = vl <= ncDim ? vl : ncDim;
    uint64_t ncCnt = CeilDiv(ncDim, maxPoolGradParams.singleCoreNc);
    if (ncCnt >= totalCoreNum) {
        return SetNormalParamsUB();
    }
    // 2. Cut nc&do between core
    uint64_t doCntNeed = CeilDiv(totalCoreNum, ncCnt);            
    // 2.1 Dim no overlap
    if (!isDOverlap && (0 != doCntNeed)) {
        uint64_t singleCoreDo = doDim / doCntNeed;
        maxPoolGradParams.singleCoreDo = singleCoreDo < 1 ? 1 : singleCoreDo;
    }

    // 2.2 D dim overlap, can cut AdaptiveMaxPool3DGrad count of d Dim between core
    if (isDOverlap && (0 != doCntNeed)) {
        maxPoolGradParams.singleCoreDo = doDim / maxPoolGradParams.dGcd;
    }
    return SetNormalParamsUB();
}

void AdaptiveMaxPool3DGradNormalTiling::SetOtherTilingParams()
{
    maxPoolGradParams.ncCnt = CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.singleCoreNc);
    maxPoolGradParams.doCnt = CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.singleCoreDo);
    maxPoolGradParams.hoCnt = CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.singleCoreHo);
    maxPoolGradParams.woCnt = CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.singleCoreWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1) * maxPoolGradParams.baseNc;  
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

void AdaptiveMaxPool3DGradNormalTiling::SetNormalTilingData()
{
    tilingData.set_singleCoreNc(maxPoolGradParams.singleCoreNc);
    tilingData.set_singleCoreDo(maxPoolGradParams.singleCoreDo);
    tilingData.set_singleCoreHo(maxPoolGradParams.singleCoreHo);
    tilingData.set_singleCoreWo(maxPoolGradParams.singleCoreWo);
}

void AdaptiveMaxPool3DGradNormalTiling::PrintNormalTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "TilingData singleCoreNc: %lu, singleCoreDo: %lu, singleCoreHo: %lu, singleCoreWo: %lu.",
        tilingData.get_singleCoreNc(), tilingData.get_singleCoreDo(),
        tilingData.get_singleCoreHo(), tilingData.get_singleCoreWo());
}

ge::graphStatus AdaptiveMaxPool3DGradNormalTiling::DoOpTiling()
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

REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3DGrad", AdaptiveMaxPool3DGradNormalTiling, 0);
} // namespace optiling
