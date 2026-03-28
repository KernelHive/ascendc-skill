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
 * \file conv3d_tiling_algorithm_hw_mode.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_tiling_algorithm_hw_mode.h"

using namespace std;

namespace conv3d_tiling {

void Conv3dTilingAlgorithmHwMode::InitPingPong()
{
    this->doubleBufferValue.pbAL1 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbAL1 = (CalcL1SizeForL0Tiling(CONST_HO_1, tilingIns_->cubeInfo.m0, tilingIns_->cubeInfo.n0)
                                     <= tilingIns_->platformInfo.l1Size) ? DOUBLE_BUFFER_NUM : 1;
    this->doubleBufferValue.pbBL1 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbAL0 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbBL0 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbCL0 = 1;
}

void Conv3dTilingAlgorithmHwMode::GetL0TilingRange()
{
    // Get nL0 range
    uint64_t nL0Max = min(tilingIns_->platformInfo.l0BSize / (tilingIns_->cubeInfo.k0 * this->doubleBufferValue.pbBL0 *
                                                                weightDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.m0 * this->doubleBufferValue.pbCL0 *
                                                                g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeCalc.singleCo1, nL0Max / tilingIns_->cubeInfo.n0,
                                 l0TilingRange.nL0Range);
    VectorElementMultip(l0TilingRange.nL0Range, tilingIns_->cubeInfo.n0);
 
    // Get hoL0 range
    l0TilingRange.hoL0Range = {CONST_HO_1};
    // Get woL0 range
    uint64_t woL0Max = min(tilingIns_->platformInfo.l0ASize / (tilingIns_->cubeInfo.k0 * this->doubleBufferValue.pbAL0 *
                          fMapDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.n0 * this->doubleBufferValue.pbCL0 *
                          g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType)));
    woL0Max = min(woL0Max, static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo));
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeInfo.singleWo / tilingIns_->cubeInfo.m0,
                                 woL0Max / tilingIns_->cubeInfo.m0, l0TilingRange.woL0Range);
    if (l0TilingRange.woL0Range.empty()) {
        l0TilingRange.woL0Range.push_back(1);
    }
    VectorElementMultip(l0TilingRange.woL0Range, tilingIns_->cubeInfo.m0);
}

void Conv3dTilingAlgorithmHwMode::L0TilingDecision()
{
    l0TilingIdx.woL0Idx = 0;
    l0TilingIdx.nL0Idx = 0;
 
    l0TilingParams.kL0 = 1 * MKN_VALUE_DEFAULT;
    l0TilingParams.woL0 = l0TilingRange.woL0Range[l0TilingIdx.woL0Idx];
    l0TilingParams.hoL0 = l0TilingRange.hoL0Range[0];
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
    l0TilingParams.orgCoAlignN0 = AlignB(tilingIns_->shapeInfo.orgCo, tilingIns_->cubeInfo.n0);
 
    bool updateWoL0Index = false;
    bool l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.woL0 * l0TilingParams.hoL0,
                                                 l0TilingParams.kL0, l0TilingParams.nL0);
    uint64_t usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.hoL0, l0TilingParams.woL0, l0TilingParams.nL0);
    uint64_t usedBTSize = CalcBTSize(l0TilingParams.nL0);
 
    uint64_t woL0RangeLen = l0TilingRange.woL0Range.size();
    uint64_t nL0RangeLen = l0TilingRange.nL0Range.size();
 
    while (l0BufferNotOverflowFlag && usedL1Size <= tilingIns_->platformInfo.l1Size &&
           usedBTSize <= tilingIns_->platformInfo.btSize) {
        if (l0TilingParams.woL0 <= l0TilingParams.nL0 || l0TilingIdx.nL0Idx == (nL0RangeLen - 1)) {
            updateWoL0Index = true;
        } else {
            updateWoL0Index = false;
        }
 
        if (updateWoL0Index) {
            ++l0TilingIdx.woL0Idx;
        } else {
            ++l0TilingIdx.nL0Idx;
        }
 
        if (l0TilingIdx.woL0Idx < woL0RangeLen && l0TilingIdx.nL0Idx < nL0RangeLen) {
            l0TilingParams.woL0 = l0TilingRange.woL0Range[l0TilingIdx.woL0Idx];
            l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
            l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.woL0 * l0TilingParams.hoL0,
                                                    l0TilingParams.kL0, l0TilingParams.nL0);
            usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.hoL0, l0TilingParams.woL0, l0TilingParams.nL0);
            usedBTSize = CalcBTSize(l0TilingParams.nL0);
        } else {
            break;
        }
    }
    if (updateWoL0Index) {
        l0TilingIdx.woL0Idx = l0TilingIdx.woL0Idx == 0 ? 0 : l0TilingIdx.woL0Idx - 1;
    } else {
        l0TilingIdx.nL0Idx = l0TilingIdx.nL0Idx == 0 ? 0 : l0TilingIdx.nL0Idx - 1;
    }
    
    l0TilingParams.woL0 = l0TilingRange.woL0Range[l0TilingIdx.woL0Idx];
    l0TilingParams.mL0 = l0TilingParams.woL0 * l0TilingParams.hoL0;
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
    tilingIns_->l0TilingInfo.mL0 = l0TilingParams.woL0;
    tilingIns_->l0TilingInfo.nL0 = l0TilingParams.nL0;
    tilingIns_->l0TilingInfo.nL0xk0 = l0TilingParams.nL0 * tilingIns_->cubeInfo.k0;
}

uint64_t Conv3dTilingAlgorithmHwMode::CalcL1SizeForL0Tiling(uint64_t currhoL0, uint64_t currwoL0, uint64_t currnL0) const
{
    // Calculate AL1 size
    uint64_t hiAL1Min = InferHiL1(currhoL0, tilingIns_->shapeInfo.orgHi);
    uint64_t wiAL1Min = InferWiL1(currwoL0, tilingIns_->shapeInfo.orgWi);
    uint64_t usedL1Size = hiAL1Min * wiAL1Min * tilingIns_->cubeInfo.k0 * fMapDTypeSize * this->doubleBufferValue.pbAL1;
    if (tilingIns_->hasBias) {
        uint64_t biasSize = currnL0 * biasDTypeSize;
        usedL1Size += biasSize;
    }
    return usedL1Size;
}

int64_t Conv3dTilingAlgorithmHwMode::InitCalcL1ParamsForFmap()
{
    // cal fmap full load in L1 size
    uint64_t hiL1FullLoad = InferHiL1(tilingIns_->shapeInfo.singleHo, tilingIns_->shapeInfo.orgHi);
    uint64_t wiL1FullLoad = InferWiL1(tilingIns_->shapeInfo.singleWo, tilingIns_->shapeInfo.orgWi);
    this->l1TilingCalc.fmapFullLoadL1Size = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        hiL1FullLoad * wiL1FullLoad * tilingIns_->cubeInfo.k0 * this->fMapDTypeSize;
    // cal min/kfullload fmap size in L1(mL0)
    uint64_t hiL1MinLoad = InferHiL1(this->l0TilingParams.hoL0, tilingIns_->shapeInfo.orgHi);
    uint64_t wiL1MinLoad = InferWiL1(this->l0TilingParams.woL0, tilingIns_->shapeInfo.orgWi);
    this->l1TilingCalc.fmapMinLoadL1Size = tilingIns_->cubeInfo.k0 * hiL1MinLoad * wiL1MinLoad *
        this->fMapDTypeSize * this->doubleBufferValue.pbAL1;
    this->l1TilingCalc.fmapKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.fmapMinLoadL1Size;

    return 0;
}

void Conv3dTilingAlgorithmHwMode::GetL1TilingRangeForM()
{
    // cal woAL1Value
    uint64_t multiWoAL1Max = tilingIns_->shapeInfo.singleWo / this->l0TilingParams.woL0;
    CalcCommFactor(multiWoAL1Max, multiWoAL1Max, this->l1TilingRange.woAL1ValueRange);
    if (this->l1TilingRange.woAL1ValueRange.empty()) {
        this->l1TilingRange.woAL1ValueRange.push_back(1);
    }
    VectorElementMultip(this->l1TilingRange.woAL1ValueRange, l0TilingParams.woL0);
    // cal hoAL1Value
    this->l1TilingRange.hoAL1ValueRange = {CONST_HO_1};
}

void Conv3dTilingAlgorithmHwMode::InitL1TiLingMap()
{
    this->l1TilingInitMap[L1TilingMode::NONE_FULL_LOAD].SetIdx(0, 0, 0, 0, 0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_AL1].SetIdx(this->l1TilingRange.kAL1Range.size() - 1, 0,
                                                              this->l1TilingRange.hoAL1ValueRange.size() - 1,
                                                              this->l1TilingRange.woAL1ValueRange.size() - 1,
                                                              0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_BL1].SetIdx(0, this->l1TilingRange.kBL1Range.size() - 1,
                                                              0, 0,
                                                              this->l1TilingRange.nBL1ValueRange.size() - 1);
    this->l1TilingInitMap[L1TilingMode::ALL_FULL_LOAD].SetIdx(this->l1TilingRange.kAL1Range.size() - 1,
                                                              this->l1TilingRange.kBL1Range.size() - 1,
                                                              this->l1TilingRange.hoAL1ValueRange.size() - 1,
                                                              this->l1TilingRange.woAL1ValueRange.size() - 1,
                                                              this->l1TilingRange.nBL1ValueRange.size() - 1);
}

void Conv3dTilingAlgorithmHwMode::InitABL1TilingMode()
{
    // init L1 fmap weight full load case and init mode
    if (tilingIns_->shapeInfo.singleWo % tilingIns_->cubeInfo.m0 != 0) {
        if (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->doubleBufferValue.pbBL1 = 1;
        } else {
            this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
        }
        return;
    }
    Conv3dTilingAlgorithm::InitABL1TilingMode();
}

int64_t Conv3dTilingAlgorithmHwMode::ProcessAllL1FullLoad()
{
    // when fmap and weight can both full load, it can set res and return directly
    this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.kBL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.woAL1Value = tilingIns_->shapeInfo.singleWo / tilingIns_->cubeInfo.m0 *
                                        tilingIns_->cubeInfo.m0;
    this->l1TilingParams.nBL1Value = tilingIns_->cubeInfo.n0 * tilingIns_->shapeCalc.singleCo1;
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    return 0;
}

int64_t Conv3dTilingAlgorithmHwMode::ProcessFmapL1FullLoad()
{
    // when only fmap full load in L1, nfirset and iter kbl1 then nbl1
    this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
                                this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.woAL1Value = tilingIns_->shapeInfo.singleWo / tilingIns_->cubeInfo.m0 *
                                        tilingIns_->cubeInfo.m0;
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    // speical case, when min weight can not load in L1, bypass
    if (CoreL1TilingMinWeightBypass()) {
        this->l1TilingParams.kBL1 = 0;
        this->l1TilingParams.nBL1Value = 0;
        this->l1TilingFlag.isWeightBypass = true;
        return 0;
    }
    // normal case
    FmapL1FullLoadIter();
    return 0;
}

void Conv3dTilingAlgorithmHwMode::MAL1IdxIter()
{
    // iter woAL1 and hoAL1
    while (this->l1TilingIdx.woAL1Idx < this->l1TilingRange.woAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.woAL1Idx++;
    }
    this->l1TilingIdx.woAL1Idx = this->l1TilingIdx.woAL1Idx == 0 ? 0 : this->l1TilingIdx.woAL1Idx - 1;
}


uint64_t Conv3dTilingAlgorithmHwMode::CalcCurrFmapL1Size() const
{
    // check if l1 buffer is overflow in current tiling decision
    uint64_t hoL1Load = this->l1TilingRange.hoAL1ValueRange.at(this->l1TilingIdx.hoAL1Idx);
    uint64_t hiL1Load = InferHiL1(hoL1Load, tilingIns_->shapeInfo.orgHi);
    uint64_t woL1Load = this->l1TilingRange.woAL1ValueRange.at(this->l1TilingIdx.woAL1Idx);
    uint64_t wiL1Load = InferWiL1(woL1Load, tilingIns_->shapeInfo.orgWi);
    uint64_t currentFmL1Size =
        (this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) / this->l1TilingCalc.ci0HkWk) *
        hiL1Load * wiL1Load * this->fMapDTypeSize * this->doubleBufferValue.pbAL1 * tilingIns_->cubeInfo.k0;
    return currentFmL1Size;
}

int64_t Conv3dTilingAlgorithmHwMode::KABL1FullLoadIter()
{
    uint64_t fmapSingleCoreL1Load = L1NoFullLoadFmapSize();
    uint64_t weightSingleCoreL1Load = this->l1TilingCalc.weightFullLoadL1Size;
    this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
    this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
    uint64_t tmpWoAL1Idx = KABL1FullLoadIterN();
    uint64_t tmpNBL1Idx = KABL1FullLoadIterM();
    // iterN
    // fmap_mte2_size: dout*fmapSingleCoreL1Load
    // weight_mte2_size: dout*weightSingleCoreL1Load*(loop m)
    uint64_t fmapMte2Size, weightMte2Size, bothFullloadIterNMte2Size;
    uint64_t loopTime = CeilDiv(tilingIns_->shapeInfo.singleWo, this->l1TilingRange.woAL1ValueRange.at(tmpWoAL1Idx)) *
        CeilDiv(tilingIns_->shapeInfo.singleHo, this->l1TilingRange.hoAL1ValueRange.at(this->l1TilingIdx.hoAL1Idx));
    if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load,
                             static_cast<uint64_t>(tilingIns_->shapeInfo.singleDo)) ||
        MulWithOverflowCheck(weightMte2Size, weightSingleCoreL1Load, loopTime * tilingIns_->shapeInfo.singleDo) ||
        AddWithOverflowCheck(bothFullloadIterNMte2Size, fmapMte2Size, weightMte2Size)) {
        TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, KABL1FullLoadIter failed!");
        return -1;
    }
    // iterM
    // fmap_mte2_size: dout*fmapSingleCoreL1Load*(loop n)
    // weight_mte2_size: weightSingleCoreL1Load
    uint64_t bothFullloadIterMMte2Size;
    loopTime = CeilDiv(tilingIns_->shapeInfo.singleCo, this->l1TilingRange.nBL1ValueRange.at(tmpNBL1Idx));
    if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load, loopTime * tilingIns_->shapeInfo.singleDo) ||
        AddWithOverflowCheck(bothFullloadIterMMte2Size, fmapMte2Size, weightSingleCoreL1Load)) {
        TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, KABL1FullLoadIter failed!");
        return -1;
    }
    if (bothFullloadIterNMte2Size < bothFullloadIterMMte2Size) {
        this->l1TilingIdx.woAL1Idx = tmpWoAL1Idx;
        this->l1TilingIdx.nBL1Idx = KABL1FullLoadIterM();
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    } else {
        this->l1TilingIdx.nBL1Idx = tmpNBL1Idx;
        this->l1TilingIdx.woAL1Idx = KABL1FullLoadIterN();
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    }
    return 0;
}

uint64_t Conv3dTilingAlgorithmHwMode::KABL1FullLoadIterN()
{
    // iter woAL1
    uint64_t startIdxBk = this->l1TilingIdx.woAL1Idx;
    uint64_t tmpWoAL1Idx = 0;
    while (this->l1TilingIdx.woAL1Idx < this->l1TilingRange.woAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.woAL1Idx++;
    }
    this->l1TilingIdx.woAL1Idx = this->l1TilingIdx.woAL1Idx == 0 ? 0 : this->l1TilingIdx.woAL1Idx - 1;
    tmpWoAL1Idx = this->l1TilingIdx.woAL1Idx;
    this->l1TilingIdx.woAL1Idx = startIdxBk;
    return tmpWoAL1Idx;
}

void Conv3dTilingAlgorithmHwMode::SetMAL1NBL1ValueAndMode()
{
    this->l1TilingParams.nBL1Value = this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx);
    this->l1TilingParams.woAL1Value = this->l1TilingRange.woAL1ValueRange.at(this->l1TilingIdx.woAL1Idx);
    this->l1TilingParams.hoAL1Value = this->l1TilingRange.hoAL1ValueRange.at(this->l1TilingIdx.hoAL1Idx);
    tilingIns_->l1TilingInfo.al1FullLoad = false;
    tilingIns_->l1TilingInfo.bl1FullLoad = false;
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->l1TilingParams.woAL1Value = tilingIns_->shapeInfo.singleWo / tilingIns_->cubeInfo.m0 *
                                          tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0;
        this->l1TilingParams.woAL1Value = tilingIns_->shapeInfo.singleWo / tilingIns_->cubeInfo.m0 *
                                          tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    }
    this->l1TilingParams.mAL1Value = this->l1TilingParams.woAL1Value;
}

} // namespace conv3d_tiling
