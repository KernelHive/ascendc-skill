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
 * \file conv3d_tiling_algorithm.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_tiling_algorithm.h"

using namespace std;

namespace conv3d_tiling {
Conv3dTilingAlgorithm::Conv3dTilingAlgorithm(Conv3dTilingBase* tilingIns)
{
    if (tilingIns == nullptr) {
        TILING_WARNING_LOG("tiling instance is null.");
    }
    this->tilingIns_ = tilingIns;
    this->fMapDTypeSize = g_dtypeSizeTab.at(tilingIns_->descInfo.fMapType.dtype);
    this->weightDTypeSize = g_dtypeSizeTab.at(tilingIns_->descInfo.weightType.dtype);
    this->biasDTypeSize = g_dtypeSizeTab.at(tilingIns_->descInfo.biasType.dtype);
    this->ubInDTypeSize = g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType);
    this->scaleDTypeSize = g_dtypeSizeTab.at(tilingIns_->descInfo.quantScaleType.dtype);
    this->outputDTypeSize = g_dtypeSizeTab.at(tilingIns_->descInfo.outputType.dtype);
}

// decide tiling
int64_t Conv3dTilingAlgorithm::Process()
{
    InitPingPong();
    GetL0Tiling();
    if (GetL1Tiling() == -1) {
        return -1;
    }
    GetVecTiling();
    // set pb res
    SetPBufferFlag();
    return 0;
}

int64_t Conv3dTilingAlgorithm::GetL1Tiling()
{
    if (PreProcessingL1Tiling() == -1) {
        return -1;
    }
    // tiling decision
    if (CoreL1TilingDecision() == -1) {
        return -1;
    }
    // bias load check
    BiasL1TilingDecision();
    // get kl0 tiling decision
    GetKL0TilingDecision();
    // get if weight can by pass in L1
    WeightBypassDecision();
    SetL1TilingRes();

    return 0;
}

int64_t Conv3dTilingAlgorithm::PreProcessingL1Tiling()
{
    // cal some common variables
    if (InitCalcL1Params() == -1) {
        return -1;
    }
    // get kL1，mL1 and nL1 range value
    GetL1TilingRange();
    // cal full load condition and init range start idx
    InitL1TiLing();
    return 0;
}

void Conv3dTilingAlgorithm::SetPBufferFlag()
{
    tilingIns_->dbValue.pBufferFlag = 0;
    tilingIns_->dbValue.pbBL1 = this->doubleBufferValue.pbBL1;
    tilingIns_->dbValue.pbAL1 = this->doubleBufferValue.pbAL1;
    tilingIns_->dbValue.pbCL0 = this->doubleBufferValue.pbCL0;
    tilingIns_->dbValue.pbBL0 = this->doubleBufferValue.pbBL0;
    tilingIns_->dbValue.pbAL0 = this->doubleBufferValue.pbAL0;
    tilingIns_->dbValue.pbUB = this->doubleBufferValue.pbUB;
    tilingIns_->dbValue.pBufferFlag = tilingIns_->dbValue.pBufferFlag |
                                      (tilingIns_->dbValue.pbUB == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbBL1 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbAL1 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbCL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbBL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbAL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    TILING_DEBUG_LOG("pBufferFlag: %ld, pbAL0: %d, pbBL0: %d, pbCL0: %d, pbAL1: %d, pbBL1: %d, pbUB: %d,",
                     tilingIns_->dbValue.pBufferFlag, tilingIns_->dbValue.pbAL0, tilingIns_->dbValue.pbBL0,
                     tilingIns_->dbValue.pbCL0, tilingIns_->dbValue.pbAL1, tilingIns_->dbValue.pbBL1,
                     tilingIns_->dbValue.pbUB);
}

bool Conv3dTilingAlgorithm::CheckL1Buffer() const
{
    // check if l1 buffer is overflow in current tiling decision
    uint64_t currentFmL1Size = CalcCurrFmapL1Size();
    uint64_t currentWeightL1Size =
        this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx) * this->doubleBufferValue.pbBL1 *
        this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx) * this->weightDTypeSize;
    uint64_t currentBiasL1Size = tilingIns_->hasBias ?
        (this->l1TilingFlag.isBiasFullLoad ? tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 *
        this->biasDTypeSize : this->l0TilingParams.nL0 * this->biasDTypeSize) : 0;
    // cal current fm in L1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    }

    if (this->l1TilingFlag.isWeightBypass) {
        currentWeightL1Size = 0;
    }

    uint64_t l1SizeCur = currentFmL1Size + currentWeightL1Size + currentBiasL1Size;

    return l1SizeCur <= tilingIns_->platformInfo.l1Size;
}

uint64_t Conv3dTilingAlgorithm::CalcCurrFmapL1Size() const
{
    uint64_t hoL1Load = (this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx) /
        tilingIns_->shapeCalc.orgWo) + 2;
    uint64_t hiL1Load = InferHiL1(hoL1Load, tilingIns_->shapeInfo.orgHi);
    uint64_t currentFmL1Size =
        (this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) / this->l1TilingCalc.ci0HkWk) *
        hiL1Load * tilingIns_->shapeInfo.orgWi * this->fMapDTypeSize * this->doubleBufferValue.pbAL1 *
        tilingIns_->cubeInfo.k0;
    return currentFmL1Size;
}

int64_t Conv3dTilingAlgorithm::InitCalcL1Params()
{
    this->l1TilingCalc.ci0HkWk = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW *
        tilingIns_->cubeInfo.k0;
    this->l1TilingCalc.alignCinKhKwKd = AlignB(tilingIns_->shapeInfo.singleCi, tilingIns_->cubeInfo.k0) *
        tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * tilingIns_->shapeInfo.orgkD;
    if (InitCalcL1ParamsForFmap() == -1 || InitCalcL1ParamsForWeight() == -1) {
        return -1;
    };
    // cal bias size in L1
    this->l1TilingCalc.biasMinLoadL1Size = tilingIns_->hasBias ? this->l0TilingParams.nL0 * this->biasDTypeSize : 0;

    return 0;
}

int64_t Conv3dTilingAlgorithm::InitCalcL1ParamsForFmap()
{
    // cal fmap full load in L1 size
    uint64_t hoL1FullLoad = (tilingIns_->shapeInfo.singleM / tilingIns_->shapeCalc.orgWo) + 2;
    uint64_t hiL1FullLoad = InferHiL1(hoL1FullLoad, tilingIns_->shapeInfo.orgHi);
    this->l1TilingCalc.fmapFullLoadL1Size = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        hiL1FullLoad * tilingIns_->shapeInfo.orgWi * tilingIns_->cubeInfo.k0 * this->fMapDTypeSize;
    // cal min/kfullload fmap size in L1(mL0)
    uint64_t hoL1MinLoad = (this->l0TilingParams.mL0 / tilingIns_->shapeCalc.orgWo) + 2;
    uint64_t hiL1MinLoad = InferHiL1(hoL1MinLoad, tilingIns_->shapeInfo.orgHi);
    this->l1TilingCalc.fmapMinLoadL1Size = tilingIns_->cubeInfo.k0 * hiL1MinLoad * tilingIns_->shapeInfo.orgWi *
        this->fMapDTypeSize * this->doubleBufferValue.pbAL1;
    this->l1TilingCalc.fmapKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        tilingIns_->cubeInfo.k0 * hiL1MinLoad * tilingIns_->shapeInfo.orgWi * this->doubleBufferValue.pbAL1 *
        this->fMapDTypeSize;

    return 0;
}

int64_t Conv3dTilingAlgorithm::InitCalcL1ParamsForWeight()
{
    // cal weight full load in L1 size
    this->l1TilingCalc.weightFullLoadL1Size = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize;
    // cal min/kfullload weight size in L1
    this->l1TilingCalc.weightMinLoadL1Size = this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 *
        this->doubleBufferValue.pbBL1 * this->weightDTypeSize;
    this->l1TilingCalc.weightKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 * this->doubleBufferValue.pbBL1 * this->weightDTypeSize;

    return 0;
}

uint64_t Conv3dTilingAlgorithm::InferHiL1(uint64_t inputHoL1, uint64_t hi) const
{
    uint64_t khDilated = (tilingIns_->shapeInfo.singlekH - 1) * tilingIns_->attrInfo.dilationH + 1;
    uint64_t tmpHiL1 = (inputHoL1 - 1) * tilingIns_->attrInfo.strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

uint64_t Conv3dTilingAlgorithm::InferWiL1(uint64_t inputWoL1, uint64_t wi) const
{
    uint64_t kwDilated = (tilingIns_->shapeInfo.singlekW - 1) * tilingIns_->attrInfo.dilationW + 1;
    uint64_t tmpWiL1 = (inputWoL1 - 1) * tilingIns_->attrInfo.strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

void Conv3dTilingAlgorithm::GetL1TilingRange()
{
    // cal kAL1 and kBL1 range, cin1 factor or (1~dk) cin1
    CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kAL1Range);
    CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kBL1Range);
    for (uint64_t dk = 2; dk <= static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD); ++dk) {
        this->l1TilingRange.kAL1Range.emplace_back(dk * tilingIns_->shapeCalc.singleCi1);
        this->l1TilingRange.kBL1Range.emplace_back(dk * tilingIns_->shapeCalc.singleCi1);
    }
    // kAL1 has limit of postk due to load3d instr
    const uint64_t limitKAL1 = (POSTK_LIMIT + tilingIns_->cubeInfo.k0) / this->l1TilingCalc.ci0HkWk;
    std::vector<uint64_t>::iterator up = std::upper_bound(this->l1TilingRange.kAL1Range.begin(),
                                                          this->l1TilingRange.kAL1Range.end(), limitKAL1);
    this->l1TilingRange.kAL1Range.erase(up, this->l1TilingRange.kAL1Range.end());
    this->l1TilingRange.kAL1Range.shrink_to_fit();

    VectorElementMultip(this->l1TilingRange.kAL1Range, this->l1TilingCalc.ci0HkWk);
    VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0HkWk);

    // cal mAL1Value and nBL1Value
    uint64_t multiNBL1Max = CeilDiv(tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0,
                                    this->l0TilingParams.nL0);
    CalcCommFactor(multiNBL1Max, multiNBL1Max, this->l1TilingRange.nBL1ValueRange);
    VectorElementMultip(this->l1TilingRange.nBL1ValueRange, l0TilingParams.nL0);

    GetL1TilingRangeForM();
}

void Conv3dTilingAlgorithm::GetL1TilingRangeForM()
{
    uint64_t multiMAL1Max = CeilDiv(CeilDiv(tilingIns_->shapeInfo.singleM,
                                            tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0,
                                    this->l0TilingParams.mL0);
    CalcCommFactor(multiMAL1Max, multiMAL1Max, this->l1TilingRange.mAL1ValueRange);
    VectorElementMultip(this->l1TilingRange.mAL1ValueRange, l0TilingParams.mL0);
}

void Conv3dTilingAlgorithm::InitL1TiLing()
{
    InitL1TiLingMap();
    InitABL1TilingMode();
    this->l1TilingIdx = this->l1TilingInitMap.at(this->l1TilingFlag.abL1Mode);
}

void Conv3dTilingAlgorithm::InitL1TiLingMap()
{
    this->l1TilingInitMap[L1TilingMode::NONE_FULL_LOAD].SetIdx(0, 0, 0, 0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_AL1].SetIdx(this->l1TilingRange.kAL1Range.size() - 1, 0,
                                                              this->l1TilingRange.mAL1ValueRange.size() - 1, 0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_BL1].SetIdx(0, this->l1TilingRange.kBL1Range.size() - 1,
                                                              0, this->l1TilingRange.nBL1ValueRange.size() - 1);
    this->l1TilingInitMap[L1TilingMode::ALL_FULL_LOAD].SetIdx(this->l1TilingRange.kAL1Range.size() - 1,
                                                              this->l1TilingRange.kBL1Range.size() - 1,
                                                              this->l1TilingRange.mAL1ValueRange.size() - 1,
                                                              this->l1TilingRange.nBL1ValueRange.size() - 1);
}

void Conv3dTilingAlgorithm::InitABL1TilingMode()
{
    // init L1 fmap weight full load case and init mode
    if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightFullLoadL1Size +
        this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::ALL_FULL_LOAD;
        this->doubleBufferValue.pbAL1 = 1;
        this->doubleBufferValue.pbBL1 = 1;
        return;
    }
    if (this->l1TilingCalc.fmapFullLoadL1Size <= this->l1TilingCalc.weightFullLoadL1Size) {
        if (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->doubleBufferValue.pbBL1 = 1;
            return;
        } else if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.biasMinLoadL1Size <=
            tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
            this->doubleBufferValue.pbAL1 = 1;
            return;
        }
    } else {
        if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.biasMinLoadL1Size <=
            tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
            this->doubleBufferValue.pbAL1 = 1;
            return;
        } else if (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->doubleBufferValue.pbBL1 = 1;
            return;
        }
    }
    // other case None can full load in L1 case
    this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
    return;
}

void Conv3dTilingAlgorithm::FmapL1FullLoadIter()
{
    // iter kBL1
    while (this->l1TilingIdx.kBL1Idx < this->l1TilingRange.kBL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
    if (this->l1TilingIdx.kBL1Idx != this->l1TilingRange.kBL1Range.size() - 1) {
        return;
    }
    // iter nBL1
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
}

bool Conv3dTilingAlgorithm::CoreL1TilingMinWeightBypass() const
{
    return (this->l1TilingCalc.fmapFullLoadL1Size +
            this->l1TilingCalc.weightMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size) > tilingIns_->platformInfo.l1Size;
}

int64_t Conv3dTilingAlgorithm::CoreL1TilingDecision()
{
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        return ProcessAllL1FullLoad();
    }
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        return ProcessFmapL1FullLoad();
    }
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        return ProcessWeightL1FullLoad();
    }
    return ProcessNoneL1FullLoad();
}

int64_t Conv3dTilingAlgorithm::ProcessAllL1FullLoad()
{
    // when fmap and weight can both full load, it can set res and return directly
    this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.kBL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.mAL1Value = tilingIns_->cubeInfo.m0 * tilingIns_->shapeCalc.singleM1;
    this->l1TilingParams.nBL1Value = tilingIns_->cubeInfo.n0 * tilingIns_->shapeCalc.singleCo1;
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    return 0;
}

int64_t Conv3dTilingAlgorithm::ProcessFmapL1FullLoad()
{
    // when only fmap full load in L1, nfirset and iter kbl1 then nbl1
    this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
                                this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.mAL1Value = tilingIns_->cubeInfo.m0 * tilingIns_->shapeCalc.singleM1;
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    // speical case, when min weight can not load in L1, bypass
    if (CoreL1TilingMinWeightBypass()) {
        this->l1TilingFlag.isWeightBypass = true;
        this->l1TilingParams.kBL1 = 0;
        this->l1TilingParams.nBL1Value = 0;
        return 0;
    }
    // normal case
    FmapL1FullLoadIter();
    return 0;
}

int64_t Conv3dTilingAlgorithm::ProcessWeightL1FullLoad()
{
    // when only weight full load in L1, mfirset and iter kal1 then mal1
    this->l1TilingParams.kBL1 = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    this->l1TilingParams.nBL1Value = tilingIns_->cubeInfo.n0 * tilingIns_->shapeCalc.singleCo1;
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    // normal case
    WeightL1FullLoadIter();
    return 0;
}

int64_t Conv3dTilingAlgorithm::ProcessNoneL1FullLoad()
{
    // both cannot full load, cal if K can full load in L1 and decision
    InitKL1LoadFlag();
    return L1NoFullLoadIter();
}

void Conv3dTilingAlgorithm::MAL1IdxIter()
{
    // iter mAL1
    while (this->l1TilingIdx.mAL1Idx < this->l1TilingRange.mAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.mAL1Idx++;
    }
    this->l1TilingIdx.mAL1Idx = this->l1TilingIdx.mAL1Idx == 0 ? 0 : this->l1TilingIdx.mAL1Idx - 1;
}

void Conv3dTilingAlgorithm::WeightL1FullLoadIter()
{
    // iter kAL1
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
    }
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    if (this->l1TilingIdx.kAL1Idx != this->l1TilingRange.kAL1Range.size() - 1) {
        return;
    }
    // iter mAL1
    MAL1IdxIter();
}

void Conv3dTilingAlgorithm::InitKL1LoadFlag()
{
    // check if KAL1, KBL1 can both/either/none can full load in L1
    if (this->l1TilingCalc.fmapKL1FullLoadSize + this->l1TilingCalc.weightKL1FullLoadSize +
        this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kAL1CanFullLoadFlag = true;
        this->l1TilingFlag.kBL1CanFullLoadFlag = true;
        this->l1TilingFlag.kABL1CanFullLoadFlag = true;
        return;
    }
    if (this->l1TilingCalc.fmapKL1FullLoadSize + this->l1TilingCalc.weightMinLoadL1Size +
        this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kAL1CanFullLoadFlag = true;
    }
    if (this->l1TilingCalc.fmapMinLoadL1Size + this->l1TilingCalc.weightKL1FullLoadSize +
        this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kBL1CanFullLoadFlag = true;
    }
}

void Conv3dTilingAlgorithm::KAL1FullLoadIter()
{
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    // iter kBL1
    while (this->l1TilingIdx.kBL1Idx < this->l1TilingRange.kBL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
    // iter mAL1
    MAL1IdxIter();
}

void Conv3dTilingAlgorithm::KBL1FullLoadIter()
{
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    // iter kAL1
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
    }
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    // iter nBL1
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
}

int64_t Conv3dTilingAlgorithm::KABL1FullLoadIter()
{
    uint64_t fmapSingleCoreL1Load = L1NoFullLoadFmapSize();
    uint64_t weightSingleCoreL1Load = this->l1TilingCalc.weightFullLoadL1Size;
    this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
    this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
    uint64_t tmpMAL1Idx = KABL1FullLoadIterN();
    uint64_t tmpNBL1Idx = KABL1FullLoadIterM();
    // iterN
    // fmap_mte2_size: dout*fmapSingleCoreL1Load
    // weight_mte2_size: dout*weightSingleCoreL1Load*(loop m)
    uint64_t fmapMte2Size, weightMte2Size, bothFullloadIterNMte2Size;
    uint64_t loopTime = CeilDiv(tilingIns_->shapeInfo.singleM, this->l1TilingRange.mAL1ValueRange.at(tmpMAL1Idx));
    if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load,
                             static_cast<uint64_t>(tilingIns_->shapeInfo.singleDo)) ||
        MulWithOverflowCheck(weightMte2Size, weightSingleCoreL1Load, tilingIns_->shapeInfo.singleDo * loopTime) ||
        AddWithOverflowCheck(bothFullloadIterNMte2Size, fmapMte2Size, weightMte2Size)) {
        TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, KABL1FullLoadIter failed!");
        return -1;
    }
    // iterM
    // fmap_mte2_size: dout*fmapSingleCoreL1Load*(loop n)
    // weight_mte2_size: weightSingleCoreL1Load
    loopTime = CeilDiv(tilingIns_->shapeInfo.singleCo, this->l1TilingRange.nBL1ValueRange.at(tmpNBL1Idx));
    uint64_t bothFullloadIterMMte2Size;
    if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load, tilingIns_->shapeInfo.singleDo * loopTime) ||
        AddWithOverflowCheck(bothFullloadIterMMte2Size, fmapMte2Size, weightSingleCoreL1Load)) {
        TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, KABL1FullLoadIter failed!");
        return -1;
    }
    if (bothFullloadIterNMte2Size < bothFullloadIterMMte2Size) {
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        this->l1TilingIdx.mAL1Idx = tmpMAL1Idx;
    } else {
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        this->l1TilingIdx.nBL1Idx = tmpNBL1Idx;
    }
    return 0;
}

uint64_t Conv3dTilingAlgorithm::KABL1FullLoadIterN()
{
    // iter mAL1
    uint64_t startIdxBk = this->l1TilingIdx.mAL1Idx;
    uint64_t tmpMAL1Idx = 0;
    while (this->l1TilingIdx.mAL1Idx < this->l1TilingRange.mAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.mAL1Idx++;
    }
    this->l1TilingIdx.mAL1Idx = this->l1TilingIdx.mAL1Idx == 0 ? 0 : this->l1TilingIdx.mAL1Idx - 1;
    tmpMAL1Idx = this->l1TilingIdx.mAL1Idx;
    this->l1TilingIdx.mAL1Idx = startIdxBk;
    return tmpMAL1Idx;
}

uint64_t Conv3dTilingAlgorithm::KABL1FullLoadIterM()
{
    // iter nBL1
    uint64_t startIdxBk = this->l1TilingIdx.nBL1Idx;
    uint64_t tmpNBL1Idx = 0;
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
    tmpNBL1Idx = this->l1TilingIdx.nBL1Idx;
    this->l1TilingIdx.nBL1Idx = startIdxBk;
    return tmpNBL1Idx;
}

bool Conv3dTilingAlgorithm::NoneKABL1FullLoadWeightBypass() const
{
    return (this->l1TilingCalc.fmapMinLoadL1Size +
           this->l1TilingCalc.weightMinLoadL1Size +
           this->l1TilingCalc.biasMinLoadL1Size) > tilingIns_->platformInfo.l1Size;
}

void Conv3dTilingAlgorithm::NoneKABL1FullLoadIter()
{
    if (NoneKABL1FullLoadWeightBypass()) {
        this->l1TilingFlag.isWeightBypass = true;
        this->l1TilingParams.kBL1 = 0;
        this->l1TilingParams.nBL1Value = 0;
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        return;
    }
    // iter kAL1 and kBL1 together
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
}

uint64_t Conv3dTilingAlgorithm::L1NoFullLoadFmapSize() const
{
    return this->l1TilingCalc.fmapFullLoadL1Size;
}

int64_t Conv3dTilingAlgorithm::L1NoFullLoadIter()
{
    // only kAL1 full load L1
    if (this->l1TilingFlag.kAL1CanFullLoadFlag && !this->l1TilingFlag.kBL1CanFullLoadFlag) {
        this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
        KAL1FullLoadIter();
        return 0;
    }
    // only kBL1 full load L1
    if (this->l1TilingFlag.kBL1CanFullLoadFlag && !this->l1TilingFlag.kAL1CanFullLoadFlag) {
        this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
        KBL1FullLoadIter();
        return 0;
    }
    uint64_t fmapSingleCoreL1Load = L1NoFullLoadFmapSize();
    uint64_t weightSingleCoreL1Load = this->l1TilingCalc.weightFullLoadL1Size;
    // either kAL1 or kBL1 can full load
    if ((this->l1TilingFlag.kBL1CanFullLoadFlag || this->l1TilingFlag.kAL1CanFullLoadFlag) &&
        !this->l1TilingFlag.kABL1CanFullLoadFlag) {
        // kAL1 full load
        // fmap_mte2_size: (kd*dout/din)*fmapSingleCoreDDR = dout*fmapSingleCoreL1Load
        // weight_mte2_size: dout*weightSingleCoreL1Load*(loop m)
        uint64_t fmapMte2Size, weightMte2Size, onlyKAL1FullloadMte2Size;
        uint64_t loopTime = CeilDiv(tilingIns_->shapeInfo.singleM, this->l0TilingParams.mL0);
        if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load,
                                 static_cast<uint64_t>(tilingIns_->shapeInfo.singleDo)) ||
            MulWithOverflowCheck(weightMte2Size, weightSingleCoreL1Load, tilingIns_->shapeInfo.singleDo * loopTime) ||
            AddWithOverflowCheck(onlyKAL1FullloadMte2Size, fmapMte2Size, weightMte2Size)) {
            TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, L1NoFullLoadIter failed!");
            return -1;
        }
        // kBL1 full load
        // fmap_mte2_size: dout*fmapSingleCoreL1Load*(loop n)
        // weight_mte2_size: weightSingleCoreL1Load
        loopTime = CeilDiv(tilingIns_->shapeInfo.singleCo, this->l0TilingParams.nL0);
        uint64_t onlyKBL1FullloadMte2Size;
        if (MulWithOverflowCheck(fmapMte2Size, fmapSingleCoreL1Load, tilingIns_->shapeInfo.singleDo * loopTime) ||
            AddWithOverflowCheck(onlyKBL1FullloadMte2Size, fmapMte2Size, weightSingleCoreL1Load)) {
            TILING_ERROR_LOG("total mte2 size in l1 is overflow uint64, L1NoFullLoadIter failed!");
            return -1;
        }
        if (onlyKAL1FullloadMte2Size < onlyKBL1FullloadMte2Size) {
            this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
            KAL1FullLoadIter();
        } else {
            this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
            KBL1FullLoadIter();
        }
        return 0;
    }
    // both kAL1 and kBL1 can full load
    if (this->l1TilingFlag.kABL1CanFullLoadFlag) {
        return KABL1FullLoadIter();
    }
    // None of kAL1 and kBL1 can full load in L1
    NoneKABL1FullLoadIter();
    return 0;
}

void Conv3dTilingAlgorithm::BiasL1TilingDecision()
{
    // decide bias in L1, fixpipeparams not decided yet
    if (!tilingIns_->hasBias || this->l1TilingFlag.isBiasFullLoad) {
        return;
    }
    this->l1TilingFlag.isBiasFullLoad = true;
    if (!CheckL1Buffer()) {
        this->l1TilingFlag.isBiasFullLoad = false;
    }
    return;
}

bool Conv3dTilingAlgorithm::FixL0PingpongDecision()
{
    // fix L0A/B db decision when it can full load, which db should be off
    uint64_t kl0FullLoadValue = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk;
    uint64_t kl0RangeMaxValue = this->l0TilingRange.kL0Range.at(this->l0TilingRange.kL0Range.size() - 1);
    uint64_t tmpL0ASizeNoDb = kl0RangeMaxValue * this->l0TilingParams.mL0 * this->fMapDTypeSize;
    uint64_t tmpL0BSizeNoDb = kl0RangeMaxValue * this->l0TilingParams.nL0 * this->weightDTypeSize;
    if (kl0RangeMaxValue == kl0FullLoadValue && tmpL0ASizeNoDb <= tilingIns_->platformInfo.l0ASize &&
        tmpL0BSizeNoDb <= tilingIns_->platformInfo.l0BSize && kl0RangeMaxValue < KL0_LIMIT) {
        // when krange max equals orik and which not over the l0A/B buffer, we can judge if db can be off
        uint64_t mL0FullloadValue = CeilDiv(tilingIns_->shapeInfo.singleM, tilingIns_->cubeInfo.m0) *
            tilingIns_->cubeInfo.m0;
        uint64_t nL0FullloadValue = tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0;
        if (this->l0TilingParams.mL0 == mL0FullloadValue && this->l0TilingParams.nL0 == nL0FullloadValue) {
            this->doubleBufferValue.pbAL0 = 1;
            this->doubleBufferValue.pbBL0 = 1;
        } else if (this->l0TilingParams.mL0 == mL0FullloadValue &&
            tmpL0BSizeNoDb * DOUBLE_BUFFER_NUM <= tilingIns_->platformInfo.l0BSize) {
            this->doubleBufferValue.pbAL0 = 1;
        } else if (this->l0TilingParams.nL0 == nL0FullloadValue &&
            tmpL0ASizeNoDb * DOUBLE_BUFFER_NUM <= tilingIns_->platformInfo.l0ASize) {
            this->doubleBufferValue.pbBL0 = 1;
        } else {
            return false;
        }
        // this case kl0 can be full load case and update idx and value
        this->l0TilingIdx.kL0Idx = this->l0TilingRange.kL0Range.size() - 1;
        this->l0TilingParams.kL0 = kl0RangeMaxValue;
        tilingIns_->l0TilingInfo.kL0 = this->l0TilingParams.kL0;
        tilingIns_->l0TilingInfo.kL0xorgCoAlignN0 = this->l0TilingParams.kL0 * this->l0TilingParams.orgCoAlignN0;
        return true;
    }
    return false;
}

void Conv3dTilingAlgorithm::GetKL0TilingDecision()
{
    // get k0 range according to kal1 and kbl1
    uint64_t maxKAL12L0Loop = CeilDiv(this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx),
                                      tilingIns_->cubeInfo.k0);
    uint64_t maxKBL12L0Loop = this->l1TilingFlag.isWeightBypass ? maxKAL12L0Loop :
                              CeilDiv(this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx),
                                      tilingIns_->cubeInfo.k0);
    uint64_t factorK = Gcd(maxKAL12L0Loop, maxKBL12L0Loop);
    CalcCommFactor(factorK, factorK, this->l0TilingRange.kL0Range);
    VectorElementMultip(this->l0TilingRange.kL0Range, tilingIns_->cubeInfo.k0);
    if (FixL0PingpongDecision()) {
        // when fix l0 pingpong res, kl0 decision is full load
        return;
    }

    // kL0 decision
    while (this->l0TilingIdx.kL0Idx < this->l0TilingRange.kL0Range.size() && CheckL0Buffer(this->l0TilingParams.mL0,
        this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx), this->l0TilingParams.nL0) &&
        this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx) < KL0_LIMIT) {
        this->l0TilingIdx.kL0Idx++;
    }
    this->l0TilingIdx.kL0Idx = this->l0TilingIdx.kL0Idx == 0 ? 0 : this->l0TilingIdx.kL0Idx - 1;
    this->l0TilingParams.kL0 = this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx);
    tilingIns_->l0TilingInfo.kL0 = this->l0TilingParams.kL0;
    tilingIns_->l0TilingInfo.kL0xorgCoAlignN0 = this->l0TilingParams.kL0 * this->l0TilingParams.orgCoAlignN0;
    return;
}

void Conv3dTilingAlgorithm::WeightBypassDecision()
{
    if (!this->l1TilingFlag.isWeightBypass && this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx) ==
        this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx) &&
        this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx) ==
        this->l0TilingRange.nL0Range.at(this->l0TilingIdx.nL0Idx)) {
        this->l1TilingFlag.isWeightBypass = true;
    }

    if (!this->l1TilingFlag.isWeightBypass) {
        return;
    }
    // when weight can by pass, BL1 db set to default 1
    this->doubleBufferValue.pbBL1 = 1;
    // update L1 Tiling when weight by pass
    // iter kAL1
    uint64_t tmpKAL1Idx = this->l1TilingIdx.kAL1Idx;
    for (this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1; this->l1TilingIdx.kAL1Idx >= tmpKAL1Idx;
         this->l1TilingIdx.kAL1Idx--) {
        if (this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) % tilingIns_->l0TilingInfo.kL0 == 0 &&
            CheckL1Buffer()) {
            break;
        }
    }
    if (this->l1TilingIdx.kAL1Idx != this->l1TilingRange.kAL1Range.size() - 1) {
        BiasL1TilingDecision();
        return;
    }
    // iter mAL1
    MAL1IdxIter();
    BiasL1TilingDecision();
    return;
}

void Conv3dTilingAlgorithm::SetKAL1KBL1TailRes()
{
    this->l1TilingParams.kAL1 = this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx);
    uint64_t kAL1TailCheck = this->l1TilingCalc.alignCinKhKwKd % this->l1TilingParams.kAL1;
    this->l1TilingParams.kAL1Tail = kAL1TailCheck == 0 ? this->l1TilingParams.kAL1 : kAL1TailCheck;

    this->l1TilingParams.kBL1 = this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx);
    uint64_t kBL1TailCheck = this->l1TilingCalc.alignCinKhKwKd % this->l1TilingParams.kBL1;
    this->l1TilingParams.kBL1Tail = kBL1TailCheck == 0 ? this->l1TilingParams.kBL1 : kBL1TailCheck;
}

void Conv3dTilingAlgorithm::SetMAL1NBL1ValueAndMode()
{
    this->l1TilingParams.nBL1Value = this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx);
    this->l1TilingParams.mAL1Value = this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx);
    tilingIns_->l1TilingInfo.al1FullLoad = false;
    tilingIns_->l1TilingInfo.bl1FullLoad = false;
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->l1TilingParams.mAL1Value = CeilDiv(tilingIns_->shapeInfo.singleM,
                                                 tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0;
        this->l1TilingParams.mAL1Value = CeilDiv(tilingIns_->shapeInfo.singleM,
                                                 tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    }
}

void Conv3dTilingAlgorithm::SetL1TilingRes()
{
    SetKAL1KBL1TailRes();
    SetMAL1NBL1ValueAndMode();

    if (this->l1TilingFlag.isWeightBypass) {
        this->l1TilingParams.nBL1Value = 0;
        this->l1TilingParams.kBL1 = 0;
    }

    tilingIns_->l1TilingInfo.kAL1 = this->l1TilingParams.kAL1;
    tilingIns_->l1TilingInfo.kBL1 = this->l1TilingParams.kBL1;
    tilingIns_->l1TilingInfo.mAL1Value = this->l1TilingParams.mAL1Value;
    tilingIns_->l1TilingInfo.nBL1Value = this->l1TilingParams.nBL1Value;
    tilingIns_->l1TilingInfo.mAL1DivmL0 = CeilDiv(this->l1TilingParams.mAL1Value, this->l0TilingParams.mL0);
    tilingIns_->l1TilingInfo.nBL1DivnL0 = CeilDiv(this->l1TilingParams.nBL1Value, this->l0TilingParams.nL0);
    tilingIns_->l1TilingInfo.cin1InAL1 = this->l1TilingParams.kAL1 / this->l1TilingCalc.ci0HkWk;
    tilingIns_->l1TilingInfo.kAL1Tail = this->l1TilingParams.kAL1Tail;
    tilingIns_->l1TilingInfo.cin1InAL1Tail = this->l1TilingParams.kAL1Tail / this->l1TilingCalc.ci0HkWk;
    tilingIns_->l1TilingInfo.KBL1Divk0 = this->l1TilingParams.kBL1 / tilingIns_->cubeInfo.k0;
    tilingIns_->l1TilingInfo.kBL1Tail = this->l1TilingParams.kBL1Tail;
    tilingIns_->l1TilingInfo.KBL1TailDivk0 = this->l1TilingParams.kBL1Tail / tilingIns_->cubeInfo.k0;
    tilingIns_->l1TilingInfo.iterateMNOrder = this->l1TilingFlag.iterateMNOrder;
    tilingIns_->l1TilingInfo.biasFullLoadFlag = this->l1TilingFlag.isBiasFullLoad;
    tilingIns_->l1TilingInfo.fixpParamsFullLoadFlag = false;
    tilingIns_->l1TilingInfo.isWeightBypass = this->l1TilingFlag.isWeightBypass;
}

bool Conv3dTilingAlgorithm::CheckL0Buffer(uint64_t currmL0, uint64_t currkL0, uint64_t currnL0)
{
    if (CalcAL0Size(currmL0, currkL0) > tilingIns_->platformInfo.l0ASize ||
        CalcBL0Size(currkL0, currnL0) > tilingIns_->platformInfo.l0BSize ||
        CalcCL0Size(currmL0, currnL0) > tilingIns_->platformInfo.l0CSize) {
        return false;
    } else {
        return true;
    }
}
 
uint64_t Conv3dTilingAlgorithm::CalcAL0Size(uint64_t currmL0, uint64_t currkL0) const
{
    return currmL0 * currkL0 * this->doubleBufferValue.pbAL0 * fMapDTypeSize;
}

uint64_t Conv3dTilingAlgorithm::CalcBL0Size(uint64_t currkL0, uint64_t currnL0) const
{
    return currkL0 * currnL0 * this->doubleBufferValue.pbBL0 * weightDTypeSize;
}
 
uint64_t Conv3dTilingAlgorithm::CalcCL0Size(uint64_t currmL0, uint64_t currnL0) const
{
    return currmL0 * currnL0 * this->doubleBufferValue.pbCL0 * g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType);
}
 
uint64_t Conv3dTilingAlgorithm::CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0) const
{
    // Calculate AL1 size
    uint64_t mAL1Min = currmL0;
    uint64_t hoAL1Min = mAL1Min / tilingIns_->shapeCalc.orgWo + CONST_VALUE_2;
    uint64_t hiAL1Min = InferHiL1(hoAL1Min, tilingIns_->shapeInfo.orgHi);
    uint64_t usedL1Size = hiAL1Min * tilingIns_->shapeInfo.orgWi *
                          tilingIns_->cubeInfo.k0 * fMapDTypeSize *
                          this->doubleBufferValue.pbAL1;
    if (tilingIns_->hasBias) {
        uint64_t biasSize = currnL0 * biasDTypeSize;
        usedL1Size += biasSize;
    }
    return usedL1Size;
}
 
uint64_t Conv3dTilingAlgorithm::CalcBTSize(uint64_t currnL0) const
{
    return tilingIns_->hasBias ? currnL0 * biasDTypeSize : 0;
}
 
int64_t Conv3dTilingAlgorithm::GetL0Tiling()
{
    GetL0TilingRange();
    L0TilingDecision();
    CheckL0CDoubleBuffer();
    return 0;
}
 
void Conv3dTilingAlgorithm::InitPingPong()
{
    this->doubleBufferValue.pbAL1 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbAL1 = (CalcL1SizeForL0Tiling(tilingIns_->cubeInfo.m0, tilingIns_->cubeInfo.n0) <=
                                            tilingIns_->platformInfo.l1Size) ? DOUBLE_BUFFER_NUM : 1;
    this->doubleBufferValue.pbBL1 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbAL0 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbBL0 = DOUBLE_BUFFER_NUM;
    this->doubleBufferValue.pbCL0 = 1;
    this->doubleBufferValue.pbUB = 1;
}
 
void Conv3dTilingAlgorithm::GetL0TilingRange()
{
    // Get nL0 range
    uint64_t nL0Max = min(tilingIns_->platformInfo.l0BSize / (tilingIns_->cubeInfo.k0 * this->doubleBufferValue.pbBL0 *
                                                                weightDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.m0 * this->doubleBufferValue.pbCL0 *
                                                                g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeCalc.singleCo1, nL0Max / tilingIns_->cubeInfo.n0,
                                 l0TilingRange.nL0Range);
    VectorElementMultip(l0TilingRange.nL0Range, tilingIns_->cubeInfo.n0);
 
    // Get mL0 range
    uint64_t mL0Max = min(tilingIns_->platformInfo.l0ASize / (tilingIns_->cubeInfo.k0 * this->doubleBufferValue.pbAL0 *
                          fMapDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.n0 * this->doubleBufferValue.pbCL0 *
                          g_dtypeSizeTab.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(CeilDiv(tilingIns_->shapeInfo.singleM, tilingIns_->cubeInfo.m0),
                                 mL0Max / tilingIns_->cubeInfo.m0, l0TilingRange.mL0Range);
    VectorElementMultip(l0TilingRange.mL0Range, tilingIns_->cubeInfo.m0);
}
 
void Conv3dTilingAlgorithm::L0TilingDecision()
{
    l0TilingIdx.mL0Idx = 0;
    l0TilingIdx.nL0Idx = 0;
 
    l0TilingParams.kL0 = 1 * MKN_VALUE_DEFAULT;
    l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
    l0TilingParams.orgCoAlignN0 = AlignB(tilingIns_->shapeInfo.orgCo, tilingIns_->cubeInfo.n0);
 
    bool updateML0Index = false;
    bool l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.mL0, l0TilingParams.kL0, l0TilingParams.nL0);
    uint64_t usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.mL0, l0TilingParams.nL0);
    uint64_t usedBTSize = CalcBTSize(l0TilingParams.nL0);
 
    uint64_t mL0RangeLen = l0TilingRange.mL0Range.size();
    uint64_t nL0RangeLen = l0TilingRange.nL0Range.size();
 
    while (l0BufferNotOverflowFlag && usedL1Size <= tilingIns_->platformInfo.l1Size &&
           usedBTSize <= tilingIns_->platformInfo.btSize) {
        if (l0TilingParams.mL0 <= l0TilingParams.nL0 || l0TilingIdx.nL0Idx == (nL0RangeLen - 1)) {
            updateML0Index = true;
        } else {
            updateML0Index = false;
        }
 
        if (updateML0Index) {
            ++l0TilingIdx.mL0Idx;
        } else {
            ++l0TilingIdx.nL0Idx;
        }
 
        if (l0TilingIdx.mL0Idx < mL0RangeLen && l0TilingIdx.nL0Idx < nL0RangeLen) {
            l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
            l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
            l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.mL0, l0TilingParams.kL0, l0TilingParams.nL0);
            usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.mL0, l0TilingParams.nL0);
            usedBTSize = CalcBTSize(l0TilingParams.nL0);
        } else {
            break;
        }
    }
    if (updateML0Index) {
        l0TilingIdx.mL0Idx = l0TilingIdx.mL0Idx == 0 ? 0 : l0TilingIdx.mL0Idx - 1;
    } else {
        l0TilingIdx.nL0Idx = l0TilingIdx.nL0Idx == 0 ? 0 : l0TilingIdx.nL0Idx - 1;
    }

    l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];
    tilingIns_->l0TilingInfo.mL0 = l0TilingParams.mL0;
    tilingIns_->l0TilingInfo.nL0 = l0TilingParams.nL0;
    tilingIns_->l0TilingInfo.nL0xk0 = l0TilingParams.nL0 * tilingIns_->cubeInfo.k0;
}
 
void Conv3dTilingAlgorithm::CheckL0CDoubleBuffer()
{
    if (CalcCL0Size(l0TilingParams.mL0, l0TilingParams.nL0) <=
        (tilingIns_->platformInfo.l0CSize / DOUBLE_BUFFER_NUM)) {
        this->doubleBufferValue.pbCL0 = DOUBLE_BUFFER_NUM;
    }
}

void Conv3dTilingAlgorithm::GetVecTiling()
{
    if (tilingIns_->quantType == 0) {
        return;
    }

    uint32_t maxLoadSize = 8192;
    uint64_t scaleAndBiasDTypeSize = this->scaleDTypeSize + this->biasDTypeSize;
    uint64_t scaleAndbiasLoadLength = 0;
    if (scaleAndBiasDTypeSize * tilingIns_->shapeInfo.singleCo <= maxLoadSize) {
        tilingIns_->ubTilingInfo.scaleAndBiasLoadType = SINGLECO_LOAD;
        scaleAndbiasLoadLength = tilingIns_->shapeInfo.singleCo;
    } else if (scaleAndBiasDTypeSize * tilingIns_->l0TilingInfo.nL0 <= maxLoadSize) {
        tilingIns_->ubTilingInfo.scaleAndBiasLoadType = NL0_LOAD;
        scaleAndbiasLoadLength = tilingIns_->l0TilingInfo.nL0;
    } else {
        tilingIns_->ubTilingInfo.scaleAndBiasLoadType = NUB_LOAD;
    }
    
    for (uint64_t m = 1; m <= tilingIns_->l0TilingInfo.mL0; m++) { //优先全载m方向，切分n方向，再考虑是否可以开doubleBuffer
        for (uint64_t n = 1; n <= tilingIns_->l0TilingInfo.nL0 / tilingIns_->cubeInfo.n0; n++) {
            uint64_t mUB = CeilDiv(tilingIns_->l0TilingInfo.mL0, m);
            uint64_t nUB = CeilDiv(tilingIns_->l0TilingInfo.nL0, n);
            uint64_t usedInSize = this->ubInDTypeSize * mUB * nUB;
            uint64_t usedOutSize = this->outputDTypeSize * mUB * nUB;
            if (tilingIns_->ubTilingInfo.scaleAndBiasLoadType == NUB_LOAD) {
                scaleAndbiasLoadLength = nUB;
            }
            uint64_t usedScaleAndBiasSize = scaleAndBiasDTypeSize * scaleAndbiasLoadLength;
            if (tilingIns_->descInfo.biasType.dtype == ConvDtype::BF16) { // bf16 to fp32
                usedScaleAndBiasSize += g_dtypeSizeTab.at(ConvDtype::FLOAT32) * scaleAndbiasLoadLength;
            }
            uint64_t usedUBSize = usedInSize + usedOutSize + usedScaleAndBiasSize;

            if (usedUBSize <= tilingIns_->platformInfo.ubSize) {
                this->tilingIns_->ubTilingInfo.mUB = mUB;
                this->tilingIns_->ubTilingInfo.nUB = nUB;
                TILING_INFO_LOG("Conv3D AscendC: vec tiling Info: quantType %u, ubInTypeSize %lu, scaleTypeSize %lu, biasTypeSize %lu, "
                    "outputTypeSize %lu, pbUB %u, mUB %lu, nUB %lu, usedUBSize %lu, scaleAndBiasLoadType %u, "
                    "hasBias %u. usedInSize: %lu, usedOutSize: %lu, usedScaleAndBiasSize %lu. \n",
                    tilingIns_->quantType, this->ubInDTypeSize, this->scaleDTypeSize, this->biasDTypeSize, this->outputDTypeSize,
                    this->doubleBufferValue.pbUB, tilingIns_->ubTilingInfo.mUB, tilingIns_->ubTilingInfo.nUB, 
                    usedUBSize, tilingIns_->ubTilingInfo.scaleAndBiasLoadType, tilingIns_->hasBias,
                    usedInSize, usedOutSize, usedScaleAndBiasSize);
                return;
            }
        }
    }
}
} // namespace conv3d_tiling
