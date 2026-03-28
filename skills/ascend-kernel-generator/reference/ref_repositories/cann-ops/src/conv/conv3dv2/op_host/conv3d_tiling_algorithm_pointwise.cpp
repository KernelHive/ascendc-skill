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
 * \file conv3d_tiling_algorithm_pointwise.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_tiling_algorithm.h"
#include "conv3d_tiling_algorithm_pointwise.h"

using namespace std;

namespace conv3d_tiling {

bool Conv3dTilingAlgorithmPointWise::CheckL1Buffer() const
{
    // MaL1 * KaL1 * dtype_size
    uint64_t currentFmL1Size =
        this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx) * this->doubleBufferValue.pbAL1 *
        this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) *
        this->fMapDTypeSize;
    uint64_t currentWeightL1Size =
        this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx) * this->doubleBufferValue.pbBL1 *
        this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx) * this->weightDTypeSize;
    uint64_t currentBiasL1Size = tilingIns_->hasBias ?
        (this->l1TilingFlag.isBiasFullLoad ? tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 *
        this->biasDTypeSize : this->l0TilingParams.nL0 * this->biasDTypeSize) * POINT_WISE_BIAS_CUBE_UNIT : 0;
    // cal current fm in L1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    }

    uint64_t l1SizeCur = currentFmL1Size + currentWeightL1Size + currentBiasL1Size;

    return l1SizeCur <= tilingIns_->platformInfo.l1Size;
}

uint64_t Conv3dTilingAlgorithmPointWise::CalcBTSize(uint64_t currnL0) const
{
    return 0;
}

void Conv3dTilingAlgorithmPointWise::WeightBypassDecision()
{
    return;
}

int64_t Conv3dTilingAlgorithmPointWise::InitCalcL1Params()
{
    this->l1TilingCalc.ci0HkWk = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW *
        tilingIns_->cubeInfo.k0;
    this->l1TilingCalc.alignCinKhKwKd = AlignB(tilingIns_->shapeInfo.singleCi, tilingIns_->cubeInfo.k0) *
        tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * tilingIns_->shapeInfo.orgkD;
    // cal fmap weight full load in L1 size
    if ((tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
         tilingIns_->shapeCalc.singleM1 * tilingIns_->cubeInfo.m0 * tilingIns_->cubeInfo.k0 * this->fMapDTypeSize) /
         this->fMapDTypeSize !=
        (tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
         tilingIns_->shapeCalc.singleM1 * tilingIns_->cubeInfo.m0 * tilingIns_->cubeInfo.k0)) {
        TILING_ERROR_LOG("fmap size in l1 is overflow uint64, initcalc l1 params failed!");
        return -1;
    }
    this->l1TilingCalc.fmapFullLoadL1Size = tilingIns_->shapeInfo.singlekD *
         AlignB(tilingIns_->shapeCalc.singleCi1 * tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) * 
         tilingIns_->shapeCalc.singleM1 * tilingIns_->cubeInfo.m0 * this->fMapDTypeSize;
    if ((tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 * this->l1TilingCalc.ci0HkWk *
        tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize) / weightDTypeSize !=
        (tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 * this->l1TilingCalc.ci0HkWk *
        tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0)) {
        TILING_ERROR_LOG("weight size in l1 is overflow uint64, initcalc l1 params failed!");
        return -1;
    }
    this->l1TilingCalc.weightFullLoadL1Size = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize;
    // cal min/kfullload fmap size in L1(mL0)

    this->l1TilingCalc.fmapMinLoadL1Size = tilingIns_->cubeInfo.k0 * this->l0TilingParams.mL0 *
        this->fMapDTypeSize * this->doubleBufferValue.pbAL1;

    this->l1TilingCalc.fmapKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        tilingIns_->cubeInfo.k0 * this->l0TilingParams.mL0 * this->doubleBufferValue.pbAL1 *
        this->fMapDTypeSize;
    // cal min/kfullload weiht size in L1
    this->l1TilingCalc.weightMinLoadL1Size = this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 *
        this->doubleBufferValue.pbBL1 * this->weightDTypeSize;
    this->l1TilingCalc.weightKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 * this->doubleBufferValue.pbBL1 * this->weightDTypeSize;
    // cal bias size in L1
    this->l1TilingCalc.biasMinLoadL1Size = tilingIns_->hasBias ? this->l0TilingParams.nL0 * POINT_WISE_BIAS_CUBE_UNIT *
                                            this->biasDTypeSize : 0;

    return 0;
}

void Conv3dTilingAlgorithmPointWise::InitABL1TilingMode()
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
        } else if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
            this->doubleBufferValue.pbAL1 = 1;
            return;
        }
    } else {
        if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
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


void Conv3dTilingAlgorithmPointWise::GetL1TilingRange()
{
    if (tilingIns_->descInfo.fMapType.dtype == ConvDtype::FLOAT32) {
        CalcFactorPointWise(tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kAL1Range);
        CalcFactorPointWise(tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kBL1Range);
    } else {
        CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kAL1Range);
        CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kBL1Range);
    }

    VectorElementMultip(this->l1TilingRange.kAL1Range, this->l1TilingCalc.ci0HkWk);
    VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0HkWk);

    // cal mAL1Value and nBL1Value
    uint64_t multiNBL1Max = CeilDiv(tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0,
                                    this->l0TilingParams.nL0);
    CalcCommFactor(multiNBL1Max, multiNBL1Max, this->l1TilingRange.nBL1ValueRange);
    VectorElementMultip(this->l1TilingRange.nBL1ValueRange, l0TilingParams.nL0);
    uint64_t multiMAL1Max = CeilDiv(CeilDiv(tilingIns_->shapeInfo.singleM,
                                            tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0,
                                    this->l0TilingParams.mL0);
    CalcCommFactor(multiMAL1Max, multiMAL1Max, this->l1TilingRange.mAL1ValueRange);
    VectorElementMultip(this->l1TilingRange.mAL1ValueRange, l0TilingParams.mL0);
}

void Conv3dTilingAlgorithmPointWise::GetKL0TilingRangeCommon(uint64_t k0)
{
    uint64_t maxKAL12L0Loop = CeilDiv(this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx),
                                    k0);
    uint64_t maxKBL12L0Loop = CeilDiv(this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx),
                                    k0);
    uint64_t factorK = Gcd(maxKAL12L0Loop, maxKBL12L0Loop);
    CalcCommFactor(factorK, factorK, this->l0TilingRange.kL0Range);
    VectorElementMultip(this->l0TilingRange.kL0Range, k0);
}

void Conv3dTilingAlgorithmPointWise::GetKL0TilingDecision()
{
     // get k0 range according to kal1 and kbl1
    if (tilingIns_->descInfo.fMapType.dtype == ConvDtype::FLOAT32) {
        GetKL0TilingRangeCommon(POINT_WISE_ALIGN_UNIT);
    } else {
        GetKL0TilingRangeCommon(tilingIns_->cubeInfo.k0);
    }
    
    if (FixL0PingpongDecision()) {
        // when fix l0 pingpong res, kl0 decision is full load
        return;
    }

    // kL0 decision
    while (this->l0TilingIdx.kL0Idx < this->l0TilingRange.kL0Range.size() &&
           CheckL0Buffer(this->l0TilingParams.mL0, this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx),
                         this->l0TilingParams.nL0)) {
        this->l0TilingIdx.kL0Idx++;
    }
    this->l0TilingIdx.kL0Idx = this->l0TilingIdx.kL0Idx == 0 ? 0 : this->l0TilingIdx.kL0Idx - 1;
    this->l0TilingParams.kL0 = this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx);
    tilingIns_->l0TilingInfo.kL0 = this->l0TilingParams.kL0;
    tilingIns_->l0TilingInfo.kL0xorgCoAlignN0 = this->l0TilingParams.kL0 * this->l0TilingParams.orgCoAlignN0;
    return;
}

uint64_t Conv3dTilingAlgorithmPointWise::CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0) const
{
    uint64_t usedL1Size = currmL0 * AlignB(tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) *
                          fMapDTypeSize * this->doubleBufferValue.pbAL1;
    usedL1Size += currnL0 * AlignB(tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) *
                    weightDTypeSize * this->doubleBufferValue.pbBL1;
    if (tilingIns_->hasBias) {
        uint64_t biasSize = currnL0 * biasDTypeSize;
        usedL1Size += biasSize;
    }
    return usedL1Size;
}

uint64_t Conv3dTilingAlgorithmPointWise::L1NoFullLoadFmapSize() const
{
    uint64_t fmapSingleCoreL1Load =  AlignB(tilingIns_->shapeInfo.singleM, tilingIns_->cubeInfo.m0) *
        AlignB(tilingIns_->shapeCalc.singleCi1 * tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) *
        tilingIns_->shapeInfo.singlekD * fMapDTypeSize;
    return fmapSingleCoreL1Load;
}

bool Conv3dTilingAlgorithmPointWise::CoreL1TilingMinWeightBypass() const
{
    return false;
}

bool Conv3dTilingAlgorithmPointWise::NoneKABL1FullLoadWeightBypass() const
{
    return false;
}

void Conv3dTilingAlgorithmPointWise::SetKAL1KBL1TailRes()
{
    this->l1TilingParams.kAL1 = this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx);
    uint64_t kAL1TailCheck = this->tilingIns_->shapeInfo.singleCi % this->l1TilingParams.kAL1;
    this->l1TilingParams.kAL1Tail = kAL1TailCheck == 0 ? this->l1TilingParams.kAL1 : kAL1TailCheck;

    this->l1TilingParams.kBL1 = this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx);
    uint64_t kBL1TailCheck = this->tilingIns_->shapeInfo.singleCi % this->l1TilingParams.kBL1;
    this->l1TilingParams.kBL1Tail = kBL1TailCheck == 0 ? this->l1TilingParams.kBL1 : kBL1TailCheck;
}
} // namespace conv3d_tiling