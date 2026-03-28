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
 * \file conv3d_tiling_base.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H
#define ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H

#include "conv3d_tiling_util.h"
#include "conv3dv2_tiling.h"

namespace conv3d_tiling {

constexpr uint32_t NUB_LOAD = 0; // LoadChannelType::NORMAL
constexpr uint32_t SINGLECO_LOAD = 1; // LoadChannelType::LOAD_TOTAL_CORE
constexpr uint32_t NL0_LOAD = 2; // LoadChannelType::LOAD_TOTAL_LC0
constexpr uint32_t KL0_LIMIT = 4096;  // MMad limits

struct Conv3DL1Tiling {
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t mAL1Value = 0;
    uint64_t nBL1Value = 0;
    uint64_t mAL1DivmL0 = 0;
    uint64_t nBL1DivnL0 = 0;
    uint64_t cin1InAL1 = 0;
    uint64_t kAL1Tail = 0;
    uint64_t cin1InAL1Tail = 0;
    uint64_t KBL1Divk0 = 0;
    uint64_t kBL1Tail = 0;
    uint64_t KBL1TailDivk0 = 0;

    IterateMNOrder iterateMNOrder;
    bool isWeightBypass;
    bool biasFullLoadFlag;
    bool fixpParamsFullLoadFlag;
    bool al1FullLoad;
    bool bl1FullLoad;
};

struct Conv3DL0Tiling {
    uint64_t mL0 = 0;
    uint64_t kL0 = 0;
    uint64_t nL0 = 0;
    uint64_t nL0xk0 = 0;
    uint64_t kL0xorgCoAlignN0 = 0;
};

struct Conv3DUBTiling {
    uint64_t mUB = 0;
    uint64_t nUB = 0;
    uint32_t scaleAndBiasLoadType = NUB_LOAD;
};

struct Conv3DInputshape {
    int64_t orgkH = -1L;
    int64_t orgkW = -1L;
    int64_t orgkD = -1L;
    int64_t orgCo = -1L;
    int64_t coutOpt = -1L;
    int64_t orgCi = -1L;
    int64_t cinOpt = -1L;
    int64_t orgDi = -1L;
    int64_t orgHi = -1L;
    int64_t orgWi = -1L;

    int64_t singlekH = -1L;
    int64_t singlekW = -1L;
    int64_t singlekD = -1L;
    int64_t singleCi = -1L;
    int64_t singleCo = -1L;
    int64_t singleDo = -1L;
    int64_t singleM = -1L;
    int64_t singleHo = -1L;
    int64_t singleWo = -1L;
    int64_t singleCoreGroupOpt = -1L;
};

struct Conv3DInputAttr {
    int64_t groups = 1;
    int64_t groupOpt = 1;

    int64_t padHead = 0;
    int64_t padTail = 0;
    int64_t padUp = 0;
    int64_t padDown = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;

    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t strideD = 1;

    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t dilationD = 1;
    int8_t offsetx = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
};

struct Conv3DCalcShape {
    uint64_t singleCi1 = 0;
    uint64_t singleCo1 = 0;
    uint64_t singleM1 = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint64_t orgDo = 0;
};

struct Conv3DDesc {
    ConvType fMapType  = {ConvFormat::NDC1HWC0, ConvDtype::FLOAT16, TPosition::GM};
    ConvType weightType  = {ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT16, TPosition::GM};
    ConvType biasType = {ConvFormat::ND, ConvDtype::FLOAT16, TPosition::GM};
    ConvType outputType = {ConvFormat::NDC1HWC0, ConvDtype::FLOAT16, TPosition::CO1};
    ConvType quantScaleType = {ConvFormat::ND, ConvDtype::INT64, TPosition::GM};
};

struct DoubleBufferValue {
    uint8_t pbAL1 = 1;
    uint8_t pbBL1 = 1;
    uint8_t pbAL0 = 2;
    uint8_t pbBL0 = 2;
    uint8_t pbCL0 = 1;
    uint8_t pbUB = 1;
    uint64_t pBufferFlag;
};

struct CubeInfo {
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    ConvDtype madType;
    ConvDtype biasType;
    uint32_t minBurstNum;
};

class Conv3dTilingBase {
public:
    Conv3dTilingBase();
    explicit Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform);
    explicit Conv3dTilingBase(const PlatformInfo& platform);
    virtual ~Conv3dTilingBase() = default;
    virtual int64_t GetTiling(optiling::TConv3DTiling& tiling) = 0;
    void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw);
    void SetOrgFmapShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi);
    void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo);
    void SetOutputOrder(int8_t outputOrder);
    void SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetScaleType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetQuantType();
    void SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight);
    void SetDilation(int64_t dilationH, int64_t dilationW, int64_t dilationD);
    void SetStride(int64_t strideH, int64_t strideW, int64_t strideD);
    void SetHF32(bool hf32Enable, bool hf32TransMode);
    bool CalOptGroupParams(const Conv3DOriGroupInfo &oriGroupInfo, Conv3DGroupOptInfo &groupOptInfo) const;
    void SetGroups(int64_t groups);
    void SetOptGroupInfo(int64_t groupOpt, int64_t singleCoreGroupOpt, int64_t cinOpt, int64_t coutOpt);

    Conv3DDesc descInfo;
    Conv3DInputshape shapeInfo;
    Conv3DCalcShape shapeCalc;
    Conv3DInputAttr attrInfo;
    CubeInfo cubeInfo;
    Conv3DL1Tiling l1TilingInfo;
    Conv3DL0Tiling l0TilingInfo;
    Conv3DUBTiling ubTilingInfo;
    DoubleBufferValue dbValue;
    PlatformInfo platformInfo;

    bool hasBias = false;
    bool hasQuantScale = false;
    uint8_t quantType = 0;

    bool hf32Enable_ = false;
    bool hf32TransMode_ = false;
    int8_t outputOrder_ = M_Mode;

protected:
    virtual int64_t Compute() = 0;
    void SetFinalTilingBasicInfo(optiling::TConv3DTiling& tiling);
    void SetFinalTilingDecisionInfo(optiling::TConv3DTiling& tiling);
    void SetFinalTiling(optiling::TConv3DTiling& tiling);
    void PrintTilingDataBasicInfo() const;
    void PrintTilingDataDecision() const;
    void PrintTilingData() const;
    bool CheckInputParam();
    bool CheckInputParamPointWise();
    void GetCubeInfo();
    bool ShapeInitCalc();
    bool CheckParamsOverflow();

private:
    bool CheckInputAttr();
    bool CheckInputAttrPointWise();
    bool CheckOrgInputInfo();
    bool CheckOrgInputShapeWithPad();
    bool CheckOrgInputInfoPointWise();
    bool CheckSingleInputInfo();
    bool CheckSingleInputInfoPointWise();
    bool CheckInputConstraint();
    bool CheckInputShape();
    bool CheckInputShapePointWise();
    bool CheckInputFormat();
    bool CheckInputFormatPointWise();
    bool CheckParamsDtype();
    bool CheckParamsDtypePointWise();
    bool CheckLoad3DLimits();
    bool CheckInstructionLimits();
    bool CheckHF32();
};
} // namespace conv3d_tiling

#endif