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
 * \file conv3d_tiling_base.cpp
 * \brief
 */

#include "conv3d_tiling_base.h"

using namespace std;

namespace conv3d_tiling {

Conv3dTilingBase::Conv3dTilingBase()
{
    this->platformInfo.l1Size = L1_SIZE;
    this->platformInfo.l0ASize = L0A_SIZE;
    this->platformInfo.l0BSize = L0B_SIZE;
    this->platformInfo.l0CSize = L0C_SIZE;
    this->platformInfo.ubSize = UB_SIZE;
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

Conv3dTilingBase::Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
{
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize);

    this->platformInfo.socVersion = socVersion;
    this->platformInfo.l1Size = static_cast<uint64_t>(l1Size);
    this->platformInfo.l0ASize = static_cast<uint64_t>(l0ASize);
    this->platformInfo.l0BSize = static_cast<uint64_t>(l0BSize);
    this->platformInfo.l0CSize = static_cast<uint64_t>(l0CSize);
    this->platformInfo.ubSize = static_cast<uint64_t>(ubSize);
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

Conv3dTilingBase::Conv3dTilingBase(const PlatformInfo& platform)
{
    this->platformInfo.socVersion = platform.socVersion;
    this->platformInfo.l1Size = static_cast<uint64_t>(platform.l1Size);
    this->platformInfo.l0ASize = static_cast<uint64_t>(platform.l0ASize);
    this->platformInfo.l0BSize = static_cast<uint64_t>(platform.l0BSize);
    this->platformInfo.l0CSize = static_cast<uint64_t>(platform.l0CSize);
    this->platformInfo.ubSize = static_cast<uint64_t>(platform.ubSize);
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

void Conv3dTilingBase::SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw)
{
    this->shapeInfo.orgkH = orgKh;
    this->shapeInfo.orgkW = orgKw;
    this->shapeInfo.orgkD = orgKd;
    this->shapeInfo.orgCo = orgCo;
}

void Conv3dTilingBase::SetSingleWeightShape(int64_t singleCi, int64_t singleKd,
    int64_t singleKh, int64_t singleKw)
{
    this->shapeInfo.singlekH = singleKh;
    this->shapeInfo.singlekW = singleKw;
    this->shapeInfo.singlekD = singleKd;
    this->shapeInfo.singleCi = singleCi;
}

void Conv3dTilingBase::SetOrgFmapShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
{
    this->shapeInfo.orgCi = orgCi;
    this->shapeInfo.orgHi = orgHi;
    this->shapeInfo.orgWi = orgWi;
    this->shapeInfo.orgDi = orgDi;
}

void Conv3dTilingBase::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleM = singleM;
}

void Conv3dTilingBase::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleHo = singleHo;
    this->shapeInfo.singleWo = singleWo;
    this->shapeInfo.singleM = singleHo * singleWo;
}

void Conv3dTilingBase::SetOutputOrder(int8_t outputOrder)
{
    this->outputOrder_ = outputOrder;
}

void Conv3dTilingBase::SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.weightType.pos = pos;
    this->descInfo.weightType.format = format;
    this->descInfo.weightType.dtype = dtype;
}

void Conv3dTilingBase::SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.fMapType.pos = pos;
    this->descInfo.fMapType.format = format;
    this->descInfo.fMapType.dtype = dtype;
}

void Conv3dTilingBase::SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight)
{
    this->attrInfo.padHead = padHead;
    this->attrInfo.padTail = padTail;
    this->attrInfo.padUp = padUp;
    this->attrInfo.padDown = padDown;
    this->attrInfo.padLeft = padLeft;
    this->attrInfo.padRight = padRight;
}

void Conv3dTilingBase::SetDilation(int64_t dilationH, int64_t dilationW, int64_t dilationD)
{
    this->attrInfo.dilationH = dilationH;
    this->attrInfo.dilationW = dilationW;
    this->attrInfo.dilationD = dilationD;
}

void Conv3dTilingBase::SetStride(int64_t strideH, int64_t strideW, int64_t strideD)
{
    this->attrInfo.strideH = strideH;
    this->attrInfo.strideW = strideW;
    this->attrInfo.strideD = strideD;
}

void Conv3dTilingBase::SetQuantType()
{
    if (descInfo.fMapType.dtype == ConvDtype::INT8 &&
        descInfo.weightType.dtype == ConvDtype::INT8 &&
        descInfo.outputType.dtype == ConvDtype::BF16) {
        this->quantType = 6; // QuantType::PER_CHANNEL_NO_OFFSET
    }
}

void Conv3dTilingBase::SetScaleType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->hasQuantScale = true;
    this->descInfo.quantScaleType.pos = pos;
    this->descInfo.quantScaleType.format = format;
    this->descInfo.quantScaleType.dtype = dtype;
}

void Conv3dTilingBase::SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->hasBias = true;
    this->descInfo.biasType.pos = pos;
    this->descInfo.biasType.format = format;
    this->descInfo.biasType.dtype = dtype;
}

void Conv3dTilingBase::SetHF32(bool hf32Enable, bool hf32TransMode = false)
{
    this->hf32Enable_ = hf32Enable;
    this->attrInfo.hf32Enable = hf32Enable;
    this->hf32TransMode_ = hf32TransMode;
    this->attrInfo.hf32TransMode = hf32TransMode;
}

bool Conv3dTilingBase::CalOptGroupParams(const Conv3DOriGroupInfo &oriGroupInfo,
                                                        Conv3DGroupOptInfo &groupOptInfo) const
{
    // user need to pass correct parms.
    if (oriGroupInfo.groups < 1 || oriGroupInfo.cin < 1 || oriGroupInfo.cout < 1) {
        TILING_DEBUG_LOG("Conv3D AscendC: unSupported parms in groupOpt:" \
                        " groups: %ld, cin: %ld, cout: %ld, only support greater than zero", 
                        oriGroupInfo.groups, oriGroupInfo.cin, oriGroupInfo.cout);
        return false;
    }

    if (oriGroupInfo.dtype != ConvDtype::BF16 && oriGroupInfo.dtype != ConvDtype::FLOAT32 &&
        oriGroupInfo.dtype != ConvDtype::FLOAT16 && oriGroupInfo.dtype != ConvDtype::INT8) {
        TILING_DEBUG_LOG("Conv3D AscendC: only support dType: INT8, BF16, FLOAT32 and FLOAT16");
        return false;
    }

    if (oriGroupInfo.groups == 1) {
        groupOptInfo.groupOpt = oriGroupInfo.groups;
        groupOptInfo.cinOpt = oriGroupInfo.cin;
        groupOptInfo.coutOpt = oriGroupInfo.cout;
    } else {
        if ((oriGroupInfo.cin % oriGroupInfo.groups != 0)
            || (oriGroupInfo.cout % oriGroupInfo.groups != 0)) {
            TILING_DEBUG_LOG("Conv3D AscendC: cIn and cOut should be multiple of groups when groups greater than one.");
            return false;
        }
        int64_t k0 = CUBE_MKN_TAB.GetMKN(oriGroupInfo.dtype, MKN_K_INDEX);
        int64_t n0 = CUBE_MKN_TAB.GetMKN(oriGroupInfo.dtype, MKN_N_INDEX);
        int64_t cinOriPerGroup = oriGroupInfo.cin / oriGroupInfo.groups;
        int64_t coutOriPerGroup = oriGroupInfo.cout / oriGroupInfo.groups;

        // calc cinOpt, coutOpt, groupOpt
        int64_t enlarge = std::min(LCM((LCM(cinOriPerGroup, k0) / cinOriPerGroup),
                            (LCM(coutOriPerGroup, n0) / coutOriPerGroup)), oriGroupInfo.groups);
        groupOptInfo.groupOpt = CeilDiv(oriGroupInfo.groups, enlarge);
        if (MulWithOverflowCheck(groupOptInfo.cinOpt, cinOriPerGroup, enlarge) ||
            MulWithOverflowCheck(groupOptInfo.coutOpt, coutOriPerGroup, enlarge)) {
            TILING_DEBUG_LOG("Conv3D AscendC: cinOpt and coutOpt should not exceed INT64_MAX.");
            return false;
        }
    }
    return true;
}

void Conv3dTilingBase::SetGroups(int64_t groups)
{
    // this should be called after SetOrgWeightShape and SetOrgFmapShape
    this->attrInfo.groups = groups;
    if (groups == 1) {
        this->attrInfo.groupOpt = this->attrInfo.groups;
        this->shapeInfo.cinOpt = this->shapeInfo.orgCi;
        this->shapeInfo.coutOpt = this->shapeInfo.orgCo;
    }
}

void Conv3dTilingBase::SetOptGroupInfo(int64_t groupOpt, int64_t singleCoreGroupOpt, int64_t cinOpt, int64_t coutOpt)
{
    this->attrInfo.groupOpt = groupOpt;
    this->shapeInfo.singleCoreGroupOpt = singleCoreGroupOpt;
    this->shapeInfo.cinOpt = cinOpt;
    this->shapeInfo.coutOpt = coutOpt;
}

void Conv3dTilingBase::SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype)
{
    this->descInfo.outputType.pos = pos;
    this->descInfo.outputType.format = format;
    this->descInfo.outputType.dtype = dtype;
}

bool Conv3dTilingBase::CheckInputParam()
{
    if (!CheckInputAttr() || !CheckInputShape() || !CheckInputFormat() || !CheckParamsDtype()) {
        return false;
    }

    if (!CheckInstructionLimits()) {
        return false;
    }

    if (!CheckHF32()) {
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputParamPointWise()
{
    if (!CheckInputAttrPointWise() || !CheckInputShapePointWise() || !CheckInputFormatPointWise() || !CheckParamsDtypePointWise()) {
        return false;
    }

    if (!CheckHF32()) {
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputAttr()
{
    if (this->attrInfo.groups < 1 || this->attrInfo.groupOpt < 1) {
        TILING_ERROR_LOG(
            "Illlegal attrs have set: groups=%ld, groupOpt=%ld.", 
            this->attrInfo.groups, this->attrInfo.groupOpt);
        return false;
    }

    bool padInvalidFlag = (this->attrInfo.padLeft < 0 || this->attrInfo.padRight < 0 ||
        this->attrInfo.padUp < 0 || this->attrInfo.padDown < 0 ||
        this->attrInfo.padHead < 0 || this->attrInfo.padTail < 0);
    if (padInvalidFlag) {
        TILING_ERROR_LOG(
            "Illlegal attrs have set: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld, padHead=%ld, padTail=%ld,\
            which must >= 0.", this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft,
            this->attrInfo.padRight, this->attrInfo.padHead, this->attrInfo.padTail);
        return false;
    }

    if (this->attrInfo.strideH <= 0 || this->attrInfo.strideW <= 0 || this->attrInfo.strideD <= 0) {
        TILING_ERROR_LOG("Illegal attrs have set: strideH=%ld, strideW=%ld, strideD=%ld, which must > 0.",
                         this->attrInfo.strideH, this->attrInfo.strideW, this->attrInfo.strideD);
        return false;
    }

    if (this->attrInfo.dilationH <= 0 || this->attrInfo.dilationW <= 0 || this->attrInfo.dilationD <= 0) {
        TILING_ERROR_LOG("Illegal attrs have set: dilationH=%ld, dilationW=%ld, dilationD=%ld, which must > 0.",
                         this->attrInfo.dilationH, this->attrInfo.dilationW, this->attrInfo.dilationD);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputAttrPointWise()
{
    if (this->attrInfo.groups != 1) {
        TILING_ERROR_LOG(
            "[PointWise] Illlegal attrs have set: groups=%ld.", 
            this->attrInfo.groups);
        return false;
    }

    bool padInvalidFlag = (this->attrInfo.padLeft != 0 || this->attrInfo.padRight != 0 ||
        this->attrInfo.padUp != 0 || this->attrInfo.padDown != 0 ||
        this->attrInfo.padHead != 0 || this->attrInfo.padTail != 0);
    if (padInvalidFlag) {
        TILING_ERROR_LOG(
            "[PointWise] Illlegal attrs have set: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld, padHead=%ld, padTail=%ld,\
            which must = 0.", this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft,
            this->attrInfo.padRight, this->attrInfo.padHead, this->attrInfo.padTail);
        return false;
    }

    if (this->attrInfo.strideH != 1 || this->attrInfo.strideW != 1 || this->attrInfo.strideD != 1) {
        TILING_ERROR_LOG("[PointWise] Illegal attrs have set: strideH=%ld, strideW=%ld, strideD=%ld, which must = 1.",
                         this->attrInfo.strideH, this->attrInfo.strideW, this->attrInfo.strideD);
        return false;
    }

    if (this->attrInfo.dilationH != 1 || this->attrInfo.dilationW != 1 || this->attrInfo.dilationD != 1) {
        TILING_ERROR_LOG("[PointWise] Illegal attrs have set: dilationH=%ld, dilationW=%ld, dilationD=%ld, which must = 1.",
                         this->attrInfo.dilationH, this->attrInfo.dilationW, this->attrInfo.dilationD);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckOrgInputInfo()
{
    if (this->shapeInfo.orgkH <= 0 || this->shapeInfo.orgkW <= 0 || 
        this->shapeInfo.orgCo <= 0 || this->shapeInfo.orgkD <= 0) {
        TILING_ERROR_LOG("Illegal org weight shapes have set: Co=%ld, kH=%ld, kW=%ld, kD=%ld, which must > 0.",
            this->shapeInfo.orgCo, this->shapeInfo.orgkH, this->shapeInfo.orgkW, this->shapeInfo.orgkD);
        return false;
    }

    if (this->shapeInfo.orgCi <= 0 || this->shapeInfo.orgHi <= 0 ||
        this->shapeInfo.orgWi <= 0 || this->shapeInfo.orgDi <= 0) {
        TILING_ERROR_LOG("Illegal org featureMap shapes have set: Ci=%ld, Hi=%ld, Wi=%ld, Di=%ld, which must > 0.",
            this->shapeInfo.orgCi, this->shapeInfo.orgHi, this->shapeInfo.orgWi, this->shapeInfo.orgDi);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckOrgInputInfoPointWise()
{
    if (this->shapeInfo.orgkH != 1 || this->shapeInfo.orgkW != 1 || this->shapeInfo.orgkD != 1) {
        TILING_ERROR_LOG("[PointWise] Illegal org weight shapes have set: kH=%ld, kW=%ld, kD=%ld, which must = 1.",
            this->shapeInfo.orgkH, this->shapeInfo.orgkW, this->shapeInfo.orgkD);
        return false;
    }
    return CheckOrgInputInfo();
}

bool Conv3dTilingBase::CheckSingleInputInfo()
{
    if (this->shapeInfo.singlekH <= 0 || this->shapeInfo.singlekW <= 0 ||
        this->shapeInfo.singleCo <= 0 || this->shapeInfo.singlekD <= 0) {
        TILING_ERROR_LOG("Illegal singleCore weight shapes have set: Co=%ld, kH=%ld, kW=%ld, kD=%ld, which must > 0.",
            this->shapeInfo.singleCo, this->shapeInfo.singlekH, this->shapeInfo.singlekW, this->shapeInfo.singlekD);
        return false;
    }

    if (this->shapeInfo.singleCi <= 0 || this->shapeInfo.singleDo <= 0) {
        TILING_ERROR_LOG("Illegal singleCore featureMap shapes have set: Ci=%ld, Do=%ld, which must > 0.",
            this->shapeInfo.singleCi, this->shapeInfo.singleDo);
        return false;
    }

    if (this->shapeInfo.cinOpt <= 0 || this->shapeInfo.coutOpt <= 0) {
        TILING_ERROR_LOG("Illegal cinOpt and coutOpt have set: cinOpt=%ld, coutOpt=%ld.",
            this->shapeInfo.cinOpt, this->shapeInfo.coutOpt);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckSingleInputInfoPointWise()
{
    if (this->shapeInfo.singlekH != 1 || this->shapeInfo.singlekW != 1 || this->shapeInfo.singlekD != 1) {
        TILING_ERROR_LOG("[PointWise] Illegal singleCore weight shapes have set: kH=%ld, kW=%ld, kD=%ld, which must = 1.",
            this->shapeInfo.singlekH, this->shapeInfo.singlekW, this->shapeInfo.singlekD);
        return false;
    }
    return CheckSingleInputInfo();
}

bool Conv3dTilingBase::CheckInputConstraint()
{
    if (this->shapeInfo.singlekH != this->shapeInfo.orgkH || this->shapeInfo.singlekW != this->shapeInfo.orgkW
        || this->shapeInfo.singlekD != this->shapeInfo.orgkD) {
        TILING_ERROR_LOG("Support singlekH = orgkH, singlekW = orgkW, singlekD = orgkD,\
            current singlekH: %ld, orgkH: %ld, singlekW: %ld, orgkW: %ld, singlekD: %ld, orgkD: %ld", this->shapeInfo.singlekH,
            this->shapeInfo.orgkH, this->shapeInfo.singlekW, this->shapeInfo.orgkW, this->shapeInfo.singlekD,
            this->shapeInfo.orgkD);
        return false;
    }

    if (this->attrInfo.groups == 1 && (this->shapeInfo.orgCi != this->shapeInfo.cinOpt
        || this->shapeInfo.orgCo != this->shapeInfo.coutOpt)) {
        TILING_ERROR_LOG("cinOpt not equal to orgCi or coutOpt not equal to orgCo in groups 1");
        return false;
    }

    if (this->shapeInfo.singleCi != this->shapeInfo.cinOpt) {
        TILING_ERROR_LOG("Support singleCi = cinOpt, current singleCi: %ld, cinOpt: %ld", this->shapeInfo.singleCi,
                         this->shapeInfo.cinOpt);
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckOrgInputShapeWithPad() 
{
  int64_t idPad = this->shapeInfo.orgDi + this->attrInfo.padHead + this->attrInfo.padTail - this->attrInfo.dilationD * (this->shapeInfo.orgkD - 1) - 1;
  int64_t ihPad = this->shapeInfo.orgHi + this->attrInfo.padUp + this->attrInfo.padDown - this->attrInfo.dilationH * (this->shapeInfo.orgkH - 1) - 1;
  int64_t iwPad = this->shapeInfo.orgWi + this->attrInfo.padLeft + this->attrInfo.padRight - this->attrInfo.dilationW * (this->shapeInfo.orgkW - 1) - 1;
  if (idPad < 0 || ihPad < 0 || iwPad < 0) {
    TILING_ERROR_LOG(
        "Fmap size(DHW) after padding should be greater than or equal to filter size(DHW). idPad %ld, ihPad %ld, iwPad %ld",
        idPad,
        ihPad,
        iwPad);
    return false;
  }
  return true;
}

bool Conv3dTilingBase::CheckInputShape()
{
    if (!CheckOrgInputInfo()) {
        return false;
    }

    if (!CheckOrgInputShapeWithPad()) {
        return false;
    }

    if (!CheckSingleInputInfo()) {
        return false;
    }

    if (!CheckInputConstraint()) {
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputShapePointWise()
{
    if (!CheckOrgInputInfoPointWise()) {
        return false;
    }

    if (!CheckSingleInputInfoPointWise()) {
        return false;
    }

    if (!CheckInputConstraint()) {
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputFormat()
{
    if (this->descInfo.weightType.format != ConvFormat::FRACTAL_Z_3D) {
        TILING_ERROR_LOG("unSupported weight format: %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }

    if (this->descInfo.fMapType.format != ConvFormat::NDC1HWC0) {
        TILING_ERROR_LOG("unSupported feature map format: %s.",
                         g_formatToStr.at(this->descInfo.fMapType.format).c_str());
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputFormatPointWise()
{
    if (this->descInfo.weightType.format != ConvFormat::NCDHW) {
        TILING_ERROR_LOG("[PointWise] weight format should be NCDHW, now is %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }
    if (this->descInfo.outputType.format != ConvFormat::NCDHW) {
        TILING_ERROR_LOG("[PointWise] output format should be NCDHW, now is %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckParamsDtype()
{
    if (this->hasQuantScale) {
        vector<ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype,
            this->descInfo.biasType.dtype, this->descInfo.quantScaleType.dtype,
            this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS_SCALE.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS_SCALE[kindsId], COUNT_PARAMS_BIAS_SCALE)) {
                return true;
            }
        }
        TILING_ERROR_LOG("unSupported params data type [fmap, weight, bias, scale, output]: [%s, %s, %s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.biasType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.quantScaleType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    } else if (this->hasBias) {
        vector<ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype,
            this->descInfo.biasType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_BIAS)) {
                return true;
            }
        }
        TILING_ERROR_LOG("unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.biasType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    } else {
        vector<ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_NO_BIAS)) {
                return true;
            }
        }
        TILING_ERROR_LOG("unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    }
    return false;
}

bool Conv3dTilingBase::CheckParamsDtypePointWise()
{
    if (this->hasBias) {
        vector<ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype,
            this->descInfo.biasType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < POINTWISE_SUPPORTED_TYPES_WITH_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, POINTWISE_SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_BIAS)) {
                return true;
            }
        }
        TILING_ERROR_LOG("[PointWise] unSupported params data type [fmap, weight, bias, output]: [%s, %s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.biasType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    } else {
        vector<ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < POINTWISE_SUPPORTED_TYPES_WITHOUT_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, POINTWISE_SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_NO_BIAS)) {
                return true;
            }
        }
        TILING_ERROR_LOG("[PointWise] unSupported params data type [fmap, weight, output]: [%s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    }
    return false;
}

bool Conv3dTilingBase::CheckLoad3DLimits()
{
    if (static_cast<uint32_t>(this->attrInfo.strideH) > LOAD3D_MAX_STRIDE_H_W ||
        static_cast<uint32_t>(this->attrInfo.strideW) > LOAD3D_MAX_STRIDE_H_W) {
        TILING_ERROR_LOG(
            "Attrs not satisfying load3d's limits: strideH=%ld, strideW=%ld, which must <= %u.",
            this->attrInfo.strideH, this->attrInfo.strideW, LOAD3D_MAX_STRIDE_H_W);
        return false;
    }

    if (static_cast<uint32_t>(this->attrInfo.dilationH) > LOAD3D_MAX_DILATION_H_W ||
        static_cast<uint32_t>(this->attrInfo.dilationW) > LOAD3D_MAX_DILATION_H_W) {
        TILING_ERROR_LOG(
            "Attrs not satisfying load3d's limits: dilationH=%ld, dilationW=%ld, which must <= %u.",
            this->attrInfo.dilationH, this->attrInfo.dilationW, LOAD3D_MAX_DILATION_H_W);
        return false;
    }

    bool padLoad3dInvalid = (static_cast<uint32_t>(this->attrInfo.padLeft) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padRight) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padUp) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padDown) > LOAD3D_MAX_PAD);
    if (padLoad3dInvalid) {
        TILING_ERROR_LOG(
            "Attrs not satisfying load3d's limits: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld,"\
            "which must <= %u.",
            this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft, this->attrInfo.padRight,
            LOAD3D_MAX_PAD);
        return false;
    }

    if (this->shapeInfo.orgkH > LOAD3D_MAX_FILTER_H_W || this->shapeInfo.orgkW > LOAD3D_MAX_FILTER_H_W) {
        TILING_ERROR_LOG("Weight shape not satisfy Load3D's limits: kh=%ld, kw=%ld, which must <= %u.",
                         this->shapeInfo.orgkH, this->shapeInfo.orgkW, LOAD3D_MAX_FILTER_H_W);
        return false;
    }

    auto k0Size = C0_BYTE_SIZE / g_dtypeSizeTab.at(this->descInfo.fMapType.dtype);
    uint64_t tmpkHWSize = this->shapeInfo.orgkH * this->shapeInfo.orgkW * k0Size;
    if (tmpkHWSize > LOAD3D_MAX_DDR2L1_SIZE) {
        TILING_ERROR_LOG("Weight shape not satisfying load3d's limits: kH*kW*k0=%lu, which must <= %u.",
                         tmpkHWSize, LOAD3D_MAX_DDR2L1_SIZE);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInstructionLimits()
{
    if (!CheckLoad3DLimits()) {
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckHF32()
{
    if (this->hf32TransMode_) {
        TILING_ERROR_LOG("hf32TransMode only support false.");
        return false;
    }
    return true;
}

void Conv3dTilingBase::SetFinalTilingBasicInfo(optiling::TConv3DTiling& tiling)
{
    // set origin shape info
    tiling.set_groups(this->attrInfo.groups);
    tiling.set_groupOpt(this->attrInfo.groupOpt);
    tiling.set_orgDo(this->shapeCalc.orgDo);
    tiling.set_orgCo(this->shapeInfo.orgCo);
    tiling.set_coutOpt(this->shapeInfo.coutOpt);
    tiling.set_orgHo(this->shapeCalc.orgHo);
    tiling.set_orgWo(this->shapeCalc.orgWo);
    tiling.set_orgCi(this->shapeInfo.orgCi);
    tiling.set_cinOpt(this->shapeInfo.cinOpt);
    tiling.set_orgDi(this->shapeInfo.orgDi);
    tiling.set_orgHi(this->shapeInfo.orgHi);
    tiling.set_orgWi(this->shapeInfo.orgWi);
    tiling.set_kernelD(this->shapeInfo.orgkD);
    tiling.set_kernelH(this->shapeInfo.orgkH);
    tiling.set_kernelW(this->shapeInfo.orgkW);
    tiling.set_orgHixWi(this->shapeInfo.orgHi * this->shapeInfo.orgWi);
    tiling.set_orgHoxWo(this->shapeCalc.orgHo * this->shapeCalc.orgWo);
    tiling.set_cin1xOriHixOriWixk0(this->shapeCalc.singleCi1 * this->shapeInfo.orgHi * this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_oriHixOriWixk0(this->shapeInfo.orgHi * this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_oriWixk0(this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_kernelHxkernelW(this->shapeInfo.orgkH * this->shapeInfo.orgkW);
    // set single core info
    tiling.set_singleCoreCo(this->shapeInfo.singleCo);
    tiling.set_singleCoreDo(this->shapeInfo.singleDo);
    if (outputOrder_ == M_Mode) {
        tiling.set_singleCoreM(this->shapeInfo.singleM);
    } else {
        tiling.set_singleCoreM(this->shapeInfo.singleHo);
    }
    tiling.set_singleCoreGroupOpt(this->shapeInfo.singleCoreGroupOpt);
    // set conv attr
    tiling.set_strideH(this->attrInfo.strideH);
    tiling.set_strideW(this->attrInfo.strideW);
    tiling.set_strideD(this->attrInfo.strideD);
    tiling.set_dilationH(this->attrInfo.dilationH);
    tiling.set_dilationW(this->attrInfo.dilationW);
    tiling.set_dilationD(this->attrInfo.dilationD);
    tiling.set_padHead(this->attrInfo.padHead);
    tiling.set_padTail(this->attrInfo.padTail);
    tiling.set_padUp(this->attrInfo.padUp);
    tiling.set_padDown(this->attrInfo.padDown);
    tiling.set_padLeft(this->attrInfo.padLeft);
    tiling.set_padRight(this->attrInfo.padRight);
    tiling.set_offsetx(this->attrInfo.offsetx);
}

void Conv3dTilingBase::SetFinalTilingDecisionInfo(optiling::TConv3DTiling& tiling)
{
    // set L1 tiling decision
    tiling.set_kAL1(this->l1TilingInfo.kAL1);
    tiling.set_kBL1(this->l1TilingInfo.kBL1);
    tiling.set_mAL1(this->l1TilingInfo.mAL1Value);
    tiling.set_nBL1(this->l1TilingInfo.nBL1Value);
    tiling.set_mAL1DivmL0(this->l1TilingInfo.mAL1DivmL0);
    tiling.set_nBL1DivnL0(this->l1TilingInfo.nBL1DivnL0);
    tiling.set_cin1InAL1(this->l1TilingInfo.cin1InAL1);
    tiling.set_kAL1Tail(this->l1TilingInfo.kAL1Tail);
    tiling.set_cin1InAL1Tail(this->l1TilingInfo.cin1InAL1Tail);
    tiling.set_KBL1Divk0(this->l1TilingInfo.KBL1Divk0);
    tiling.set_kBL1Tail(this->l1TilingInfo.kBL1Tail);
    tiling.set_KBL1TailDivk0(this->l1TilingInfo.KBL1TailDivk0);
    tiling.set_iterateMNOrder(static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    tiling.set_biasFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    tiling.set_fixpParamsFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    tiling.set_bl1BypassFlag(static_cast<uint8_t>(this->l1TilingInfo.isWeightBypass));
    tiling.set_al1FullLoad(static_cast<uint8_t>(this->l1TilingInfo.al1FullLoad));
    tiling.set_bl1FullLoad(static_cast<uint8_t>(this->l1TilingInfo.bl1FullLoad));

    // set L0 tiling decision
    tiling.set_mL0(this->l0TilingInfo.mL0);
    tiling.set_kL0(this->l0TilingInfo.kL0);
    tiling.set_nL0(this->l0TilingInfo.nL0);
    tiling.set_nL0xk0(this->l0TilingInfo.nL0xk0);
    tiling.set_kL0xorgCoAlignN0(this->l0TilingInfo.kL0xorgCoAlignN0);

    // set UB tiling decision
    tiling.set_mUB(this->ubTilingInfo.mUB);
    tiling.set_nUB(this->ubTilingInfo.nUB);
    tiling.set_quantType(this->quantType);
    tiling.set_scaleAndBiasLoadType(this->ubTilingInfo.scaleAndBiasLoadType);
 
    // set pb
    tiling.set_pBufferFlag(this->dbValue.pBufferFlag);
    // set hf32 mode
    tiling.set_hf32Enable(this->attrInfo.hf32Enable);
    tiling.set_hf32TransMode(0);
}

void Conv3dTilingBase::SetFinalTiling(optiling::TConv3DTiling& tiling)
{
    SetFinalTilingBasicInfo(tiling);
    SetFinalTilingDecisionInfo(tiling);
}

void Conv3dTilingBase::PrintTilingDataBasicInfo() const
{
    TILING_DEBUG_LOG("groups: %ld", this->attrInfo.groups);
    TILING_DEBUG_LOG("groupOpt: %ld", this->attrInfo.groupOpt);
    TILING_DEBUG_LOG("orgDo: %lu", this->shapeCalc.orgDo);
    TILING_DEBUG_LOG("orgCo: %ld", this->shapeInfo.orgCo);
    TILING_DEBUG_LOG("coutOpt: %ld", this->shapeInfo.coutOpt);
    TILING_DEBUG_LOG("orgHo: %lu", this->shapeCalc.orgHo);
    TILING_DEBUG_LOG("orgWo: %lu", this->shapeCalc.orgWo);
    TILING_DEBUG_LOG("orgCi: %ld", this->shapeInfo.orgCi);
    TILING_DEBUG_LOG("cinOpt: %ld", this->shapeInfo.cinOpt);
    TILING_DEBUG_LOG("orgDi: %ld", this->shapeInfo.orgDi);
    TILING_DEBUG_LOG("orgHi: %ld", this->shapeInfo.orgHi);
    TILING_DEBUG_LOG("orgWi: %ld", this->shapeInfo.orgWi);
    TILING_DEBUG_LOG("orgkD: %ld", this->shapeInfo.orgkD);
    TILING_DEBUG_LOG("orgkH: %ld", this->shapeInfo.orgkH);
    TILING_DEBUG_LOG("orgkW: %ld", this->shapeInfo.orgkW);
    TILING_DEBUG_LOG("singleCo: %ld", this->shapeInfo.singleCo);
    TILING_DEBUG_LOG("singleDo: %ld", this->shapeInfo.singleDo);
    TILING_DEBUG_LOG("singleM: %ld", this->shapeInfo.singleM);
    TILING_DEBUG_LOG("singleCoreGroupOpt: %ld", this->shapeInfo.singleCoreGroupOpt);
    TILING_DEBUG_LOG("strideH: %ld", this->attrInfo.strideH);
    TILING_DEBUG_LOG("strideW: %ld", this->attrInfo.strideW);
    TILING_DEBUG_LOG("strideD: %ld", this->attrInfo.strideD);
    TILING_DEBUG_LOG("dilationH: %ld", this->attrInfo.dilationH);
    TILING_DEBUG_LOG("dilationW: %ld", this->attrInfo.dilationW);
    TILING_DEBUG_LOG("dilationD: %ld", this->attrInfo.dilationD);
    TILING_DEBUG_LOG("padHead: %ld", this->attrInfo.padHead);
    TILING_DEBUG_LOG("padTail: %ld", this->attrInfo.padTail);
    TILING_DEBUG_LOG("padUp: %ld", this->attrInfo.padUp);
    TILING_DEBUG_LOG("padDown: %ld", this->attrInfo.padDown);
    TILING_DEBUG_LOG("padLeft: %ld", this->attrInfo.padLeft);
    TILING_DEBUG_LOG("padRight: %ld", this->attrInfo.padRight);
    TILING_DEBUG_LOG("mAL1DivmL0: %lu", this->l1TilingInfo.mAL1DivmL0);
    TILING_DEBUG_LOG("nBL1DivnL0: %lu", this->l1TilingInfo.nBL1DivnL0);
    TILING_DEBUG_LOG("cin1InAL1: %lu", this->l1TilingInfo.cin1InAL1);
    TILING_DEBUG_LOG("kAL1Tail: %lu", this->l1TilingInfo.kAL1Tail);
    TILING_DEBUG_LOG("cin1InAL1Tail: %lu", this->l1TilingInfo.cin1InAL1Tail);
    TILING_DEBUG_LOG("KBL1Divk0: %lu", this->l1TilingInfo.KBL1Divk0);
    TILING_DEBUG_LOG("kBL1Tail: %lu", this->l1TilingInfo.kBL1Tail);
    TILING_DEBUG_LOG("KBL1TailDivk0: %lu", this->l1TilingInfo.KBL1TailDivk0);
    TILING_DEBUG_LOG("nL0xk0: %lu", this->l0TilingInfo.nL0xk0);
    TILING_DEBUG_LOG("kL0xorgCoAlignN0: %lu", this->l0TilingInfo.kL0xorgCoAlignN0);
    TILING_DEBUG_LOG("offsetx: %d", this->attrInfo.offsetx);
}

void Conv3dTilingBase::PrintTilingDataDecision() const
{
    TILING_DEBUG_LOG("kAL1: %lu", this->l1TilingInfo.kAL1);
    TILING_DEBUG_LOG("kBL1: %lu", this->l1TilingInfo.kBL1);
    TILING_DEBUG_LOG("mAL1Value: %lu", this->l1TilingInfo.mAL1Value);
    TILING_DEBUG_LOG("nBL1Value: %lu", this->l1TilingInfo.nBL1Value);
    TILING_DEBUG_LOG("iterateMNOrder: %u", static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    TILING_DEBUG_LOG("biasFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    TILING_DEBUG_LOG("fixpParamsFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    TILING_DEBUG_LOG("isWeightBypass: %u", static_cast<uint8_t>(this->l1TilingInfo.isWeightBypass));
    TILING_DEBUG_LOG("al1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.al1FullLoad));
    TILING_DEBUG_LOG("bl1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.bl1FullLoad));
    TILING_DEBUG_LOG("mL0: %lu", this->l0TilingInfo.mL0);
    TILING_DEBUG_LOG("kL0: %lu", this->l0TilingInfo.kL0);
    TILING_DEBUG_LOG("nL0: %lu", this->l0TilingInfo.nL0);
    TILING_DEBUG_LOG("mUB: %lu", this->ubTilingInfo.mUB);
    TILING_DEBUG_LOG("nUB: %lu", this->ubTilingInfo.nUB);
    TILING_DEBUG_LOG("quantType: %u", this->quantType);
    TILING_DEBUG_LOG("scaleAndBiasLoadType: %u", this->ubTilingInfo.scaleAndBiasLoadType);
    TILING_DEBUG_LOG("pBufferFlag: %lu", this->dbValue.pBufferFlag);
    TILING_DEBUG_LOG("hf32Enable: %u", this->attrInfo.hf32Enable);
    TILING_DEBUG_LOG("hf32TransMode: %u", this->attrInfo.hf32TransMode);
}
void Conv3dTilingBase::PrintTilingData() const
{
    PrintTilingDataBasicInfo();
    PrintTilingDataDecision();
}

void Conv3dTilingBase::GetCubeInfo()
{
    this->cubeInfo.m0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_M_INDEX);
    this->cubeInfo.k0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_K_INDEX);
    this->cubeInfo.n0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_N_INDEX);
    this->cubeInfo.biasType = CUBE_TYPE_TAB.ToBiasType(this->descInfo.fMapType.dtype);
    this->cubeInfo.madType = CUBE_TYPE_TAB.ToMadType(this->descInfo.fMapType.dtype);
    this->cubeInfo.minBurstNum = MIN_BURST_SIZE / g_dtypeSizeTab.at(descInfo.fMapType.dtype);
}

bool Conv3dTilingBase::ShapeInitCalc()
{
    this->shapeCalc.orgHo = (this->shapeInfo.orgHi + this->attrInfo.padUp + this->attrInfo.padDown -
                             this->attrInfo.dilationH * (this->shapeInfo.orgkH - 1) - 1) / this->attrInfo.strideH + 1;
    this->shapeCalc.orgWo = (this->shapeInfo.orgWi + this->attrInfo.padLeft + this->attrInfo.padRight -
                             this->attrInfo.dilationW * (this->shapeInfo.orgkW - 1) - 1) / this->attrInfo.strideW + 1;
    this->shapeCalc.orgDo = (this->shapeInfo.orgDi + this->attrInfo.padHead + this->attrInfo.padTail -
                             this->attrInfo.dilationD * (this->shapeInfo.orgkD - 1) - 1) / this->attrInfo.strideD + 1;
    if (this->shapeCalc.orgHo <= 0 || this->shapeCalc.orgWo <= 0 || this->shapeCalc.orgDo <= 0) {
        TILING_ERROR_LOG("Illegal org output shapes cal get: Ho=%lu, Wo=%lu, Do=%lu, which should > 0.",
                         this->shapeCalc.orgHo, this->shapeCalc.orgWo, this->shapeCalc.orgDo);
        return false;
    }
    this->shapeCalc.singleCi1 = CeilDiv(this->shapeInfo.singleCi, this->cubeInfo.k0);
    this->shapeCalc.singleCo1 = CeilDiv(this->shapeInfo.singleCo, this->cubeInfo.n0);
    this->shapeCalc.singleM1 = CeilDiv(this->shapeInfo.singleM, this->cubeInfo.m0);

    return true;
}

bool Conv3dTilingBase::CheckParamsOverflow()
{
    uint64_t prod;
    bool isOverflow = MulWithOverflowCheck(
        prod, static_cast<uint64_t>(this->shapeInfo.orgDi),
        static_cast<uint64_t>(this->attrInfo.groupOpt), CeilDiv(this->shapeInfo.cinOpt, this->cubeInfo.k0),
        static_cast<uint64_t>(this->shapeInfo.orgHi), static_cast<uint64_t>(this->shapeInfo.orgWi),
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.fMapType.dtype)
    ) || MulWithOverflowCheck(
        prod, static_cast<uint64_t>(this->attrInfo.groupOpt), static_cast<uint64_t>(this->shapeInfo.singlekD),
        CeilDiv(this->shapeInfo.cinOpt, this->cubeInfo.k0),
        static_cast<uint64_t>(this->shapeInfo.singlekH), static_cast<uint64_t>(this->shapeInfo.singlekW),
        CeilDiv(this->shapeInfo.coutOpt, this->cubeInfo.n0), static_cast<uint64_t>(this->cubeInfo.n0) *
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.weightType.dtype)
    ) || MulWithOverflowCheck(
        prod, this->shapeCalc.orgDo, static_cast<uint64_t>(this->attrInfo.groupOpt),
        CeilDiv(this->shapeInfo.coutOpt, this->cubeInfo.k0), this->shapeCalc.orgHo, this->shapeCalc.orgWo,
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.outputType.dtype)
    );
    if (isOverflow) {
        TILING_ERROR_LOG("Overflow detected, fmap or weight size exceeds UINT64_MAX.");
        return false;
    }
    return true;
}

} // namespace conv3d_tiling
