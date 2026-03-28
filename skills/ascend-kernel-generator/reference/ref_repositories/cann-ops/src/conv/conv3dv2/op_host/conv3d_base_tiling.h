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
 * \file conv3d_base_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BASE_TILING_H

#include "tiling/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "conv3dv2_tiling.h"
#include "conv3d_tiling.h"
#include "conv3d_tiling_utils.h"

namespace optiling {
struct CubeTilingCommonParseInfo {
    int32_t fmapC1 = 0;
    bool correctRangeFlag = false;
    std::string tilingType = "";
    std::vector<std::string> varMap;
    std::vector<std::string> tilingKeyList;
    std::vector<std::vector<std::string>> customVarsList;
    std::vector<std::vector<int64_t>> defaultRangeList;
    std::vector<std::vector<int64_t>> tilingRangeList;
    std::vector<int32_t> blockDimList;
    std::vector<std::vector<int32_t>> repoSeedsList;
    std::vector<std::vector<int64_t>> repoRangeList;
    std::vector<std::vector<int64_t>> costRangeList;
};

struct Conv3DTilingParseInfo: CubeTilingCommonParseInfo {
        uint32_t aicoreNum = 0;
        uint64_t l2Size = 0;
        uint64_t l1Size = 0;
        uint64_t l0aSize = 0;
        uint64_t l0bSize = 0;
        uint64_t l0cSize = 0;
        uint64_t ubSize = 0;
        uint64_t btSize = 0;
        uint64_t l2Rate = 0;
        std::string socVersion = "";
        std::string shortSocVersion = "";
    };

struct Conv3DAttrInfo {
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t dilationD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t strideD = 1;
    uint32_t padh = 0;
    uint32_t padt = 0;
    uint32_t padu = 0;
    uint32_t padd = 0;
    uint32_t padl = 0;
    uint32_t padr = 0;
    uint32_t groups = 0;
    uint64_t groupOpt = 0;
    bool compileHf32Mode = false;
};

struct Conv3DOrignalFormat {
    // for fmap
    uint32_t FORMAT_FMAP_N_INDEX = 0;
    uint32_t FORMAT_FMAP_C_INDEX = 0;
    uint32_t FORMAT_FMAP_D_INDEX = 0;
    uint32_t FORMAT_FMAP_H_INDEX = 0;
    uint32_t FORMAT_FMAP_W_INDEX = 0;
    // for weight
    uint32_t FORMAT_WEIGHT_N_INDEX = 0;
    uint32_t FORMAT_WEIGHT_C_INDEX = 0;
    uint32_t FORMAT_WEIGHT_D_INDEX = 0;
    uint32_t FORMAT_WEIGHT_H_INDEX = 0;
    uint32_t FORMAT_WEIGHT_W_INDEX = 0;
    // for stride and dilation
    uint32_t FORMAT_DATA_D_INDEX = 0;
    uint32_t FORMAT_DATA_H_INDEX = 0;
    uint32_t FORMAT_DATA_W_INDEX = 0;
};

namespace conv3d_ops_tiling {
class Conv3dBaseTiling : public TilingBaseClass {
public:
    explicit Conv3dBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {};
    ~Conv3dBaseTiling() override {};

protected:
    bool IsCapable() override {return true;};
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    conv3d_tiling::Conv3dTiling conv3dApiTiling_;
    Conv3DTilingParseInfo opInfo_;
    Conv3DTilingParseInfo opRunInfo_;
    Conv3DAscendcShapesInfo shapeInfo_;
    Conv3DAttrInfo attrInfo_;
    Conv3DTilingData tilingData_;
    Conv3DDescInfo descInfo_;
    Conv3DTilingFlag flagInfo_;
    Conv3DOrignalFormat originalFormat_;

    // blockdim decision
    BlockDimRange blockDimRanges;
    BlockDimConstParas blockDimConst;
    std::vector<uint32_t> blockDimInit;
    BlockDimRes blockDimRes;

    bool isPointWise = false;
    int8_t outputOrder_ = M_Mode;

private:
    bool CheckDims(const gert::Shape& inputShape);
    ge::graphStatus CheckStrideLegal();
    ge::graphStatus CheckDilationLegal();
    ge::graphStatus CheckPadLegal();
    ge::graphStatus CheckFmapShape();
    ge::graphStatus CheckFmapNCDHWShape();
    ge::graphStatus CheckWeightShape();
    ge::graphStatus CheckWeightNCDHWShape();
    ge::graphStatus CheckInputShapeWithPad();
    ge::graphStatus CheckScaleShape();
    ge::graphStatus CheckBiasShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus CheckOutputNCDHWShape();
    ge::graphStatus CheckInputDesc();
    ge::graphStatus CheckParamsDtype();
    ge::graphStatus CheckLoad3DLimits();
    ge::graphStatus CheckInstructionLimits();
    ge::graphStatus InitConv3dApiTiling();
    ge::graphStatus GetConv3dApiTiling();
    ge::graphStatus CheckInputLimitsHwMode();
    ge::graphStatus SetTilingKey();
    ge::graphStatus GetGroupConvOpt();
    ge::graphStatus CheckGroupOpt();
    ge::graphStatus CheckParamsOverflow();
    ge::graphStatus CheckPointWise();
    ge::graphStatus InitOutputOrder();
    uint64_t CalcMinL1LoadSize(int8_t outputOrder);
    void SetSingleOutputShapeByMode();
    void InitConv3dOriginFormat();
    void InitPointWiseFlag();
    void GetShapeInfo();
    void GetAttrsInfo();
    void GetDescInfo();
    void PrintTilingInfo();
    void PrintOpTilingData();
    void PrintApiTilingDataShapeInfo();
    void PrintApiTilingDataDecisionInfo();
    void PrintApiTilingDataScalarInfo();
    void PrintLibApiTilingData();

    // blockdim decision
    bool IsExceedMinBurstNum(uint64_t input);
    uint64_t GetMinBurstNum();
    uint64_t CalcFixParamSize() const;
    uint64_t CalcTotalCost(uint32_t batchDim, uint32_t mDim, uint32_t nDim, uint32_t doDim, uint32_t groupDim);
    void BlockDimDecision();
    void GetBlockDimRange();
    void GetBlockDimInit();
    void BlockDimDecisionBackTrack(const std::vector<std::vector<uint32_t>> &inputRanges, uint32_t rangeIdx,
                                   std::vector<uint32_t> &record);
    void CoreBlockDimDecision();
    void BlockDimFactorMix(uint32_t orgDim, std::vector<uint32_t> &inputRange, const std::vector<uint32_t> &mixRange);
    void GetBlockDimRangeforGroupRange(std::vector<uint32_t> &groupRange);
};
}
}
#endif