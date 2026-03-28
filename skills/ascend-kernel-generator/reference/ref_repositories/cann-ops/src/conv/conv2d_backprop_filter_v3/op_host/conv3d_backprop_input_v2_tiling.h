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
 * \file conv3d_backprop_input_v2_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_TILING_H
#define CONV3D_BACKPROP_INPUT_V2_TILING_H

#include "cache_tiling.h"
#include "conv3d_backprop_input.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "cube/include/cube_run_info.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TConv3DInputV2Tiling)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, cin);
TILING_DATA_FIELD_DEF(uint32_t, cout);
TILING_DATA_FIELD_DEF(uint32_t, cout1);
TILING_DATA_FIELD_DEF(uint32_t, cin1);
TILING_DATA_FIELD_DEF(uint32_t, cout1G);
TILING_DATA_FIELD_DEF(uint32_t, cin1G);
TILING_DATA_FIELD_DEF(uint32_t, c0);
TILING_DATA_FIELD_DEF(uint32_t, c0Bits);
TILING_DATA_FIELD_DEF(uint32_t, dout);
TILING_DATA_FIELD_DEF(uint32_t, ho);
TILING_DATA_FIELD_DEF(uint32_t, wo);
TILING_DATA_FIELD_DEF(uint32_t, di);
TILING_DATA_FIELD_DEF(uint32_t, hi);
TILING_DATA_FIELD_DEF(uint32_t, wi);
TILING_DATA_FIELD_DEF(uint32_t, dk);
TILING_DATA_FIELD_DEF(uint32_t, hk);
TILING_DATA_FIELD_DEF(uint32_t, wk);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, strideD);
TILING_DATA_FIELD_DEF(uint32_t, strideH);
TILING_DATA_FIELD_DEF(uint32_t, strideW);
TILING_DATA_FIELD_DEF(uint32_t, padFront);
TILING_DATA_FIELD_DEF(uint32_t, padBack);
TILING_DATA_FIELD_DEF(uint32_t, padUp);
TILING_DATA_FIELD_DEF(uint32_t, padDown);
TILING_DATA_FIELD_DEF(uint32_t, padLeft);
TILING_DATA_FIELD_DEF(uint32_t, padRight);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadTail);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadUp);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadDown);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadLeft);
TILING_DATA_FIELD_DEF(uint32_t, backpropPadRight);
TILING_DATA_FIELD_DEF(uint32_t, dilationD);
TILING_DATA_FIELD_DEF(uint32_t, dilationH);
TILING_DATA_FIELD_DEF(uint32_t, dilationW);
TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout1);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCin1);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreDin);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, baseD);
TILING_DATA_FIELD_DEF(uint32_t, baseBatch);
TILING_DATA_FIELD_DEF(uint32_t, baseGroup);
TILING_DATA_FIELD_DEF(uint32_t, stepM);
TILING_DATA_FIELD_DEF(uint32_t, stepN);
TILING_DATA_FIELD_DEF(uint32_t, stepKa);
TILING_DATA_FIELD_DEF(uint32_t, stepKb);
TILING_DATA_FIELD_DEF(uint32_t, stepBatch);
TILING_DATA_FIELD_DEF(uint32_t, stepGroup);
TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
TILING_DATA_FIELD_DEF(int32_t, hf32Flag);
TILING_DATA_FIELD_DEF(int32_t, initOutputFlag);
TILING_DATA_FIELD_DEF(int32_t, reserved);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TConv3DInputV2TilingOp, TConv3DInputV2Tiling);

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2Params)
TILING_DATA_FIELD_DEF(uint32_t, batchDim);
TILING_DATA_FIELD_DEF(uint32_t, groupDim);
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, dDim);
TILING_DATA_FIELD_DEF(uint64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropInputV2ParamsOp, Conv3DBackpropInputV2Params)

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(Conv3DBackpropInputV2Params, params);
TILING_DATA_FIELD_DEF_STRUCT(TConv3DInputV2Tiling, conv3DDxTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropInputV2, Conv3DBackpropInputV2TilingData)
REGISTER_TILING_DATA_CLASS(Conv3DTransposeV2, Conv3DBackpropInputV2TilingData)

struct TilingValue
{
    uint64_t coreNum;
    uint32_t batchDim;
    uint32_t groupDim;
    uint32_t mDim;
    uint32_t kDim;
    uint32_t nDim;
    uint32_t dDim;
    uint64_t singleCoreBatch;
    uint32_t singleCoreGroup;
    uint64_t singleCoreM;
    uint32_t singleCoreCout;
    uint32_t singleCoreCout1;
    uint64_t singleCoreCin;
    uint32_t singleCoreCin1;
    uint32_t singleCoreDin;
    uint32_t singleCoreHo;
    uint32_t al0Pbuffer;
    uint32_t bl0Pbuffer;
    uint32_t cl0Pbuffer;
    uint32_t al1Pbuffer;
    uint32_t bl1Pbuffer;
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
    uint32_t baseD;
    uint32_t baseBatch;
    uint32_t baseGroup;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t stepKa;
    uint32_t stepKb;
    uint32_t stepBatch;
    uint32_t stepGroup;
    uint32_t iterateOrder;
};

class Conv3DBackpropInputV2Tiling : public TilingBaseClass {
public:
    explicit Conv3DBackpropInputV2Tiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~Conv3DBackpropInputV2Tiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();
    ge::graphStatus CheckContext();
    bool AnalyzeDtype() const;
    void InitCompileInfo() const;
    void SetDxTilingFromTbeTiling();
    bool GetTbeTiling();
    bool SetPlatformInfoForTbeTiling(cachetiling::PlatformInfo &platformInstance);
    void PrintTilingData();
    void PrintTbeTiling();
    int32_t GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLits) const;
    int32_t CalFmapH(const int32_t& mL1Size) const;
    void GetCoreDim(int32_t& batchDim, int32_t& dDim, int32_t& mDim, int32_t& nDim, const int32_t curCoreNum);
    void UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValue& tilingParams);
    void UpdateBaseStep(uint32_t& stepKa, uint32_t& stepKb, TilingValue& tilingParams);
    void CalCoreDimTiling(TilingValue& tilingParams, const uint32_t coreNum, bool& enableTbeBlock);
    void SetTilingParamByDimInfo(TilingValue& tilingParams,
        const int32_t batchDim, const int32_t dDim, const int32_t mDim, const int32_t nDim);
    bool CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize);
    void AlignCout1(uint32_t &cout1A, uint32_t &cout1B, bool adaptFP32);
    void UpdateStepFp32(uint32_t &stepKa, uint32_t &stepKb, TilingValue &tilingParams);
    bool CalBaseBlockTiling(TilingValue& tilingParams);
    void CalTbeBlockTiling(TilingValue& tilingParams);
    void InitTilingValue(TilingValue& tilingParams, const uint32_t coreNum);
    void SetRunInfoTiling(TConv3DInputV2Tiling& dxt);
    void SetBackpropPadInfo(TConv3DInputV2Tiling& dxt);
    void SetTilingValue(TConv3DInputV2Tiling& dxt, const TilingValue& tilingParams);
    bool SetTbeTilingParam(cachetiling::Conv3DBpInputTilingParam &tilingParams);
    bool CheckAttrs(Conv3DDxParas &conv3ddxParas);
    bool CheckOutputHeight(Conv3DDxParas &conv3ddxParas);
    bool CheckPadRange(Conv3DDxParas &conv3ddxParas);
    bool CheckTranspose() const;
    bool CheckKernelSplitEnable() const;
    bool GetImplMode(Conv3DDxParas &conv3ddxParas);
    void SetInitOutput(const Conv3DDxParas &conv3ddxParas);

    uint8_t loadB2Condition_ = 0;
    bool enableKernelSplit_ = false;
    bool c0DbFlag_ = false;
    bool useBasicBlock_ = false;
    int32_t coreNum_ = 1;
    int32_t initOutputFlag = 0;

    int32_t blockSize_ = 16;
    uint32_t dtypeByte_ = 2;
    const char *opName_ = "";
    cachetiling::OpType opType_ = cachetiling::kConv3DBackpropInput;
    Conv3DBackpropInputV2TilingData tilingData_ = {};
    RunInfoPara runInfo_ = {};
    cachetiling::Conv3DBpInputTiling tbeTiling_ = {};
};

class Conv3DTransposeV2Tiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv3DTransposeV2Tiling(gert::TilingContext *context) : Conv3DBackpropInputV2Tiling(context)
    {
        Reset();
        opType_ = cachetiling::kConv3DTranspose;
    }
    ~Conv3DTransposeV2Tiling() override = default;
};
}  // namespace optiling
#endif  // CONV3D_BACKPROP_INPUT_V2_TILING_H