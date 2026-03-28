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
 * \file conv3d_backprop_filter_v2_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_TILING_H
#define CONV3D_BACKPROP_FILTER_V2_TILING_H
#include "tiling/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "cube/include/cube_run_info.h"
#include "cube/algorithm/hash/tiling_cache.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TConv3DDwTiling)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, cin1G);
    TILING_DATA_FIELD_DEF(uint32_t, cout1G);
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
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, channelSize);
    TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, baseM);
    TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TILING_DATA_FIELD_DEF(uint32_t, baseN);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, stepM);
    TILING_DATA_FIELD_DEF(uint32_t, stepN);
    TILING_DATA_FIELD_DEF(uint32_t, stepKa);
    TILING_DATA_FIELD_DEF(uint32_t, stepKb);
    TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
    TILING_DATA_FIELD_DEF(uint32_t, bl1Bound);
    TILING_DATA_FIELD_DEF(uint32_t, hf32Flag);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreDk);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TConv3DDwTilingOp, TConv3DDwTiling);

BEGIN_TILING_DATA_DEF(Conv3DBackpropFilterV2Params)
    TILING_DATA_FIELD_DEF(uint64_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, dkDim);
    TILING_DATA_FIELD_DEF(uint32_t, totalL1Size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropFilterV2ParamsOp, Conv3DBackpropFilterV2Params)

BEGIN_TILING_DATA_DEF(TConv3DDwBasicBlockTiling)
    TILING_DATA_FIELD_DEF(uint32_t, coreBindOrder);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TConv3DDwBasicBlockTilingOp, TConv3DDwBasicBlockTiling)

BEGIN_TILING_DATA_DEF(Conv3DBackpropFilterV2TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(Conv3DBackpropFilterV2Params, params);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DDwTiling, dwTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DDwBasicBlockTiling, basicBlockTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropFilterV2, Conv3DBackpropFilterV2TilingData)

struct TilingValueDw
{
    uint64_t batchDim;
    uint32_t groupDim;
    uint32_t dkDim;
    uint32_t mDim;
    uint32_t kDim;
    uint32_t nDim;
    uint32_t singleCoreBatch;
    uint32_t singleCoreGroup;
    uint32_t singleCoreDk;
    uint32_t singleCoreCout;
    uint64_t singleCoreCin;
    uint32_t singleCoreHo;
    uint32_t al0Pbuffer;
    uint32_t bl0Pbuffer;
    uint32_t cl0Pbuffer;
    uint32_t al1Pbuffer;
    uint32_t bl1Pbuffer;
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t stepKa;
    uint32_t stepKb;
    uint32_t iterateOrder;
    uint32_t bl1Bound;
};

struct CoreDimDw
{
    int32_t batchDim = 1;
    int32_t dDim = 1;
    int32_t mDim = 1;
    int32_t kDim = 1;
    int32_t nDim = 1;
};

class Conv3DBackpropFilterV2Tiling : public TilingBaseClass {
public:
    explicit Conv3DBackpropFilterV2Tiling(gert::TilingContext *context) : TilingBaseClass(context) { Reset(); }
    ~Conv3DBackpropFilterV2Tiling() override = default;

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
    bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeInputs();
    void SetShapeTiling(TConv3DDwTiling &dwt);
    void SetAttrTiling(TConv3DDwTiling &dwt);
    void GetBatchDim(CoreDimDw& coreDim, int32_t dMaxFactor, int32_t batchDepthMaxFactor);
    void GetCoreDim(CoreDimDw& coreDim, uint32_t curCoreNum);
    void SetTilingParamByDimInfo(TilingValueDw& tilingParams, CoreDimDw& coreDim);
    bool CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize);
    int32_t GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLists);
    void CalCoreDimTiling(TilingValueDw& tilingParams, bool& enableTbeBlock);
    uint32_t CalCin(const uint32_t& nL1Size);
    int64_t CalBL1Bound(TilingValueDw &tilingParams);
    void UpdateBaseBlock(uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, TilingValueDw& tilingParams);
    void UpdateBaseStep(uint32_t &stepKa, uint32_t &stepKb, TilingValueDw &tilingParams);
    bool CalBaseBlockTiling(TilingValueDw& tilingParams);
    void InitTilingValue(TilingValueDw& tilingParams);
    void InitSinglecoreParam(TilingValueDw& tilingParams);
    void InitCalTilingValue(TilingValueDw& tilingParams);
    void SetTilingValue(TConv3DDwTiling& dwt, const TilingValueDw& tilingParams);
    void SetDwTilingFromTbeTiling();
    bool GetTbeTiling();
    bool SetPlatformInfoForTbeTiling(cachetiling::PlatformInfo& platformInstance);
    void PrintTilingData();
    bool SetTbeTilingParam(cachetiling::Conv2DBpFilterTilingParam& tilingParams);
    void ReCalDilation(cachetiling::Conv3DBpFilterTilingParam &tilingParams);

    bool enableDeterministic_ = false;
    uint32_t libApiWorkSpaceSize_ = 0;
    uint32_t coreNum_ = 1;
    ge::DataType aDtype_ = ge::DT_FLOAT16;
    const char *opName_ = "";
    int32_t dtypeByte_ = 2;
    Conv3DBackpropFilterV2TilingData tilingData_;
    Conv3dBpFilterRunInfo runInfo_;
    cachetiling::Conv3DBpFilterTiling tbeTiling_;
};
}  // namespace optiling
#endif  // CONV3D_BACKPROP_FILTER_V2_TILING_H