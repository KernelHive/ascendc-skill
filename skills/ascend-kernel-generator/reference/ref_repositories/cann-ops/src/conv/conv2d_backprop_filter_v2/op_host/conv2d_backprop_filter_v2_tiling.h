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
 * \file conv2d_backprop_filter_v2_tiling.h
 * \brief
 */
#ifndef CONV2D_BACKPROP_FILTER_V2_TILING_H
#define CONV2D_BACKPROP_FILTER_V2_TILING_H
#include "tiling/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "cube/include/cube_run_info.h"
#include "cube/algorithm/hash/tiling_cache.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TConvTiling)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, Cin);
    TILING_DATA_FIELD_DEF(uint32_t, Cout);
    TILING_DATA_FIELD_DEF(uint32_t, Ho);
    TILING_DATA_FIELD_DEF(uint32_t, Wo);
    TILING_DATA_FIELD_DEF(uint32_t, Hi);
    TILING_DATA_FIELD_DEF(uint32_t, Wi);
    TILING_DATA_FIELD_DEF(uint32_t, Hk);
    TILING_DATA_FIELD_DEF(uint32_t, Wk);
    TILING_DATA_FIELD_DEF(uint32_t, group);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, padT);
    TILING_DATA_FIELD_DEF(uint32_t, padB);
    TILING_DATA_FIELD_DEF(uint32_t, padL);
    TILING_DATA_FIELD_DEF(uint32_t, padR);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, channelSize);
    TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCin);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
    TILING_DATA_FIELD_DEF(uint32_t, baseM);
    TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TILING_DATA_FIELD_DEF(uint32_t, baseN);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, baseBatch);
    TILING_DATA_FIELD_DEF(uint32_t, baseGroup);
    TILING_DATA_FIELD_DEF(uint32_t, stepM);
    TILING_DATA_FIELD_DEF(uint32_t, stepN);
    TILING_DATA_FIELD_DEF(uint32_t, stepKa);
    TILING_DATA_FIELD_DEF(uint32_t, stepKb);
    TILING_DATA_FIELD_DEF(uint32_t, stepBatch);
    TILING_DATA_FIELD_DEF(uint32_t, stepGroup);
    TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TConvTilingOp, TConvTiling);

BEGIN_TILING_DATA_DEF(Conv2DBackpropFilterV2Params)
    TILING_DATA_FIELD_DEF(uint32_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2DBackpropFilterV2ParamsOp, Conv2DBackpropFilterV2Params)

BEGIN_TILING_DATA_DEF(Conv2DBackpropFilterV2TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(Conv2DBackpropFilterV2Params, params);
    TILING_DATA_FIELD_DEF_STRUCT(TConvTiling, dwTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2DBackpropFilterV2, Conv2DBackpropFilterV2TilingData)

class Conv2DBackpropFilterV2Tiling : public TilingBaseClass {
public:
    explicit Conv2DBackpropFilterV2Tiling(gert::TilingContext *context) : TilingBaseClass(context) { Reset(); }
    ~Conv2DBackpropFilterV2Tiling() override = default;

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
    void SetDwTilingFromTbeTiling();
    bool GetTbeTiling();
    bool SetPlatformInfoForTbeTiling(cachetiling::PlatformInfo& platformInstance);
    void PrintTilingData();
    void PrintTbeTiling();
    bool SetTbeTilingParam(cachetiling::Conv2DBpFilterTilingParam& tilingParams);
    void GetChannelVal(cachetiling::Conv2DBpFilterTilingParam& tilingParams);
    bool CheckTilingData() const;

    int32_t blockSize_;
    uint32_t channelVal_;
    const char *opName_;
    Conv2DBackpropFilterV2TilingData tilingData_;
    Conv2dBpFilterRunInfo runInfo_;
    cachetiling::Conv2DBpFilterTiling tbeTiling_;
    Conv2DBackPropCompileInfo compileInfo_;
};
}  // namespace optiling
#endif  // CONV2D_BACKPROP_FILTER_V2_TILING_H