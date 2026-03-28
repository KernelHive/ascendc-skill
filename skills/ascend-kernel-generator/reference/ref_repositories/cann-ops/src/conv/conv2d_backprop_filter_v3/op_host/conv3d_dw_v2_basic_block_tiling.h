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
 * \file conv3d_dw_v2_basic_block_tiling.h
 * \brief
 */
#ifndef CONV3D_DW_BASIC_BLOCK_TILING_H
#define CONV3D_DW_BASIC_BLOCK_TILING_H
#include "tiling/tiling_base.h"
#include "cache_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "cube/include/cube_run_info.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "conv3d_backprop_filter_v2_tiling.h"

namespace optiling {

struct BasicBlockTilingParams
{
    uint32_t usedCoreNum = 0;
    uint64_t totalCnt = 0;
    uint32_t blockBaseM = 128;
    uint32_t blockBaseN = 128;
    uint32_t blockBaseK = 128;
    uint64_t singleCoreM = 128;
    uint64_t singleCoreN = 128;
    uint64_t singleCoreK = 128;
    uint32_t depthA1 = 1;
    uint32_t depthB1 = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t dbL1A = 1;
    uint32_t dbL1B = 1;
    uint32_t dbL0C = 1;
    uint32_t iterateOrder = 0;
    uint32_t coreBindDirection = 1;
    uint32_t coreBindOrder = 1;
};

struct MatMulInfo
{
    uint64_t mValue = 0;
    uint64_t kValue = 0;
    uint64_t nValue = 0;
};

class Conv3DDWV2BasicBlockTiling : public conv_bp_v2::Conv3DBackpropFilterV2Tiling {
public:
    explicit Conv3DDWV2BasicBlockTiling(gert::TilingContext *context) : Conv3DBackpropFilterV2Tiling(context) { Reset(); }
    ~Conv3DDWV2BasicBlockTiling() override = default;

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void InitBaseMNK();
    void UpdateStepMNK();
    void UpdateSingleCoreInfo();
    void MultiCoreSplitK();
    void MultiCoreSplitMN();
    void PrintBasickBlockTilingData();
    void SetBasicBlockAttrsTiling();
    void ShrinkBaseBlock();
    void ShrinkBlockBaseMN();
    bool ShrinkBlockBaseK();
    uint32_t CalculateBl1Cin1CopyLen(uint32_t newBaseN);
    uint64_t CalculateL1SizeGap();
    uint64_t IsCurBlockL1Invalid();
    uint64_t CalBL1Bound();

    BasicBlockTilingParams blockTiling_;
    MatMulInfo mmInfo_;
};
}  // namespace optiling
#endif  // CONV3D_DW_BASIC_BLOCK_TILING_H