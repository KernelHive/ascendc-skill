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
 * \file weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2CustomNzSplitKTilingData)

TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, reverse1);
TILING_DATA_FIELD_DEF(uint16_t, reverse2);
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);

TILING_DATA_FIELD_DEF(uint16_t, postSingleN); // 后处理单次UB的N方向计算量
TILING_DATA_FIELD_DEF(uint16_t, postSingleM); // 后处理单次UB的M方向计算量

TILING_DATA_FIELD_DEF(uint32_t, vecSingleN); // vec单次N方向计算量
TILING_DATA_FIELD_DEF(uint32_t, singleK);  // vec/cube单次K方向计算量
TILING_DATA_FIELD_DEF(uint32_t, cubeSingleM); // cube单次M方向计算量
TILING_DATA_FIELD_DEF(uint32_t, cubeSingleN); // cube单次N方向计算量
TILING_DATA_FIELD_DEF(uint32_t, cubeBaseK); // cube K方向基本块
TILING_DATA_FIELD_DEF(uint64_t, postSingleCoreN); // 后处理单核N方向计算量

TILING_DATA_FIELD_DEF(uint64_t, cubeSingleCoreNLoop); // cube单核内N方向的外层循环数，L1之外的循环
TILING_DATA_FIELD_DEF(uint64_t, cubeSingleCoreNTailLoop); // cube尾核上单核N方向外层循环数
TILING_DATA_FIELD_DEF(uint64_t, singleCoreKLoop); // vec/cube单核上K方向外层循环数
TILING_DATA_FIELD_DEF(uint64_t, singleCoreKTailLoop); // vec/cube尾核上单核K方向外层循环数
TILING_DATA_FIELD_DEF(uint64_t, vecSingleCoreNLoop); // vec单核上N方向外层循环数
TILING_DATA_FIELD_DEF(uint64_t, vecSingleCoreNTailLoop); // vec尾核上单核N方向外层循环数

TILING_DATA_FIELD_DEF(uint64_t, cubeSingleCoreN); // cube单核N方向计算量
TILING_DATA_FIELD_DEF(uint64_t, cubeSingleCoreNTail); // cube尾核N方向计算量
TILING_DATA_FIELD_DEF(uint64_t, cubeSingleCoreNOriTail); // cube尾核N方向计算量,原始大小
TILING_DATA_FIELD_DEF(uint64_t, singleCoreK); // vec/cube单核K方向计算量
TILING_DATA_FIELD_DEF(uint64_t, singleCoreKTail); // vec/cube尾核K方向计算量
TILING_DATA_FIELD_DEF(uint64_t, singleCoreKOriTail); // vec/cube尾核K方向计算量, 原始大小
TILING_DATA_FIELD_DEF(uint64_t, vectorSingleCoreN); // vec单核N方向计算量
TILING_DATA_FIELD_DEF(uint64_t, vectorSingleCoreNTail); // vec尾核N方向计算量

TILING_DATA_FIELD_DEF(uint64_t, mSize); // 输入m的大小
TILING_DATA_FIELD_DEF(uint64_t, kSize); // 输入k的大小
TILING_DATA_FIELD_DEF(uint64_t, nSize); // 输入n的大小
TILING_DATA_FIELD_DEF(uint64_t, nSizeAlign); // 输入n的对齐后大小
TILING_DATA_FIELD_DEF(uint64_t, kSizeAlign); // 输入k的对齐后大小

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000010001000012021, WeightQuantBatchMatmulV2CustomNzSplitKTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000010001000012001, WeightQuantBatchMatmulV2CustomNzSplitKTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000010000000012021, WeightQuantBatchMatmulV2CustomNzSplitKTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000010000000012001, WeightQuantBatchMatmulV2CustomNzSplitKTilingData)

class WeightQuantBatchMatmulV2CustomNzSplitK : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2CustomNzSplitK(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2Tiling(context) {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2CustomNzSplitK() override = default;

protected:
    uint64_t cubeSingleN_;
    bool al1FullLoad_;
    std::unique_ptr<WeightQuantBatchMatmulV2CustomNzSplitKTilingData> tilingData_;
    //std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo> compileInfoPtr_;

    void Reset();
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    void GetMatMulTiling();
};
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H

