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
 * \file weight_quant_batch_matmul_v2_tiling_msd_group.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2MsdGroupTilingData)
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, vec1SingleCoreM);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, reserve1);
TILING_DATA_FIELD_DEF(uint16_t, reserve2);
TILING_DATA_FIELD_DEF(uint32_t, reserve3);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_711300, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_710300, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000100000000003021, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000100000000003001, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000101000000003020, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000101000000003000, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000101000000003021, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000101000000003001, WeightQuantBatchMatmulV2MsdGroupTilingData)

class WeightQuantBatchMatmulV2TilingMsdGroup : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingMsdGroup(gert::TilingContext *context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    };
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2TilingMsdGroup() override = default;

protected:
    std::unique_ptr<WeightQuantBatchMatmulV2MsdGroupTilingData> tilingData_;

    void Reset();

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    bool CheckL1Size() const;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        if (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP &&
           (matmulInfoPtr_->innerPrecise != 0 || matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)) { 
            // 在A16W4 pergroup 高性能/高精度 tilingkey
            TilingKeyConfigure tilingKeyConfigure;
            SetCommonTilingKeyElement(tilingKeyConfigure);
            tilingKeyConfigure.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::MULTI_SCALE_DEQUANT) * 10 +
                static_cast<uint8_t>(OptimizationAlgorithmSubCategory::VDEFAULT);
            tilingKeyConfigure.templateCustom = static_cast<uint8_t>(matmulInfoPtr_->innerPrecise) * 1000; // 1000:第6位
            return tilingKeyConfigure.GenTilingKey();
        } else {
            return RecursiveSum(matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->antiQuantType,
                    matmulInfoPtr_->hasAntiQuantOffset, matmulInfoPtr_->quantType,
                    KernelTemplateType::MSD_GROUP);
        }
    }

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = 64 * 1024 * 1024;  // workspace 固定使用 64 * 1024 * 1024
        return ge::GRAPH_SUCCESS;
    }

    bool GetMatMulTiling();
};

}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H

