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
 * \file weight_quant_batch_matmul_v2_tiling_msd.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2MsdTilingData)
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint16_t, v1BaseM);
TILING_DATA_FIELD_DEF(uint16_t, preloadTimes);
TILING_DATA_FIELD_DEF(uint16_t, taskNSize);
TILING_DATA_FIELD_DEF(uint16_t, taskSingleCoreNSize);
TILING_DATA_FIELD_DEF(uint16_t, postProcessBaseM);
TILING_DATA_FIELD_DEF(uint16_t, postProcessSingleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, preProcessUsedVecNum);
TILING_DATA_FIELD_DEF(uint32_t, v1BaseK);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupPack);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_511200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_511210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_611200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_610200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_8611200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_8610200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_8611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_8610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10611200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10611300, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10610200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10610300, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_28611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_20611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_28610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_20610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_18611200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_18610200, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_18611210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_18610210, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000111000000003021, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000110000000003021, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000111000000003020, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000111000000003001, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000110000000003001, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000111000000003000, WeightQuantBatchMatmulV2MsdTilingData)

class WeightQuantBatchMatmulV2Msd : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2Msd(gert::TilingContext *context) : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2Msd() override = default;

protected:
    uint32_t order_ = 2;  // 展开的阶数
    uint32_t blkDim_ = 0;
    bool splitKFlag_;
    bool highPrecision_;
    uint64_t cubeBaseN_;
    std::unique_ptr<WeightQuantBatchMatmulV2MsdTilingData> tilingData_;

    void Reset();
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoMSDGeneralOpTiling();
    ge::graphStatus DoOpTiling() override;
    uint64_t SplitKByKBlock(uint64_t kBlockNum) const;
    ge::graphStatus DoMSDGroupSplitKOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    bool CheckCacheTiling();
    bool CheckInt4MatmulTiling() const;
    bool CheckInt8MatmulTiling(uint64_t singleCoreNCalc) const;
    bool InvokeCacheTiling();
    bool GetMatMulTiling();
    void ReviseMMTiling() const;
    bool GetTilingFromCache();
    uint64_t GetInnerPreciseTilingKey() const;
};
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H
