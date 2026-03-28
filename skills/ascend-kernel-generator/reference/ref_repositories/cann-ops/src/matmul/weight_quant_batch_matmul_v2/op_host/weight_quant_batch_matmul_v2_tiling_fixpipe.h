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
 * \file weight_quant_batch_matmul_v2_tiling_fixpipe.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_FIXPIPE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_FIXPIPE_H

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2FixpipeTilingData)
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, nBlockNum);
TILING_DATA_FIELD_DEF(uint16_t, baseK);
TILING_DATA_FIELD_DEF(uint16_t, baseM);
TILING_DATA_FIELD_DEF(uint16_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200000000012000, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200001000012000, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200000000012010, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200001000012010, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200000000012020, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200001000012020, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200000000012030, WeightQuantBatchMatmulV2FixpipeTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1000200001000012030, WeightQuantBatchMatmulV2FixpipeTilingData)

class WeightQuantBatchMatmulV2TilingFixpipe : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingFixpipe(gert::TilingContext *context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    };
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2TilingFixpipe() override = default;

protected:
    uint64_t aFullLoad_;
    std::unique_ptr<WeightQuantBatchMatmulV2FixpipeTilingData> tilingData_;

    void Reset()
    {
        TilingBaseClass::Reset(context_);
        aFullLoad_ = 0;

        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to memset tiling data"), return;);
    }

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    bool CheckDtypeIsCapable() const;

    bool CheckShapeIsCapable() const;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
};
}  // namespace optiling

#endif
