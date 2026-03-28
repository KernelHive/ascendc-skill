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
 * \file weight_quant_batch_matmul_v2_tiling_splitk.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {
class WeightQuantBatchMatmulV2TilingSplitK : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingSplitK(gert::TilingContext *context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2TilingSplitK() override = default;

protected:
    std::unique_ptr<WeightQuantBatchMatmulV2TilingData> tilingData_;
    void Reset();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool GetMatMulTiling();
    uint64_t GetTilingKey() const override;
};
} // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H
