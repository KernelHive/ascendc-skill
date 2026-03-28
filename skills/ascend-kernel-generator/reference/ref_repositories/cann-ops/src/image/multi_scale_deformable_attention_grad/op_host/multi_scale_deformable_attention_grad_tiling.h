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
 * \file multi_scale_deformable_attention_grad.h
 * \brief tiling of MultiScaleDeformableAttentionGrad op
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttentionGradTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, batchSize)
    TILING_DATA_FIELD_DEF(uint64_t, numKeys)
    TILING_DATA_FIELD_DEF(uint64_t, numHeads)
    TILING_DATA_FIELD_DEF(uint64_t, embedDims)
    TILING_DATA_FIELD_DEF(uint64_t, numLevels)
    TILING_DATA_FIELD_DEF(uint64_t, numQueries)
    TILING_DATA_FIELD_DEF(uint64_t, numPoints)
    TILING_DATA_FIELD_DEF(uint64_t, maxUbNum)
    TILING_DATA_FIELD_DEF(uint64_t, coreNum)
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttentionGrad, MultiScaleDeformableAttentionGradTilingData)
}

struct MultiScaleDeformableAttentionGradCompileInfo {
    uint64_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
};

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H