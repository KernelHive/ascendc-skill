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
 * \file multi_scale_deformable_attn_function_tiling.h
 * \brief
 */
#ifndef MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H
#define MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttnFunctionTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, batchSize)
    TILING_DATA_FIELD_DEF(uint64_t, numKeys)
    TILING_DATA_FIELD_DEF(uint64_t, numHeads)
    TILING_DATA_FIELD_DEF(uint64_t, embedDims)
    TILING_DATA_FIELD_DEF(uint64_t, numLevels)
    TILING_DATA_FIELD_DEF(uint64_t, numQueries)
    TILING_DATA_FIELD_DEF(uint64_t, numPoints)
    TILING_DATA_FIELD_DEF(uint64_t, coreNum)
    TILING_DATA_FIELD_DEF(uint64_t, pointLoops)
    TILING_DATA_FIELD_DEF(uint64_t, realLevels)
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttnFunction, MultiScaleDeformableAttnFunctionTilingData)
}

struct MultiScaleDeformableAttnFunctionCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatform = 0;
};

#endif  // MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H