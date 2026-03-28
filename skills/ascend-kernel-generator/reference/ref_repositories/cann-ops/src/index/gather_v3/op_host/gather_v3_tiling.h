/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file gather_v3_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V3_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V3_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GatherV3TilingData)
    TILING_DATA_FIELD_DEF(int64_t, tilingKey);
    TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, ubLineLimit);
    TILING_DATA_FIELD_DEF(int64_t, xBufferSize);
    TILING_DATA_FIELD_DEF(int64_t, yBufferSize);
    TILING_DATA_FIELD_DEF(int64_t, idxBufferSize);
    TILING_DATA_FIELD_DEF(int64_t, bSize);
    TILING_DATA_FIELD_DEF(int64_t, pSize);
    TILING_DATA_FIELD_DEF(int64_t, gxSize);
    TILING_DATA_FIELD_DEF(int64_t, gySize);
    TILING_DATA_FIELD_DEF(int64_t, aSize);
    TILING_DATA_FIELD_DEF(int64_t, bTileNum);
    TILING_DATA_FIELD_DEF(int64_t, pTileNum);
    TILING_DATA_FIELD_DEF(int64_t, gTileNum);
    TILING_DATA_FIELD_DEF(int64_t, aTileNum);
    TILING_DATA_FIELD_DEF(int64_t, bTileSize);
    TILING_DATA_FIELD_DEF(int64_t, pTileSize);
    TILING_DATA_FIELD_DEF(int64_t, gTileSize);
    TILING_DATA_FIELD_DEF(int64_t, aTileSize);
    TILING_DATA_FIELD_DEF(int64_t, bTileHead);
    TILING_DATA_FIELD_DEF(int64_t, pTileHead);
    TILING_DATA_FIELD_DEF(int64_t, gTileHead);
    TILING_DATA_FIELD_DEF(int64_t, aTileHead);
    TILING_DATA_FIELD_DEF(int64_t, bufferNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV3, GatherV3TilingData)

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V3_TILING_H_