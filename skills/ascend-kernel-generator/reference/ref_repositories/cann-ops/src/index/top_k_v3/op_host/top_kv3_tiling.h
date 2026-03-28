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
 * @file top_kv3_tiling.h
 */
#ifndef TOP_K_V3_TILING_H
#define TOP_K_V3_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TopKV3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRow);
    TILING_DATA_FIELD_DEF(uint32_t, numCol);
    TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
    TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
    TILING_DATA_FIELD_DEF(int32_t, kValue);
    TILING_DATA_FIELD_DEF(uint32_t, largest);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TopKV3, TopKV3TilingData)
}  // namespace optiling

#endif // TOP_K_V3_TILING_H
