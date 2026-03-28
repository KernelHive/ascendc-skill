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
 * @file less_tiling.h
 */
#ifndef LESS_TILING_H
#define LESS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LessTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubPartDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreLoopNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreLoopNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, isTailBlock);
    TILING_DATA_FIELD_DEF(uint32_t, bigprocessDataNumComputes);
    TILING_DATA_FIELD_DEF(uint32_t, smallprocessDataNumComputes);
    TILING_DATA_FIELD_DEF(uint32_t, tailbigprocessDataNumComputes);
    TILING_DATA_FIELD_DEF(uint32_t, tailsmallprocessDataNumComputes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Less, LessTilingData)
} // namespace optiling
#endif
