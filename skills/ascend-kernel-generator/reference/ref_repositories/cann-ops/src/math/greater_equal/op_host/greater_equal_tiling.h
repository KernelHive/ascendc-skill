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
 * @file greater_equal_tiling.h
 */
#ifndef GREATEREQUAL_TILING_H
#define GREATEREQUAL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GreaterEqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GreaterEqual, GreaterEqualTilingData)
}
#endif // GREATEREQUAL_TILING_H