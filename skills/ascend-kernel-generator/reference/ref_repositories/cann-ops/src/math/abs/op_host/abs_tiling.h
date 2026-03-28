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
 * @file abs_tiling.h
 */
#ifndef ABS_TILING_H
#define ABS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AbsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, bigDataCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallBlockLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigBlockLength);
  TILING_DATA_FIELD_DEF(uint32_t, smallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, smallLasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigLasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, dataWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Abs, AbsTilingData)
}
#endif // ADD_CUSTOM_TILING_H