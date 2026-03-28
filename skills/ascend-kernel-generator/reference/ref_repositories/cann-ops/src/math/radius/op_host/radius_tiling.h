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
 * @file radius_tiling.h
 */
#ifndef RADIUS_TILING_H
#define RADIUS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RadiusTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, xSize);
  TILING_DATA_FIELD_DEF(uint32_t, ySize);
  TILING_DATA_FIELD_DEF(uint32_t, itemLength);
  TILING_DATA_FIELD_DEF(uint32_t, max_num_neighbors);
  TILING_DATA_FIELD_DEF(uint32_t, ignore_same_index);
  TILING_DATA_FIELD_DEF(uint32_t, ptrXLen);
  TILING_DATA_FIELD_DEF(uint32_t, ptrYLen);
  TILING_DATA_FIELD_DEF(float, r);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Radius, RadiusTilingData)
} // namespace optiling
#endif // RADIUS_TILING_H
