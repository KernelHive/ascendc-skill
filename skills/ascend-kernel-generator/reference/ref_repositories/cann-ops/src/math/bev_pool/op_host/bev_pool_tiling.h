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
 * @file bev_pool_tiling.h
 */
#ifndef BEV_POOL_TILING_H
#define BEV_POOL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BevPoolTilingData)
  TILING_DATA_FIELD_DEF(int32_t, B);
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, D);
  TILING_DATA_FIELD_DEF(int32_t, fH);
  TILING_DATA_FIELD_DEF(int32_t, fW);
  TILING_DATA_FIELD_DEF(int32_t, C);
  TILING_DATA_FIELD_DEF(int32_t, D_Z);
  TILING_DATA_FIELD_DEF(int32_t, D_Y);
  TILING_DATA_FIELD_DEF(int32_t, D_X);
  TILING_DATA_FIELD_DEF(int32_t, N_points);
  TILING_DATA_FIELD_DEF(int32_t, N_pillar);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BevPool, BevPoolTilingData)
}
#endif // BEV_POOL_TILING_H
