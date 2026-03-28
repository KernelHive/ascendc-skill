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
 * @file gcd_tiling.h
 */
#ifndef GCD_TILING_H
#define GCD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
  TILING_DATA_FIELD_DEF(int, N0);
  TILING_DATA_FIELD_DEF(int, N1);
  TILING_DATA_FIELD_DEF(int, N2);
  TILING_DATA_FIELD_DEF(int, N3);
  TILING_DATA_FIELD_DEF(int, N4);
  TILING_DATA_FIELD_DEF(int, broadcast_mask);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
#endif // GCD_TILING_H
