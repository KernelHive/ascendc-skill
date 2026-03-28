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
 * @file add_custom_tiling.h
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)
}
#endif // ADD_CUSTOM_TILING_H