/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KL_DIV_TARGET_BACKWARD_TILING_H
#define KL_DIV_TARGET_BACKWARD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(KlDivTargetBackwardTilingData)
  TILING_DATA_FIELD_DEF(int64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(int64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(int64_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(int64_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(int64_t, tileDataNum);
  TILING_DATA_FIELD_DEF(int64_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(int64_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(int64_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(int64_t, inputNum);
  TILING_DATA_FIELD_DEF(uint32_t, reduction);
  TILING_DATA_FIELD_DEF(uint32_t, logTarget);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KlDivTargetBackward, KlDivTargetBackwardTilingData)
}
#endif