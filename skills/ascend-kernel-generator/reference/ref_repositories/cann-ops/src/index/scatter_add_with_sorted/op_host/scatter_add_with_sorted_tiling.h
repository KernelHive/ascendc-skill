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
 * @file scatter_add_with_sorted_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddWithSortedTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, extraTaskCore);
  TILING_DATA_FIELD_DEF(uint64_t, eachCount);
  TILING_DATA_FIELD_DEF(uint64_t, lastCount);
  TILING_DATA_FIELD_DEF(uint64_t, inputCount);
  TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
  TILING_DATA_FIELD_DEF(uint64_t, updatesCount);
  TILING_DATA_FIELD_DEF(uint64_t, inputOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, updatesOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, updatesAlign);
  TILING_DATA_FIELD_DEF(uint64_t, maxSize);
  TILING_DATA_FIELD_DEF(uint64_t, eachNum);
  TILING_DATA_FIELD_DEF(uint64_t, eachLoop);
  TILING_DATA_FIELD_DEF(uint64_t, eachTail);
  TILING_DATA_FIELD_DEF(uint64_t, lastNum);
  TILING_DATA_FIELD_DEF(uint64_t, lastLoop);
  TILING_DATA_FIELD_DEF(uint64_t, lastTail);
  TILING_DATA_FIELD_DEF(uint64_t, updatesLoop);
  TILING_DATA_FIELD_DEF(uint64_t, updatesEach);
  TILING_DATA_FIELD_DEF(uint64_t, updatesLast);
END_TILING_DATA_DEF;
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ADD_WITH_SORTED_H
