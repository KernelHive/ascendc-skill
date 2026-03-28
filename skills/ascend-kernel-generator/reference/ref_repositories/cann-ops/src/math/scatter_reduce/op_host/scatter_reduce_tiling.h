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
 * @file scatter_reduce_tiling.h
 */
#ifndef SCATTER_REDUCE_TILING_H
#define SCATTER_REDUCE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterReduceTilingData)
  TILING_DATA_FIELD_DEF(int32_t, batchSize);
  TILING_DATA_FIELD_DEF(int32_t, dimSizeX);
  TILING_DATA_FIELD_DEF(int32_t, dimSizeSrc);
  TILING_DATA_FIELD_DEF(int32_t, strideSize);
  TILING_DATA_FIELD_DEF(int32_t, reduction);
  TILING_DATA_FIELD_DEF(bool, includeSelf);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterReduce, ScatterReduceTilingData)
}  // namespace optiling
#endif // SCATTER_REDUCE_TILING_H