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
 * @file inplace_index_add_with_sorted_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_INDEX_ADD_WITH_SORTED_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_INDEX_ADD_WITH_SORTED_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceIndexAddWithSortedTilingData)
    TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, enableAlpha);
    TILING_DATA_FIELD_DEF(int64_t, eachIndexCount);
    TILING_DATA_FIELD_DEF(int64_t, lastIndexCount);
    TILING_DATA_FIELD_DEF(int64_t, batchCount);
    TILING_DATA_FIELD_DEF(int64_t, eachBatchNum);
    TILING_DATA_FIELD_DEF(int64_t, lastBatchNum);
    TILING_DATA_FIELD_DEF(int64_t, inputCount);
    TILING_DATA_FIELD_DEF(int64_t, indicesCount);
    TILING_DATA_FIELD_DEF(int64_t, updatesCount);
    TILING_DATA_FIELD_DEF(int64_t, updatesOneTime);
    TILING_DATA_FIELD_DEF(int64_t, maxSize);
    TILING_DATA_FIELD_DEF(int64_t, eachNum);
    TILING_DATA_FIELD_DEF(int64_t, eachLoop);
    TILING_DATA_FIELD_DEF(int64_t, eachTail);
    TILING_DATA_FIELD_DEF(int64_t, varDimNum);
    TILING_DATA_FIELD_DEF(int64_t, eachUBIndexRound);
    TILING_DATA_FIELD_DEF(int64_t, eachUBIndexCount);
    TILING_DATA_FIELD_DEF(int64_t, eachUBIndexTail);
    TILING_DATA_FIELD_DEF(int64_t, lastUBIndexRound);
    TILING_DATA_FIELD_DEF(int64_t, lastUBIndexCount);
    TILING_DATA_FIELD_DEF(int64_t, lastUBIndexTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceIndexAddWithSorted, InplaceIndexAddWithSortedTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_INPLACE_INDEX_ADD_WITH_SORTED_H