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
 * @file coalesce_sparse_tiling.h
 */
#ifndef COALESCE_SPARSE_TILING_H
#define COALESCE_SPARSE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CoalesceSparseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, m);
TILING_DATA_FIELD_DEF(uint64_t, n);
TILING_DATA_FIELD_DEF(uint64_t, valueSize);
TILING_DATA_FIELD_DEF(uint64_t, taskNum);
TILING_DATA_FIELD_DEF(uint64_t, taskTail);
TILING_DATA_FIELD_DEF(uint64_t, moveOneSize);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTimes);
TILING_DATA_FIELD_DEF(uint64_t, moveValueLen);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CoalesceSparse, CoalesceSparseTilingData)
} // namespace optiling
#endif // COALESCE_SPARSE_TILING_H
