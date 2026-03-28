/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file feeds_repeat_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FeedsRepeatTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, length);            // feeds_repeat_times的元素个数
    TILING_DATA_FIELD_DEF(uint32_t, length_aligned);    // length与32B对齐
    TILING_DATA_FIELD_DEF(int64_t, elem_row);          // feeds第0维一行切片的数据量
    TILING_DATA_FIELD_DEF(int64_t, elem_per_loop);     // 实际一个核一次循环处理的数据量
    TILING_DATA_FIELD_DEF(int64_t, max_core_num);       // 可用最大核数（全部使用）
    TILING_DATA_FIELD_DEF(int64_t, core_per_group);     // 核数大于feeds行数时，同时处理一行的平均核数
    TILING_DATA_FIELD_DEF(int64_t, core_moreover);      // 核数大于feeds行数时，核数除以行数的余量
    TILING_DATA_FIELD_DEF(int64_t, empty_size);         // 需要清零的行的结束位置，即output_feeds_size
    TILING_DATA_FIELD_DEF(int64_t, row_per_core);       // 每个核处理的平均核数
    TILING_DATA_FIELD_DEF(int64_t, row_left);           // 行数除以核数的余量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FeedsRepeat, FeedsRepeatTilingData)
}  // namespace optiling

struct FeedsRepeatCompileInfo {
    uint64_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
};

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H