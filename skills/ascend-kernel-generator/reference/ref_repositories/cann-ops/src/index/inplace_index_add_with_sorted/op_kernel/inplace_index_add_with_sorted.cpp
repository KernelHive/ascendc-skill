/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inplace_index_add_with_sorted_fix.h"
#include "inplace_index_add_with_sorted_avg.h"

extern "C" __global__ __aicore__ void inplace_index_add_with_sorted(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices,
                                    GM_ADDR pos, GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 使能VectorCore

#define INIT_AND_PROCESS                                                               \
    op.Init(var, value, sorted_indices, pos, alpha);                             \
    op.Process()
    if (TILING_KEY_IS(1)) {
        // InplaceIndexAddWithSorted FLOAT axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<float> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(4)) {
        // InplaceIndexAddWithSorted INT16 axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<int16_t> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(5)) {
        // InplaceIndexAddWithSorted INT32 axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<int32_t> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        // InplaceIndexAddWithSorted HALF axis = 0, same index on the same core
        InplaceIndexAddWithSortedFix<half> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        // InplaceIndexAddWithSorted BF16 axis = 0, same index on the same core
        InplaceIndexAddWithSortedFix<bfloat16_t> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(6)) {
        // InplaceIndexAddWithSorted FLOAT axis = 0, same index on the same core
        InplaceIndexAddWithSortedFix<float> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    }
}