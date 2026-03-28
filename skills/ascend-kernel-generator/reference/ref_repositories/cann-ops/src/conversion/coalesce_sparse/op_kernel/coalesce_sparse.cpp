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
 * @file coalesce_sparse.cpp
 */
#include "coalesce_sparse.h"

extern "C" __global__ __aicore__ void coalesce_sparse(GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices,
                                                      GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const CoalesceSparseTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(0)) {
        KernelCoalesceSparse<int64_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(1)) {
        KernelCoalesceSparse<int64_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(2)) {
        KernelCoalesceSparse<int64_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(3)) {
        KernelCoalesceSparse<int64_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(4)) {
        KernelCoalesceSparse<int64_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(5)) {
        KernelCoalesceSparse<int64_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(6)) {
        KernelCoalesceSparse<int32_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(7)) {
        KernelCoalesceSparse<int32_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(8)) {
        KernelCoalesceSparse<int32_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(9)) {
        KernelCoalesceSparse<int32_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(10)) {
        KernelCoalesceSparse<int32_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(11)) {
        KernelCoalesceSparse<int32_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
}