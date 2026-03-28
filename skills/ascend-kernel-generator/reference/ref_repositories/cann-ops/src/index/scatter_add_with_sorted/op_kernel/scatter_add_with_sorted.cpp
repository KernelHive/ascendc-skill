/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter_add_float_with_sorted.h"
#include "scatter_add_int_with_sorted.h"
#include "scatter_with_sorted.h"

#define CALL_OP_IMPL_SCATTER(...)                                        \
  do {                                                                   \
    KernelScatterWithSorted<__VA_ARGS__> op;                             \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
    op.Process();                                                        \
  } while (0)

#define CALL_OP_IMPL_FLOAT(...)                                          \
  do {                                                                   \
    KernelScatterAddFloatWithSorted<__VA_ARGS__> op;                     \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
    op.Process();                                                        \
  } while (0)

#define CALL_OP_IMPL_INT(...)                                            \
  do {                                                                   \
    KernelScatterAddIntWithSorted<__VA_ARGS__> op;                       \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, output);      \
    op.Process();                                                        \
  } while (0)

extern "C" __global__ __aicore__ void scatter_add_with_sorted(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, 
                                                              GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterAddWithSortedTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CALL_OP_IMPL_SCATTER(float);
    } else if (TILING_KEY_IS(2)) {
        CALL_OP_IMPL_SCATTER(half);
    } else if (TILING_KEY_IS(3)) {
        CALL_OP_IMPL_SCATTER(int);
    } else if (TILING_KEY_IS(4)) {
        CALL_OP_IMPL_SCATTER(uint8_t);
    } else if (TILING_KEY_IS(5)) {
        CALL_OP_IMPL_SCATTER(int8_t);
    } else if (TILING_KEY_IS(6)) {
        CALL_OP_IMPL_SCATTER(bfloat16_t);
    } else if (TILING_KEY_IS(11)) {
        CALL_OP_IMPL_FLOAT(float);
    } else if (TILING_KEY_IS(12)) {
        CALL_OP_IMPL_FLOAT(half);
    } else if (TILING_KEY_IS(13)) {
        CALL_OP_IMPL_INT(int);
    } else if (TILING_KEY_IS(14)) {
        CALL_OP_IMPL_INT(uint8_t);
    } else if (TILING_KEY_IS(15)) {
        CALL_OP_IMPL_INT(int8_t);
    } else if (TILING_KEY_IS(16)) {
        CALL_OP_IMPL_FLOAT(bfloat16_t);
    }
}
