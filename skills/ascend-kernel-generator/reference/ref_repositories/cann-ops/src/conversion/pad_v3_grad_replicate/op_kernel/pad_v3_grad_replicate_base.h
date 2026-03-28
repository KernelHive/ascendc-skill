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
 * \file pad_v3_grad_replicate_base.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_BASE_
#define _PAD_V3_GRAD_REPLICATE_BASE_

#include "kernel_operator.h"

constexpr int32_t INPUT_NUM = 2;
constexpr int32_t OUTPUT_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PADDING_INPUT_INDEX = 2;
constexpr int32_t Y_OUTPUT_INDEX = 0;
constexpr int32_t BUFFER_APPLY_NUM = 2;
constexpr int32_t COPY_ROWS_AND_COLS = 16;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t COPY_LOOP = 16;
constexpr uint32_t CAL_COUNT = 32;
constexpr uint32_t FLOAT_BLOCK_NUM = 8;
constexpr uint32_t HALF_BLOCK_NUM = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t TRANSDATA_BASE_H = 16;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t MINI_SHAPE_MAX_ROWS = 64;
constexpr uint32_t SMALL_WIDTH_LIMIT = 64;
constexpr uint32_t SMALL_HEIGHT_LIMIT = 64;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
};
#endif  // _PAD_V3_GRAD_REPLICATE_BASE_