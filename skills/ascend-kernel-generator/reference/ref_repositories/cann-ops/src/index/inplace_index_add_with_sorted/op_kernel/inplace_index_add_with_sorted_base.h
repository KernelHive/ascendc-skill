/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_

#include "kernel_operator.h"
#define IS_CAST_FLOAT ((is_same<T, half>::value) || (is_same<T, bfloat16_t>::value))
using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr int64_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t INDEX_UB_NUM = 1536;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : (a + b - 1) / b);
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlignA2B(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : CeilDiv(a, b) * b);
};

#endif  // INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_