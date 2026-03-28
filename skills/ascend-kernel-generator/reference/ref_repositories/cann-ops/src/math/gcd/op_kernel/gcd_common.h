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
 * @file gcd_common.h
 */
#ifndef GCD_COMMON_H
#define GCD_COMMON_H

#include "kernel_operator.h"

template<typename T>
struct GcdConfig {
    using FpType = T;
    static constexpr int TILE_SIZE = 1024;
};

template<>
struct GcdConfig<int16_t> {
    using FpType = half;
    static constexpr int TILE_SIZE = 4096;
};

template<>
struct GcdConfig<int32_t> {
    using FpType = float;
    static constexpr int TILE_SIZE = 2048;
};

#endif // GCD_COMMON_H