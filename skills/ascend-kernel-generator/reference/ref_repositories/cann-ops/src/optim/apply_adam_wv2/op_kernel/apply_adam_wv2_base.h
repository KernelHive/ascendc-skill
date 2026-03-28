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
 * @file apply_adam_wv2_base.h
 */

#ifndef APPLYADAM_W_V2_BASE_H
#define APPLYADAM_W_V2_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ApplyAdamWV2 {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BLOCK_SIZE_FOR_FLOAT32 = 8;
constexpr int32_t IN_BUFFER_NUM = 5;
constexpr int32_t OUT_BUFFER_NUM = 4;
constexpr int32_t VAR_ORDER_IN_LOCAL_TENSOR = 0;
constexpr int32_t EXP_AVG_ORDER_IN_LOCAL_TENSOR = 1;
constexpr int32_t EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR = 2;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR = 3;
constexpr int32_t GRAD_NORM_ORDER_IN_LOCAL_TENSOR = 4;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR = 3;
}  // namespace ApplyAdamWV2

#endif  // APPLYADAM_W_V2_BASE_H