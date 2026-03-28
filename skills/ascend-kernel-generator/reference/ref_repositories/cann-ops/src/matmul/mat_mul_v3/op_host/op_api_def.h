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
 * \file op_api_def.h
 * \brief
 */

#ifndef OP_API_DEF_H_
#define OP_API_DEF_H_

namespace op {
    const size_t BN_MIN_SUPPORT_DIMS_NUMS = 2;
    const size_t MAX_SUPPORT_DIMS_NUMS = 8;
    const int8_t FP16FP32_KEEP_DTYPE = -1;
    const int8_t KEEP_DTYPE = 0;
    const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
    const int8_t USE_FP16 = 2;
    const int8_t USE_HF32 = 3;
    constexpr size_t MAX_MASK_LEN64 = 64;
}  // namespace op
#endif  // OP_API_DEF_H_
