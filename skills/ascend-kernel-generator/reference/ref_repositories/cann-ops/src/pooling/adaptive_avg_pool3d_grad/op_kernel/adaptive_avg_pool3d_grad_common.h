/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_common.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H
#define ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H


#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline uint64_t start_index(uint64_t a, uint64_t b, uint64_t c)
{
    ASSERT_MSG(b != 0, "Division by zero error!");
    return (a / b) * c + ((a % b) * c) / b;
}

__aicore__ inline uint64_t end_index(uint64_t a, uint64_t b, uint64_t c)
{
    ASSERT_MSG(b != 0, "Division by zero error!");
    return 1 + ((a + 1) * c - 1) / b;
}

#endif //ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H