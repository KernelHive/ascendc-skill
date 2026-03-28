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
 * @file complex_mat_dot.cpp
 */

#include "complex_mat_dot_aiv.h"

extern "C" __global__ __aicore__ void complex_mat_dot(GM_ADDR matx, GM_ADDR maty,
                                                    GM_ADDR result, GM_ADDR workspace, GM_ADDR tilingGm)
{
    if (TILING_KEY_IS(0)) {
        auto coreIdx = GetBlockIdx(); // 0 ~ 39
        auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

        uint32_t m = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf)); // num of float elements
        uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 4)); // num of float elements

        uint64_t offset = (*(__gm__ uint64_t *)((__gm__ uint8_t *)tilingBuf + 8 + 8 * coreIdx)); // FP32
        uint32_t calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 8 + 40 * 8 + 4 * coreIdx)); // complex num

        ComplexMatDot::ComplexMatDotAIV<float> op;
        op.Init({matx, maty, result, m, n, offset, calNum});
        op.Process();
    }
}