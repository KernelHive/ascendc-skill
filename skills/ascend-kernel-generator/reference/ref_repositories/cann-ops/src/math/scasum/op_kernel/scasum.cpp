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
 * @file scasum.cpp
 */

#include "scasum_aiv.h"

extern "C" __global__ __aicore__ void scasum(GM_ADDR inGM, GM_ADDR outGM,
                                            GM_ADDR workspace, GM_ADDR tilingGM)
{
    if (TILING_KEY_IS(0)) {
        SetSysWorkspace(workspace);
        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }
        auto vecIdx = GetBlockIdx();
        auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGM);

        uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));
        uint32_t coreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 4));
        uint32_t offset = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 8 + 4 * vecIdx));
        uint32_t calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 8 + 40 * 4 + 4 * vecIdx));

        Scasum::ScasumAIV<float> op;
        op.Init(inGM, outGM, n, offset, calNum);
        op.Process();
    }
}