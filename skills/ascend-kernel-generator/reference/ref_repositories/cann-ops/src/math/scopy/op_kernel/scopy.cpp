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
 * @file scopy.cpp
 */

#include "scopy_aiv.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void scopy(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    if (TILING_KEY_IS(0)) {
        ScopyAIV<float> op;
        op.Init(x, y, workSpace, tilingGm);
        op.Process();
    }
}