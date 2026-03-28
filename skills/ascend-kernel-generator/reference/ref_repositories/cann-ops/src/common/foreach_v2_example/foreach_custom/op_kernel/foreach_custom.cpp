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
 * \file foreach_custom.cpp
 * \brief
 */

 #include "kernel_operator.h"
 
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_custom.h"

using namespace AscendC;
using namespace ForeachCustom;

template <typename T>
__aicore__ void CustomAdapter(
    const LocalTensor<T>& dstLocal, 
    const LocalTensor<T>& srcLocal, 
    const uint32_t uValue,
    const LocalTensor<T>& tempLocal) {
    uint32_t oneDataCount =  BYTE_REPEATE / sizeof(T);
    uint32_t castTimes = uValue / oneDataCount;
    uint32_t castTimesRemainder = uValue % oneDataCount;
    uint32_t offset = 0;

    for (int i = 0; i < castTimes; i++) {
        Mul(tempLocal, srcLocal[offset], srcLocal[offset], static_cast<int32_t>(oneDataCount));
        pipe_barrier(PIPE_V);
        Abs(srcLocal[offset], srcLocal[offset], static_cast<int32_t>(oneDataCount));
        pipe_barrier(PIPE_V);
        Sub(dstLocal[offset], tempLocal, srcLocal[offset], static_cast<int32_t>(oneDataCount));
        pipe_barrier(PIPE_V);
        offset += static_cast<int32_t>(oneDataCount);
    }

    if (castTimesRemainder) {
        Mul(tempLocal, srcLocal[offset], srcLocal[offset], static_cast<int32_t>(castTimesRemainder));
        pipe_barrier(PIPE_V);
        Abs(srcLocal[offset], srcLocal[offset], static_cast<int32_t>(castTimesRemainder));
        pipe_barrier(PIPE_V);
        Sub(dstLocal[offset], tempLocal, srcLocal[offset], static_cast<int32_t>(castTimesRemainder));
        pipe_barrier(PIPE_V);
    }
}

extern "C" __global__ __aicore__ void foreach_custom(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachCustomNd<half, half, CustomAdapter<half>, 2, 1, true, true> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachCustomNd<float, float, CustomAdapter<float>, 2, 1, true, true> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachCustomNd<bfloat16_t, float, CustomAdapter<float>, 2, 1, true, true> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}