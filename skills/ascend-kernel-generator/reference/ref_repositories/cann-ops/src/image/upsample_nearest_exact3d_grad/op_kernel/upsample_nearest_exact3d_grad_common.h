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
 * \file upsample_nearest_exact3d_grad_common.h
 * \brief
 */
#ifndef UPSAMPLE_NEARESTEXACT3D_GRAD_COMMON_H
#define UPSAMPLE_NEARESTEXACT3D_GRAD_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;
constexpr int64_t BLOCK_SIZE = 32;

__aicore__ inline int64_t ROUND_UP(int64_t x, int64_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}

template <typename T>
__aicore__ inline void InitGmZero(
    const GlobalTensor<T> &outGm, TBuf<TPosition::VECCALC> &TmpZeroTBuf, int64_t zeroLen, int64_t outOffset)
{
    int64_t alignLen_ = BLOCK_SIZE / sizeof(T);
    LocalTensor<T> temp_zero_tensor = TmpZeroTBuf.Get<T>();

    Duplicate(temp_zero_tensor, (T)0.0, zeroLen);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outGm[outOffset], temp_zero_tensor, ROUND_UP(zeroLen, alignLen_));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

    PipeBarrier<PIPE_ALL>();
}
#endif  // UPSAMPLE_NEARESTEXACT3D_GRAD_COMMON_H