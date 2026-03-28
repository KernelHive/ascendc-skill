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
 * \file instance_norm_helper.h
 * \brief
 */

#ifndef INSTANCE_NORM_HELPER_H_
#define INSTANCE_NORM_HELPER_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using true_type = IntegralConstant<bool, true>;
using false_type = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public false_type {};
template <typename Tp>
struct IsSame<Tp, Tp> : public true_type {};

constexpr int BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline void DataCopyCustomUB2GM(
    const GlobalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t count)
{
    // only support count greater than 32byte
    int32_t numPerBlock = BLOCK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        int32_t num = count / numPerBlock * numPerBlock;
        DataCopy(dstTensor, srcTensor, num);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for (int32_t i = 0; i < numPerBlock; i++) {
            T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
            srcTensor.SetValue(i, tensorValue);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
    }
}

template <typename T>
__aicore__ inline void DataCopyCustomGM2UB(
    const LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, const uint32_t count)
{
    // only support count greater than 32byte
    int32_t numPerBlock = BLOCK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        int32_t num = AlignUp(count, numPerBlock);
        DataCopy(dstTensor, srcTensor, num);
    }
}

#endif  // INSTANCE_NORM_HELPER_H_
