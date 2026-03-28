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
 * \file group_norm_silu_base.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_OPEN_BASE_H_
#define GROUP_NORM_SILU_OPEN_BASE_H_

#include "kernel_operator.h"

namespace GroupNormSilu {
using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 32;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
constexpr bool IS_DATA_COPY_PAD_SUPPORT = true;
#else
constexpr bool IS_DATA_COPY_PAD_SUPPORT = false;
#endif

template <typename T>
class GroupNormSiluBase {
public:
    __aicore__ inline GroupNormSiluBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    __aicore__ inline RoundMode GetRoundMode() {
#if __CCE_AICORE__ == 220
        return RoundMode::CAST_ROUND;
#else
        return RoundMode::CAST_NONE;
#endif
    };

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyInData(const LocalTensor<T1>& dstUB, const GlobalTensor<T1>& srcGM,
                                      const int64_t dataCount) {
        if constexpr (isAlign) {
            DataCopy(dstUB, srcGM, dataCount);
        } else {
            if constexpr (IS_DATA_COPY_PAD_SUPPORT) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
                DataCopyPad(dstUB, srcGM, copyParams, padParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstUB, srcGM, dataCountAlign);
            }
        }
    }

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyOutData(const GlobalTensor<T1>& dstGM, const LocalTensor<T1>& srcUB,
                                       const int64_t dataCount) {
        if constexpr (isAlign) {
            DataCopy(dstGM, srcUB, dataCount);
        } else {
            if constexpr (IS_DATA_COPY_PAD_SUPPORT) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPad(dstGM, srcUB, copyParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstGM, srcUB, dataCountAlign);
            }
        }
    }
};

}  // namespace GroupNormSilu

#endif  // GROUP_NORM_SILU_OPEN_BASE_H_
