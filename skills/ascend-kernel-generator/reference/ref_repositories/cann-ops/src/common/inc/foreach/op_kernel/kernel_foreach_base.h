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
 * \file kernel_foreach_base.h
 * \brief
 */

#ifndef KERNEL_FOREACH_BASE_H
#define KERNEL_FOREACH_BASE_H

#include <type_traits>
#include "kernel_operator.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;

template <typename T>
class KernelForeachBase {
protected:
    __aicore__ inline KernelForeachBase() {};

    __aicore__ inline void Init(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

protected:
    TPipe pipe;

    int64_t blockIdx = 0;

    // tiling params
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;

    uint64_t totalTensorUbSize = 0;
    uint32_t maxDataCount = 0;
    uint32_t maxCastDataCount = 0;
};

template <typename T>
__aicore__ inline void KernelForeachBase<T>::Init(
    const ForeachCommonTilingData* tilingData) {
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);
    
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            maxDataCount = totalTensorUbSize / sizeof(T);        
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
    #else 
        maxDataCount = inputsTensorUbSize / sizeof(T);
    #endif
}

template <typename T>
__aicore__ inline void KernelForeachBase<T>::ParseTilingData(
    const ForeachCommonTilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline __gm__ T* KernelForeachBase<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_BASE_H