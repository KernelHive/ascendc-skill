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
 * \file foreach_non_finite_check_and_unscale_n_d.h
 * \brief
 */
#ifndef FOREACH_NON_FINITE_CHECK_AND_UNSCALE_N_D_H
#define FOREACH_NON_FINITE_CHECK_AND_UNSCALE_N_D_H

#include "kernel_operator.h"

namespace ForeachNonFiniteCheckAndUnscale {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = 32;

template <typename T>
class ForeachNonFiniteCheckAndUnscaleND {
public:
    __aicore__ inline ForeachNonFiniteCheckAndUnscaleND(){};
    __aicore__ inline void Init(GM_ADDR scaled_grads, GM_ADDR found_inf, GM_ADDR inv_scale, GM_ADDR workspace,
                                const ForeachNonFiniteCheckAndUnscaleTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount);
    __aicore__ inline void Compute(uint16_t index, int64_t dataCount);
    __aicore__ inline void CopyOut(uint16_t index, int64_t dataCount);
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
    __aicore__ inline void CastToFloat(uint16_t index, int64_t dataCount);
    __aicore__ inline void CastToOriginalType(uint16_t index, int64_t dataCount);
#endif
    __aicore__ inline bool IsNonFinite(float value);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> copyInQueue;
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
    TQue<QuePosition::VECIN, BUFFER_NUM> calcInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> calcOutQueue;
#endif
    TQue<QuePosition::VECOUT, BUFFER_NUM> copyOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tempValQueue;
    GlobalTensor<T> scaledGradsGM;
    GlobalTensor<float> foundInfGM;
    GlobalTensor<float> invScaleGM;
    GM_ADDR scaledGradsPtr = nullptr;
    int64_t blockIdx = 0;
    float invScaleVal = 0;
    bool haveFoundInf = false;
    int32_t perBlockCount = 0;
    uint32_t maxDataCount = 0;
    // tiling params
    const ForeachNonFiniteCheckAndUnscaleTilingData* __restrict tilingDataInClass = nullptr;
    uint32_t scaledGradsUbSize = 0;
    uint32_t reduceTempValUbSize = 0;
    const int64_t* __restrict tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
};

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::Init(
    GM_ADDR scaled_grads, GM_ADDR found_inf, GM_ADDR inv_scale, GM_ADDR workspace,
    const ForeachNonFiniteCheckAndUnscaleTilingData* __restrict tilingData) {
    tilingDataInClass = tilingData;
    blockIdx = GetBlockIdx();
    scaledGradsPtr = scaled_grads;
    ParseTilingData();
    foundInfGM.SetGlobalBuffer((__gm__ float*)found_inf, 1);
    invScaleGM.SetGlobalBuffer((__gm__ float*)inv_scale, 1);
    pipe.InitBuffer(copyInQueue, BUFFER_NUM, scaledGradsUbSize);
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
    pipe.InitBuffer(calcInQueue, BUFFER_NUM, 2 * scaledGradsUbSize);
    pipe.InitBuffer(calcOutQueue, BUFFER_NUM, 2 * scaledGradsUbSize);
#endif
    pipe.InitBuffer(copyOutQueue, BUFFER_NUM, scaledGradsUbSize);
    pipe.InitBuffer(tempValQueue, BUFFER_NUM, reduceTempValUbSize);
    invScaleVal = invScaleGM.GetValue(0);
    perBlockCount = BYTE_BLOCK / sizeof(T);
    maxDataCount = scaledGradsUbSize / sizeof(T);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::Process() {
    for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == tensorStart) {
            cursorStart = tensorStartOffset;
        }
        if (i == tensorEnd && tensorEndOffset < cursorEnd) {
            cursorEnd = tensorEndOffset;
        }
        dataCount = cursorEnd - cursorStart + 1;
        scaledGradsGM.SetGlobalBuffer(GetTensorAddr(i) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::SingleTensorProcess(int64_t dataCount) {
    // Batch handling and calculation.
    uint32_t copyTimes = CeilA2B(dataCount, maxDataCount);
    for (uint32_t i = 0; i < copyTimes; i++) {
        int64_t tempCount = maxDataCount;
        if ((i + 1 == copyTimes) && (dataCount % maxDataCount)) {
            tempCount = dataCount % maxDataCount;
        }
        CopyIn(i, tempCount);
        int64_t alignCount = CeilA2B(tempCount, perBlockCount) * perBlockCount;
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
        CastToFloat(i, alignCount);
#endif
        Compute(i, alignCount);
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
        CastToOriginalType(i, alignCount);
#endif
        CopyOut(i, tempCount);
    }
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::ParseTilingData() {
    scaledGradsUbSize = tilingDataInClass->scaledGradsUbSize;
    reduceTempValUbSize = tilingDataInClass->reduceTempValUbSize;
    tensorDataCountList = tilingDataInClass->tensorDataCountList;
    tensorStart = tilingDataInClass->tensorStartList[blockIdx];
    tensorEnd = tilingDataInClass->tensorEndList[blockIdx];
    tensorStartOffset = tilingDataInClass->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingDataInClass->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::CopyIn(uint16_t index, int64_t dataCount) {
    LocalTensor<T> copyInLT = copyInQueue.AllocTensor<T>();
    if (dataCount % perBlockCount) {
        struct DataCopyParams copyParams = {1, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        struct DataCopyPadParams padParams = {true, 0, 0, 0};
        int64_t alignDataCount = CeilA2B(dataCount, perBlockCount) * perBlockCount;
        padParams.rightPadding = alignDataCount - dataCount;
        DataCopyPad(copyInLT, scaledGradsGM[index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(copyInLT, scaledGradsGM[index * maxDataCount], dataCount);
    }
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID1);
    WaitFlag<HardEvent::MTE2_S>(eventID1);
    copyInQueue.EnQue(copyInLT);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::Compute(uint16_t index, int64_t dataCount) {
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
    LocalTensor<float> computeInLT = calcInQueue.DeQue<float>();
    LocalTensor<float> computeOutLT = calcOutQueue.AllocTensor<float>();
#else
    LocalTensor<float> computeInLT = copyInQueue.DeQue<float>();
    LocalTensor<float> computeOutLT = copyOutQueue.AllocTensor<float>();
#endif
    if (!haveFoundInf) {  // Inside the same Core, just find it once.
        LocalTensor<float> workLocal = tempValQueue.AllocTensor<float>();
        ReduceMax<float>(workLocal, computeInLT, workLocal, dataCount, false);
        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID1);
        WaitFlag<HardEvent::V_S>(eventID1);
        float maxValue = workLocal.GetValue(0);
        if (IsNonFinite(maxValue)) {
            foundInfGM.SetValue(0, 1.0);
            haveFoundInf = true;
        }
        if (!haveFoundInf) {
            ReduceMin<float>(workLocal, computeInLT, workLocal, dataCount, false);
            event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID2);
            WaitFlag<HardEvent::V_S>(eventID2);
            float minValue = workLocal.GetValue(0);
            if (IsNonFinite(minValue)) {
                foundInfGM.SetValue(0, 1.0);
                haveFoundInf = true;
            }
        }
        tempValQueue.FreeTensor(workLocal);
    }
    Muls(computeOutLT, computeInLT, invScaleVal, dataCount);
#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
    calcOutQueue.EnQue(computeOutLT);
    calcInQueue.FreeTensor(computeInLT);
#else
    copyOutQueue.EnQue(computeOutLT);
    copyInQueue.FreeTensor(computeInLT);
#endif
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::CopyOut(uint16_t index, int64_t dataCount) {
    LocalTensor<T> copyOutLT = copyOutQueue.DeQue<T>();
    if (dataCount % perBlockCount) {
        struct DataCopyParams copyParams = {1, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        DataCopyPad(scaledGradsGM[index * maxDataCount], copyOutLT, copyParams);
    } else {
        DataCopy(scaledGradsGM[index * maxDataCount], copyOutLT, dataCount);
    }

    copyOutQueue.FreeTensor(copyOutLT);
}

#if ORIG_DTYPE_SCALED_GRADS != DT_FLOAT
template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::CastToFloat(uint16_t index, int64_t dataCount) {
    LocalTensor<T> copyInLT = copyInQueue.DeQue<T>();
    LocalTensor<float> calcInLT = calcInQueue.AllocTensor<float>();
    Cast(calcInLT, copyInLT, RoundMode::CAST_NONE, dataCount);
    calcInQueue.EnQue(calcInLT);
    copyInQueue.FreeTensor(copyInLT);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleND<T>::CastToOriginalType(uint16_t index, int64_t dataCount) {
    LocalTensor<float> calcOutLT = calcOutQueue.DeQue<float>();
    LocalTensor<T> copyOutLT = copyOutQueue.AllocTensor<T>();
    Cast(copyOutLT, calcOutLT, RoundMode::CAST_RINT, dataCount);
    copyOutQueue.EnQue(copyOutLT);
    calcOutQueue.FreeTensor(calcOutLT);
}
#endif

template <typename T>
__aicore__ inline bool ForeachNonFiniteCheckAndUnscaleND<T>::IsNonFinite(float value) {
    uint32_t tempValue = *((uint32_t*)&value);
    if ((tempValue & 0x7FFFFFFF) >> 23 == 0xFF) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
__aicore__ inline __gm__ T* ForeachNonFiniteCheckAndUnscaleND<T>::GetTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(scaledGradsPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

}  // namespace ForeachNonFiniteCheckAndUnscale

#endif  // FOREACH_NON_FINITE_CHECK_AND_UNSCALE_N_D_H