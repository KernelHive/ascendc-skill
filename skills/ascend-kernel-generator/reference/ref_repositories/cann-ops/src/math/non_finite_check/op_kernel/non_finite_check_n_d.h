/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file non_finite_check_n_d.h
 * \brief
 */
#ifndef NON_FINITE_CHECK_N_D_H
#define NON_FINITE_CHECK_N_D_H

#include "kernel_operator.h"

namespace NonFiniteCheck {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = 32;

template <typename T>
class NonFiniteCheckND {
public:
    __aicore__ inline NonFiniteCheckND(){};
    __aicore__ inline void Init(GM_ADDR tensor_list, GM_ADDR found_flag,
                                const NonFiniteCheckTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a + b - 1) / b);
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlignA2B(T1 a, T2 b) {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : CeilDiv(a, b) * b);
    };

    __aicore__ inline void ParseTilingData();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount);
    template <typename T2>
    __aicore__ inline void Compute(uint16_t index, int64_t dataCount);
    __aicore__ inline bool IsNonFinite(float value);
    __aicore__ inline bool IsNonFinite(half value);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> copyInQueue;
    TBuf<QuePosition::VECCALC> tempValBuf;
    GlobalTensor<T> tensorListGM;
    GlobalTensor<float> foundFlagGM;
    GM_ADDR tensorListPtr = nullptr;
    int64_t blockIdx = 0;
    bool haveFoundInf = false;
    int32_t perBlockCount = 0;

    // tiling params
    const NonFiniteCheckTilingData* __restrict tilingDataInClass = nullptr;
    uint32_t maxProcCount = 0;
    uint32_t tempValUbSize = 0;
    const int64_t* __restrict tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
};

template <typename T>
__aicore__ inline void NonFiniteCheckND<T>::Init(GM_ADDR tensor_list, GM_ADDR found_flag,
                                                 const NonFiniteCheckTilingData* __restrict tilingData) {
    tilingDataInClass = tilingData;
    blockIdx = GetBlockIdx();
    tensorListPtr = tensor_list;
    ParseTilingData();
    foundFlagGM.SetGlobalBuffer((__gm__ float*)found_flag, 1);
#if defined(ORIG_DTYPE_TENSOR_LIST) && ORIG_DTYPE_TENSOR_LIST == DT_FLOAT16
    pipe.InitBuffer(copyInQueue, BUFFER_NUM, maxProcCount * sizeof(half));
#else
    pipe.InitBuffer(copyInQueue, BUFFER_NUM, maxProcCount * sizeof(float));
#endif
    pipe.InitBuffer(tempValBuf, tempValUbSize);
    perBlockCount = BYTE_BLOCK / sizeof(T);
}

template <typename T>
__aicore__ inline void NonFiniteCheckND<T>::Process() {
    if (blockIdx == 0) {
        LocalTensor<float> dupLT = tempValBuf.Get<float>();
        Duplicate(dupLT, float(0.0f), BYTE_BLOCK / sizeof(float));
        event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);
        struct DataCopyExtParams copyParams = {1, sizeof(float), 0, 0, 0};
        DataCopyPad(foundFlagGM, dupLT, copyParams);
    }
    SyncAll();

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
        tensorListGM.SetGlobalBuffer(GetTensorAddr(i) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T>
__aicore__ inline void NonFiniteCheckND<T>::ParseTilingData() {
    maxProcCount = tilingDataInClass->maxProcCount;
    tempValUbSize = tilingDataInClass->tempValUbSize;
    tensorDataCountList = tilingDataInClass->tensorDataCountList;
    tensorStart = tilingDataInClass->tensorStartList[blockIdx];
    tensorEnd = tilingDataInClass->tensorEndList[blockIdx];
    tensorStartOffset = tilingDataInClass->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingDataInClass->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void NonFiniteCheckND<T>::SingleTensorProcess(int64_t dataCount) {
    // Batch handling and calculation.
    uint32_t copyTimes = CeilDiv(dataCount, maxProcCount);
    for (uint32_t i = 0; i < copyTimes; i++) {
        int64_t tempCount = maxProcCount;
        if ((i + 1 == copyTimes) && (dataCount % maxProcCount)) {
            tempCount = dataCount % maxProcCount;
        }
        CopyIn(i, tempCount);
        int64_t realProcCount = CeilAlignA2B(tempCount, perBlockCount);
#if defined(ORIG_DTYPE_TENSOR_LIST) && ORIG_DTYPE_TENSOR_LIST == DT_FLOAT16
        Compute<half>(i, realProcCount);
#else
        Compute<float>(i, realProcCount);
#endif
    }
}

template <typename T>
__aicore__ inline void NonFiniteCheckND<T>::CopyIn(uint16_t index, int64_t dataCount) {
#if defined(ORIG_DTYPE_TENSOR_LIST) && ORIG_DTYPE_TENSOR_LIST == DT_BF16
    LocalTensor<float> copyInLTFloat = copyInQueue.AllocTensor<float>();
    LocalTensor<T> copyInLT = copyInLTFloat.ReinterpretCast<T>()[maxProcCount];
#else
    LocalTensor<T> copyInLT = copyInQueue.AllocTensor<T>();
#endif
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventID1);
    WaitFlag<HardEvent::S_MTE2>(eventID1);
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID2);
    WaitFlag<HardEvent::V_MTE2>(eventID2);
    if (dataCount % perBlockCount) {
        struct DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        struct DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};
        padParams.rightPadding = CeilAlignA2B(dataCount, perBlockCount) - dataCount;
        DataCopyPad(copyInLT, tensorListGM[index * maxProcCount], copyParams, padParams);
    } else {
        DataCopy(copyInLT, tensorListGM[index * maxProcCount], dataCount);
    }
#if defined(ORIG_DTYPE_TENSOR_LIST) && ORIG_DTYPE_TENSOR_LIST == DT_BF16
    event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID3);
    WaitFlag<HardEvent::MTE2_V>(eventID3);
    Cast(copyInLTFloat, copyInLT, RoundMode::CAST_NONE, CeilAlignA2B(dataCount, perBlockCount));
    copyInQueue.EnQue(copyInLTFloat);
#else
    copyInQueue.EnQue(copyInLT);
#endif
}

template <typename T>
template <typename T2>
__aicore__ inline void NonFiniteCheckND<T>::Compute(uint16_t index, int64_t dataCount) {
    LocalTensor<T2> computeInLT = copyInQueue.DeQue<T2>();
    if (!haveFoundInf) {  // Inside the same Core, just find it once.
        LocalTensor<T2> workLocal = tempValBuf.Get<T2>();
        ReduceMax<T2>(workLocal, computeInLT, workLocal, dataCount, false);
        event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID1);
        WaitFlag<HardEvent::V_S>(eventID1);
        T2 maxValue = workLocal.GetValue(0);
        if (IsNonFinite(maxValue)) {
            foundFlagGM.SetValue(0, 1.0);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(foundFlagGM);
            haveFoundInf = true;
        }
        if (!haveFoundInf) {
            ReduceMin<T2>(workLocal, computeInLT, workLocal, dataCount, false);
            event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID2);
            WaitFlag<HardEvent::V_S>(eventID2);
            T2 minValue = workLocal.GetValue(0);
            if (IsNonFinite(minValue)) {
                foundFlagGM.SetValue(0, 1.0);
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(foundFlagGM);
                haveFoundInf = true;
            }
        }
    }
    copyInQueue.FreeTensor(computeInLT);
}

template <typename T>
__aicore__ inline bool NonFiniteCheckND<T>::IsNonFinite(float value) {
    uint32_t tempValue = *((uint32_t*)&value);
    if ((tempValue & 0x7FFFFFFF) >> 23 == 0xFF) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
__aicore__ inline bool NonFiniteCheckND<T>::IsNonFinite(half value) {
    uint16_t tempValue = *((uint16_t*)&value);
    if ((tempValue & 0x7FFF) >> 10 == 0x1F) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
__aicore__ inline __gm__ T* NonFiniteCheckND<T>::GetTensorAddr(uint16_t index) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

}  // namespace NonFiniteCheck

#endif  // NON_FINITE_CHECK_N_D_H