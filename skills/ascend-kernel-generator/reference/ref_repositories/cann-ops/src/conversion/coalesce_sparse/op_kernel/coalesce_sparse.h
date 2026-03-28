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
 * @file coalesce_sparse.h
 */
#ifndef COALESCE_SPARSE_H
#define COALESCE_SPARSE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 1u;

template <typename uIdxType, typename idxType, typename dataType>
class KernelCoalesceSparse {
public:
    __aicore__  inline KernelCoalesceSparse() = default;
    __aicore__  inline void Init(GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices,
                                 GM_ADDR newValue, const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__  inline void Process();

private:
    __aicore__  inline void InitTilingValue(const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__  inline void CopyIn(uint64_t repeatTime, uint64_t moveLen);
    __aicore__  inline void ComputeAndCopyOut(uint64_t repeatTime, uint64_t taskLen);
    __aicore__ inline void valueMove(uint64_t valueOffset, uint64_t ubValueOffset, uint64_t moveValueLen);
    __aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> uniqueIndicesQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> valueQueue;

    GlobalTensor<uIdxType> uniqueIndicesGm;
    GlobalTensor<idxType> indicesGm;
    GlobalTensor<dataType> valueGm;

    GlobalTensor<idxType> newIndicesGm;
    GlobalTensor<dataType> newValueGm;

    uint64_t usedCoreNum {0};
    uint64_t m {0};
    uint64_t n {0};
    uint64_t valueSize {0};
    uint64_t taskNum {0};
    uint64_t taskTail {0};
    uint64_t moveOneSize {0};
    uint64_t taskRepeatTimes {0};
    uint64_t taskRepeatTail {0};
    uint64_t taskTailRepeatTimes {0};
    uint64_t taskTailRepeatTail {0};
    uint64_t moveValueTimes {0};
    uint64_t moveValueLen {0};
    uint64_t moveValueTail {0};

    uint64_t blockSize {32};
    uint64_t taskLen {0};
    uint64_t coreTaskTimes {0};
    uint64_t coreTaskTail {0};
    uint64_t unquieBlockPreSize {0};
    uint64_t idBlockPreSize {0};
    uint32_t indicesUbStride {0};
    uint64_t mByte {0};
    uint64_t indicesAlign32 {0};
    const CoalesceSparseTilingData* __restrict tilingDevice {nullptr};
};

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::Init(
    GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices, GM_ADDR newValue,
    const CoalesceSparseTilingData* __restrict tilingData) {
    InitTilingValue(tilingData);
    uint64_t coreId = GetBlockIdx();
    uint64_t beginOffset = coreId * taskNum;
    uint64_t indicesBeginOffset = beginOffset * m;
    uint64_t valueBeginOffset = beginOffset * valueSize;

    if (coreId < usedCoreNum - 1) {
        taskLen = taskNum;
        coreTaskTimes = taskRepeatTimes;
        coreTaskTail = taskRepeatTail;
    }
    else if (coreId == usedCoreNum - 1) {
        taskLen = taskTail;
        coreTaskTimes = taskTailRepeatTimes;
        coreTaskTail = taskTailRepeatTail;
    }
    //SetGlobalBuffer
    this->uniqueIndicesGm.SetGlobalBuffer((__gm__ uIdxType *)uniqueIndices + beginOffset, taskLen);
    this->indicesGm.SetGlobalBuffer((__gm__ idxType *)indices + indicesBeginOffset, taskLen * m);
    this->valueGm.SetGlobalBuffer((__gm__ dataType *)values + valueBeginOffset, taskLen * valueSize);
    this->newIndicesGm.SetGlobalBuffer((__gm__ idxType *)newIndices);
    this->newValueGm.SetGlobalBuffer((__gm__ dataType *)newValue);
    indicesUbStride = (uint32_t)(CeilDiv(m * sizeof(idxType), blockSize));
    // //block_pre_size
    unquieBlockPreSize = blockSize  / sizeof(uIdxType);
    idBlockPreSize = blockSize  / sizeof(idxType);
    indicesAlign32 = indicesUbStride * idBlockPreSize;
    // //InitBuffer Need Align32
    this->pipe.InitBuffer(this->uniqueIndicesQueue, BUFFER_NUM, moveOneSize * sizeof(uIdxType));
    this->pipe.InitBuffer(this->indicesQueue, BUFFER_NUM, moveOneSize * indicesUbStride * blockSize);
    // moveValueLen is align 32
    uint64_t moveValueLenAlign32 = CeilDiv(moveValueLen * sizeof(dataType), blockSize) * blockSize;
    this->pipe.InitBuffer(this->valueQueue, BUFFER_NUM, moveValueLenAlign32);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::InitTilingValue(
    const CoalesceSparseTilingData* __restrict tilingData) {
    //Get tilingData
    this->tilingDevice = tilingData;
    usedCoreNum = tilingDevice->usedCoreNum;
    m = tilingDevice->m;
    n = tilingDevice->n;
    mByte = m * sizeof(idxType);
    valueSize = tilingDevice->valueSize;
    taskNum = tilingDevice->taskNum;
    taskTail = tilingDevice->taskTail;
    moveOneSize = tilingDevice->moveOneSize;
    taskRepeatTimes = tilingDevice->taskRepeatTimes;
    taskRepeatTail = tilingDevice->taskRepeatTail;
    taskTailRepeatTimes = tilingDevice->taskTailRepeatTimes;
    taskTailRepeatTail = tilingDevice->taskTailRepeatTail;
    moveValueTimes = tilingDevice->moveValueTimes;
    moveValueLen = tilingDevice->moveValueLen;
    moveValueTail = tilingDevice->moveValueTail;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::Process() {
    dataType initValue = 0;
    if (GetBlockIdx() == 0) {
        InitOutput<dataType>(newValueGm, valueSize * n, initValue);
    }
    SyncAll();
    // taskAlign32
    for (uint64_t i = 0; i < coreTaskTimes; i++){
        CopyIn(i, moveOneSize);
        ComputeAndCopyOut(i, moveOneSize);
    }
    // taskTail
    if (coreTaskTail != 0){
        CopyIn(coreTaskTimes, coreTaskTail);
        ComputeAndCopyOut(coreTaskTimes, coreTaskTail);
    }
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::CopyIn(uint64_t repeatTime, uint64_t moveLen) {

    uint64_t taksOffset = repeatTime * moveOneSize;
    uint64_t uniqueIndicesOffset = taksOffset;
    uint64_t indicesOffset = taksOffset * m;

    DataCopyExtParams copyParams_indices {(uint16_t)moveLen, (uint32_t)(mByte), 0, 0, 0};
    DataCopyPadExtParams<idxType> indices_padParams{true, 0, 0, 0};

    LocalTensor<uIdxType> uniqueIndicesLocal = uniqueIndicesQueue.AllocTensor<uIdxType>();
    LocalTensor<idxType> indicesLocal = indicesQueue.AllocTensor<idxType>();

    uint64_t unquieIndicesAlign32 = CeilDiv(moveLen, unquieBlockPreSize) * unquieBlockPreSize;

    DataCopy(uniqueIndicesLocal, uniqueIndicesGm[uniqueIndicesOffset], unquieIndicesAlign32);
    DataCopyPad(indicesLocal, indicesGm[indicesOffset], copyParams_indices, indices_padParams);
    uniqueIndicesQueue.EnQue<uIdxType>(uniqueIndicesLocal);
    indicesQueue.EnQue<idxType>(indicesLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::ComputeAndCopyOut(uint64_t repeatTime, uint64_t taskLen) {
    //maybe the value_size too big and should move many times ---> moveValueTimes and moveValueTail
    //need moveValueTimes, moveValueLen and moveValueTail tiling
    LocalTensor<uIdxType> uniqueIndicesLocal = uniqueIndicesQueue.DeQue<uIdxType>();
    LocalTensor<idxType> indicesLocal = indicesQueue.DeQue<idxType>();
     for(uint64_t i = 0; i < taskLen; i++) {
        int64_t uniqueIndicesId = uniqueIndicesLocal.GetValue(i);
        int64_t gmIndicesOffset = uniqueIndicesId * m;
        int64_t gmValueOffset = uniqueIndicesId * valueSize;
        DataCopyParams copyParams_indices{1, (uint16_t)(mByte), 0, 0};
        DataCopyPad(newIndicesGm[gmIndicesOffset], indicesLocal[i * indicesAlign32], copyParams_indices);
        for (int j = 0; j < moveValueTimes; j++) {
            event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            uint64_t valueOffset = gmValueOffset + j * moveValueLen;
            uint64_t ubValueOffset = repeatTime * moveOneSize + i * valueSize + j * moveValueLen;
            valueMove(valueOffset, ubValueOffset, moveValueLen);
        }
        if (moveValueTail > 0) {
            event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            uint64_t valueOffset = gmValueOffset + moveValueTimes * moveValueLen;
            uint64_t ubValueOffset = repeatTime * moveOneSize + i * valueSize + moveValueTimes * moveValueLen;
            valueMove(valueOffset, ubValueOffset, moveValueTail);
        }
    }
    uniqueIndicesQueue.FreeTensor(uniqueIndicesLocal);
    indicesQueue.FreeTensor(indicesLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::valueMove(uint64_t valueOffset, uint64_t ubValueOffset, uint64_t moveValueLen) {
    uint64_t valueByte = moveValueLen * sizeof(dataType);
    LocalTensor<dataType> valueLocal = valueQueue.AllocTensor<dataType>();
    DataCopyExtParams copyParams_value_ {(uint16_t)1, (uint32_t)(valueByte) , 0, 0, 0};
    DataCopyPadExtParams<dataType> values_padParams{true, 0, 0, 0};
    DataCopyPad(valueLocal, valueGm[ubValueOffset], copyParams_value_, values_padParams);
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    DataCopyParams copyParams_value{1, (uint16_t)(valueByte), 0, 0};
    SetAtomicAdd<dataType>();
    DataCopyPad(newValueGm[valueOffset], valueLocal, copyParams_value);
    SetAtomicNone();
    valueQueue.FreeTensor(valueLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline uint64_t KernelCoalesceSparse<uIdxType, idxType, dataType>::CeilDiv(uint64_t x, uint64_t y) {
    return y == 0 ? x : (x + y - 1) / y;
}
#endif // COALESCE_SPARSE