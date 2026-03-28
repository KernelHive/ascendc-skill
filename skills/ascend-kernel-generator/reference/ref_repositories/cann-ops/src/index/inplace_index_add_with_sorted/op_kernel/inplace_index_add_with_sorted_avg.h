/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_INDEX_ADD_WITH_SORTED_AVG_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_AVG_H_

#include "inplace_index_add_with_sorted_base.h"

using namespace AscendC;

template <typename T>
class InplaceIndexAddWithSortedAvg {
public:
    __aicore__ inline InplaceIndexAddWithSortedAvg(TPipe* pipeIn,
            const InplaceIndexAddWithSortedTilingData* __restrict tilingData) {
        pipe = pipeIn;
        coreId = GetBlockIdx();
        usedCoreNum = tilingData->usedCoreNum;
        enableAlpha = tilingData->enableAlpha;
        eachIndexCountAvg = tilingData->eachIndexCount;
        lastIndexCountAvg = tilingData->lastIndexCount;
        inputCountAvg = tilingData->inputCount;
        indicesCountAvg = tilingData->indicesCount;
        updatesCountAvg = tilingData->updatesCount;
        updatesOneTime = tilingData->updatesOneTime;
        maxSize = tilingData->maxSize;
        eachNumAvg = tilingData->eachNum;
        eachLoopAvg = tilingData->eachLoop;
        eachTailAvg = tilingData->eachTail;
        eachUBIndexRoundAvg = tilingData->eachUBIndexRound;
        eachUBIndexCountAvg = tilingData->eachUBIndexCount;
        eachUBIndexTailAvg = tilingData->eachUBIndexTail;
        lastUBIndexRoundAvg = tilingData->lastUBIndexRound;
        lastUBIndexCountAvg = tilingData->lastUBIndexCount;
        lastUBIndexTailAvg = tilingData->lastUBIndexTail;

        currentUBIndexRound = coreId == (usedCoreNum - 1) ? lastUBIndexRoundAvg : eachUBIndexRoundAvg;
        currentIndexCount = coreId == (usedCoreNum - 1) ? lastUBIndexCountAvg : eachUBIndexCountAvg;
        currentUBIndexTail = coreId == (usedCoreNum - 1) ? lastUBIndexTailAvg : eachUBIndexTailAvg;
    }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha) {
        inputGmAvg.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCountAvg);
        valueGmAvg.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), updatesCountAvg);
        if (enableAlpha == 1) {
            alphaGmAvg.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha), 1);
            alphaDataAvg = alphaGmAvg.GetValue(0);
        }
        indicesGmAvg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices), indicesCountAvg);
        posGmAvg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos), indicesCountAvg);

        pipe->InitBuffer(inQueueValueAvg, BUFFER_NUM, maxSize * sizeof(T));
        pipe->InitBuffer(inQueueIndex, 1, NUM_TWO * INDEX_UB_NUM * sizeof(int32_t));
        pipe->InitBuffer(outQueueOutAvg, 1, maxSize * sizeof(T));
    }

    __aicore__ inline void Process() {
        for (int64_t i = 0; i < eachLoopAvg; ++i) {
            AvgProcess(i);
        }
    }

private:
    __aicore__ inline void AvgProcess(int64_t loopIdx) {
        int32_t updateIndex, updatePos, lastUpdateIndex = -1;
        int64_t currentEachNum = loopIdx == eachLoopAvg - 1 ? eachTailAvg : eachNumAvg;
        int64_t currentEachNumAlign = CeilAlignA2B(currentEachNum, BLOCK_SIZE);
        bool firstMov = true;
        for (int64_t idxRound = 0; idxRound < currentUBIndexRound; ++idxRound) {
            currentEachIndex = idxRound == currentUBIndexRound -1 ? currentUBIndexTail : currentIndexCount;
            indexLocal = inQueueIndex.AllocTensor<int32_t>();
            int64_t indexOffset = coreId * eachIndexCountAvg + idxRound * eachUBIndexCountAvg;
            DataCopyPadExtParams<int32_t> tPadParams = {false, 0, 0, static_cast<int32_t>(0)};
            DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentEachIndex * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(indexLocal, indicesGmAvg[indexOffset], updatesExtParams, tPadParams);
            DataCopyPad(indexLocal[INDEX_UB_NUM], posGmAvg[indexOffset], updatesExtParams, tPadParams);
            event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (int64_t j = 0; j < currentEachIndex; ++j) {
                int64_t indicesOffset = INDEX_UB_NUM + j;
                updateIndex = indexLocal.GetValue(j);
                bool ifSyncIn = firstMov || updateIndex != lastUpdateIndex;
                bool ifSyncOut = (!firstMov && updateIndex != lastUpdateIndex);
                if (ifSyncOut) {
                    outQueueOutAvg.EnQue(outLocal);
                    CopyOut(lastUpdateIndex, loopIdx, currentEachNum);
                }
                updatePos = indexLocal.GetValue(indicesOffset);
                event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                if (ifSyncIn) {
                    outLocal = outQueueOutAvg.AllocTensor<T>();
                    Duplicate(outLocal, (T)0, currentEachNumAlign);
                }
                CopyValueIn(updatePos, loopIdx, currentEachNum);
                Compute(currentEachNumAlign);
                lastUpdateIndex = updateIndex;
                firstMov = false;
            }
            inQueueIndex.FreeTensor(indexLocal);
        }
        outQueueOutAvg.EnQue(outLocal);
        CopyOut(updateIndex, loopIdx, currentEachNum);
    }

    __aicore__ inline void CopyValueIn(int32_t index, int64_t progress, int64_t dataLen) {
        valueLocal = inQueueValueAvg.AllocTensor<T>();
        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(valueLocal, valueGmAvg[index * updatesOneTime + progress * maxSize], updatesExtParams, tPadParams);
        inQueueValueAvg.EnQue(valueLocal);
    }

    __aicore__ inline void Compute(int64_t dataLen) {
        valueLocal = inQueueValueAvg.DeQue<T>();
        if (enableAlpha == 1) {
            Muls(valueLocal, valueLocal, alphaDataAvg, dataLen);
            PipeBarrier<PIPE_V>();
        }
        Add(outLocal, outLocal, valueLocal, dataLen);
        inQueueValueAvg.FreeTensor(valueLocal);
    }

    __aicore__ inline void CopyOut(int32_t index, int64_t progress, int64_t dataLen) {
        outLocal = outQueueOutAvg.DeQue<T>();
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        SetAtomicAdd<T>();
        DataCopyPad(inputGmAvg[index * updatesOneTime + progress * maxSize], outLocal, updatesExtParams);
        SetAtomicNone();
        outQueueOutAvg.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;
    GlobalTensor<T> inputGmAvg, valueGmAvg, alphaGmAvg;
    GlobalTensor<int32_t> indicesGmAvg, posGmAvg;
    LocalTensor<int32_t> indexLocal;
    LocalTensor<T> valueLocal, outLocal;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueValueAvg, inQueueIndex;
    TQue<QuePosition::VECOUT, 1> outQueueOutAvg;

    int64_t coreId;
    int32_t usedCoreNum;
    int32_t enableAlpha;
    int64_t eachIndexCountAvg;
    int64_t lastIndexCountAvg;
    int64_t inputCountAvg;
    int64_t indicesCountAvg;
    int64_t updatesCountAvg;
    int64_t updatesOneTime;
    int64_t maxSize;
    int64_t eachNumAvg;
    int64_t eachLoopAvg;
    int64_t eachTailAvg;
    int64_t currentEachIndex;
    int64_t currentUBIndexRound;
    int64_t currentIndexCount;
    int64_t currentUBIndexTail;
    int64_t eachUBIndexRoundAvg;
    int64_t eachUBIndexCountAvg;
    int64_t eachUBIndexTailAvg;
    int64_t lastUBIndexRoundAvg;
    int64_t lastUBIndexCountAvg;
    int64_t lastUBIndexTailAvg;
    T alphaDataAvg;
};

#endif  // INPLACE_INDEX_ADD_WITH_SORTED_AVG_H_