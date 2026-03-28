/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_INDEX_ADD_WITH_SORTED_FIX_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_FIX_H_

#include "inplace_index_add_with_sorted_base.h"

using namespace AscendC;

template <typename T>
class InplaceIndexAddWithSortedFix {
public:
    __aicore__ inline InplaceIndexAddWithSortedFix(TPipe* pipeIn,
            const InplaceIndexAddWithSortedTilingData* __restrict tilingData) {
        pipe = pipeIn;
        coreId = GetBlockIdx();
        usedCoreNum = tilingData->usedCoreNum;
        enableAlpha = tilingData->enableAlpha;
        eachIndexCountFix = tilingData->eachIndexCount;
        lastIndexCountFix = tilingData->lastIndexCount;
        inputCountFix = tilingData->inputCount;
        indicesCountFix = tilingData->indicesCount;
        updatesCountFix = tilingData->updatesCount;
        updatesOneTime = tilingData->updatesOneTime;
        maxSize = tilingData->maxSize;
        eachNumFix = tilingData->eachNum;
        eachLoopFix = tilingData->eachLoop;
        eachTailFix = tilingData->eachTail;
        eachUBIndexRound = tilingData->eachUBIndexRound;
        eachUBIndexCount = tilingData->eachUBIndexCount;
        eachUBIndexTail = tilingData->eachUBIndexTail;
        lastUBIndexRound = tilingData->lastUBIndexRound;
        lastUBIndexCount = tilingData->lastUBIndexCount;
        lastUBIndexTail = tilingData->lastUBIndexTail;

        currentUBIndexRound = coreId == (usedCoreNum - 1) ? lastUBIndexRound : eachUBIndexRound;
        currentIndexCount = coreId == (usedCoreNum - 1) ? lastUBIndexCount : eachUBIndexCount;
        currentUBIndexTail = coreId == (usedCoreNum - 1) ? lastUBIndexTail : eachUBIndexTail;
    }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha) {
        inputGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCountFix);
        valueGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), updatesCountFix);
        if (enableAlpha == 1) {
            alphaGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha), 1);
            alphaDataFix = alphaGmFix.GetValue(0);
            if (alphaDataFix == (float)1.0) {
                // 经测试，即使在torch侧不传alpha位置参数，aclnn侧传给算子的alpha的值为1，因此判断为1时，不做vmuls，省一点计算时间
                enableAlpha = 0;
            }
        }
        indicesGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices), indicesCountFix);
        posGmFix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos), indicesCountFix);
        finalIndex = indicesGmFix.GetValue(indicesCountFix - 1);
        pipe->InitBuffer(inQueueSelfFix, BUFFER_NUM, maxSize * sizeof(T));
        pipe->InitBuffer(inQueueValueFix, BUFFER_NUM, maxSize * sizeof(T));
        pipe->InitBuffer(inQueueIndex, 1, NUM_TWO * INDEX_UB_NUM * sizeof(int32_t));
        pipe->InitBuffer(outQueueOutFix, BUFFER_NUM, maxSize * sizeof(T));
        if constexpr (IS_CAST_FLOAT) {
            pipe->InitBuffer(calcSelfBuf, maxSize * sizeof(float));
            pipe->InitBuffer(calcValueBuf, maxSize * sizeof(float));
            calcSelfLocal = calcSelfBuf.Get<float>();
            calcValueLocal = calcValueBuf.Get<float>();
        }
    }

    __aicore__ inline void Process() {
        if (coreId != 0) {
            int64_t lastOffset = eachIndexCountFix * coreId - 1;
            lastCoreIndex = indicesGmFix.GetValue(lastOffset);
        }
        for (int64_t i = 0; i < eachLoopFix; ++i) {
            currentEachNum = i == eachLoopFix - 1 ? eachTailFix : eachNumFix;
            currentEachNumAlign = CeilAlignA2B(currentEachNum, BLOCK_SIZE);
            AvgProcess(i);
            // 尾核肯定只需要遍历自己的索引，如果lastCoreIndex == finalIndex，说明尾核什么都不用做
            if (coreId == usedCoreNum - 1 && lastCoreIndex != finalIndex) {
                CopyOut(finalIndex, i, currentEachNum);
                continue;
            } else if (coreId == usedCoreNum - 1) {
                continue;
            }
            PipeBarrier<PIPE_ALL>();
            // 每个核接下来要遍历后面的核分到的索引，直到出现遍历到的索引值改变
            SameIndexProcess(i);
        }
    }

private:
    __aicore__ inline void AvgProcess(int64_t loopIdx) {
        int32_t updateIndex, updatePos, lastUpdateIndex = -1;
        bool firstMov = true;
        for (int64_t idxRound = 0; idxRound < currentUBIndexRound; ++idxRound) {
            currentEachIndex = idxRound == currentUBIndexRound -1 ? currentUBIndexTail : currentIndexCount;
            indexLocal = inQueueIndex.AllocTensor<int32_t>();
            int64_t indexOffset = coreId * eachIndexCountFix + idxRound * eachUBIndexCount;
            DataCopyPadExtParams<int32_t> tPadParams = {false, 0, 0, static_cast<int32_t>(0)};
            DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentEachIndex * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(indexLocal, indicesGmFix[indexOffset], updatesExtParams, tPadParams);
            DataCopyPad(indexLocal[INDEX_UB_NUM], posGmFix[indexOffset], updatesExtParams, tPadParams);
            event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (int64_t j = 0; j < currentEachIndex; ++j) {
                int64_t indicesOffset = INDEX_UB_NUM + j;
                updateIndex = indexLocal.GetValue(j);
                // 当前索引与上一个AIV核的最后一个索引值相同，则跳过该索引，不做处理，确保同索引分核
                if (coreId != 0 && updateIndex == lastCoreIndex) {
                    continue;
                }
                // 当遇到updateIndex != lastUpdateIndex情况时，说明相同index的一块数据已经搬完，
                // 需要对将已经算完的outLocal进行搬出，再进行后面的搬入和计算
                bool ifSyncIn = firstMov || updateIndex != lastUpdateIndex;
                bool ifSyncOut = (!firstMov && updateIndex != lastUpdateIndex);
                if (ifSyncOut) {
                    CopyOut(lastUpdateIndex, loopIdx, currentEachNum);
                }
                updatePos = indexLocal.GetValue(indicesOffset);
                event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                // 每一个i循环相当于self的第i块数据，需要搬入，但每个j循环中，如果需要更新的index相同，则不需要重复搬入
                if (ifSyncIn) {
                    CopySelfIn(updateIndex, loopIdx, currentEachNum);
                }
                // Value每次j循环都需要重新搬入，计算也每个循环都会计算
                CopyValueIn(updatePos, loopIdx, currentEachNum);
                Compute(currentEachNumAlign);
                lastUpdateIndex = updateIndex;
                firstMov = false;
            }
            // 最后一次的循环累加结果的搬出由SameIndexProcess完成
            inQueueIndex.FreeTensor(indexLocal);
        }
    }

    __aicore__ inline void SameIndexProcess(int64_t loopIdx) {
        // 每个核接下来要遍历后面的核分到的索引，直到出现遍历到的索引值改变
        int32_t lastIndex = indicesGmFix.GetValue(eachIndexCountFix * (coreId + 1) - 1);
        int32_t nextIndex = indicesGmFix.GetValue(eachIndexCountFix * (coreId + 1));
        bool ifContinue = lastIndex == nextIndex;
        int32_t updatePos = 0;
        int64_t offset = 0;
        // 先确保进入下面的循环 ifcontinue一定为true，当lastCoreIndex == lastIndex，说明该核的所有索引由上一个核处理，直接返回
        if ((coreId != 0 && lastCoreIndex == lastIndex)) {
            return;
        }
        if (!ifContinue) {
            CopyOut(lastIndex, loopIdx, currentEachNum);
            return;
        }
        // 由于接下来一定只遇到相同的index，因此在外循环开始时搬入self的第i块数据，内循环结束之后再搬出，
        // 并且需要重置临时变量和if continue的条件
        while (ifContinue) {
            int64_t indicesOffset = eachIndexCountFix * (coreId + 1) + offset;
            updatePos = posGmFix.GetValue(indicesOffset);
            nextIndex = indicesGmFix.GetValue(indicesOffset);
            event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            CopyValueIn(updatePos, loopIdx, currentEachNum);
            Compute(currentEachNumAlign);
            offset += 1;
            indicesOffset = eachIndexCountFix * (coreId + 1) + offset;
            if (indicesOffset >= indicesCountFix) {
                break;
            }
            nextIndex = indicesGmFix.GetValue(indicesOffset);
            ifContinue = lastIndex == nextIndex;
        }
        // while 循环跳出，表示self的第i块数据累加完成
        CopyOut(lastIndex, loopIdx, currentEachNum);
        offset = 0;
        ifContinue = true;
    }

    __aicore__ inline void CopySelfIn(int32_t index, int64_t progress, int64_t dataLen) {
        inputLocal = inQueueSelfFix.AllocTensor<T>();
        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputLocal, inputGmFix[index * updatesOneTime + progress * maxSize], updatesExtParams, tPadParams);
        inQueueSelfFix.EnQue(inputLocal);
        inputLocal = inQueueSelfFix.DeQue<T>();
        if constexpr (IS_CAST_FLOAT) {
            Cast(calcSelfLocal, inputLocal, RoundMode::CAST_NONE, CeilAlignA2B(dataLen, BLOCK_SIZE));
        }
        outLocal = outQueueOutFix.AllocTensor<T>();
    }

    __aicore__ inline void CopyValueIn(int32_t index, int64_t progress, int64_t dataLen) {
        valueLocal = inQueueValueFix.AllocTensor<T>();
        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(valueLocal, valueGmFix[index * updatesOneTime + progress * maxSize], updatesExtParams, tPadParams);
        inQueueValueFix.EnQue(valueLocal);
    }

    __aicore__ inline void Compute(int64_t dataLen) {
        valueLocal = inQueueValueFix.DeQue<T>();
        if constexpr (IS_CAST_FLOAT) {
            PipeBarrier<PIPE_V>();
            Cast(calcValueLocal, valueLocal, RoundMode::CAST_NONE, dataLen);
            if (enableAlpha == 1) {
                PipeBarrier<PIPE_V>();
                Muls(calcValueLocal, calcValueLocal, alphaDataFix, dataLen);
            }
            PipeBarrier<PIPE_V>();
            inQueueValueFix.FreeTensor(valueLocal);
            Add(calcSelfLocal, calcSelfLocal, calcValueLocal, dataLen);
        } else {
            if (enableAlpha == 1) {
                Muls(valueLocal, valueLocal, alphaDataFix, dataLen);
                PipeBarrier<PIPE_V>();
            }
            Add(inputLocal, inputLocal, valueLocal, dataLen);
            inQueueValueFix.FreeTensor(valueLocal);
        }
    }

    __aicore__ inline void CopyOut(int32_t index, int64_t progress, int64_t dataLen) {
        if constexpr (IS_CAST_FLOAT) {
            inQueueSelfFix.FreeTensor(inputLocal);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, calcSelfLocal, RoundMode::CAST_RINT, CeilAlignA2B(dataLen, BLOCK_SIZE));
        } else {
            PipeBarrier<PIPE_V>();
            DataCopy(outLocal, inputLocal, CeilAlignA2B(dataLen, BLOCK_SIZE));
            inQueueSelfFix.FreeTensor(inputLocal);
        }
        outQueueOutFix.EnQue(outLocal);
        outLocal = outQueueOutFix.DeQue<T>();
        DataCopyExtParams updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputGmFix[index * updatesOneTime + progress * maxSize], outLocal, updatesExtParams);
        outQueueOutFix.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;
    GlobalTensor<T> inputGmFix, valueGmFix;
    GlobalTensor<float> alphaGmFix;
    GlobalTensor<int32_t> indicesGmFix, posGmFix;
    LocalTensor<T> inputLocal, valueLocal, outLocal;
    LocalTensor<float> calcSelfLocal, calcValueLocal;
    LocalTensor<int32_t> indexLocal;
    TBuf<QuePosition::VECCALC> calcSelfBuf, calcValueBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelfFix, inQueueValueFix, inQueueIndex;
    TQue<QuePosition::VECOUT, 1> outQueueOutFix;

    int64_t coreId;
    int32_t usedCoreNum;
    int32_t enableAlpha;
    int64_t eachIndexCountFix;
    int64_t lastIndexCountFix;
    int64_t inputCountFix;
    int64_t indicesCountFix;
    int64_t updatesCountFix;
    int64_t updatesOneTime;
    int64_t maxSize;
    int64_t eachNumFix;
    int64_t eachLoopFix;
    int64_t eachTailFix;
    int64_t currentEachIndex;
    int32_t finalIndex;
    int64_t currentUBIndexRound;
    int64_t currentIndexCount;
    int64_t currentUBIndexTail;
    int64_t currentEachNum;
    int64_t currentEachNumAlign;
    int32_t lastCoreIndex = -1;
    int64_t eachUBIndexRound = 1;
    int64_t eachUBIndexCount = 1;
    int64_t eachUBIndexTail = 0;
    int64_t lastUBIndexRound = 1;
    int64_t lastUBIndexCount = 1;
    int64_t lastUBIndexTail = 0;
    float alphaDataFix;
};

#endif  // INPLACE_INDEX_ADD_WITH_SORTED_FIX_H_