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
 * \file scatter_add_float_with_sorted.h
 * \brief
 */
#ifndef SCATTER_ADD_FLOAT_WITH_SORTED_H
#define SCATTER_ADD_FLOAT_WITH_SORTED_H
#include "kernel_operator.h"
#define IS_CAST_FLOAT ((is_same<T, half>::value) || (is_same<T, bfloat16_t>::value))
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T>
class KernelScatterAddFloatWithSorted {
public:
    __aicore__ inline KernelScatterAddFloatWithSorted() {}
    __aicore__ inline void Init(const ScatterAddWithSortedTilingData* __restrict tiling_data, TPipe *tmpPipe, GM_ADDR var,
                                GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pipe = tmpPipe;
        coreId = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        eachCount = tiling_data->eachCount;
        lastCount = tiling_data->lastCount;
        eachNum = tiling_data->eachNum;
        eachLoop = tiling_data->eachLoop;
        eachTail = tiling_data->eachTail;
        lastNum = tiling_data->lastNum;
        lastLoop = tiling_data->lastLoop;
        lastTail = tiling_data->lastTail;
        inputCount = tiling_data->inputCount;
        indicesCount = tiling_data->indicesCount;
        updatesCount = tiling_data->updatesCount;
        inputOneTime = tiling_data->inputOneTime;
        updatesOneTime = tiling_data->updatesOneTime;
        updatesAlign = tiling_data->updatesAlign;
        updatesLoop = tiling_data->updatesLoop;
        updatesEach = tiling_data->updatesEach;
        updatesLast = tiling_data->updatesLast;

        currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
        currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
        currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCount);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), updatesCount);
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(sorted_index), indicesCount);
        posGm.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pos), indicesCount);

        pipe->InitBuffer(inQueueSelf, BUFFER_NUM, updatesAlign * sizeof(T));
        pipe->InitBuffer(inQueueUpdates, BUFFER_NUM, updatesAlign * sizeof(T));
        pipe->InitBuffer(inQueueIndics, BUFFER_NUM, (currentEach + 1) * sizeof(int));
        pipe->InitBuffer(inQueuePos, BUFFER_NUM, currentEach * sizeof(int));
        pipe->InitBuffer(outQueueSelf, BUFFER_NUM, updatesAlign * sizeof(T));
        pipe->InitBuffer(calcSelfBuf, updatesAlign * sizeof(float));
        pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(float));

        indicesLocal = inQueueIndics.AllocTensor<int>();
        posLocal = inQueuePos.AllocTensor<int>();
        calcSelfLocal = calcSelfBuf.Get<float>();
        calcUpdatesLocal = calcUpdatesBuf.Get<float>();

        tPadParams = {false, 0, 0, static_cast<T>(0)};
        uPadParams = {false, 0, 0, static_cast<int>(0)};
    }

    __aicore__ inline void CopyInputFromGm2Ub(int index, int updatesOffset) {
        auto inputLocal = inQueueSelf.AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm[index * inputOneTime + updatesOffset], updatesExtParams, tPadParams);
        inQueueSelf.EnQue(inputLocal);
        
        auto dequeLocal = inQueueSelf.DeQue<T>();
        if constexpr (IS_CAST_FLOAT) {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Cast(calcSelfLocal, dequeLocal, RoundMode::CAST_NONE, updatesAlign);
        } else {
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            DataCopy(calcSelfLocal, dequeLocal, updatesAlign);
        }
        inQueueSelf.FreeTensor(dequeLocal);
    }

    __aicore__ inline void CopyInputFromUb2Gm(int index, int updatesOffset) {
        auto outputLocal = outQueueSelf.AllocTensor<T>();
        if constexpr (IS_CAST_FLOAT) {
            Cast(outputLocal, calcSelfLocal, RoundMode::CAST_RINT, updatesAlign);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        } else {
            DataCopy(outputLocal, calcSelfLocal, updatesAlign);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        }
        outQueueSelf.EnQue(outputLocal);
        
        auto dequeLocal = outQueueSelf.DeQue<T>();
        DataCopyPad(inputGm[index * inputOneTime + updatesOffset], dequeLocal, updatesExtParams);
        outQueueSelf.FreeTensor(dequeLocal);
    }

    __aicore__ inline void CopyUpdateFromGm2Ub(int index, int updatesOffset) {
        auto updatesLocal = inQueueUpdates.AllocTensor<T>();
        DataCopyPad(updatesLocal, updatesGm[index * updatesOneTime + updatesOffset], updatesExtParams, tPadParams);
        inQueueUpdates.EnQue(updatesLocal);
    }

    __aicore__ inline void ComputeUpdate() {
        auto updatesLocal = inQueueUpdates.DeQue<T>();
        if constexpr (IS_CAST_FLOAT) {
            Cast(calcUpdatesLocal, updatesLocal, RoundMode::CAST_NONE, updatesAlign);
            PipeBarrier<PIPE_V>();
            Add(calcSelfLocal, calcSelfLocal, calcUpdatesLocal, updatesAlign);
        } else {
            Add(calcSelfLocal, calcSelfLocal, updatesLocal, updatesAlign);
        }
        inQueueUpdates.FreeTensor(updatesLocal);
    }

    __aicore__ inline void SingleLoop(int updatesOffset)
    {
        int updateIndex = -1;
        int start, last, offset = 0;
        for (size_t i = 0; i < currentLoop; ++i) {
            uint64_t indicesOffset = eachCount * coreId + i * currentEach;
            currentNum = currentEach;
            if (i == currentLoop - 1) {
                currentNum = currentTail;
            }
            indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(int)), 0, 0, 0};
            indicesExtParams2 = {(uint16_t)1, static_cast<uint32_t>((currentNum + 1) * sizeof(int)), 0, 0, 0};
            if (coreId == 0 && i == 0) {
                DataCopyPad(indicesLocal, indicesGm[indicesOffset], indicesExtParams, uPadParams);
            } else {
                DataCopyPad(indicesLocal, indicesGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                offset = 1;
            }
            DataCopyPad(posLocal, posGm[indicesOffset], indicesExtParams, uPadParams);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            start = indicesLocal.GetValue(0);
            for (size_t j = 0; j < currentNum; ++j) {
                last = start;
                start = indicesLocal.GetValue(j + offset);
                if (coreId != 0 && start == last && updateIndex == -1) {
                    continue;
                }
                if (start != last && updateIndex != -1) {
                    CopyInputFromUb2Gm(last, updatesOffset);
                }
                if ((coreId + i + j == 0) || (start != last)) {
                    CopyInputFromGm2Ub(start, updatesOffset);
                }
                updateIndex = posLocal.GetValue(j);
                CopyUpdateFromGm2Ub(updateIndex, updatesOffset);
                ComputeUpdate();
                PipeBarrier<PIPE_V>();
            }            
        }

        last = start;
        
        bool run = (updateIndex != -1);
        while(run) {
            coreId = coreId + 1;
            currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
            currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
            currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;
            if (coreId >= usedCoreNum) {
                CopyInputFromUb2Gm(last, updatesOffset);
                break;
            }
            for (size_t i = 0; i < currentLoop; ++i) {
                uint64_t indicesOffset = eachCount * coreId + i * currentEach;
                currentNum = currentEach;
                if (i == currentLoop - 1) {
                    currentNum = currentTail;
                }
                indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(int)), 0, 0, 0};
                indicesExtParams2 = {(uint16_t)1, static_cast<uint32_t>((currentNum + 1) * sizeof(int)), 0, 0, 0};
                DataCopyPad(indicesLocal, indicesGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                DataCopyPad(posLocal, posGm[indicesOffset], indicesExtParams, uPadParams);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                for (size_t j = 0; j < currentNum; ++j) {
                    last = start;
                    start = indicesLocal.GetValue(j + 1);
                    if (start != last) {
                        CopyInputFromUb2Gm(last, updatesOffset);
                        run = false;
                        break;
                    }
                    updateIndex = posLocal.GetValue(j);
                    CopyUpdateFromGm2Ub(updateIndex, updatesOffset);
                    ComputeUpdate();
                    PipeBarrier<PIPE_V>();
                }
                if (!run) {
                    break;
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(updatesEach * sizeof(T)), 0, 0, 0};
        for (size_t i = 0; i < updatesLoop; ++i) {
            if (i == updatesLoop - 1) {
                updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(updatesLast * sizeof(T)), 0, 0, 0};
            }
            coreId = GetBlockIdx();
            currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
            currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
            currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;
            SingleLoop(updatesEach * i);
            PipeBarrier<PIPE_ALL>();
        }
        inQueueIndics.FreeTensor(indicesLocal);
        inQueuePos.FreeTensor(posLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf, inQueueIndics, inQueuePos, inQueueUpdates;
    TBuf<QuePosition::VECCALC> calcSelfBuf, calcUpdatesBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueSelf;
    GlobalTensor<T> inputGm, updatesGm;
    GlobalTensor<int> indicesGm, posGm;
    LocalTensor<int> indicesLocal, posLocal;
    LocalTensor<float> calcSelfLocal, calcUpdatesLocal;
    DataCopyPadExtParams<T> tPadParams;
    DataCopyPadExtParams<int> uPadParams;
    DataCopyExtParams indicesExtParams, indicesExtParams2, updatesExtParams;
    uint32_t coreId;
    uint64_t usedCoreNum;
    uint64_t currentNum;
    uint64_t currentEach;
    uint32_t currentLoop;
    uint64_t currentTail;
    uint64_t eachCount;
    uint64_t lastCount;
    uint64_t eachNum;
    uint64_t eachLoop;
    uint64_t eachTail;
    uint64_t lastNum;
    uint64_t lastLoop;
    uint64_t lastTail;
    uint64_t inputCount;
    uint64_t indicesCount;
    uint64_t updatesCount;
    uint64_t inputOneTime;
    uint64_t updatesOneTime;
    uint64_t updatesAlign;
    uint64_t updatesLoop;
    uint64_t updatesEach;
    uint64_t updatesLast;
};

#endif  // SCATTER_ADD_FLOAT_WITH_SORTED_H
