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
 * \file scatter_with_sorted.h
 * \brief
 */
#ifndef SCATTER_WITH_SORTED_H
#define SCATTER_WITH_SORTED_H
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class KernelScatterWithSorted {
public:
    __aicore__ inline KernelScatterWithSorted() {}
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

        pipe->InitBuffer(inQueueIndics, BUFFER_NUM, (currentEach + 1) * sizeof(int));
        pipe->InitBuffer(inQueuePos, BUFFER_NUM, (currentEach + 1) * sizeof(int));
        pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(T));

        indicesLocal = inQueueIndics.AllocTensor<int>();
        posLocal = inQueuePos.AllocTensor<int>();
        updatesLocal = calcUpdatesBuf.Get<T>();

        tPadParams = {false, 0, 0, static_cast<T>(0)};
        uPadParams = {false, 0, 0, static_cast<int>(0)};
    }

    __aicore__ inline void CopyLastUpdate(int updateIndex, int inputIndex) {
        PipeBarrier<PIPE_ALL>();
        updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(updatesEach * sizeof(T)), 0, 0, 0};
        for (size_t i = 0; i < updatesLoop; ++i) {
            if (i == updatesLoop - 1) {
                updatesExtParams = {(uint16_t)1, static_cast<uint32_t>(updatesLast * sizeof(T)), 0, 0, 0};
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            DataCopyPad(updatesLocal, updatesGm[updateIndex * updatesOneTime + updatesEach * i], updatesExtParams, tPadParams);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            DataCopyPad(inputGm[inputIndex * inputOneTime + updatesEach * i], updatesLocal, updatesExtParams);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        int updateIndex = -1;
        int start, last, offset = 0;
        bool enter = coreId == 0;
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
                DataCopyPad(posLocal, posGm[indicesOffset], indicesExtParams, uPadParams);
            } else {
                DataCopyPad(indicesLocal, indicesGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                DataCopyPad(posLocal, posGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                offset = 1;
            }
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            start = indicesLocal.GetValue(0);
            for (size_t j = offset; j < currentNum + offset; ++j) {
                last = start;
                start = indicesLocal.GetValue(j);
                if (start != last) {
                    if (enter) {
                        updateIndex = posLocal.GetValue(j - 1);
                        CopyLastUpdate(updateIndex, last);
                    }
                    enter = true;
                }
            }
        }

        last = start;
        
        bool run = enter;
        while(run) {
            coreId = coreId + 1;
            currentEach = coreId == (usedCoreNum - 1) ? lastNum : eachNum;
            currentLoop = coreId == (usedCoreNum - 1) ? lastLoop : eachLoop;
            currentTail = coreId == (usedCoreNum - 1) ? lastTail : eachTail;
            if (coreId >= usedCoreNum) {
                updateIndex = posLocal.GetValue(currentNum + offset - 1);
                CopyLastUpdate(updateIndex, last);
                break;
            }
            offset = 1;
            for (size_t i = 0; i < currentLoop; ++i) {
                uint64_t indicesOffset = eachCount * coreId + i * currentEach;
                currentNum = currentEach;
                if (i == currentLoop - 1) {
                    currentNum = currentTail;
                }
                indicesExtParams2 = {(uint16_t)1, static_cast<uint32_t>((currentNum + 1) * sizeof(int)), 0, 0, 0};
                DataCopyPad(indicesLocal, indicesGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                DataCopyPad(posLocal, posGm[indicesOffset - 1], indicesExtParams2, uPadParams);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                for (size_t j = 0; j < currentNum; ++j) {
                    last = start;
                    start = indicesLocal.GetValue(j + 1);
                    if (start != last) {
                        updateIndex = posLocal.GetValue(j);
                        CopyLastUpdate(updateIndex, last);
                        run = false;
                        break;
                    }
                }
                last = start;
                if (!run) {
                    break;
                }
            }
        }

        inQueueIndics.FreeTensor(indicesLocal);
        inQueuePos.FreeTensor(posLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndics, inQueuePos;
    TBuf<QuePosition::VECCALC> calcUpdatesBuf;
    GlobalTensor<T> inputGm, updatesGm;
    GlobalTensor<int> indicesGm, posGm;
    LocalTensor<int> indicesLocal, posLocal;
    LocalTensor<T> updatesLocal;
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

#endif  // SCATTER_WITH_SORTED_H
