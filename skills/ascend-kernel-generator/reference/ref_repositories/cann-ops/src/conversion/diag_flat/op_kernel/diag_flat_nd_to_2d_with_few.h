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
 * \file diag_flat_nd_to_2d_with_few.h
 * \brief
 */
#ifndef DIAG_FLAT_ND_TO_2D_WITH_FEW
#define DIAG_FLAT_ND_TO_2D_WITH_FEW

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "platform.h"

using namespace AscendC;

namespace DiagFlat {

template <typename T>
class DiagFlatNDTo2DWithFew {
public:
    __aicore__ inline DiagFlatNDTo2DWithFew(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DiagV2TilingData *tilingData);
    __aicore__ inline void ClearOuput();
    template <typename U>
    __aicore__ inline void MemSetZero(GlobalTensor<U> gmTensor, int64_t size);
    __aicore__ inline void InitGm(GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void CopyIn(int64_t iter, int64_t rollback);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t iter);
    __aicore__ inline void CopyOutTail(int64_t iter);
    __aicore__ inline void GmSetZeroPad(GM_ADDR output);
    __aicore__ inline void GmSetZero(GM_ADDR output);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmOutput_;
    GlobalTensor<int16_t> gmOutputInt16_;
    GlobalTensor<int32_t> gmOutputInt32_;
    int64_t offset_{0};

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> workQueue_;

    // tiling params
    int64_t inputNum_{0};
    int64_t usedCoreNum_{1};
    int64_t totalCoreNum_{1};
    int64_t ONCE_HANDLE_NUM{0};
};

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::ParseTilingData(const DiagV2TilingData *tilingData)
{
    inputNum_ = tilingData->inputNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    offset_ = tilingData->diagonal;
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::GmSetZeroPad(GM_ADDR output)
{
    int64_t inputAllNum = (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_));
    int64_t cleanNum = (inputAllNum + totalCoreNum_ - 1) / totalCoreNum_;
    cleanNum = (cleanNum + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
    int64_t usedCoreNum = (inputAllNum + cleanNum - 1) / cleanNum;

    // tail core
    if (GetBlockIdx() == usedCoreNum - 1) {
        int64_t lastCleanNum = inputAllNum % cleanNum;
        if (lastCleanNum == 0) {
            lastCleanNum = cleanNum;
        }
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * cleanNum));
        if constexpr (sizeof(T) == 1) {
            InitOutput<int16_t>(gmOutputInt16_[GetBlockIdx() * cleanNum / 2], lastCleanNum / 2, 0);
            if (lastCleanNum % 2 == 1) {
                gmOutputInt16_.SetGlobalBuffer((__gm__ int16_t *)(output + inputAllNum - 2));
                InitOutput<int16_t>(gmOutputInt16_, 1, 0);
            }
        } else if constexpr (sizeof(T) < 8) {
            InitOutput<T>(gmOutput_, lastCleanNum, 0);
        } else if constexpr (sizeof(T) == 8) {
            InitOutput<int32_t>(gmOutputInt32_[GetBlockIdx() * cleanNum * 2], lastCleanNum * 2, 0);
        }

    } else if (GetBlockIdx() < usedCoreNum) {
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * cleanNum));
        if constexpr (sizeof(T) == 1) {
            InitOutput<int16_t>(gmOutputInt16_[GetBlockIdx() * cleanNum / 2], cleanNum / 2, 0);
        } else if constexpr (sizeof(T) < 8) {
            InitOutput<T>(gmOutput_, cleanNum, 0);
        } else if constexpr (sizeof(T) == 8) {
            InitOutput<int32_t>(gmOutputInt32_[GetBlockIdx() * cleanNum * 2], cleanNum * 2, 0);
        }
    }
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::GmSetZero(GM_ADDR output)
{
    int64_t inputSum = (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_));
    int64_t minCleanNum = 256 / sizeof(T);
    int64_t newCleanNum = (inputSum + totalCoreNum_ - 1) / totalCoreNum_;
    newCleanNum = (newCleanNum + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
    int64_t usedCoreNum = (inputSum + newCleanNum - 1) / newCleanNum;

    // tail core
    if (GetBlockIdx() == usedCoreNum - 1) {
        int64_t lastCleanNum = inputSum - (usedCoreNum - 1) * newCleanNum;
        int64_t bkOffset = 0;
        if (lastCleanNum % ONCE_HANDLE_NUM != 0) {
            bkOffset = ONCE_HANDLE_NUM - lastCleanNum % ONCE_HANDLE_NUM;
            lastCleanNum = (lastCleanNum + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
        }
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * newCleanNum) - bkOffset);
        MemSetZero<T>(gmOutput_, lastCleanNum);

        // main core
    } else if (GetBlockIdx() < usedCoreNum) {
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * newCleanNum));
        MemSetZero<T>(gmOutput_, newCleanNum);
    }
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::InitGm(GM_ADDR output, GM_ADDR workspace)
{
    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)workspace, totalCoreNum_ * 32 / sizeof(int32_t));

    MemSetZero<int32_t>(syncGlobal_, totalCoreNum_ * EACH_CORE_HANDLE_NUM);

    // set workspace for sync
    pipe.InitBuffer(workQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));
    
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        GmSetZeroPad(output);
    } else {
        GmSetZero(output);
    }

    // 区分芯片使用syncAll同步
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        SyncAll();
    } else {
        LocalTensor<int32_t> workLocal = workQueue_.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        workQueue_.FreeTensor(workLocal);
    }
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData)
{
    // get tiling data
    ParseTilingData(tilingData);
    ONCE_HANDLE_NUM = ONE_BLK_SIZE / sizeof(T);
    gmOutputInt32_.SetGlobalBuffer((__gm__ int32_t *)output);
    gmOutputInt16_.SetGlobalBuffer((__gm__ int16_t *)output);
    // init output
    if (inputNum_ + offset_ != 1) {
        InitGm(output, workspace);
    }

    // init gmInput_ gmOutput_
    gmInput_.SetGlobalBuffer((__gm__ T *)input);
    gmOutput_.SetGlobalBuffer((__gm__ T *)output + (offset_ > 0 ? 0 : abs(offset_)) * (inputNum_ + abs(offset_)),
        (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_)));

    pipe.InitBuffer(inputQueue_, 1, ONCE_HANDLE_NUM * sizeof(T));
    pipe.InitBuffer(outputQueue_, 1, ONCE_HANDLE_NUM * ONCE_HANDLE_NUM * sizeof(T));
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::CopyIn(int64_t iter, int64_t rollback)
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    DataCopy(ubInput, gmInput_[iter * ONCE_HANDLE_NUM - rollback], ONCE_HANDLE_NUM);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::Compute()
{
    ClearOuput();
    LocalTensor<T> ubInput = inputQueue_.DeQue<T>();
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int i = 0; i < ONCE_HANDLE_NUM; i++) {
        ubOutput.SetValue(ONCE_HANDLE_NUM * i + i, ubInput.GetValue(i));
    }
    outputQueue_.EnQue<T>(ubOutput);
    inputQueue_.FreeTensor(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::CopyOut(int64_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    int64_t gmOffset = 0;
    int64_t width = inputNum_ + abs(offset_);
    for (int64_t i = 0; i < ONCE_HANDLE_NUM; i++) {
        pipe_barrier(PIPE_MTE3);
        int64_t rowIdx = iter * ONCE_HANDLE_NUM + i;
        gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + iter * ONCE_HANDLE_NUM;
        DataCopy(gmOutput_[gmOffset], ubOutput[ONCE_HANDLE_NUM * i], ONCE_HANDLE_NUM);
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::CopyOutTail(int64_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    int64_t gmOffset = 0;
    int64_t width = inputNum_ + abs(offset_);
    // no align
    if (inputNum_ % ONCE_HANDLE_NUM != 0) {
        if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
            int64_t rowIdx = iter * ONCE_HANDLE_NUM;
            int64_t gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + iter * ONCE_HANDLE_NUM;
            uint16_t blockCount = inputNum_ % ONCE_HANDLE_NUM;
            uint16_t blockLen = (inputNum_ % ONCE_HANDLE_NUM) * sizeof(T);
            uint16_t srcStride = 0;
            uint16_t dstStride = width * sizeof(T) - blockLen;
            DataCopyParams copyParams {blockCount, blockLen, srcStride, dstStride};
            if (sizeof(T) < 8) {
                DataCopyPad(gmOutput_[gmOffset], ubOutput, copyParams);
            } else {
                LocalTensor<int32_t> ubOutputCast = ubOutput.template ReinterpretCast<int32_t>();
                DataCopyPad(gmOutputInt32_[gmOffset * 2], ubOutputCast, copyParams);
            }
        } else {
            int64_t rowBack = iter == 0 ? 0 : ONCE_HANDLE_NUM - inputNum_ % ONCE_HANDLE_NUM;
            int64_t iterBack = iter - 1 < 0 ? 0 : iter - 1;
            int64_t offset = iter == 0 ? 0 : inputNum_ % ONCE_HANDLE_NUM;
            for (int64_t i = 0; i < ONCE_HANDLE_NUM; i++) {
                pipe_barrier(PIPE_MTE3);
                int64_t rowIdx = iter * ONCE_HANDLE_NUM + i - rowBack;
                gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + iterBack * ONCE_HANDLE_NUM + offset;
                DataCopy(gmOutput_[gmOffset], ubOutput[ONCE_HANDLE_NUM * i], ONCE_HANDLE_NUM);
            }
        }
    } else {
        for (int64_t i = 0; i < ONCE_HANDLE_NUM; i++) {
            pipe_barrier(PIPE_MTE3);
            int64_t rowIdx = iter * ONCE_HANDLE_NUM + i;
            gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + iter * ONCE_HANDLE_NUM;
            DataCopy(gmOutput_[gmOffset], ubOutput[ONCE_HANDLE_NUM * i], ONCE_HANDLE_NUM);
        }
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    int64_t loopSize = (inputNum_ + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM;
    for (int64_t i = 0; i < loopSize - 1; i++) {
        CopyIn(i, 0);
        Compute();
        CopyOut(i);
    }
    int64_t rollback = 0;
    if (inputNum_ > ONCE_HANDLE_NUM && inputNum_ % ONCE_HANDLE_NUM != 0 &&
        !PlatformSocInfo::IsDataCopyPadSupport()) {
        rollback = ONCE_HANDLE_NUM - inputNum_ % ONCE_HANDLE_NUM;
    }
    CopyIn(loopSize - 1, rollback);
    Compute();
    CopyOutTail(loopSize - 1);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::ClearOuput()
{
    LocalTensor<T> ubOutput = outputQueue_.AllocTensor<T>();
    LocalTensor<int16_t> ubOutputCast = ubOutput.template ReinterpretCast<int16_t>();

    // construct ONCE_HANDLE_NUM*ONCE_HANDLE_NUM zeor matrix
    constexpr uint8_t BLOCK_LENGTH = 32;
    constexpr uint64_t INT16_MAX_MASK = 128;
    uint64_t countOfInt16 = ONCE_HANDLE_NUM * sizeof(T) / sizeof(int16_t);
    uint64_t loops = (countOfInt16 + INT16_MAX_MASK - 1) / INT16_MAX_MASK;
    int16_t scalarValue = 0;
    uint64_t mask = countOfInt16 > INT16_MAX_MASK ? INT16_MAX_MASK : countOfInt16;
    uint8_t repeatTimes = ONCE_HANDLE_NUM;
    uint16_t dstBlockStride = 1;
    uint8_t dstRepeatStride = ONCE_HANDLE_NUM * sizeof(T) / BLOCK_LENGTH;
    for (int i = 0; i < loops; i++) {
        Duplicate(ubOutputCast[mask * i], scalarValue, mask, repeatTimes, dstBlockStride, dstRepeatStride);
    }
    outputQueue_.EnQue<T>(ubOutput);
}

template <typename T>
template <typename U>
__aicore__ inline void DiagFlatNDTo2DWithFew<T>::MemSetZero(GlobalTensor<U> gmTensor, int64_t size)
{
    if (g_coreType == AIC) {
        return;
    }
    int64_t int16Size = (size * sizeof(U) + sizeof(int16_t) - 1) / sizeof(int16_t);
    LocalTensor<int16_t> popBuffer;
    bool ret = PopStackBuffer<int16_t, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(int16_t);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = int16Size / popSize;
    uint32_t tail = int16Size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    DuplicateImpl<int16_t>((__ubuf__ int16_t *)popBuffer.GetPhyAddr(), 0, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    wait_flag(PIPE_V, PIPE_MTE3, eventIDVToMTE3);
    uint32_t comOffset = 0;
    // compute the main block
    for (int index = 0; index < round; ++index) {
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((roundSize * sizeof(int16_t) + ONE_BLK_SIZE - 1) / (ONE_BLK_SIZE)), 0, 0});
        comOffset += roundSize;
    }
    // compute the tail block
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyUB2GMImpl((__gm__ int16_t *)gmTensor.GetPhyAddr() + comOffset,
            (__ubuf__ int16_t *)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((tail * sizeof(int16_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE), 0, 0});
    }
}

}  // namespace DiagFlat
#endif  // DIAG_FLAT_ND_TO_2D_WITH_FEW