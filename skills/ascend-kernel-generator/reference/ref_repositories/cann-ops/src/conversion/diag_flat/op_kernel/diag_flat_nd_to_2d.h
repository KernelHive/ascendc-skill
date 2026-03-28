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
 * \file diag_flat_nd_to_2d.h
 * \brief
 */
#ifndef DIAG_FLAT_ND_TO_2D
#define DIAG_FLAT_ND_TO_2D

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "platform.h"

using namespace AscendC;

namespace DiagFlat {
template <typename T>
class DiagFlatNDTo2D {
public:
    __aicore__ inline DiagFlatNDTo2D(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DiagV2TilingData *tilingData);
    __aicore__ inline void ConstructAssistMatrix();
    __aicore__ inline void InitGm(GM_ADDR output, GM_ADDR workspace);
    template <typename U>
    __aicore__ inline void MemSetZero(GlobalTensor<U> gmTensor, int64_t size);
    __aicore__ inline void CopyIn(int64_t iter);
    __aicore__ inline void Compute(int64_t iter);
    __aicore__ inline void CopyOut(int64_t iter);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> assistBuf_;
    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmOutput_;
    int64_t offset_{0};

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> workQueue_;

    // tiling params
    int64_t inputNum_{0};
    int64_t inputIdx_{0};
    int64_t usedCoreNum_{1};
    int64_t normalCoreHandleNum_{64};
    int64_t lastCoreHandleNum_{64};
    int64_t curHandleNum_{64};
    int64_t totalCoreNum_{1};
    const int32_t ONCE_HANDLE_NUM{64};
};

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::ParseTilingData(const DiagV2TilingData *tilingData)
{
    inputNum_ = tilingData->inputNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    normalCoreHandleNum_ = tilingData->normalCoreHandleNum;
    lastCoreHandleNum_ = tilingData->lastCoreHandleNum;
    offset_ = tilingData->diagonal;
    inputIdx_ = GetBlockIdx() * normalCoreHandleNum_;
    // tail core
    if (GetBlockIdx() == usedCoreNum_ - 1) {
        // rollback
        if (lastCoreHandleNum_ % ONCE_HANDLE_NUM != 0) {
            inputIdx_ -= (ONCE_HANDLE_NUM - (lastCoreHandleNum_ % ONCE_HANDLE_NUM));
        }
        // align to 64
        curHandleNum_ = (lastCoreHandleNum_ + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
        // main core
    } else {
        curHandleNum_ = normalCoreHandleNum_;
    }
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::InitGm(GM_ADDR output, GM_ADDR workspace)
{
    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)workspace, totalCoreNum_ * 32 / sizeof(int32_t));
    MemSetZero<int32_t>(syncGlobal_, totalCoreNum_ * EACH_CORE_HANDLE_NUM);

    // set workspace for sync
    pipe.InitBuffer(workQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));

    int64_t inputSumNum = (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_));
    int64_t minCleanNum = 256 / sizeof(T);
    int64_t newCleanNum = (inputSumNum + totalCoreNum_ - 1) / totalCoreNum_;
    newCleanNum = (newCleanNum + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
    int64_t usedCoreNum = (inputSumNum + newCleanNum - 1) / newCleanNum;

    // tail core
    if (GetBlockIdx() == usedCoreNum - 1) {
        int64_t lastCleanNum = inputSumNum - (usedCoreNum - 1) * newCleanNum;
        int64_t backOffset = 0;
        if (lastCleanNum % ONCE_HANDLE_NUM != 0) {
            backOffset = ONCE_HANDLE_NUM - lastCleanNum % ONCE_HANDLE_NUM;
            lastCleanNum = (lastCleanNum + ONCE_HANDLE_NUM - 1) / ONCE_HANDLE_NUM * ONCE_HANDLE_NUM;
        }
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * newCleanNum) - backOffset);
        MemSetZero<T>(gmOutput_, lastCleanNum);

        // main core
    } else if (GetBlockIdx() < usedCoreNum) {
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * newCleanNum));
        MemSetZero<T>(gmOutput_, newCleanNum);
    }

    ConstructAssistMatrix();
    
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
__aicore__ inline void DiagFlatNDTo2D<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    // init output
    InitGm(output, workspace);

    // init gm buffer
    gmInput_.SetGlobalBuffer((__gm__ T *)input + inputIdx_, inputNum_);
    gmOutput_.SetGlobalBuffer(
        (__gm__ T *)output + (inputIdx_ + (offset_ > 0 ? 0 : abs(offset_))) * (inputNum_ + abs(offset_)),
        (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_)));

    // init ub buffer
    pipe.InitBuffer(inputQueue_, 1, ONCE_HANDLE_NUM * sizeof(T));
    pipe.InitBuffer(outputQueue_, 1, ONCE_HANDLE_NUM * ONCE_HANDLE_NUM * sizeof(T));
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::CopyIn(int64_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    DataCopy(ubInput, gmInput_[iter * ONCE_HANDLE_NUM], ONCE_HANDLE_NUM);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::Compute(int64_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.DeQue<T>();
    LocalTensor<T> ubOutput = outputQueue_.AllocTensor<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    uint64_t countOfInt16 = ONCE_HANDLE_NUM * sizeof(T) / sizeof(int16_t);
    constexpr uint64_t INT16_MAX_MASK = 128;
    uint64_t loops = (countOfInt16 + INT16_MAX_MASK - 1) / INT16_MAX_MASK;
    uint64_t mask = countOfInt16 > INT16_MAX_MASK ? INT16_MAX_MASK : countOfInt16;
    uint8_t repeatTime = ONCE_HANDLE_NUM;
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    uint8_t src0RepStride = 0;
    uint8_t src1RepStride = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    BinaryRepeatParams params = {
        dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride};

    LocalTensor<int16_t> ubInputCast = ubInput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> ubOutputCast = ubOutput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> ubAssistCast = ubAssist.template ReinterpretCast<int16_t>();
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (int i = 0; i < loops; i++) {
        And(ubOutputCast[mask * i], ubInputCast[mask * i], ubAssistCast[mask * i], mask, repeatTime, params);
    }

    outputQueue_.EnQue<T>(ubOutput);
    inputQueue_.FreeTensor(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::CopyOut(int64_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();

    int64_t width = inputNum_ + abs(offset_);
    uint16_t blockLen = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    uint16_t dstStride = width * sizeof(T) / ONE_BLK_SIZE - blockLen;
    // align
    if ((width * sizeof(T) % ONE_BLK_SIZE == 0 && dstStride < 65535)) {
        uint16_t blockCount = ONCE_HANDLE_NUM;
        uint16_t srcStride = 0;
        DataCopyParams intriParams = {blockCount, blockLen, srcStride, dstStride};
        int64_t rowIdx = iter * ONCE_HANDLE_NUM;
        int64_t gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + (iter * ONCE_HANDLE_NUM + inputIdx_);
        DataCopy(gmOutput_[gmOffset], ubOutput, intriParams);
        // non-align
    } else {
        if constexpr (PlatformSocInfo::IsDataCopyPadSupport() && sizeof(T) < 8) {
            int64_t rowIdx = iter * ONCE_HANDLE_NUM;
            int64_t gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + (iter * ONCE_HANDLE_NUM + inputIdx_);
            uint16_t blockCount = ONCE_HANDLE_NUM;
            uint16_t blockLen = ONCE_HANDLE_NUM * sizeof(T);
            uint16_t srcStride = 0;
            uint16_t dstStride = width * sizeof(T) - blockLen;
            DataCopyParams copyParams {blockCount, blockLen, srcStride, dstStride};
            DataCopyPad(gmOutput_[gmOffset], ubOutput, copyParams);
        } else {
            for (int64_t i = 0; i < ONCE_HANDLE_NUM; i++) {
                int64_t rowIdx = iter * ONCE_HANDLE_NUM + i;
                int64_t gmOffset = width * rowIdx + (offset_ > 0 ? offset_ : 0) + (iter * ONCE_HANDLE_NUM + inputIdx_);
                DataCopy(gmOutput_[gmOffset], ubOutput[ONCE_HANDLE_NUM * i], ONCE_HANDLE_NUM);
            }
        }
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    int64_t loops = curHandleNum_ / ONCE_HANDLE_NUM;
    for (int64_t i = 0; i < loops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <typename T>
__aicore__ inline void DiagFlatNDTo2D<T>::ConstructAssistMatrix()
{
    // init buffer according to inputBytesSize
    pipe.InitBuffer(assistBuf_, ONCE_HANDLE_NUM * ONCE_HANDLE_NUM * sizeof(T));

    // construct zeor matrix
    LocalTensor<int16_t> ubAssist = assistBuf_.Get<int16_t>();
    constexpr uint64_t INT16_MAX_MASK = 128;
    uint64_t countOfInt16 = ONCE_HANDLE_NUM * sizeof(T) / sizeof(int16_t);
    uint64_t loops = (countOfInt16 + INT16_MAX_MASK - 1) / INT16_MAX_MASK;
    int16_t scalarValue = 0;
    uint64_t mask = countOfInt16 > INT16_MAX_MASK ? INT16_MAX_MASK : countOfInt16;
    uint8_t repeatTimes = ONCE_HANDLE_NUM;
    uint16_t dstBlockStride = 1;
    uint8_t dstRepeatStride = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int i = 0; i < loops; i++) {
        Duplicate(ubAssist[mask * i], scalarValue, mask, repeatTimes, dstBlockStride, dstRepeatStride);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // set value as -1
    if (sizeof(T) == 1) {
        int8_t value = -1;
        LocalTensor<int8_t> ubTmp = assistBuf_.Get<int8_t>();
        for (int i = 0; i < ONCE_HANDLE_NUM; i++) {
            ubTmp.SetValue(ONCE_HANDLE_NUM * i + i, value);
        }
    } else if (sizeof(T) == 2) {
        int16_t value = -1;
        LocalTensor<int16_t> ubTmp = assistBuf_.Get<int16_t>();
        for (int i = 0; i < ONCE_HANDLE_NUM; i++) {
            ubTmp.SetValue(ONCE_HANDLE_NUM * i + i, value);
        }
    } else if (sizeof(T) == 4) {
        int32_t value = -1;
        LocalTensor<int32_t> ubTmp = assistBuf_.Get<int32_t>();
        for (int i = 0; i < ONCE_HANDLE_NUM; i++) {
            ubTmp.SetValue(ONCE_HANDLE_NUM * i + i, value);
        }

    } else if (sizeof(T) == 8) {
        int64_t value = -1;
        LocalTensor<int64_t> ubTmp = assistBuf_.Get<int64_t>();
        for (int i = 0; i < ONCE_HANDLE_NUM; i++) {
            ubTmp.SetValue(ONCE_HANDLE_NUM * i + i, value);
        }
    }
}

template <typename T>
template <typename U>
__aicore__ inline void DiagFlatNDTo2D<T>::MemSetZero(GlobalTensor<U> gmTensor, int64_t size)
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
#endif  // DIAG_FLAT_ND_TO_2D