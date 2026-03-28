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
 * \file diag_flat_nd_to_2d_b16_more64.h
 * \brief
 */
#ifndef DIAG_FLAT_ND_TO_2D_B16_MORE_64
#define DIAG_FLAT_ND_TO_2D_B16_MORE_64

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "platform.h"

using namespace AscendC;

namespace DiagFlat {

template <typename T>
class DiagFlatND2To2DB16More64 {
public:
    __aicore__ inline DiagFlatND2To2DB16More64() = default;
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint16_t iter);
    __aicore__ inline void Compute(uint16_t iter);
    __aicore__ inline void CopyOut(uint16_t iter);
    __aicore__ inline void CreateAuxMatrix();
    template <typename U>
    __aicore__ inline void MemSetZero(GlobalTensor<U> gmTensor, int64_t size);
    __aicore__ inline void ParseTilingData(const DiagV2TilingData *tilingData);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> assistBuf_;

    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmOutput_;
    GlobalTensor<T> gmAssist_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<int32_t> syncGlobal_;

    TQue<QuePosition::VECIN, 1> workQueue_;

    int64_t offset_{0};
    int64_t inputNum_{0};
    int64_t inputIdx_{0};
    int64_t usedCoreNum_{1};
    int64_t normalCoreHandleNum_{64};
    int64_t lastCoreHandleNum_{64};
    int64_t curHandleNum_{64};
    int64_t totalCoreNum_{1};
};

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::ParseTilingData(const DiagV2TilingData *tilingData)
{
    inputNum_ = tilingData->inputNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    normalCoreHandleNum_ = tilingData->normalCoreHandleNum;
    lastCoreHandleNum_ = tilingData->lastCoreHandleNum;
    offset_ = tilingData->diagonal;

    inputIdx_ = GetBlockIdx() * normalCoreHandleNum_;
    // [尾核处理] && [非尾核处理]
    if (GetBlockIdx() == usedCoreNum_ - 1) {
        if (lastCoreHandleNum_ % 64 != 0) {
            inputIdx_ -= (64 - (lastCoreHandleNum_ % 64));
        }
        curHandleNum_ = (lastCoreHandleNum_ + 64 - 1) / 64 * 64;
    } else {
        curHandleNum_ = normalCoreHandleNum_;
    }
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const DiagV2TilingData *tilingData)
{
    // 解析tiling数据
    ParseTilingData(tilingData);

    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)workspace, totalCoreNum_ * 32 / sizeof(int32_t));
    MemSetZero<int32_t>(syncGlobal_, totalCoreNum_ * EACH_CORE_HANDLE_NUM);

    // 核间同步
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)workspace, 1024 * sizeof(int32_t));
    pipe.InitBuffer(workQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));

    // gm输出进行清零
    int32_t coreNum = totalCoreNum_;
    int64_t inputSumNum = (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_));
    int64_t cleanNum = (inputSumNum + coreNum - 1) / coreNum;

    if (GetBlockIdx() == coreNum - 1) {
        int64_t lastCleanNum = inputSumNum % cleanNum;
        if (lastCleanNum == 0) {
            lastCleanNum = cleanNum;
        }
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * cleanNum));
        MemSetZero<T>(gmOutput_, lastCleanNum);
    } else {
        gmOutput_.SetGlobalBuffer((__gm__ T *)output + (GetBlockIdx() * cleanNum));
        MemSetZero<T>(gmOutput_, cleanNum);
    }
    // 创建辅助矩阵
    CreateAuxMatrix();
    
    // 区分芯片使用syncAll同步
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
        SyncAll();
    } else {
        LocalTensor<int32_t> workLocal = workQueue_.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        workQueue_.FreeTensor(workLocal);
    }

    // 参数初始化
    gmInput_.SetGlobalBuffer((__gm__ T *)input + inputIdx_ * 2, inputNum_);

    // 首地址偏移量
    gmOutput_.SetGlobalBuffer(
        (__gm__ T *)output + (inputIdx_ + (offset_ > 0 ? 0 : abs(offset_))) * (inputNum_ + abs(offset_)) * 2,
        (inputNum_ + abs(offset_)) * (inputNum_ + abs(offset_)) * 2);

    // 申请内存空间
    pipe.InitBuffer(inputQueue_, 1, 128 * sizeof(int64_t));
    pipe.InitBuffer(outputQueue_, 1, 64 * 128 * sizeof(int64_t));
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CopyIn(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    DataCopy(ubInput, gmInput_[iter * 64 * 2], 64 * 2);
    inputQueue_.EnQue<T>(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Compute(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.DeQue<T>();
    LocalTensor<T> ubOutput = outputQueue_.AllocTensor<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    // 参数设置
    uint64_t mask = 128;
    uint64_t repeatTime = 64;
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = 32;
    uint8_t src0RepStride = 0;
    uint8_t src1RepStride = 32;

    // 类型转换
    LocalTensor<int16_t> CastUbInput = ubInput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> CastUbOutput = ubOutput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> CastUbAssist = ubAssist.template ReinterpretCast<int16_t>();

    BinaryRepeatParams params = {
        dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride};
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    And(CastUbOutput[0], CastUbInput[0], CastUbAssist[0], mask, repeatTime, params);
    And(CastUbOutput[128], CastUbInput[128], CastUbAssist[128], mask, repeatTime, params);
    And(CastUbOutput[128 * 2], CastUbInput[128 * 2], CastUbAssist[128 * 2], mask, repeatTime, params);
    And(CastUbOutput[128 * 3], CastUbInput[128 * 3], CastUbAssist[128 * 3], mask, repeatTime, params);

    outputQueue_.EnQue<T>(ubOutput);
    inputQueue_.FreeTensor(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CopyOut(uint16_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    // 区分为output的 [对齐场景,且小于65535个数] && [非对齐场景]进行拷贝策略
    /*
    [CopyOut 逻辑伪代码]:

     if {
         if (对齐场景 && gap < 65535){
             ...
         } else {
             ...
         }
     }
     */

    int64_t width = (inputNum_ + abs(offset_));
    int64_t BLOCK_SIZE = 32;
    uint16_t blockCount = 64;
    uint16_t blockLen = 64 * 2 * sizeof(int64_t) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = width * 2 * sizeof(int64_t) / BLOCK_SIZE - blockLen;

    if (width % 64 == 0 && dstStride < 65535) {
        const DataCopyParams intriParams = {blockCount, blockLen, srcStride, dstStride};
        auto rowIdx1 = iter * 64;
        int64_t gmOffset1 =
            (inputNum_ + abs(offset_)) * rowIdx1 + (offset_ > 0 ? offset_ : 0) + (iter * 64 + inputIdx_);
        DataCopy(gmOutput_[gmOffset1 * 2], ubOutput[0], intriParams);
    } else {
        for (int32_t i = 0; i < 64; i++) {
            auto rowIdx2 = iter * 64 + i;
            int64_t gmOffset2 =
                (inputNum_ + abs(offset_)) * rowIdx2 + (offset_ > 0 ? offset_ : 0) + (iter * 64 + inputIdx_);
            DataCopy(gmOutput_[gmOffset2 * 2], ubOutput[64 * i * 2], 64 * 2);
        }
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }

    int32_t loops = curHandleNum_ / 64;
    for (int32_t i = 0; i < loops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

/*
 * 辅助矩阵的创建函数
 */
template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CreateAuxMatrix()
{
    pipe.InitBuffer(assistBuf_, 64 * 128 * sizeof(int64_t));
    LocalTensor<int16_t> ubAssist = assistBuf_.Get<int16_t>();

    constexpr int64_t loops = 64 * 4;
    constexpr int16_t scalarValue = 0;
    constexpr uint8_t calCount = 64 * 2;
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int32_t i = 0; i < loops; i++) {
        Duplicate(ubAssist[calCount * i], scalarValue, calCount);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    int64_t fillVal = -1;
    constexpr int32_t repeatTime = 64;
    constexpr int32_t step = 128;
    LocalTensor<int64_t> tmpBuf = assistBuf_.Get<int64_t>();
    for (int32_t i = 0; i < repeatTime; i++) {
        tmpBuf.SetValue(step * i + i * 2, fillVal);
        tmpBuf.SetValue(step * i + (i * 2 + 1), fillVal);
    }
}

template <typename T>
template <typename U>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::MemSetZero(GlobalTensor<U> gmTensor, int64_t size)
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

#endif  // DIAG_FLAT_ND_TO_2D_B16_MORE_64
