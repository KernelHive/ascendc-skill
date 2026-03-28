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
 * \file upsample_nearest_exact3d_310p.h
 * \brief
 */
#ifndef UPSAMPLE_NEARESTEXACT3D_310P_H
#define UPSAMPLE_NEARESTEXACT3D_310P_H

#include <type_traits>
#include "kernel_operator.h"

namespace UpsampleNearestExact3d {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int8_t D_INDEX = 0;
constexpr int8_t H_INDEX = 1;
constexpr int8_t W_INDEX = 2;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr float BEST_PERFORMANCE_SCALE = 100.0f;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

const int64_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;
const int64_t DEFAULT_SYNC_UB_SIZE = 1 * 1024;

template <typename T>
class UpsampleNearestExact3dND310p {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearestExact3dND310p(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, bool isNearestExact, GM_ADDR workspace,
        const UpsampleNearestExact3dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };
    template <typename T1>
    __aicore__ inline T1 Max(T1 a, T1 b)
    {
        return a > b ? a : b;
    };
    __aicore__ inline void ClearGM();
    __aicore__ inline void ParseTilingData(const UpsampleNearestExact3dTilingData *tilingData);
    __aicore__ inline void GatherData(int64_t slideIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyParams repeatParams);
    __aicore__ inline void ComputeAndCopyOut(
        uint32_t dataCount, uint32_t srcDataLength, uint32_t blockCount, int64_t outputOffset);
    __aicore__ inline void CopyOutProcess(int64_t offsetTemp, LocalTensor<T> dstLocal);
    __aicore__ inline void CopyOut(int64_t offsetTemp, LocalTensor<T> dstLocal, int64_t copyOutCnt);
    __aicore__ inline void GetRangeW(int64_t slideIndex);
    __aicore__ inline void GetRangeH(int64_t slideIndex);
    __aicore__ inline void GetRangeD(int64_t slideIndex);
    __aicore__ inline void CalculateSrcIndexTensor(
        int64_t index, int64_t length, int8_t direction, LocalTensor<float> srcIndexTensor);
    __aicore__ inline void CalculateGatherOffsetW();

private:
    TBuf<QuePosition::VECCALC> srcIndexQueueW;
    TBuf<QuePosition::VECCALC> srcIndexQueueH;
    TBuf<QuePosition::VECCALC> srcIndexQueueD;
    TBuf<QuePosition::VECCALC> srcOffsetQueue;
    TBuf<TPosition::VECCALC> clearTensorBuff;
    TBuf<TPosition::VECCALC> syncTensorBuff;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> workQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<int32_t> syncGM;

    LocalTensor<float> srcIndexTensorW;
    LocalTensor<float> srcIndexTensorH;
    LocalTensor<float> srcIndexTensorD;
    LocalTensor<int32_t> srcOffsetTensor;
    LocalTensor<uint32_t> gatherOffsetTensor;
    LocalTensor<float> cacheTensor;

    int64_t blockIdx = 0;
    bool isExact = false;
    int64_t batches = 0;
    int64_t inputShapes[3] = {0};
    int64_t outputShapes[3] = {0};
    float scales[3] = {ZERO_FLOAT};

    int64_t slideSizeW = 0;
    int64_t tensorSizeW = 0;
    int64_t tensorSizeH = 0;
    int64_t tensorSizeD = 0;

    int64_t slideNumH = 0;
    int64_t slideNumD = 0;
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    int64_t tailStartSlideNum = 0;
    int64_t groupCoreNum = 0;
    int64_t inputRow = 0;
    int64_t tailAvergingRow = 0;
    int64_t needCoreNum = 0;

    int64_t lastStartW = -1;
    int64_t startW = 0;
    int64_t endW = 0;
    int64_t dataCount = 0;
    int64_t srcStartW = 0;
    int64_t srcEndW = 0;
    int64_t srcDataCount = 0;
    int64_t srcDataLength = 0;
    int64_t batchNum = 0;
    uint16_t srcBlockLen = 0;
    uint16_t srcStride = 0;

    int64_t indexH = 0;
    int64_t srcIndexH = 0;
    int64_t heightCount = 0;

    int64_t indexD = 0;
    int64_t srcIndexD = 0;
    int64_t depthCount = 0;

    int64_t blockSize = 8;
    int64_t totalNum = 0;
};

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::Init(
    GM_ADDR x, GM_ADDR y, bool isNearestExact, GM_ADDR workspace, const UpsampleNearestExact3dTilingData *tilingData)
{
    blockIdx = GetBlockIdx();

    isExact = isNearestExact;
    ParseTilingData(tilingData);
    totalNum = outputShapes[H_INDEX] * outputShapes[W_INDEX] * outputShapes[D_INDEX] * batches;

    pipe.InitBuffer(srcIndexQueueW, slideSizeW * sizeof(float));
    pipe.InitBuffer(srcIndexQueueH, CeilA2B(tensorSizeH * sizeof(float), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(srcIndexQueueD, CeilA2B(tensorSizeD * sizeof(float), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(srcOffsetQueue, slideSizeW * sizeof(int32_t));
    pipe.InitBuffer(inQueue, BUFFER_NUM, CeilA2B(tensorSizeW * sizeof(T), BYTE_BLOCK) * BYTE_BLOCK);
    pipe.InitBuffer(outQueue, BUFFER_NUM, slideSizeW * sizeof(T));
    pipe.InitBuffer(workQueue, BUFFER_NUM, DEFAULT_SYNC_UB_SIZE * sizeof(int32_t));
    pipe.InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));
    pipe.InitBuffer(syncTensorBuff, DEFAULT_SYNC_UB_SIZE * sizeof(int32_t));
    inTensorsGM.SetGlobalBuffer((__gm__ T *)x);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)y);
    syncGM.SetGlobalBuffer((__gm__ int32_t *)workspace, DEFAULT_SYNC_UB_SIZE * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::Process()
{
    if (blockIdx >= needCoreNum) {
        return;
    }
    ClearGM();
    LocalTensor<int32_t> syncUB = syncTensorBuff.Get<int32_t>();
    Duplicate(syncUB, (int32_t)0, DEFAULT_SYNC_UB_SIZE);
    event_t eventIdVToMTE3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopy(syncGM[0], syncUB, DEFAULT_SYNC_UB_SIZE);
    LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();
    SyncAll(syncGM, workLocal, needCoreNum);
    workQueue.FreeTensor(workLocal);

    srcIndexTensorW = srcIndexQueueW.AllocTensor<float>();
    srcIndexTensorH = srcIndexQueueH.AllocTensor<float>();
    srcIndexTensorD = srcIndexQueueD.AllocTensor<float>();
    srcOffsetTensor = srcOffsetQueue.AllocTensor<int32_t>();
    lastStartW = -1;

    int64_t slideStart = blockIdx * eachCoreSlideNum;
    int64_t slideEnd = slideStart + eachCoreSlideNum;
    // 计算批量分组的数据
    if (slideStart < slideEnd) {
        for (int64_t slideIndex = slideStart; slideIndex < slideEnd; slideIndex++) {
            GatherData(slideIndex, 0, inputRow);
        }
    }

    int64_t groupIndex = blockIdx / groupCoreNum;
    if (groupIndex < remainder) {
        // 处理尾块部分数据
        int64_t slideIndex = tailStartSlideNum + groupIndex;
        int64_t blockIdxInGroup = blockIdx % groupCoreNum;
        int64_t tailRowStart = blockIdxInGroup * tailAvergingRow;
        int64_t tailRowEnd = Min(tailRowStart + tailAvergingRow, inputRow);
        GatherData(slideIndex, tailRowStart, tailRowEnd);
    }

    srcOffsetQueue.FreeTensor(srcOffsetTensor);
    srcIndexQueueD.FreeTensor(srcIndexTensorD);
    srcIndexQueueH.FreeTensor(srcIndexTensorH);
    srcIndexQueueW.FreeTensor(srcIndexTensorW);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::ClearGM()
{
    // 清理GM
    int64_t totalBlockNum = (totalNum + blockSize - 1) / blockSize;
    int64_t preCoreBlockCnt = totalBlockNum / needCoreNum;
    int64_t tailBlockCnt = totalBlockNum % needCoreNum;
    int64_t tailDataCnt = totalNum - preCoreBlockCnt * needCoreNum * blockSize;
    int64_t realNeedCore = 1;
    if (preCoreBlockCnt > 0) {
        realNeedCore = needCoreNum;
    }

    int64_t preCoreDataCnt = preCoreBlockCnt * blockSize;
    int64_t loopCnt = preCoreDataCnt / DEFAULT_CLEAR_UB_SIZE;
    int64_t tailCnt = preCoreDataCnt % DEFAULT_CLEAR_UB_SIZE;
    int64_t offset = (blockIdx % needCoreNum) * preCoreDataCnt;

    LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
    Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);
    event_t eventIdVToMTE3_2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3_2);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3_2);
    for (int i = 0; i < loopCnt; i++) {
        DataCopy(outTensorsGM[offset], clearUb, DEFAULT_CLEAR_UB_SIZE);
        offset += DEFAULT_CLEAR_UB_SIZE;
    }
    if (tailCnt > 0) {
        tailCnt = (tailCnt + blockSize - 1) / blockSize * blockSize;
        int64_t offsetTemp = (blockIdx + 1) * preCoreDataCnt - tailCnt;
        offset = offsetTemp > 0 ? offsetTemp : 0;
        DataCopy(outTensorsGM[offset], clearUb, tailCnt);
    }
    if ((tailBlockCnt > 0) && (blockIdx == 0)) {
        tailCnt = (tailDataCnt + blockSize - 1) / blockSize * blockSize;
        offset = totalNum - tailCnt > 0 ? totalNum - tailCnt : 0;
        DataCopy(outTensorsGM[offset], clearUb, tailCnt);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::GatherData(int64_t slideIndex, int64_t rowStart, int64_t rowEnd)
{
    GetRangeH(slideIndex);
    GetRangeD(slideIndex);
    if (heightCount == 0 || depthCount == 0) {
        return;
    }

    GetRangeW(slideIndex);
    int64_t j = 0;
    while (startW < endW) {
        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
            srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(j));
            j++;
        }
        int64_t inputOffsetInBatch =
            srcIndexD * inputShapes[H_INDEX] * inputShapes[W_INDEX] + srcIndexH * inputShapes[W_INDEX] + srcStartW;
        int64_t outputOffsetInBatch =
            indexD * outputShapes[H_INDEX] * outputShapes[W_INDEX] + indexH * outputShapes[W_INDEX] + startW;
        for (int64_t batchIndex = rowStart; batchIndex < rowEnd; batchIndex += batchNum) {
            int64_t inputOffset =
                batchIndex * inputShapes[D_INDEX] * inputShapes[H_INDEX] * inputShapes[W_INDEX] + inputOffsetInBatch;
            int64_t outputOffset = batchIndex * outputShapes[D_INDEX] * outputShapes[H_INDEX] * outputShapes[W_INDEX] +
                                   outputOffsetInBatch;
            uint16_t blockCount = Min(batchNum, rowEnd - batchIndex);
            DataCopyParams repeatParams{blockCount, srcBlockLen, 0, 0};
            CopyIn(inputOffset, repeatParams);
            ComputeAndCopyOut(dataCount, srcDataLength, blockCount, outputOffset);
        }
        startW += dataCount;
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::CopyIn(int64_t inputOffset, DataCopyParams repeatParams)
{
    LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);
    DataCopy(srcLocal, inTensorsGM[inputOffset], repeatParams);
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    inQueue.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::ComputeAndCopyOut(
    uint32_t dataCount, uint32_t srcDataLength, uint32_t blockCount, int64_t outputOffset)
{
    LocalTensor<T> srcLocal = inQueue.DeQue<T>();
    for (int64_t i = 0; i < blockCount; i++) {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Duplicate(dstLocal, (T)0, slideSizeW);
        Gather(dstLocal, srcLocal, gatherOffsetTensor, static_cast<uint32_t>(i * srcDataLength), dataCount);
        outQueue.EnQue(dstLocal);
        dstLocal = outQueue.DeQue<T>();
        for (int64_t j = 0; j < depthCount; j++) {
            int64_t offset = outputOffset + j * outputShapes[H_INDEX] * outputShapes[W_INDEX];
            for (int64_t k = 0; k < heightCount; k++) {
                CopyOutProcess(offset + k * outputShapes[W_INDEX], dstLocal);
            }
        }
        outQueue.FreeTensor(dstLocal);
        outputOffset += outputShapes[D_INDEX] * outputShapes[H_INDEX] * outputShapes[W_INDEX];
    }
    inQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::CopyOutProcess(int64_t offsetTemp, LocalTensor<T> dstLocal)
{
    int64_t copyOutBlock = CeilA2B(dataCount, blockSize);
    int64_t copyOutCnt = copyOutBlock * blockSize;
    int64_t offset = offsetTemp;
    if ((offset + copyOutCnt) > totalNum) {
        // 如果拷贝的数据块超过1个整块，把前面n-1个整块先拷出
        if (copyOutBlock > 1) {
            CopyOut(offset, dstLocal, (copyOutBlock - 1) * blockSize);
            offset += (copyOutBlock - 1) * blockSize;
        }
        // 处理最后一个块
        LocalTensor<T> tailTensor = clearTensorBuff.Get<T>();
        Duplicate(tailTensor, (T)0, blockSize);
        int64_t copyOutTailCnt = dataCount - (copyOutBlock - 1) * blockSize;
        for (int64_t m = 0; m < copyOutTailCnt; m++) {
            if (totalNum < blockSize) {
                tailTensor.SetValue(m + offset, dstLocal.GetValue(m + (copyOutBlock - 1) * blockSize));
            } else {
                tailTensor.SetValue(
                    m + offset + blockSize - totalNum, dstLocal.GetValue(m + (copyOutBlock - 1) * blockSize));
            }
        }
        offset = totalNum - blockSize > 0 ? totalNum - blockSize : 0;
        LocalTensor<T> outTensor = syncTensorBuff.Get<T>();
        Adds(outTensor, tailTensor, (T)0, blockSize);
        CopyOut(offset, outTensor, blockSize);
    } else {
        CopyOut(offset, dstLocal, copyOutCnt);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::CopyOut(
    int64_t offsetTemp, LocalTensor<T> dstLocal, int64_t copyOutCnt)
{
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);
    SetAtomicAdd<T>();
    DataCopy(outTensorsGM[offsetTemp], dstLocal, copyOutCnt);
    SetAtomicNone();
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::GetRangeW(int64_t slideIndex)
{
    startW = (slideIndex / (slideNumH * slideNumD)) * slideSizeW;
    if (lastStartW != startW) {
        lastStartW = startW;
        endW = Min(startW + slideSizeW, outputShapes[W_INDEX]);
        dataCount = endW - startW;
        CalculateSrcIndexTensor(startW, dataCount, W_INDEX, srcIndexTensorW);
        srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(0));
        srcEndW = static_cast<int64_t>(srcIndexTensorW.GetValue(dataCount - 1)) + 1;
        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
            dataCount = 1;
            srcDataCount = 1;
        } else {
            srcDataCount = srcEndW - srcStartW;
        }
        CalculateGatherOffsetW();
        srcDataLength = CeilA2B(srcDataCount * sizeof(T), BYTE_BLOCK) * BYTE_BLOCK;
        batchNum = 1;
        srcBlockLen = CeilA2B(srcDataCount * sizeof(T), BYTE_BLOCK);
    }
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::GetRangeH(int64_t slideIndex)
{
    if (scales[H_INDEX] >= ONE_FLOAT) {
        indexH = slideIndex % slideNumH;
        CalculateSrcIndexTensor(indexH, 1, H_INDEX, srcIndexTensorH);
        srcIndexH = static_cast<int64_t>(srcIndexTensorH.GetValue(0));
        heightCount = 1;
        return;
    }
    srcIndexH = slideIndex % slideNumH;
    indexH = Max(static_cast<int64_t>((float)srcIndexH / scales[H_INDEX] - 2), static_cast<int64_t>(0));
    int64_t length = Min(tensorSizeH, outputShapes[H_INDEX] - indexH);
    CalculateSrcIndexTensor(indexH, length, H_INDEX, srcIndexTensorH);
    heightCount = 0;
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorH.GetValue(j));
        if (srcIndex == srcIndexH) {
            heightCount = 1;
            indexH += j;
            break;
        }
    }
    if (heightCount == 0) {
        return;
    }

    int64_t lastIndexH = Max(static_cast<int64_t>((float)(srcIndexH + 1) / scales[H_INDEX] - 2), indexH);
    lastIndexH = Min(lastIndexH, outputShapes[H_INDEX] - 1);
    length = Min(tensorSizeH, outputShapes[H_INDEX] - lastIndexH);
    CalculateSrcIndexTensor(lastIndexH, length, H_INDEX, srcIndexTensorH);
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorH.GetValue(j));
        if (srcIndex == srcIndexH) {
            lastIndexH++;
        }
    }
    heightCount = lastIndexH - indexH;
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::GetRangeD(int64_t slideIndex)
{
    if (scales[D_INDEX] >= ONE_FLOAT) {
        indexD = (slideIndex % (slideNumD * slideNumH)) / slideNumH;
        CalculateSrcIndexTensor(indexD, 1, D_INDEX, srcIndexTensorD);
        srcIndexD = static_cast<int64_t>(srcIndexTensorD.GetValue(0));
        depthCount = 1;
        return;
    }
    srcIndexD = (slideIndex % (slideNumD * slideNumH)) / slideNumH;
    indexD = Max(static_cast<int64_t>((float)srcIndexD / scales[D_INDEX] - 2), static_cast<int64_t>(0));
    int64_t length = Min(tensorSizeD, outputShapes[D_INDEX] - indexD);
    CalculateSrcIndexTensor(indexD, length, D_INDEX, srcIndexTensorD);
    depthCount = 0;
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorD.GetValue(j));
        if (srcIndex == srcIndexD) {
            depthCount = 1;
            indexD += j;
            break;
        }
    }
    if (depthCount == 0) {
        return;
    }

    int64_t lastIndexD = Max(static_cast<int64_t>((float)(srcIndexD + 1) / scales[D_INDEX] - 2), indexD);
    lastIndexD = Min(lastIndexD, outputShapes[D_INDEX] - 1);
    length = Min(tensorSizeD, outputShapes[D_INDEX] - lastIndexD);
    CalculateSrcIndexTensor(lastIndexD, length, D_INDEX, srcIndexTensorD);
    for (int64_t j = 0; j < length; j++) {
        int64_t srcIndex = static_cast<int64_t>(srcIndexTensorD.GetValue(j));
        if (srcIndex == srcIndexD) {
            lastIndexD++;
        }
    }
    depthCount = lastIndexD - indexD;
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::CalculateSrcIndexTensor(
    int64_t index, int64_t length, int8_t direction, LocalTensor<float> srcIndexTensor)
{
    ArithProgression(srcIndexTensor, static_cast<float>(index), static_cast<float>(1), length);
    PipeBarrier<PIPE_V>();
    if (isExact) {
        Adds(srcIndexTensor, srcIndexTensor, static_cast<float>(0.5), length);
        PipeBarrier<PIPE_V>();
    }
    Muls(srcIndexTensor, srcIndexTensor, scales[direction], length);
    PipeBarrier<PIPE_V>();
    Floor(srcIndexTensor, srcIndexTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(srcIndexTensor, srcIndexTensor, static_cast<float>(inputShapes[direction] - 1), length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::CalculateGatherOffsetW()
{
    Cast(srcOffsetTensor, srcIndexTensorW, RoundMode::CAST_FLOOR, dataCount);
    PipeBarrier<PIPE_V>();
    Adds(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(-srcStartW), dataCount);
    PipeBarrier<PIPE_V>();
    Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(sizeof(T)), dataCount);
    PipeBarrier<PIPE_V>();
    gatherOffsetTensor = srcOffsetTensor.ReinterpretCast<uint32_t>();
}

template <typename T>
__aicore__ inline void UpsampleNearestExact3dND310p<T>::ParseTilingData(
    const UpsampleNearestExact3dTilingData *tilingData)
{
    batches = tilingData->batches;
    for (int8_t i = 0; i < 3; i++) {
        outputShapes[i] = tilingData->outputShapes[i];
        inputShapes[i] = tilingData->inputShapes[i];
    }

    scales[W_INDEX] = tilingData->scaleW;
    scales[H_INDEX] = tilingData->scaleH;
    scales[D_INDEX] = tilingData->scaleD;
    slideSizeW = tilingData->slideSizeW;
    tensorSizeW = tilingData->tensorSizeW;
    tensorSizeH = tilingData->tensorSizeH;
    tensorSizeD = tilingData->tensorSizeD;

    slideNumH = tilingData->slideNumH;
    slideNumD = tilingData->slideNumD;
    eachCoreSlideNum = tilingData->eachCoreSlideNum;
    remainder = tilingData->remainder;
    tailStartSlideNum = tilingData->tailStartSlideNum;
    groupCoreNum = tilingData->groupCoreNum;
    inputRow = tilingData->inputRow;
    tailAvergingRow = tilingData->tailAvergingRow;
    needCoreNum = tilingData->needCoreNum;

    blockSize = 32 / sizeof(T);
}
}  // namespace UpsampleNearestExact3d
#endif