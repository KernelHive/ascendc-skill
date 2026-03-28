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
 * \file upsample_nearest.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST_H
#define UPSAMPLE_NEAREST_H

#include <type_traits>
#include "kernel_operator.h"

namespace UpsampleNearest {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t NO_BUFFER_NUM = 1;
constexpr int64_t EACH_SLICE_HANDLE_MIN_NUM = 16;

constexpr int8_t W_DIRECTION = 0;
constexpr int8_t H_DIRECTION = 1;

const int64_t DEFAULT_UB_MAX_DATA_COUNT = 2048;
const int64_t DEFAULT_UB_MAX_COPY_SIZE = 64 * 1024;  // 64kb

template <typename T, int32_t MODE>
class UpsampleNearestND {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearestND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleNearestTilingData *tilingData);
    __aicore__ inline void Process();

private:
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

    __aicore__ inline void ParseTilingData(const UpsampleNearestTilingData *tilingData);

    __aicore__ inline void CalculateIdxTensor(int64_t index, int64_t length, int8_t direction);
    __aicore__ inline void NearestComputeBase();
    __aicore__ inline void NearestComputeSmallCW();
    __aicore__ inline void NearestComputeSmallNCH();
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyInBatch(int64_t indexInput, int64_t calCount, uint16_t blockCnt);
    __aicore__ inline void CopyOutBatch(int64_t indexOutput, int64_t calCount);
    __aicore__ inline void CopyOutBase(LocalTensor<T> dstDataLocal, int64_t indexOutput, int64_t calCount);
    __aicore__ inline void ProcessOutput(
        int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);
    __aicore__ inline void ProcessOutputBase(
        int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallC(
        int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);
    __aicore__ inline void ProcessOutputSmallCW(
        int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH);

private:
    TBuf<QuePosition::VECCALC> centerQueueW;
    TBuf<QuePosition::VECCALC> xIntQueueW;

    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> xIntQueueH;
    TBuf<QuePosition::VECCALC> gatherQueue;
    TBuf<QuePosition::VECCALC> offsetQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    int64_t blockIdx = 0;
    int64_t slideSize = 512;
    float scaleW;
    float scaleH;
    int64_t dataType;

    int64_t tailColStart;
    int64_t tailColEnd;
    int64_t tailRowStart;
    int64_t tailRowEnd;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int32_t blockSize = 8;
    int64_t inputBatchSize;
    int64_t outputBatchSize;
    bool exactMode;

    int64_t maxCopyCount;
};

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleNearestTilingData *tilingData)
{
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    pipe.InitBuffer(centerQueueW, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));  // 8k
    pipe.InitBuffer(xIntQueueW, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));    // 8k

    pipe.InitBuffer(centerQueueH, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));  // 8k
    pipe.InitBuffer(xIntQueueH, DEFAULT_UB_MAX_DATA_COUNT * sizeof(float));    // 8k

    maxCopyCount = DEFAULT_UB_MAX_COPY_SIZE / sizeof(T);
    if (MODE == 1) {
        maxCopyCount = maxCopyCount / 2;
        pipe.InitBuffer(dataQueue, NO_BUFFER_NUM, maxCopyCount * sizeof(T));  // 32k
        pipe.InitBuffer(outQueue, NO_BUFFER_NUM, maxCopyCount * sizeof(T));   // 32k
        pipe.InitBuffer(gatherQueue, maxCopyCount * sizeof(uint32_t));        // 32k
    } else if (MODE == 3) {
        maxCopyCount = maxCopyCount / 4;                                   // 4k
        pipe.InitBuffer(dataQueue, BUFFER_NUM, maxCopyCount * sizeof(T));  // 32k
        pipe.InitBuffer(outQueue, BUFFER_NUM, maxCopyCount * sizeof(T));   // 32k
        pipe.InitBuffer(gatherQueue, maxCopyCount * sizeof(uint32_t));     // 16k
        pipe.InitBuffer(offsetQueue, maxCopyCount * sizeof(uint32_t));     // 16k
    } else {
        pipe.InitBuffer(dataQueue, BUFFER_NUM, maxCopyCount * sizeof(T));  // 128k
    }

    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::Process()
{
    if (tailColStart >= tailColEnd) {
        return;
    }

    if (MODE == 1) {
        NearestComputeSmallCW();
    }
    if (MODE == 3) {
        NearestComputeSmallNCH();
    } else {
        NearestComputeBase();
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeSmallNCH()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;

    for (int64_t indexH = startIdxH; indexH < endIdxH; indexH++) {
        CalculateIdxTensor(indexH, 1, H_DIRECTION);
        LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(0));
        for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
            int64_t lengthW = Min(slideSize, endIdxW - indexW);
            CalculateIdxTensor(indexW, lengthW, W_DIRECTION);
            LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
            int64_t srcStartW = static_cast<int64_t>(srcTensorW.GetValue(0));

            LocalTensor<int32_t> srcOffsetTensor = offsetQueue.Get<int32_t>();
            Cast(srcOffsetTensor, srcTensorW, RoundMode::CAST_FLOOR, lengthW);
            PipeBarrier<PIPE_V>();
            Adds(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(-srcStartW), lengthW);
            PipeBarrier<PIPE_V>();
            Muls(srcOffsetTensor, srcOffsetTensor, static_cast<int32_t>(sizeof(T)), lengthW);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> gatherOffsetTensor = srcOffsetTensor.ReinterpretCast<uint32_t>();

            for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                for (int64_t channelIdx = 0; channelIdx < inputC; channelIdx++) {
                    int64_t indexInput =
                        batchIdx * inputC * inputBatchSize + channelIdx * inputBatchSize + srcH * inputW + srcStartW;
                    int64_t indexOutput =
                        batchIdx * inputC * outputBatchSize + channelIdx * outputBatchSize + indexH * outputW + indexW;

                    CopyIn(indexInput, lengthW);

                    LocalTensor<T> srcLocal = dataQueue.DeQue<T>();
                    LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
                    Gather(dstDataLocal, srcLocal, gatherOffsetTensor, static_cast<uint32_t>(0), lengthW);
                    outQueue.EnQue(dstDataLocal);
                    dataQueue.FreeTensor(srcLocal);

                    CopyOutBatch(indexOutput, lengthW);
                }
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeSmallCW()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;

    for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
        int64_t lengthW = Min(slideSize, endIdxW - indexW);
        CalculateIdxTensor(indexW, lengthW, W_DIRECTION);

        LocalTensor<int32_t> gatherTensor = gatherQueue.Get<int32_t>();
        LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
        int64_t minW = static_cast<int64_t>(srcTensorW.GetValue(0));
        for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
            int32_t srcW = static_cast<int32_t>(srcTensorW.GetValue(offsetW));
            int32_t inputOffset = (srcW - minW) * inputC;
            if (inputC % blockSize == 0) {
                ArithProgression(gatherTensor[offsetW * inputC], inputOffset, (int32_t)1, inputC);
            } else {
                for (int64_t i = 0; i < inputC; i++) {
                    gatherTensor.SetValue(offsetW * inputC + i, inputOffset + i);
                }
            }
        }
        int64_t maxDataOutEachRow = lengthW * inputC;
        Muls(gatherTensor, gatherTensor, (int32_t)sizeof(T), maxDataOutEachRow);

        for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
            int64_t lengthH = Min(slideSize, endIdxH - indexH);
            CalculateIdxTensor(indexH, lengthH, H_DIRECTION);
            for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                ProcessOutput(batchIdx, indexW, indexH, lengthW, lengthH);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::NearestComputeBase()
{
    int64_t startIdxW = tailColStart;
    int64_t startIdxH = tailRowStart;
    int64_t endIdxW = tailColEnd;
    int64_t endIdxH = tailRowEnd;

    for (int64_t indexH = startIdxH; indexH < endIdxH; indexH += slideSize) {
        int64_t lengthH = Min(slideSize, endIdxH - indexH);
        CalculateIdxTensor(indexH, lengthH, H_DIRECTION);
        for (int64_t indexW = startIdxW; indexW < endIdxW; indexW += slideSize) {
            int64_t lengthW = Min(slideSize, endIdxW - indexW);
            CalculateIdxTensor(indexW, lengthW, W_DIRECTION);
            for (int64_t batchIdx = 0; batchIdx < inputN; batchIdx++) {
                ProcessOutput(batchIdx, indexW, indexH, lengthW, lengthH);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CalculateIdxTensor(int64_t index, int64_t length, int8_t direction)
{
    length = Max(length, EACH_SLICE_HANDLE_MIN_NUM);
    float scale = scaleW;
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> xIntTensor = xIntQueueW.Get<float>();
    float inputSizeBound = static_cast<float>(inputW) - (float)1.0;
    if (direction == H_DIRECTION) {
        scale = scaleH;
        centerTensor = centerQueueH.Get<float>();
        xIntTensor = xIntQueueH.Get<float>();
        inputSizeBound = static_cast<float>(inputH) - (float)1.0;
    }

    ArithProgression(centerTensor, static_cast<float>(index), (float)1.0, length);
    PipeBarrier<PIPE_V>();

    // 计算center下标
    if (exactMode) {
        // exact模式
        Adds(centerTensor, centerTensor, (float)0.5, length);
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        // 普通模式
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    }

    Floor(xIntTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();
    Mins(xIntTensor, xIntTensor, inputSizeBound, length);
    PipeBarrier<PIPE_V>();
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutput(
    int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH)
{
    if (MODE == 1) {
        ProcessOutputSmallCW(batchIdx, indexW, indexH, lengthW, lengthH);
    } else if (MODE == 2) {
        ProcessOutputSmallC(batchIdx, indexW, indexH, lengthW, lengthH);
    } else {
        ProcessOutputBase(batchIdx, indexW, indexH, lengthW, lengthH);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputBase(
    int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();
    int32_t loopCnt = (inputC + maxCopyCount - 1) / maxCopyCount;

    for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
        int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;
        int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
        for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
            int64_t srcW = static_cast<int64_t>(srcTensorW.GetValue(offsetW));

            int64_t inputOffset = inputOffsetBase + srcW * inputC;
            int64_t outputOffset = outputOffsetBase + offsetW * inputC;
            for (int32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
                int64_t startIdx = loopIdx * maxCopyCount;
                int64_t calCount = Min(maxCopyCount, inputC - startIdx);

                int64_t indexInput = inputOffset + startIdx;
                CopyIn(indexInput, calCount);
                int64_t indexOutput = outputOffset + startIdx;
                CopyOut(indexOutput, calCount);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputSmallC(
    int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();

    int64_t minW = static_cast<int64_t>(srcTensorW.GetValue(0));
    int64_t maxW = static_cast<int64_t>(srcTensorW.GetValue(lengthW - 1));

    int64_t inputCBlock = (inputC + blockSize - 1) / blockSize * blockSize;
    int64_t maxDataCopyW = Min(maxCopyCount / inputCBlock, maxW - minW + 1);

    LocalTensor<T> srcDataLocal;
    for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
        int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;

        int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;
        int64_t indexInput = inputOffsetBase + minW * inputC;
        int64_t originW = minW + maxDataCopyW;
        CopyInBatch(indexInput, inputC, maxDataCopyW);
        srcDataLocal = dataQueue.DeQue<T>();

        int64_t copyBlockCount = maxDataCopyW;
        for (int64_t offsetW = 0; offsetW < lengthW; offsetW++) {
            int64_t srcW = static_cast<int64_t>(srcTensorW.GetValue(offsetW));
            int64_t indexOutput = outputOffsetBase + offsetW * inputC;
            if (srcW >= originW) {
                dataQueue.FreeTensor(srcDataLocal);
                indexInput = inputOffsetBase + srcW * inputC;
                if ((copyBlockCount + srcW) > maxW) {
                    copyBlockCount = maxW - srcW + 1;
                }
                CopyInBatch(indexInput, inputC, copyBlockCount);
                originW = srcW + copyBlockCount;
                srcDataLocal = dataQueue.DeQue<T>();
            }
            int64_t indexInputOffset = (srcW + copyBlockCount - originW) * inputCBlock;
            CopyOutBase(srcDataLocal[indexInputOffset], indexOutput, inputC);
        }
        dataQueue.FreeTensor(srcDataLocal);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ProcessOutputSmallCW(
    int64_t batchIdx, int64_t indexW, int64_t indexH, int64_t lengthW, int64_t lengthH)
{
    LocalTensor<float> srcTensorW = xIntQueueW.Get<float>();
    LocalTensor<float> srcTensorH = xIntQueueH.Get<float>();

    int64_t minW = static_cast<int64_t>(srcTensorW.GetValue(0));
    int64_t maxW = static_cast<int64_t>(srcTensorW.GetValue(lengthW - 1));

    int64_t maxDataCountEachRow = (maxW - minW + 1) * inputC;
    int64_t maxDataOutEachRow = lengthW * inputC;
    LocalTensor<uint32_t> gatherTensor = gatherQueue.Get<uint32_t>();
    for (int64_t offsetH = 0; offsetH < lengthH; offsetH++) {
        int64_t srcH = static_cast<int64_t>(srcTensorH.GetValue(offsetH));
        int64_t inputOffsetBase = (inputBatchSize * batchIdx + (srcH * inputW)) * inputC;
        int64_t outputOffsetBase = (outputBatchSize * batchIdx + ((indexH + offsetH) * outputW + indexW)) * inputC;

        int64_t indexInput = inputOffsetBase + minW * inputC;
        CopyIn(indexInput, maxDataCountEachRow);

        LocalTensor<T> srcDataLocal = dataQueue.DeQue<T>();
        LocalTensor<T> dstDataLocal = outQueue.AllocTensor<T>();
        Gather(dstDataLocal, srcDataLocal, gatherTensor, (uint32_t)0, maxDataOutEachRow);
        outQueue.EnQue(dstDataLocal);
        dataQueue.FreeTensor(srcDataLocal);
        CopyOutBatch(outputOffsetBase, maxDataOutEachRow);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyIn(int64_t indexInput, int64_t calCount)
{
    LocalTensor<T> srcDataLocal = dataQueue.AllocTensor<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcDataLocal, inTensorsGM[indexInput], copyParams, padParams);
    }
    dataQueue.EnQue(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOut(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = dataQueue.DeQue<T>();

    CopyOutBase(dstDataLocal, indexOutput, calCount);

    dataQueue.FreeTensor(dstDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOutBase(
    LocalTensor<T> dstDataLocal, int64_t indexOutput, int64_t calCount)
{
    event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[indexOutput], dstDataLocal, copyParams);
    }
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyInBatch(int64_t indexInput, int64_t calCount, uint16_t blockCnt)
{
    LocalTensor<T> srcDataLocal = dataQueue.AllocTensor<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(srcDataLocal, inTensorsGM[indexInput], calCount * blockCnt);
    } else {
        DataCopyExtParams copyParams{blockCnt, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(srcDataLocal, inTensorsGM[indexInput], copyParams, padParams);
    }
    dataQueue.EnQue(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::CopyOutBatch(int64_t indexOutput, int64_t calCount)
{
    LocalTensor<T> dstDataLocal = outQueue.DeQue<T>();
    if ((calCount % blockSize) == 0) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[indexOutput], dstDataLocal, copyParams);
    }
    outQueue.FreeTensor(dstDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND<T, MODE>::ParseTilingData(const UpsampleNearestTilingData *tilingData)
{
    slideSize = DEFAULT_UB_MAX_DATA_COUNT;
    dataType = tilingData->dataType;
    scaleW = tilingData->scaleW;
    scaleH = tilingData->scaleH;
    exactMode = tilingData->exactMode;

    inputN = tilingData->inputShapes[0];
    inputH = tilingData->inputShapes[1];
    inputW = tilingData->inputShapes[2];
    inputC = tilingData->inputShapes[3];
    outputH = tilingData->outputShapes[1];
    outputW = tilingData->outputShapes[2];
    if (MODE == 3) {
        inputN = tilingData->inputShapes[0];
        inputC = tilingData->inputShapes[1];
        inputH = tilingData->inputShapes[2];
        inputW = tilingData->inputShapes[3];
        outputH = tilingData->outputShapes[2];
        outputW = tilingData->outputShapes[3];
    }

    inputBatchSize = inputH * inputW;
    outputBatchSize = outputH * outputW;

    tailColStart = tilingData->tailColStartList[blockIdx];
    tailColEnd = tilingData->tailColEndList[blockIdx];
    tailRowStart = tilingData->tailRowStartList[blockIdx];
    tailRowEnd = tilingData->tailRowEndList[blockIdx];

    blockSize = 32 / sizeof(T);
}
}  // namespace UpsampleNearest

#endif  // UPSAMPLE_NEAREST
