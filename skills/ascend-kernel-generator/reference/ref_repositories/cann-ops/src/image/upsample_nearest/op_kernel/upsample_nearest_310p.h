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
 * \file upsample_nearest_310p.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST_310P_H
#define UPSAMPLE_NEAREST_310P_H

#include <type_traits>
#include "kernel_operator.h"

namespace UpsampleNearest {
using namespace AscendC;

constexpr uint8_t BUFFER_NUM = 2;

constexpr uint16_t DEFAULT_UB_MAX_DATA_COUNT = 512;
constexpr uint32_t DEFAULT_CLEAR_UB_SIZE = 10 * 1024;

constexpr uint8_t H_DIRECTION = 0;
constexpr uint8_t W_DIRECTION = 1;
constexpr uint32_t COPY_BLOCK = 32;

template <typename T, int32_t MODE>
class UpsampleNearestND310p {
public:
    TPipe pipe;

    __aicore__ inline UpsampleNearestND310p(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const UpsampleNearestTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const UpsampleNearestTilingData *tilingData);
    __aicore__ inline void ComputeNearest();
    __aicore__ inline void ClearGM();
    __aicore__ inline void CalcTensors(int64_t nIdx, int64_t indexH, int64_t indexW, int64_t lengthH, int64_t lengthW);
    __aicore__ inline void CalcTensorsC(int64_t indexInput, int64_t indexOutput, int64_t calcCount);
    __aicore__ inline void CalcIdxTensor(int64_t dstStartIndex, int64_t length, uint8_t direction);
    __aicore__ inline void CopyIn(int64_t indexInput, int64_t calcCount);
    __aicore__ inline void CopyOut(int64_t indexOutput, int64_t calcCount, int64_t copyLength);
    template <typename T1>
    __aicore__ inline uint32_t Ceil(T1 x)
    {
        int32_t floor_v = int32_t(x);
        if (x == floor_v) {
            return floor_v;
        }
        return floor_v + 1;
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

private:
    TBuf<QuePosition::VECCALC> dstToSrcQueueH;
    TBuf<QuePosition::VECCALC> dstToSrcQueueW;
    TBuf<QuePosition::VECCALC> centerQueueH;
    TBuf<QuePosition::VECCALC> centerQueueW;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TQue<QuePosition::VECIN, 1> syncWorkQueue;
    TQue<QuePosition::VECIN, 1> clearWorkspaceQueue;

    TBuf<TPosition::VECCALC> clearTensorBuff;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<int32_t> syncTensorsGM;

    LocalTensor<T> clearTensor;

    int64_t blockIdx = 0;
    int64_t slideSize = DEFAULT_UB_MAX_DATA_COUNT;

    float scaleH = 0;
    float scaleW = 0;

    int64_t inputN = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t inputC = 0;

    int64_t outputH = 0;
    int64_t outputW = 0;

    bool exactMode = true;

    int64_t dataTypeSize = 4;
    int64_t blockSize = 8;

    // 计算batch的偏移
    int64_t outputNBatchOffset = 0;
    int64_t outputHBatchOffset = 0;
    int64_t inputNBatchOffset = 0;
    int64_t inputHBatchOffset = 0;

    // c方向上的切割块数量
    int64_t cTilingLoopCnt = 0;
    int64_t cTailCnt = 0;

    uint32_t dataType = 0;

    int64_t needCoreNum = 0;

    int64_t tailRowStart = 0;
    int64_t tailRowEnd = 0;
    int64_t tailColStart = 0;
    int64_t tailColEnd = 0;

    int64_t maxDataCount = DEFAULT_UB_MAX_DATA_COUNT;
};

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleNearestTilingData *tilingData)
{
    // 获取当前核的索引
    blockIdx = GetBlockIdx();

    // 加载传入的参数
    ParseTilingData(tilingData);

    // 进行内存分配，使用双buffer
    pipe.InitBuffer(inputQueue, BUFFER_NUM, maxDataCount * sizeof(float));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, maxDataCount * sizeof(float));
    pipe.InitBuffer(syncWorkQueue, 1, 8 * 32 * sizeof(int32_t));
    pipe.InitBuffer(clearWorkspaceQueue, 1, 512);

    pipe.InitBuffer(dstToSrcQueueH, maxDataCount * sizeof(float));
    pipe.InitBuffer(dstToSrcQueueW, maxDataCount * sizeof(float));
    pipe.InitBuffer(centerQueueH, maxDataCount * sizeof(float));
    pipe.InitBuffer(centerQueueW, maxDataCount * sizeof(float));

    pipe.InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));

    // 设置输入和输出存储
    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);

    syncTensorsGM.SetGlobalBuffer((__gm__ int32_t *)workspace, needCoreNum * 8 * 32);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::Process()
{
    ClearGM();

    LocalTensor<int32_t> syncLocalTensor = syncWorkQueue.AllocTensor<int32_t>();

    SyncAll(syncTensorsGM, syncLocalTensor, int32_t(needCoreNum));
    syncWorkQueue.FreeTensor(syncLocalTensor);

    ComputeNearest();
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::ClearGM()
{
    // clear gm 拆到各工作核上分开处理
    if (blockIdx >= needCoreNum) {
        return;
    }

    LocalTensor<T> clearUb = clearTensorBuff.Get<T>();

    Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

    // 计算数据元素个数
    int64_t totalNum = inputN * inputC * outputH * outputW;
    // 计算须正常处理的32byte的总块数
    int64_t totalBlockNum = totalNum / blockSize;
    // 处理完块数后的尾块
    int64_t tailCnt = totalNum % blockSize;
    // 需要额外多清理一个数据块的核数量
    int64_t needExtraBlockCoreCnt = totalBlockNum % needCoreNum;
    // 单个核至少须清空的数据块量
    int64_t perCoreBlockCnt = totalBlockNum / needCoreNum;
    // 单个核至少清空的元素数量，已为32byte的整数倍
    int64_t perCoreEleCnt = perCoreBlockCnt * blockSize;

    // clear内存的初始偏移
    int64_t offset = blockIdx * perCoreEleCnt;
    int64_t nextCoreOffset = offset + perCoreEleCnt;
    // 处理单个核平均分配的数据块
    for (int64_t clearOffset = offset; clearOffset < nextCoreOffset; clearOffset += DEFAULT_CLEAR_UB_SIZE) {
        int64_t clearLength = DEFAULT_CLEAR_UB_SIZE;
        if (clearOffset + DEFAULT_CLEAR_UB_SIZE >= nextCoreOffset) {
            clearLength = nextCoreOffset - clearOffset;
        }
        DataCopy(outTensorsGM[clearOffset], clearUb, clearLength);
    }

    // 处理剩余的数据块尾块数量
    if (blockIdx < needExtraBlockCoreCnt) {
        offset = needCoreNum * perCoreEleCnt + blockIdx * blockSize;
        DataCopy(outTensorsGM[offset], clearUb, blockSize);
    }

    // 剩余的元素由0核统一处理
    if (tailCnt > 0 && blockIdx == 0) {
        tailCnt = Ceil(float(tailCnt) / blockSize) * blockSize;
        offset = Max(int64_t(0), totalNum - tailCnt);
        DataCopy(outTensorsGM[offset], clearUb, tailCnt);
    }

    LocalTensor<int32_t> clearWorkspaceUb = clearTensorBuff.Get<int32_t>();
    DataCopy(syncTensorsGM[0], clearWorkspaceUb, needCoreNum * 8 * 32);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::ComputeNearest()
{
    for (int64_t indexH = tailRowStart; indexH < tailRowEnd; indexH += slideSize) {
        int64_t lengthH = Min(slideSize, tailRowEnd - indexH);
        CalcIdxTensor(int64_t(indexH), lengthH, H_DIRECTION);
        for (int64_t indexW = tailColStart; indexW < tailColEnd; indexW += slideSize) {
            int64_t lengthW = Min(slideSize, tailColEnd - indexW);
            CalcIdxTensor(int64_t(indexW), lengthW, W_DIRECTION);
            for (int64_t indexN = 0; indexN < inputN; ++indexN) {
                CalcTensors(indexN, indexH, indexW, lengthH, lengthW);
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::CalcTensors(
    int64_t nIdx, int64_t indexH, int64_t indexW, int64_t lengthH, int64_t lengthW)
{
    LocalTensor<float> dstToSrcTensorH = dstToSrcQueueH.Get<float>();
    LocalTensor<float> dstToSrcTensorW = dstToSrcQueueW.Get<float>();

    for (int64_t offsetH = 0; offsetH < lengthH; ++offsetH) {
        int64_t srcH = static_cast<int64_t>(dstToSrcTensorH.GetValue(offsetH));

        int64_t indexInputBase = nIdx * inputNBatchOffset + srcH * inputHBatchOffset;
        int64_t indexOutputBase = nIdx * outputNBatchOffset + (indexH + offsetH) * outputHBatchOffset + indexW * inputC;
        for (int64_t offsetW = 0; offsetW < lengthW; ++offsetW) {
            int64_t srcW = static_cast<int64_t>(dstToSrcTensorW.GetValue(offsetW));

            int64_t indexInput = indexInputBase + srcW * inputC;
            int64_t indexOutput = indexOutputBase + offsetW * inputC;

            for (int64_t offsetC = 0; offsetC < inputC; offsetC += slideSize) {
                int64_t copyLength = Min(slideSize, inputC - offsetC);
                CalcTensorsC(indexInput, indexOutput, copyLength);
                indexInput += copyLength;
                indexOutput += copyLength;
            }
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::CalcTensorsC(
    int64_t indexInput, int64_t indexOutput, int64_t calcCount)
{
    int64_t copyLength = Ceil(float(calcCount) / blockSize) * blockSize;
    CopyIn(indexInput, copyLength);

    LocalTensor<T> dstDataLocal = outputQueue.AllocTensor<T>();

    Duplicate<T>(dstDataLocal, T(0), copyLength);

    LocalTensor<T> srcDataLocal = inputQueue.DeQue<T>();
    Add(dstDataLocal, dstDataLocal, srcDataLocal, calcCount);

    PipeBarrier<PIPE_V>();

    outputQueue.EnQue(dstDataLocal);

    inputQueue.FreeTensor(srcDataLocal);
    CopyOut(indexOutput, calcCount, copyLength);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::CalcIdxTensor(
    int64_t dstStartIndex, int64_t length, uint8_t direction)
{
    LocalTensor<float> centerTensor = centerQueueW.Get<float>();
    LocalTensor<float> dstToSrcTensor = dstToSrcQueueW.Get<float>();

    float scale = scaleW;
    float maxValue = static_cast<float>(inputW) - (float)1.0;

    if (direction == H_DIRECTION) {
        centerTensor = centerQueueH.Get<float>();
        dstToSrcTensor = dstToSrcQueueH.Get<float>();
        scale = scaleH;
        maxValue = static_cast<float>(inputH) - (float)1.0;
    }

    ArithProgression<float>(
        centerTensor, static_cast<float>(dstStartIndex), static_cast<float>(1), static_cast<int32_t>(length));
    PipeBarrier<PIPE_V>();

    if (exactMode) {
        Adds(centerTensor, centerTensor, (float)0.5, length);
        PipeBarrier<PIPE_V>();
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(centerTensor, centerTensor, scale, length);
        PipeBarrier<PIPE_V>();
    }

    Floor(dstToSrcTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();

    Mins(dstToSrcTensor, dstToSrcTensor, maxValue, length);
    PipeBarrier<PIPE_V>();
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::CopyIn(int64_t indexInput, int64_t calcCount)
{
    LocalTensor<T> srcDataLocal = inputQueue.AllocTensor<T>();

    DataCopy(srcDataLocal, inTensorsGM[indexInput], calcCount);

    inputQueue.EnQue(srcDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::CopyOut(
    int64_t indexOutput, int64_t calcCount, int64_t copyLength)
{
    LocalTensor<T> dstDataLocal = outputQueue.DeQue<T>();

    if (calcCount == copyLength) {
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, calcCount);
    } else {
        SetAtomicAdd<T>();
        DataCopy(outTensorsGM[indexOutput], dstDataLocal, copyLength);
        SetAtomicNone();
    }

    outputQueue.FreeTensor(dstDataLocal);
}

template <typename T, int32_t MODE>
__aicore__ inline void UpsampleNearestND310p<T, MODE>::ParseTilingData(const UpsampleNearestTilingData *tilingData)
{
    scaleH = tilingData->scaleH;
    scaleW = tilingData->scaleW;
    exactMode = tilingData->exactMode;

    needCoreNum = tilingData->needCoreNum;

    inputN = tilingData->inputShapes[0];
    inputH = tilingData->inputShapes[1];
    inputW = tilingData->inputShapes[2];
    inputC = tilingData->inputShapes[3];
    outputH = tilingData->outputShapes[1];
    outputW = tilingData->outputShapes[2];

    inputHBatchOffset = inputW * inputC;
    inputNBatchOffset = inputH * inputHBatchOffset;
    outputHBatchOffset = outputW * inputC;
    outputNBatchOffset = outputH * outputHBatchOffset;

    tailRowStart = tilingData->tailRowStartList[blockIdx];
    tailRowEnd = tilingData->tailRowEndList[blockIdx];
    tailColStart = tilingData->tailColStartList[blockIdx];
    tailColEnd = tilingData->tailColEndList[blockIdx];

    dataTypeSize = sizeof(T);

    blockSize = COPY_BLOCK / dataTypeSize;
}
}  // namespace UpsampleNearest
#endif